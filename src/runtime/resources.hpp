// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 fsr4-vk contributors.
// See LICENSE in the project root.

#pragma once

#include "assets/assets.hpp"
#include "runtime/device-dispatch.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <vulkan/vulkan.h>

namespace fsr4 {
inline uint32_t roundUp(uint32_t value, uint32_t multiple) {
    return (value + multiple - 1) / multiple * multiple;
}

inline uint32_t divideUp(uint32_t value, uint32_t divisor) {
    return (value + divisor - 1) / divisor;
}

inline float sanitizePreExposure(float value) {
    return std::isfinite(value) && value > 0.0f ? value : 1.0f;
}

inline const char* resolutionPreset(uint32_t input_width, uint32_t output_width) {
    const float ratio = float(roundUp(output_width, 8)) / input_width;
    if (ratio >= 2.99f) {
        return "ultraperf";
    }
    if (ratio >= 1.99f) {
        return "performance";
    }
    if (ratio >= 1.69f) {
        return "balanced";
    }
    // Allow rounded Quality sizes, such as 900 input pixels for 1344 output pixels.
    if (ratio >= 1.49f) {
        return "quality";
    }
    return "native";
}

inline std::array<uint32_t, 2> modelBucket(uint32_t width, uint32_t height) {
    if (width > 3840 || height > 2160) {
        return {7680, 4320};
    }
    return width > 1920 || height > 1080 ? std::array<uint32_t, 2>{3840, 2160}
                                         : std::array<uint32_t, 2>{1920, 1080};
}

inline const char* resolutionBucket(uint32_t width, uint32_t height) {
    if (width > 3840 || height > 2160) {
        return "4320";
    }
    return width > 1920 || height > 1080 ? "2160" : "1080";
}

inline uint64_t scratchCapacity(uint32_t width, uint32_t height) {
    // Fixed tensor layouts from the SDK's three resolution tiers.
    if (width > 3840 || height > 2160) {
        return 332352256ull;
    }
    return width > 1920 || height > 1080 ? 83232256ull : 20880256ull;
}

inline void validateDimensions(uint32_t input_width,
                               uint32_t input_height,
                               uint32_t output_width,
                               uint32_t output_height) {
    const bool dimensions_too_small =
        input_width < 64 || input_height < 64 || output_width < 64 || output_height < 64;
    const bool output_exceeds_capacity = output_width > 7680 || output_height > 4320;
    const bool input_exceeds_output = input_width > output_width || input_height > output_height;

    if (dimensions_too_small || output_exceeds_capacity || input_exceeds_output) {
        throw std::invalid_argument(
            "supported dimension bounds: 64..7680 x 64..4320, input <= output");
    }
}

} // namespace fsr4

namespace fsr4vk {

inline constexpr std::uint32_t kAmdVendorId = 0x1002;

inline constexpr bool isAmdvlkOrProprietaryDriver(std::uint32_t vendor_id, VkDriverId driver_id) {
    return vendor_id == kAmdVendorId &&
           (driver_id == VK_DRIVER_ID_AMD_PROPRIETARY || driver_id == VK_DRIVER_ID_AMD_OPEN_SOURCE);
}

inline constexpr bool allowWindowsAmdShaders(std::uint32_t vendor_id,
                                             VkDriverId driver_id,
                                             bool force,
                                             bool assets_available = true) {
    return assets_available && (force || isAmdvlkOrProprietaryDriver(vendor_id, driver_id));
}

// Identify RDNA2 GPUs and APUs by PCI device ID.
//
// This predicate is RDNA2-only *by design*, and the exclusions are deliberate and
// asserted: RDNA1 (Navi 10, 0x731f), RDNA3 (Navi 31, 0x744c) and RDNA4 (this
// machine's RX 9070 XT, deviceID 0x7550, which RADV names "(RADV GFX1201)") are
// all "not RDNA2". The reason is narrow: the verdict only feeds the wave32 pin and
// the AMD shader-pack allowance, and "wave32 was slightly faster" was measured on
// RDNA2 silicon, where the native wavefront is 32 lanes. RDNA3 and RDNA4 can run
// 64-lane wavefronts, so pinning 32 there is a performance decision this project
// has no measurement for - and the RX 9070 XT here shows exactly that shape:
// default subgroup 64, range 32-64, rdna2=no. Nothing else is generation-blind:
// allowWindowsAmdShaders keys on the *driver*, so an RDNA3 or RDNA4 card on the
// AMD proprietary driver still gets the AMD-optimised shader packs.
inline constexpr bool isRdna2DeviceId(std::uint32_t device_id) {
    return (device_id >= 0x73a0 && device_id <= 0x73ff) ||
           (device_id >= 0x7420 && device_id <= 0x743f) ||
           device_id == 0x1506 || // Mendocino (GFX10.3.7)
           device_id == 0x163f || // Van Gogh
           device_id == 0x164e || // Raphael (GFX10.3.6)
           device_id == 0x1681;   // Rembrandt / Yellow Carp
}

inline constexpr bool isAmdRdna2Device(std::uint32_t vendor_id, std::uint32_t device_id) {
    return vendor_id == kAmdVendorId && isRdna2DeviceId(device_id);
}

static_assert(isAmdRdna2Device(kAmdVendorId, 0x73bf));  // Navi 21
static_assert(isAmdRdna2Device(kAmdVendorId, 0x73df));  // Navi 22
static_assert(isAmdRdna2Device(kAmdVendorId, 0x73ff));  // Navi 23
static_assert(isAmdRdna2Device(kAmdVendorId, 0x7422));  // Navi 24
static_assert(!isAmdRdna2Device(kAmdVendorId, 0x731f)); // Navi 10 (RDNA1)
static_assert(!isAmdRdna2Device(kAmdVendorId, 0x744c)); // Navi 31 (RDNA3)
// The RDNA4 card this project has measured, straight off tools/cap-probe.cpp:
// device="AMD Radeon RX 9070 XT (RADV GFX1201)" device_id=0x7550 type=discrete
// subgroup_size=64 subgroup_range=32-64 -> rdna2=no.
static_assert(!isAmdRdna2Device(kAmdVendorId, 0x7550)); // RX 9070 XT (GFX1201, RDNA4)
// Outside the AMD vendor table entirely: the Intel integrated device measured here
// (vendor 0x8086, device_id 0xa780), which must never reach an AMD-specific rule.
static_assert(!isAmdRdna2Device(0x8086, 0x73bf));

// Whether an AMD device reports itself integrated, which is how an APU's graphics
// block arrives (a Ryzen A1..Z2-class part shows up as INTEGRATED_GPU rather than
// as a discrete card). This is a diagnosis for the log line, not a verdict, and
// deliberately not folded into the RDNA2 lookup above: widening that table needs a
// measured device ID, and the identity fields cannot substitute for one. Probing
// it on this machine refuted the obvious route - RADV on the RX 9070 XT (deviceID
// 0x7550) writes a deviceUUID of 00000000 03000000 0000000000000000, which carries
// no device ID at all, while the Intel RPL-S device (vendor 0x8086, deviceID 0xa780)
// writes 868080a7..., i.e. vendor and device IDs as two little-endian pairs. Two
// drivers, two incompatible layouts, and neither of them is an AMD integrated
// device, so no driver-independent rule can name an APU family this table lacks.
inline constexpr bool isAmdIntegratedDevice(std::uint32_t vendor_id,
                                            VkPhysicalDeviceType device_type) {
    return vendor_id == kAmdVendorId &&
           device_type == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU;
}

static_assert(isAmdIntegratedDevice(kAmdVendorId, VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU));
static_assert(!isAmdIntegratedDevice(kAmdVendorId, VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU));
// The Intel integrated device this project has measured: integrated, but not AMD,
// so it is not an AMD APU and must not be folded into an AMD-specific rule.
static_assert(!isAmdIntegratedDevice(0x8086, VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU));
static_assert(isAmdvlkOrProprietaryDriver(kAmdVendorId, VK_DRIVER_ID_AMD_PROPRIETARY));
static_assert(isAmdvlkOrProprietaryDriver(kAmdVendorId, VK_DRIVER_ID_AMD_OPEN_SOURCE));
static_assert(!isAmdvlkOrProprietaryDriver(kAmdVendorId, VK_DRIVER_ID_MESA_RADV));
static_assert(!allowWindowsAmdShaders(kAmdVendorId, VK_DRIVER_ID_MESA_RADV, false));
static_assert(allowWindowsAmdShaders(kAmdVendorId, VK_DRIVER_ID_MESA_RADV, true));
static_assert(!allowWindowsAmdShaders(kAmdVendorId, VK_DRIVER_ID_AMD_PROPRIETARY, true, false));

} // namespace fsr4vk

namespace fsr4_411 {
namespace fs = std::filesystem;

constexpr uint32_t kDescriptorCount = 128;
constexpr uint32_t kNumDescriptorSets = 6;
constexpr VkDeviceSize kConstantBufSize = 4096;

struct Buffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;

    VkDeviceSize size = 0;
    void* mapped = nullptr;

    uint32_t memory_type_index = std::numeric_limits<uint32_t>::max();
    uint32_t memory_heap_index = std::numeric_limits<uint32_t>::max();
    VkMemoryPropertyFlags memory_property_flags = 0;
};

struct Image {
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;

    VkFormat format = VK_FORMAT_UNDEFINED;
    uint32_t width = 0;
    uint32_t height = 0;

    // FFX aliases D32 to R32_FLOAT, so keep the Vulkan aspect separately.
    VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT;

    // Optional backing size; width and height describe the active render region.
    uint32_t resource_width = 0;
    uint32_t resource_height = 0;
};

struct OptimizedConstants {
    float inv_size[2];
    float scale[2];
    float inv_scale[2];
    float jitter[2];
    float mv_scale[2];
    float tex_size[2];
    float max_render_size[2];
    float motion_vector_jitter_cancellation[2];

    uint32_t width;
    uint32_t height;
    uint32_t reset;

    uint32_t width_lr;
    uint32_t height_lr;

    float pre_exposure;
    float previous_pre_exposure;

    uint32_t rcas_enabled;
    float rcas_sharpness;

    float padding;
};

static_assert(sizeof(OptimizedConstants) == 104);

void check(VkResult result, const char* operation);

std::vector<uint8_t> readBytes(const fs::path& path);
std::vector<uint32_t> readSpirv(const fs::path& path);

std::vector<uint32_t> readModelSpirv(const fs::path& path, uint32_t api_version);

// The memory-type table is part of the per-device capability snapshot, so these
// allocation helpers take it by reference instead of a device handle they would
// have to query again on every buffer and image.
uint32_t findMemoryType(const VkPhysicalDeviceMemoryProperties& memory,
                        uint32_t allowed_type_bits,
                        VkMemoryPropertyFlags required_properties,
                        VkMemoryPropertyFlags preferred_properties = 0);

Buffer createBuffer(const VkPhysicalDeviceMemoryProperties& memory,
                    VkDevice device,
                    VkDeviceSize size,
                    VkBufferUsageFlags usage,
                    VkMemoryPropertyFlags properties,
                    bool enable_device_address = false,
                    VkMemoryPropertyFlags preferred_properties = 0);

Image createImage(const VkPhysicalDeviceMemoryProperties& memory,
                  VkDevice device,
                  uint32_t width,
                  uint32_t height,
                  VkFormat format,
                  VkImageUsageFlags usage,
                  VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT);

void destroyBuffer(VkDevice device, Buffer& buffer);

void destroyImage(VkDevice device, Image& image);

VkDeviceAddress deviceAddress(PFN_vkGetBufferDeviceAddress get_buffer_device_address,
                              VkDevice device,
                              VkBuffer buffer);

void imageBarrier(PFN_vkCmdPipelineBarrier2 pipeline_barrier2,
                  VkCommandBuffer command_buffer,
                  VkImage image,
                  VkImageLayout old_layout,
                  VkImageLayout new_layout,
                  VkPipelineStageFlags2 source_stage,
                  VkAccessFlags2 source_access,
                  VkPipelineStageFlags2 destination_stage,
                  VkAccessFlags2 destination_access,
                  VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT);

// Resample motion and restore both images to their sampled layouts.
void recordMotionResample(PFN_vkCmdPipelineBarrier2 pipeline_barrier2,
                          VkCommandBuffer command_buffer,
                          const Image& source,
                          const Image& destination,
                          uint32_t target_width,
                          uint32_t target_height,
                          bool destination_initialized);

void computeBarrier(PFN_vkCmdPipelineBarrier2 pipeline_barrier2, VkCommandBuffer command_buffer);

} // namespace fsr4_411
