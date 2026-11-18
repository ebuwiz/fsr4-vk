// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 fsr4-vk contributors.
// See LICENSE in the project root.

// One physical device probed once: identity, extensions, features and the
// cooperative matrix shapes the shipped model shaders are built around. Every
// query but the tile list is physical-device level and core, so no handle beyond
// the device is needed. The tile list belongs to VK_KHR_cooperative_matrix, which
// a loader is free not to export an entry point for, and vkGetDeviceProcAddr
// excludes physical-device-level commands by spec, so a host that wants its tiles
// confirmed has to hand over the instance that created the device. A query that
// could not be reached leaves that field unprobed rather than looking unsupported;
// only a probe that answered can rule a model out on shape grounds.
//
// The provider consults this snapshot before it is allowed to select a model,
// so an unsupported FP8 request fails with a reason instead of a shader compile
// error deep inside the runtime.

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <vulkan/vulkan.h>

#include "resources.hpp"

namespace fsr4vk {

// Cooperative matrix tile shared by every FP8 model shader variant.
constexpr uint32_t kModelMatrixExtent = 16;

// Device functions the runtime resolves through the loader's device. Group
// entries are alternatives, so "vkCmdPipelineBarrier2|vkCmdPipelineBarrier2KHR"
// accepts either spelling.
constexpr const char* kRequiredEntryPointGroups[] = {
    "vkGetDescriptorEXT",
    "vkCmdBindDescriptorBuffersEXT",
    "vkCmdSetDescriptorBufferOffsetsEXT",
    "vkGetDescriptorSetLayoutSizeEXT",
    "vkGetDescriptorSetLayoutBindingOffsetEXT",
    "vkCmdPipelineBarrier2|vkCmdPipelineBarrier2KHR",
    "vkCmdWriteTimestamp2|vkCmdWriteTimestamp2KHR",
    "vkGetBufferDeviceAddress|vkGetBufferDeviceAddressKHR",
};

// Model request, before any device is consulted.
enum class ModelChoice : std::uint8_t { Auto, ForceInt8, ForceFp8 };

// Which handle can answer the cooperative matrix tile query for a device. The
// instance has to be the one that created that physical device; handing this pair
// over is what lets a host confirm tiles on a loader such as the Mesa one, which
// exports no extension entry points of its own.
struct InstanceResolver {
    VkInstance instance = VK_NULL_HANDLE;
    PFN_vkGetInstanceProcAddr get_instance_proc_addr = nullptr;
};

// How the tile list got its answers. Only Unprobed may never be read as
// "this device has no tiles".
enum class MatrixProbeRoute : std::uint8_t { Unprobed, DynamicLookup, HostInstance };

// FSR4_VK_MODEL values. Auto keeps the historical INT8 default and must never
// probe for FP8 support; only an explicit request is allowed to gate on it.
std::optional<ModelChoice> parseModelChoice(const char* value);

struct GpuCapabilities {
    // Identity ------------------------------------------------------------
    VkPhysicalDeviceProperties properties{};
    VkPhysicalDeviceIDProperties identity{};
    VkPhysicalDeviceDriverProperties driver{};

    // Properties ----------------------------------------------------------
    // The memory-type table cannot change while the device handle is alive, so
    // it belongs to the snapshot: the buffer and image allocation paths read this
    // copy instead of asking the driver once per allocation. vkGetPhysicalDevice
    // MemoryProperties is core, so unlike the cooperative matrix tiles it is always
    // reachable from the device handle alone.
    VkPhysicalDeviceMemoryProperties memory{};
    VkPhysicalDeviceSubgroupProperties subgroups{};
    VkPhysicalDeviceSubgroupSizeControlProperties subgroup_sizes{};
    VkPhysicalDeviceFloatControlsProperties float_controls{};
    VkPhysicalDeviceCooperativeMatrixPropertiesKHR cooperative{};

    std::vector<VkExtensionProperties> extensions;

    // Features -----------------------------------------------------------
    VkPhysicalDeviceFeatures core{};
    VkPhysicalDeviceShaderFloat16Int8Features float16_int8{};
    VkPhysicalDevice8BitStorageFeatures storage8{};
    VkPhysicalDeviceVulkanMemoryModelFeatures memory_model{};
    VkPhysicalDeviceSubgroupSizeControlFeatures subgroup_size_control{};
    VkPhysicalDeviceBufferDeviceAddressFeatures buffer_device_address{};
    VkPhysicalDeviceDescriptorIndexingFeatures descriptor_indexing{};
    VkPhysicalDeviceDescriptorBufferFeaturesEXT descriptor_buffer{};
    VkPhysicalDeviceShaderFloat8FeaturesEXT shader_float8{};
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR cooperative_matrix{};

    // Cooperative matrix shapes offered for compute workloads. Empty is
    // ambiguous on its own, so matrix_probe_route records whether a query ran
    // and which handle answered it. tools/cap-probe.cpp reaches that list the
    // same way a cooperating host does: through its own instance resolver.
    std::vector<VkCooperativeMatrixPropertiesKHR> matrix_shapes;
    MatrixProbeRoute matrix_probe_route = MatrixProbeRoute::Unprobed;

    bool hasExtension(const char* wanted) const;

    // Whether a tile query actually answered for this device. An unprobed
    // list is unknown, never a verdict.
    bool matrixShapesProbed() const;

    // Name of the route that answered, for the [capabilities] log line.
    const char* matrixProbeRouteName() const;

    // The tile list as the log lines print it: a count, or "unprobed".
    std::string matrixShapeSummary() const;

    // Feature chain and subgroup contract the runtime always needs.
    bool supportsRuntimeContract() const;

    // Cooperative matrix tiles the FP8 model shaders are built around.
    bool supportsModelMatrixShapes() const;

    // Whether the device accepts a pinned stage subgroup size, which the
    // runtime needs before it forces a wave32 pipeline.
    bool supportsRequiredSubgroupSize(std::uint32_t wanted) const;

    // Whether the reported subgroup range can hold the 32 lanes the shipped
    // shaders pin. Unknown ranges answer yes: only a probe can refuse.
    bool supportsWave32Pipelines() const;

    // Full INT8 and FP8 model contracts.
    bool supportsInt8Model() const;
    bool supportsFp8Model() const;

    // Human readable reasons for rejecting either model.
    std::vector<std::string> int8Blockers() const;
    std::vector<std::string> fp8Blockers() const;

    // Vendor profiles, so shader packaging decisions can stop guessing.
    bool amdProprietaryDriver() const;
    bool rdna2Device() const;

    std::string describe() const;

    bool hasMatrixShape(VkComponentTypeKHR input_type) const;

private:
    const char* matrixInputName(VkComponentTypeKHR type) const;
    std::string missingMatrixShapeName(VkComponentTypeKHR type) const;
    std::string wave32BlockerText() const;
};

std::string joinBlockers(const std::vector<std::string>& blockers);

// Queries one physical device. Everything but the cooperative matrix tile list
// takes the device itself, so the provider can read it from the handle the loader
// hands over. The resolver is optional: without an instance the tile list stays
// unprobed on loaders that do not export the extension's entry point, which is
// the case for the Mesa Vulkan loader.
GpuCapabilities detect(VkPhysicalDevice physical_device, InstanceResolver resolver = {});

// The same device probed again answers from the snapshot rather than repeating
// every query: that is how the provider choosing a model and the runtime choosing
// shader packaging are guaranteed to agree about the hardware. The snapshot
// arrives by value, so no caller can hold a reference into the bounded cache,
// which evicts the oldest device. A cached snapshot whose tiles were never
// probed gets re-queried once a caller supplies a usable resolver, which is what
// lets the runtime - it has no instance of its own - still read the provider's
// answers. A null handle yields no value; a harness that creates its own devices
// calls clearCapabilityCache() when it is done with them.
std::optional<GpuCapabilities> detectCached(VkPhysicalDevice physical_device,
                                            InstanceResolver resolver = {});

void clearCapabilityCache();

// Entry points missing from the device the loader handed over.
std::vector<std::string> missingEntryPoints(VkDevice device,
                                            PFN_vkGetDeviceProcAddr get_device_proc_addr);

const char* modelName(ModelChoice choice);

} // namespace fsr4vk
