// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 fsr4-vk contributors.
// See LICENSE in the project root.

// AMD FidelityFX Frame Generation 4.0.2.4555 Linux Vulkan provider.
//
// Independent of the FSR 4.1.2 upscaler provider: different component, different
// version id, different shader/asset family, no shared runtime resources and no
// shared algorithm logic. The two .so/.dll outputs may be loaded by the same host
// because the FFX "modular API" gives every effect DLL the same five entry points
// (ffxCreateContext/DestroyContext/Configure/Query/Dispatch), verified against the
// pinned Frame Generation DLL's own export table. The pNext descriptor *type*
// discriminates which effect a call belongs to, so the providers do not collide.

#pragma once

#if !defined(_WIN32)
#define __declspec(x) __attribute__((visibility("default")))
#endif
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#include "ffx_api.h"
#include "ffx_api_types.h"
#include "ffx_frame_generation.h"

#include <cstdint>
#include <string>
#include <vector>
#include <vulkan/vulkan.h>

namespace ffgvk {

// Descriptor type ids shared by every effect DLL in the modular FFX API.
constexpr ffxStructType_t kBackendVkDescType = 3u;         // VkDevice backend block
constexpr ffxStructType_t kBackendVkInstanceDescType = 4u; // optional instance block
constexpr ffxStructType_t kOverrideVersionDescType = 5u;   // version override block
constexpr ffxStructType_t kGetProviderVersionQueryType = 6u; // provider-version query
constexpr ffxStructType_t kEnabledFeaturesDescType = 7u; // host-declared enabled device features

// Corrected to the SDK's own Frame Generation descriptor type ids (see vendored
// ffx_frame_generation.h). The previous pair mislabeled them: 0x00020000 is the
// Frame Generation EFFECT id, and 0x00020001 is the create-desc type — so a real
// host's ffxCreateContext(ffxCreateContextDescFrameGeneration) (type 0x00020001)
// was being refused as NO_PROVIDER, and a generate dispatch (0x00020003) was
// never recognized. These are the ids the loader and host actually pass.
constexpr ffxStructType_t kEffectId = FFX_API_EFFECT_ID_FRAMEGENERATION;                        // 0x00020000
constexpr ffxStructType_t kCreateDescType = FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATION;    // 0x00020001
constexpr ffxStructType_t kConfigureDescType = FFX_API_CONFIGURE_DESC_TYPE_FRAMEGENERATION;      // 0x00020002
constexpr ffxStructType_t kDispatchGenerateType = FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION;    // 0x00020003
constexpr ffxStructType_t kDispatchPrepareType =
    FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION_PREPARE;  // 0x00020004
constexpr ffxStructType_t kPrepareCameraInfoType =
    FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION_PREPARE_CAMERAINFO;  // 0x0002000a
constexpr ffxStructType_t kQueryGPUUsageType =
    FFX_API_QUERY_DESC_TYPE_FRAMEGENERATION_GPU_MEMORY_USAGE;  // 0x00020007
// Optional create/configure blocks the vendored header defines; handled at the host boundary
// (recognized + validated + stored) instead of silently forwarded past, while the compute
// algorithm is still refused.
constexpr ffxStructType_t kConfigureKeyValueDescType =
    FFX_API_CONFIGURE_DESC_TYPE_FRAMEGENERATION_KEYVALUE;  // 0x00020006
constexpr ffxStructType_t kRegisterDistortionDescType =
    FFX_API_CONFIGURE_DESC_TYPE_FRAMEGENERATION_REGISTERDISTORTIONRESOURCE;  // 0x00020008
constexpr ffxStructType_t kHudlessCreateDescType =
    FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATION_HUDLESS;  // 0x00020009

// The version id the pinned Frame Generation DLL itself publishes. Measured: a
// single 8-byte FFX-packed id (low half 4.0.2 == (4<<22)|(0<<12)|2 == 0x01000002,
// high/module half 0xF6000000) with the string "4.0.2" beside it. It commits to no
// algorithm model family; the module half is the Frame Generation effect number.
inline constexpr std::uint64_t kReferenceVersionId = 0xF600000001000002ULL;
constexpr const char* kComponentName = "4.0.2 Frame Generation (Vulkan)";

// The Vulkan backend block a host attaches to ffxCreateContext. The field layout
// is the modular FFX API's Vulkan backend descriptor (not an FSR-specific shape):
// a caller-owned VkDevice, its physical device, and the device-proc-addr resolver
// for that device. Every feature, extension and entry point the runtime may use
// has to be reachable through this block or enumerated from this device.
struct CreateBackendVkDesc {
    ffxApiHeader header;
    VkDevice device;
    VkPhysicalDevice physicalDevice;
    PFN_vkGetDeviceProcAddr getDeviceProcAddr;
};

// Optional instance block on the same pNext chain (FFX_API_DESC_TYPE_OVERRIDE
// is 5u and the effect descs are above these, so 4u is free). Only used to reach
// cooperative-matrix shape queries when a loader exports no extension entry point.
struct CreateBackendVkInstanceDesc {
    ffxApiHeader header;
    VkInstance instance;
    PFN_vkGetInstanceProcAddr getInstanceProcAddr;
};

struct CreateVersionOverride {
    ffxApiHeader header;
    std::uint64_t versionId;
};

// Vulkan has no query for the feature set used to create an existing VkDevice.
// Hosts attach this one custom pNext block and copy the exact bits they passed to
// VkDeviceCreateInfo. The provider can then reject missing features honestly
// instead of mistaking physical-device support for enabled functionality.
struct CreateBackendVkEnabledFeaturesDesc {
    ffxApiHeader header;
    VkBool32 shaderInt16 = VK_FALSE;
    VkBool32 shaderInt64 = VK_FALSE;
    VkBool32 shaderFloat16 = VK_FALSE;
    VkBool32 shaderInt8 = VK_FALSE;
    VkBool32 storage8Bit = VK_FALSE;
    VkBool32 storage16Bit = VK_FALSE;
    VkBool32 bufferDeviceAddress = VK_FALSE;
    VkBool32 runtimeDescriptorArray = VK_FALSE;
    VkBool32 cooperativeMatrix = VK_FALSE;
    VkBool32 vulkanMemoryModel = VK_FALSE;
    VkBool32 subgroupSizeControl = VK_FALSE;
    VkBool32 computeDerivativeGroupLinear = VK_FALSE;
    VkBool32 synchronization2 = VK_FALSE;
    VkBool32 storageImageWriteWithoutFormat = VK_FALSE;
};

// Resolve the backend, instance and override blocks out of one create/configure
// chain. Unknown pNext blocks are skipped, never rejected: the FFX API forwards
// unrecognized extensions, and refusing a block this provider does not model would
// break hosts that attach newer optional blocks.
const CreateBackendVkDesc* findBackend(const ffxApiHeader* header);
const CreateBackendVkInstanceDesc* findBackendInstance(const ffxApiHeader* header);
const CreateVersionOverride* findVersionOverride(const ffxApiHeader* header);
const CreateBackendVkEnabledFeaturesDesc* findEnabledFeatures(const ffxApiHeader* header);

// One physical/VK device probed through the caller-owned device only.
struct GpuCapabilities {
    VkDevice device = VK_NULL_HANDLE;
    std::string deviceName;
    std::string driverName;
    std::uint32_t vendorId = 0;
    std::uint32_t deviceId = 0;
    std::uint32_t apiVersion = 0;
    std::uint32_t driverVersion = 0;

    // Enabled-on-the-device feature facts the Frame Generation execution
    // contract reads (never inferred from headers or marketing specs).
    bool shaderFloat16 = false;
    bool shaderInt8 = false;
    bool shaderInt16 = false;
    bool shaderInt64 = false;
    bool shaderInt64Atomics = false;
    bool storage8Bit = false;
    bool storage16Bit = false;
    bool bufferDeviceAddress = false;
    bool descriptorIndexing = false;
    bool runtimeDescriptorArray = false;
    bool storageImageDescriptorWithoutFormat = false;
    bool cooperativeMatrix = false;
    bool vulkanMemoryModel = false;
    bool subgroupSizeControl = false;
    bool computeDerivativeGroupLinear = false;
    bool synchronization2 = false;
    bool storageImageWriteWithoutFormat = false;

    VkSubgroupFeatureFlags subgroupOperations = static_cast<VkSubgroupFeatureFlags>(0);
    std::uint32_t minSubgroupSize = 0;
    std::uint32_t maxSubgroupSize = 0;
    std::uint32_t maxWorkGroupSize = 0;
    std::vector<std::string> enabledExtensions;
};

// One VK device probed through the caller-owned device handle only (device-level
// entry points resolved from CreateBackendVkDesc::getDeviceProcAddr). The result
// reports the features ENABLED on that device, which is the only set the runtime
// may use; nothing is inferred from the SDK headers or the Vulkan spec.
GpuCapabilities queryCapabilities(const CreateBackendVkDesc& backend,
                                  const CreateBackendVkEnabledFeaturesDesc* enabled);

// The Frame Generation execution contract, derived from the measured capability
// manifest (build/fg-capabilities/execution-contract.json): the enabled feature
// set every shipped Frame Generation module needs. Empty means this device can
// run the contract; each string is a precise blocker naming the missing feature
// and how many modules demand it, so context creation can refuse before any
// pipeline creation would fail deeper down.
std::vector<std::string> executionContractBlockers(const GpuCapabilities& capabilities);

} // namespace ffgvk
