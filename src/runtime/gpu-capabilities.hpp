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

// Device functions the runtime resolves through the loader's device live in
// fsr4vk::kRequiredEntryPointGroups, which sits next to the resolver in
// device-dispatch.hpp: the same rows have to satisfy the contract check below
// and be bound by DeviceDispatch::load, and they used to be two lists that
// nothing kept in step. A group is satisfied when either spelling resolves.

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

// FSR4_VK_MODEL values. Auto keeps INT8 as its first rung, so a deployment that
// runs INT8 today keeps running it; only an explicit request is answered strictly,
// never swapped for another model behind the caller's back (see resolveModel).
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
    // Descriptor byte sizes the runtime's descriptor sets are built from. They
    // ride in the same properties chain as the rest, and a host without VK_EXT
    // descriptor_buffer leaves them zeroed - which is caught by the feature bit
    // below rather than by sizing sets of zero bytes.
    VkPhysicalDeviceDescriptorBufferPropertiesEXT descriptor_buffer_properties{};

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

    // Two further links in the same chain, added because the shipped SPIR-V modules
    // declare the capabilities and nothing in the runtime was checking them. At this
    // header revision (VK_HEADER_VERSION 357, the Vulkan 1.4 spec era) integer dot
    // product is a Vulkan 1.3 member rather than a 1.2 one, and compute shader
    // derivatives are core: the KHR struct carries only the derivative-group
    // controls left as feature bits. Float controls have no feature bit anywhere
    // in these headers - the extension is properties-only - so they gate on API
    // level instead.
    VkPhysicalDeviceVulkan13Features vulkan13{};
    VkPhysicalDeviceComputeShaderDerivativesFeaturesKHR compute_derivatives{};

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

    // The five descriptor byte sizes the runtime's sets are built from, in the
    // order createDescriptorLayouts consumes them.
    std::string descriptorSizeSummary() const;

    // Feature chain and subgroup contract the runtime needs whichever transport
    // carries its descriptors. Everything here is independent of how descriptors
    // reach a shader, including the runtime array of descriptors the converted
    // shaders are built on.
    bool supportsSharedContract() const;

    // The VK_EXT_descriptor_buffer transport: today's path, where descriptors are
    // bytes inside a bound buffer.
    bool supportsDescriptorBuffers() const;

    // Conventional descriptor sets: the fallback, and the reason a host without
    // the extension is not automatically unusable. It asks for nothing beyond
    // the shared contract.
    bool supportsDescriptorSets() const;

    // Whether either transport works here at all, which is what the [capabilities]
    // line has always called "runtime".
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

    // The contract that is not model-specific: transports, feature chain and
    // subgroup width. Both model lists start from this rather than from each
    // other, because the INT8 model's own SPIR-V requirements - integer dot
    // products and compute shader derivatives - are not FP8's, and a shared base
    // that carried them would have refused FP8 for capabilities its modules never
    // declare.
    std::vector<std::string> runtimeBlockers() const;

    // The INT8 model's own SPIR-V capability lines, kept apart from the shared
    // base because the FP8 modules declare neither integer dot products nor compute
    // shader derivatives.
    std::vector<std::string> int8ModelBlockers() const;
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

// Whether this device can read back a storage image in that format without the
// app having requested shaderStorageImageReadWithoutFormat, which decides whether
// the overlay is allowed to blend against the app's own output image. The version
// and extension facts come from the snapshot, so this makes only the per-format
// query that the decision actually needs.
bool canReadStorageImage(VkPhysicalDevice physical_device,
                         const GpuCapabilities& capabilities,
                         VkFormat format);

// Entry points missing from the device the loader handed over, for one
// transport. The provider chooses the transport from the same cached snapshot
// it chooses a model with, so the contract it validates is the contract the
// runtime is about to use, and rows belonging to the other transport are not
// held against a device that would never call them.
std::vector<std::string> missingEntryPoints(VkDevice device,
                                            PFN_vkGetDeviceProcAddr get_device_proc_addr,
                                            DescriptorBackend backend);

// Which transport this device can run. The extension path wins whenever the
// device reports it, because it is the cheaper of the two and the one every
// shipped asset was validated under; the sets path covers hosts that never
// implemented the extension.
DescriptorBackend selectDescriptorBackend(const GpuCapabilities& capabilities);

// FSR4_DESCRIPTOR_BACKEND, so both transports can be exercised on a device that
// supports both. An unrecognised value answers nothing rather than quietly
// picking a transport: the caller logs the refusal and falls back to the
// capability decision.
std::optional<DescriptorBackend> descriptorBackendOverride();

const char* modelName(ModelChoice choice);

// What the provider will run on one device, and why. INT8 is the first rung of
// the ladder because it is the cheaper contract and the one every deployment
// already runs; FP8 arrives only when the device *refuses* INT8's own SPIR-V
// capabilities (integer dot product, linear derivative groups), not because it
// looked faster. Refusing both is reported with both blocker lists, since
// "cannot run INT8" and "cannot run either" are different diagnoses and the
// second one is what a low-entry AMD APU tends to look like. Neither list is a
// driver query: both read the snapshot the provider is already holding, so
// walking the ladder costs nothing an explicit request would not have paid.
struct ModelResolution {
    std::optional<ModelChoice> chosen;
    std::string reason;
};

ModelResolution resolveModel(const GpuCapabilities& capabilities, ModelChoice requested);

} // namespace fsr4vk
