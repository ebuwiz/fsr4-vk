// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 fsr4-vk contributors.
// See LICENSE in the project root.

#include "gpu-capabilities.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace fsr4vk {

namespace {

std::string hexValue(std::uint32_t value) {
    std::ostringstream line;
    line << "0x" << std::hex << value;
    return line.str();
}

const char* deviceTypeName(VkPhysicalDeviceType type) {
    switch (type) {
    case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
        return "integrated";
    case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
        return "discrete";
    case VK_PHYSICAL_DEVICE_TYPE_CPU:
        return "cpu";
    case VK_PHYSICAL_DEVICE_TYPE_OTHER:
        return "other";
    default:
        return "unknown";
    }
}

std::string apiVersionName(std::uint32_t version) {
    return std::to_string(VK_API_VERSION_MAJOR(version)) + "." +
           std::to_string(VK_API_VERSION_MINOR(version)) + "." +
           std::to_string(VK_API_VERSION_PATCH(version));
}

// The cooperative matrix shape list belongs to VK_KHR_cooperative_matrix and
// takes the physical device itself (vulkan_core.h:13571). *Reaching* that command
// is the part a host may refuse: a loader is not required to export entry points
// for extensions - the Mesa Vulkan loader exports no CooperativeMatrix symbol at
// all, while its core vkGetPhysicalDevice* set is complete - and vkGetDeviceProcAddr
// excludes every physical-device-level command by spec. So the routes tried here
// are the instance the host supplied first, then the loader's exported symbol
// through dlsym. A route that cannot be reached or that answers with an error is
// skipped, and only a route that returned shapes sets the probe route; a probe
// that never ran leaves the list unprobed rather than reporting "no shapes".
std::vector<VkCooperativeMatrixPropertiesKHR> probeMatrixShapes(
    VkPhysicalDevice physical_device, const InstanceResolver& resolver,
    MatrixProbeRoute& route) {
    using ShapeQuery = PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR;
    if (!physical_device) {
        return {};
    }

    ShapeQuery host_query = nullptr;
    if (resolver.instance && resolver.get_instance_proc_addr) {
        host_query = reinterpret_cast<ShapeQuery>(
            resolver.get_instance_proc_addr(resolver.instance,
                                            "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR"));
    }

    const std::pair<MatrixProbeRoute, ShapeQuery> sources[] = {
        {MatrixProbeRoute::HostInstance, host_query},
        {MatrixProbeRoute::DynamicLookup,
         reinterpret_cast<ShapeQuery>(
             dlsym(RTLD_DEFAULT, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR"))},
    };

    for (const auto& [candidate_route, query] : sources) {
        if (!query) {
            continue;
        }

        uint32_t shape_count = 0;
        if (query(physical_device, &shape_count, nullptr) != VK_SUCCESS) {
            continue;
        }

        std::vector<VkCooperativeMatrixPropertiesKHR> shapes(
            shape_count,
            VkCooperativeMatrixPropertiesKHR{VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR});
        if (query(physical_device, &shape_count, shapes.data()) != VK_SUCCESS) {
            continue;
        }

        shapes.resize(shape_count);
        route = candidate_route;
        return shapes;
    }

    return {};
}

} // namespace

// Reads identity, advertised extensions and the feature chain both models need.
// No query here throws: a failed query leaves its field empty, and the
// predicates below turn that into an explicit rejection.
GpuCapabilities detect(VkPhysicalDevice physical_device, InstanceResolver resolver) {
    GpuCapabilities capabilities;

    // Both of these are core device queries: the identity block drives the
    // vendor profile and the memory-type table drives every allocation the
    // runtime makes, so neither may be left unprobed.
    vkGetPhysicalDeviceProperties(physical_device, &capabilities.properties);
    vkGetPhysicalDeviceMemoryProperties(physical_device, &capabilities.memory);

    capabilities.identity =
        VkPhysicalDeviceIDProperties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
    capabilities.driver =
        VkPhysicalDeviceDriverProperties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
    capabilities.subgroups =
        VkPhysicalDeviceSubgroupProperties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
    capabilities.subgroup_sizes =
        VkPhysicalDeviceSubgroupSizeControlProperties{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES};
    capabilities.float_controls =
        VkPhysicalDeviceFloatControlsProperties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT_CONTROLS_PROPERTIES};
    capabilities.cooperative =
        VkPhysicalDeviceCooperativeMatrixPropertiesKHR{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_PROPERTIES_KHR};
    capabilities.descriptor_buffer_properties =
        VkPhysicalDeviceDescriptorBufferPropertiesEXT{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_BUFFER_PROPERTIES_EXT};

    VkPhysicalDeviceProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    properties.pNext = &capabilities.identity;
    capabilities.identity.pNext = &capabilities.driver;
    capabilities.driver.pNext = &capabilities.subgroups;
    capabilities.subgroups.pNext = &capabilities.subgroup_sizes;
    capabilities.subgroup_sizes.pNext = &capabilities.float_controls;
    capabilities.float_controls.pNext = &capabilities.cooperative;
    capabilities.cooperative.pNext = &capabilities.descriptor_buffer_properties;
    vkGetPhysicalDeviceProperties2(physical_device, &properties);

    uint32_t extension_count = 0;
    if (vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &extension_count, nullptr) ==
        VK_SUCCESS) {
        capabilities.extensions.resize(extension_count);
        if (vkEnumerateDeviceExtensionProperties(physical_device,
                                                 nullptr,
                                                 &extension_count,
                                                 capabilities.extensions.data()) != VK_SUCCESS) {
            capabilities.extensions.clear();
        }
    }

    // One chain answers both contracts. Drivers leave structures of extensions
    // they do not implement untouched, so they read back as unsupported.
    VkPhysicalDeviceShaderFloat8FeaturesEXT shader_float8{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT};
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR cooperative{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
    VkPhysicalDeviceShaderFloat16Int8Features float16{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES};
    VkPhysicalDevice8BitStorageFeatures storage8{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES};
    VkPhysicalDeviceVulkanMemoryModelFeatures memory_model{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_MEMORY_MODEL_FEATURES};
    VkPhysicalDeviceSubgroupSizeControlFeatures subgroup_features{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES};
    VkPhysicalDeviceBufferDeviceAddressFeatures buffer_address{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES};
    VkPhysicalDeviceDescriptorIndexingFeatures descriptor_indexing{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES};
    VkPhysicalDeviceDescriptorBufferFeaturesEXT descriptor_buffer{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_BUFFER_FEATURES_EXT};
    VkPhysicalDeviceVulkan13Features vulkan13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    VkPhysicalDeviceComputeShaderDerivativesFeaturesKHR compute_derivatives{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COMPUTE_SHADER_DERIVATIVES_FEATURES_KHR};
    VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};

    features.pNext = &cooperative;
    cooperative.pNext = &shader_float8;
    shader_float8.pNext = &float16;
    float16.pNext = &storage8;
    storage8.pNext = &memory_model;
    memory_model.pNext = &subgroup_features;
    subgroup_features.pNext = &buffer_address;
    buffer_address.pNext = &descriptor_indexing;
    descriptor_indexing.pNext = &descriptor_buffer;
    descriptor_buffer.pNext = &vulkan13;
    vulkan13.pNext = &compute_derivatives;
    vkGetPhysicalDeviceFeatures2(physical_device, &features);
    capabilities.cooperative_matrix = cooperative;
    capabilities.shader_float8 = shader_float8;
    capabilities.float16_int8 = float16;
    capabilities.storage8 = storage8;
    capabilities.memory_model = memory_model;
    capabilities.subgroup_size_control = subgroup_features;
    capabilities.buffer_device_address = buffer_address;
    capabilities.descriptor_indexing = descriptor_indexing;
    capabilities.descriptor_buffer = descriptor_buffer;
    capabilities.vulkan13 = vulkan13;
    capabilities.compute_derivatives = compute_derivatives;
    capabilities.core = features.features;

    capabilities.matrix_shapes =
        probeMatrixShapes(physical_device, resolver, capabilities.matrix_probe_route);
    return capabilities;
}

// Snapshots are keyed by the handle the caller passes, so the provider choosing
// a model and the runtime choosing shader packaging cannot disagree about what
// the device reported, and the queries run once per device rather than once per
// concern. The cache is bounded, evicts the oldest device and hands out copies,
// so a handle the caller destroyed can never leave a dangling snapshot behind;
// a harness that creates its own devices clears the cache explicitly.
namespace {

constexpr std::size_t kCapabilityCacheLimit = 16;
std::mutex capability_cache_mutex;
std::vector<std::pair<VkPhysicalDevice, GpuCapabilities>> capability_cache;

} // namespace

std::optional<GpuCapabilities> detectCached(VkPhysicalDevice physical_device,
                                            InstanceResolver resolver) {
    if (!physical_device) {
        return {};
    }

    const bool resolver_usable = resolver.instance && resolver.get_instance_proc_addr;
    std::lock_guard<std::mutex> lock(capability_cache_mutex);
    for (auto& entry : capability_cache) {
        if (entry.first != physical_device) {
            continue;
        }

        // A snapshot is kept for the handle, but a tile list nobody could reach
        // is not a fact about the device: when a later caller can supply an
        // instance resolver, that entry is re-probed once so the runtime - which
        // has no instance of its own - still reads the provider's answers.
        if (entry.second.matrix_probe_route == MatrixProbeRoute::Unprobed && resolver_usable) {
            entry.second = detect(physical_device, resolver);
        }

        return entry.second;
    }

    if (capability_cache.size() >= kCapabilityCacheLimit) {
        capability_cache.erase(capability_cache.begin());
    }

    capability_cache.emplace_back(physical_device, detect(physical_device, resolver));
    return capability_cache.back().second;
}

void clearCapabilityCache() {
    std::lock_guard<std::mutex> lock(capability_cache_mutex);
    capability_cache.clear();
}

bool GpuCapabilities::hasExtension(const char* wanted) const {
    return std::any_of(extensions.begin(),
                       extensions.end(),
                       [wanted](const VkExtensionProperties& extension) {
                           return std::strcmp(extension.extensionName, wanted) == 0;
                       });
}

bool GpuCapabilities::hasMatrixShape(VkComponentTypeKHR input_type) const {
    return std::any_of(matrix_shapes.begin(), matrix_shapes.end(), [&](const auto& shape) {
        return shape.scope == VK_SCOPE_SUBGROUP_KHR && shape.MSize == kModelMatrixExtent &&
               shape.NSize == kModelMatrixExtent && shape.KSize == kModelMatrixExtent &&
               shape.AType == input_type && shape.BType == input_type &&
               shape.CType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
               shape.ResultType == VK_COMPONENT_TYPE_FLOAT32_KHR && !shape.saturatingAccumulation;
    });
}

bool GpuCapabilities::supportsModelMatrixShapes() const {
    // An empty list from a host that refused the query means unknown, so the
    // extension and feature bits stay the only signal. Only a probe that
    // answered can confirm the tiles the FP8 shaders are built around.
    if (!matrixShapesProbed()) {
        return false;
    }

    return hasMatrixShape(VK_COMPONENT_TYPE_FLOAT16_KHR) &&
           hasMatrixShape(VK_COMPONENT_TYPE_FLOAT8_E4M3_EXT);
}

const char* GpuCapabilities::matrixInputName(VkComponentTypeKHR type) const {
    switch (type) {
    case VK_COMPONENT_TYPE_FLOAT16_KHR:
        return "float16";
    case VK_COMPONENT_TYPE_FLOAT8_E4M3_EXT:
        return "float8_e4m3";
    case VK_COMPONENT_TYPE_FLOAT8_E5M2_EXT:
        return "float8_e5m2";
    case VK_COMPONENT_TYPE_SINT8_KHR:
    case VK_COMPONENT_TYPE_UINT8_KHR:
        return "int8";
    default:
        return "unknown";
    }
}

std::string GpuCapabilities::missingMatrixShapeName(VkComponentTypeKHR type) const {
    std::ostringstream reason;
    reason << "no subgroup " << kModelMatrixExtent << 'x' << kModelMatrixExtent << 'x'
           << kModelMatrixExtent << " " << matrixInputName(type) << "->float32 matrix shape";
    return reason.str();
}

// Everything but the descriptors themselves: the feature chain, the storage-image
// formats the passes write, the 8-bit and 16-bit building blocks, device addressing
// and the subgroup contract. Both transports need all of it, so it is stated once
// and the transports differ by exactly one question.
bool GpuCapabilities::supportsSharedContract() const {
    return core.shaderInt16 && core.shaderStorageImageReadWithoutFormat &&
           core.shaderStorageImageWriteWithoutFormat && float16_int8.shaderFloat16 &&
           float16_int8.shaderInt8 && storage8.storageBuffer8BitAccess &&
           buffer_device_address.bufferDeviceAddress &&
           descriptor_indexing.runtimeDescriptorArray &&
           (subgroups.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT) &&
           (subgroups.supportedOperations & VK_SUBGROUP_FEATURE_QUAD_BIT);
}

// The transport the shipped assets were validated under: descriptor bytes in a
// buffer the frame binds, which VK_EXT_descriptor_buffer both spells itself as an
// extension name and gates with a feature bit.
bool GpuCapabilities::supportsDescriptorBuffers() const {
    return supportsSharedContract() && hasExtension(VK_EXT_DESCRIPTOR_BUFFER_EXTENSION_NAME) &&
           descriptor_buffer.descriptorBuffer;
}

// The fallback: six sets of runtime arrays bound the ordinary way, which is core
// Vulkan and therefore needs nothing the extension adds. A device can lose the
// extension and still run the whole upscaler.
bool GpuCapabilities::supportsDescriptorSets() const {
    return supportsSharedContract();
}

// Whether either transport runs here, which is what the [capabilities] line calls
// "runtime". A refusal below this line is about a feature, not about a transport.
bool GpuCapabilities::supportsRuntimeContract() const {
    return supportsDescriptorBuffers() || supportsDescriptorSets();
}

std::vector<std::string> GpuCapabilities::runtimeBlockers() const {
    std::vector<std::string> blockers;

    // Neither transport is a feature the model needs: it is how the runtime moves
    // descriptors at all. A device missing VK_EXT_descriptor_buffer is refused only
    // when it also misses the core descriptor indexing that both transports ride on.
    if (!supportsDescriptorBuffers() && !supportsDescriptorSets()) {
        blockers.emplace_back("no descriptor transport: VK_EXT_descriptor_buffer is absent "
                              "and core descriptor indexing is not reported");
    }

    const std::pair<const char*, bool> required[] = {
        {"core.shaderInt16", core.shaderInt16},
        {"core.shaderStorageImageReadWithoutFormat", core.shaderStorageImageReadWithoutFormat},
        {"core.shaderStorageImageWriteWithoutFormat", core.shaderStorageImageWriteWithoutFormat},
        {"shaderFloat16", float16_int8.shaderFloat16},
        {"shaderInt8", float16_int8.shaderInt8},
        {"storageBuffer8BitAccess", storage8.storageBuffer8BitAccess},
        {"bufferDeviceAddress", buffer_device_address.bufferDeviceAddress},
        {"runtimeDescriptorArray", descriptor_indexing.runtimeDescriptorArray},
    };

    for (const auto& [name, enabled] : required) {
        if (!enabled) {
            blockers.emplace_back(std::string("feature ") + name + " is not reported");
        }
    }

    if (!(subgroups.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT)) {
        blockers.emplace_back("subgroups are not supported in compute");
    }
    if (!(subgroups.supportedOperations & VK_SUBGROUP_FEATURE_QUAD_BIT)) {
        blockers.emplace_back("quad subgroup operations are missing");
    }

    // Both models' modules declare SPV_KHR_float_controls (189 of the INT8 set,
    // 138 of the FP8 set). Float controls are properties-only: this header revision
    // has no feature bit for them, so the API level is the honest gate here, and
    // the float_controls properties the runtime reads for denormal behaviour come
    // off the same properties chain this snapshot was taken from.
    if (properties.apiVersion < VK_API_VERSION_1_2) {
        blockers.emplace_back("device API version is below 1.2, where float controls arrive");
    }

    // The passes around the model itself pin 32-lane subgroups in every preset,
    // so a reported range that excludes 32 rules the runtime out entirely. A
    // range the probe never reported is unknown, not a reason to refuse.
    if (!supportsWave32Pipelines()) {
        blockers.emplace_back(wave32BlockerText());
    }

    return blockers;
}

// The INT8 model's own capability lines, from the census of its 340 shipped
// modules: 168 declare SPV_KHR_integer_dot_product, and 93 declare
// SPV_KHR_compute_shader_derivatives - all 93 with "OpExecutionMode %main
// DerivativeGroupLinearKHR", so the feature bit that matters at this header
// revision is the linear derivative group rather than the base capability, which
// the Vulkan 1.4 era made core. No FP8 module declares either, which is why these
// lines sit apart from the shared base: a module asking for a capability the device
// never reported is a driver-side compile failure with no useful message, and that
// is what these two lines exist to replace.
std::vector<std::string> GpuCapabilities::int8ModelBlockers() const {
    std::vector<std::string> blockers;

    if (!vulkan13.shaderIntegerDotProduct) {
        blockers.emplace_back("feature shaderIntegerDotProduct is not reported");
    }
    if (!compute_derivatives.computeDerivativeGroupLinear) {
        blockers.emplace_back("feature computeDerivativeGroupLinear is not reported");
    }
    return blockers;
}

std::vector<std::string> GpuCapabilities::int8Blockers() const {
    auto blockers = runtimeBlockers();
    const auto model_blockers = int8ModelBlockers();
    blockers.insert(blockers.end(), model_blockers.begin(), model_blockers.end());
    return blockers;
}

// The FP8 gate mirrors what the shipped model shaders compile against: the
// shader_float8 and cooperative_matrix extensions, subgroups wide enough to
// require 32 lanes explicitly, and one 16x16x16 tile per input type that
// accumulates without saturation into float32. It starts from the shared
// runtime contract rather than from the INT8 list, because the two models
// declare different SPIR-V capabilities and neither should inherit the other's.
std::vector<std::string> GpuCapabilities::fp8Blockers() const {
    auto blockers = runtimeBlockers();

    if (!hasExtension(VK_EXT_SHADER_FLOAT8_EXTENSION_NAME)) {
        blockers.emplace_back(std::string("missing extension ") +
                              VK_EXT_SHADER_FLOAT8_EXTENSION_NAME);
    }
    if (!hasExtension(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME)) {
        blockers.emplace_back(std::string("missing extension ") +
                              VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
    }

    const std::pair<const char*, bool> required[] = {
        {"shaderFloat8", shader_float8.shaderFloat8},
        {"shaderFloat8CooperativeMatrix", shader_float8.shaderFloat8CooperativeMatrix},
        {"cooperativeMatrix", cooperative_matrix.cooperativeMatrix},
        {"subgroupSizeControl", subgroup_size_control.subgroupSizeControl},
        {"computeFullSubgroups", subgroup_size_control.computeFullSubgroups},
        {"vulkanMemoryModel", memory_model.vulkanMemoryModel},
    };

    for (const auto& [name, enabled] : required) {
        if (!enabled) {
            blockers.emplace_back(std::string("feature ") + name + " is not reported");
        }
    }

    if (properties.apiVersion < VK_API_VERSION_1_3) {
        blockers.emplace_back("device API version is below 1.3");
    }
    if (!(cooperative.cooperativeMatrixSupportedStages & VK_SHADER_STAGE_COMPUTE_BIT)) {
        blockers.emplace_back("cooperative matrices are not supported in compute");
    }
    if (!float_controls.shaderDenormPreserveFloat16) {
        blockers.emplace_back("float16 denormals are not preserved");
    }
    // The 32-lane range check arrives from runtimeBlockers(), which this list starts
    // from, so the pinned stage sizes are only judged once per rejection.
    if (!(subgroup_sizes.requiredSubgroupSizeStages & VK_SHADER_STAGE_COMPUTE_BIT)) {
        blockers.emplace_back("explicit subgroup sizes are unavailable in compute");
    }

    // Only a probe that answered can reject a tile; an unprobed device is judged
    // on the extension and feature bits alone.
    if (matrixShapesProbed()) {
        if (!hasMatrixShape(VK_COMPONENT_TYPE_FLOAT16_KHR)) {
            blockers.emplace_back(missingMatrixShapeName(VK_COMPONENT_TYPE_FLOAT16_KHR));
        }
        if (!hasMatrixShape(VK_COMPONENT_TYPE_FLOAT8_E4M3_EXT)) {
            blockers.emplace_back(missingMatrixShapeName(VK_COMPONENT_TYPE_FLOAT8_E4M3_EXT));
        }
    }

    return blockers;
}

bool GpuCapabilities::supportsInt8Model() const { return int8Blockers().empty(); }

bool GpuCapabilities::supportsFp8Model() const { return fp8Blockers().empty(); }

bool GpuCapabilities::amdProprietaryDriver() const {
    return isAmdvlkOrProprietaryDriver(properties.vendorID, driver.driverID);
}

bool GpuCapabilities::rdna2Device() const {
    return isAmdRdna2Device(properties.vendorID, properties.deviceID);
}

// Forcing a stage subgroup size is a contract with the driver, not with the
// shader: the value has to be accepted by the subgroup size control feature and
// sit inside the range the device reported. An unprobed snapshot has a zero
// range, so it refuses every size rather than pretending 32 lanes are legal.
bool GpuCapabilities::supportsRequiredSubgroupSize(std::uint32_t wanted) const {
    if (!subgroup_size_control.subgroupSizeControl) {
        return false;
    }
    if (subgroup_sizes.minSubgroupSize == 0 || subgroup_sizes.maxSubgroupSize == 0) {
        return false;
    }

    return wanted >= subgroup_sizes.minSubgroupSize && wanted <= subgroup_sizes.maxSubgroupSize;
}

// The passes around the model itself pin their stage subgroup size to 32 in
// every preset, so the reported range decides whether the runtime can build
// pipelines at all. This predicate deliberately ignores the feature bits, which
// the blocker lists already report on their own, so one missing capability
// never produces two reasons.
bool GpuCapabilities::supportsWave32Pipelines() const {
    if (subgroup_sizes.minSubgroupSize == 0 || subgroup_sizes.maxSubgroupSize == 0) {
        return true;
    }

    return 32 >= subgroup_sizes.minSubgroupSize && 32 <= subgroup_sizes.maxSubgroupSize;
}

std::string GpuCapabilities::wave32BlockerText() const {
    return "device subgroup range " + std::to_string(subgroup_sizes.minSubgroupSize) + '-' +
           std::to_string(subgroup_sizes.maxSubgroupSize) + " cannot pin 32-lane subgroups";
}

// Whether a tile query reached the device at all. The distinction matters more
// than the count: an empty list from a probe nobody could run says nothing about
// the hardware, while an empty list from a probe that answered rules the model
// out. matrix_probe_route names which case this snapshot is, so a log line can
// tell "the driver has no tiles" from "nobody could ask the driver".
bool GpuCapabilities::matrixShapesProbed() const {
    return matrix_probe_route != MatrixProbeRoute::Unprobed;
}

const char* GpuCapabilities::matrixProbeRouteName() const {
    switch (matrix_probe_route) {
    case MatrixProbeRoute::HostInstance:
        return "host";
    case MatrixProbeRoute::DynamicLookup:
        return "dlsym";
    default:
        return "unprobed";
    }
}

// How the tile list reads in a log line: a count only a probe can produce, and
// the word "unprobed" when nobody reached the query. Both the provider's
// [capabilities] line and the runtime's [device-policy] line print this, so the
// two places cannot drift into telling different stories about the same device.
std::string GpuCapabilities::matrixShapeSummary() const {
    return matrixShapesProbed() ? std::to_string(matrix_shapes.size()) : "unprobed";
}

// One line answers the question the runtime asks when it sizes descriptor sets:
// which byte costs each of the five descriptor types carries on this device.
std::string GpuCapabilities::descriptorSizeSummary() const {
    if (!descriptor_buffer.descriptorBuffer) {
        return "unsupported";
    }

    if (!descriptor_buffer_properties.sampledImageDescriptorSize) {
        // The extension is supported but was never enabled on the instance that
        // created the device, so the driver left the byte sizes alone. Printing
        // zeros would read like a measurement the hardware had given us.
        return "unreported";
    }

    std::ostringstream sizes;
    sizes << descriptor_buffer_properties.sampledImageDescriptorSize << '/'
          << descriptor_buffer_properties.storageImageDescriptorSize << '/'
          << descriptor_buffer_properties.uniformBufferDescriptorSize << '/'
          << descriptor_buffer_properties.storageBufferDescriptorSize << '/'
          << descriptor_buffer_properties.samplerDescriptorSize;
    return sizes.str();
}

// The overlay blends against the app's own output image, which the device may
// only allow it to read back per format. Vulkan 1.3 reports that in
// VkFormatProperties3, while a 1.1 or 1.2 host needs VK_KHR_format_feature_flags2
// advertised as well - both facts are already in the snapshot, so the check
// makes one per-format query instead of the four it used to make on its own.
// Nothing here throws: an unanswered query reads as "cannot read", which lands
// on the write-only overlay shader the same way the old code did.
bool canReadStorageImage(VkPhysicalDevice physical_device,
                         const GpuCapabilities& capabilities,
                         VkFormat format) {
    if (!physical_device) {
        return false;
    }

    if (capabilities.properties.apiVersion < VK_API_VERSION_1_3 &&
        !capabilities.hasExtension(VK_KHR_FORMAT_FEATURE_FLAGS_2_EXTENSION_NAME)) {
        return false;
    }

    // Both structs have to carry their own sType: the chain is validated by the
    // driver, and a zeroed sType leaves the flags empty, which reads as "cannot
    // read" instead of answering. The command itself returns no result code.
    VkFormatProperties3 features{VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_3};
    VkFormatProperties2 chain{VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2, &features};
    vkGetPhysicalDeviceFormatProperties2(physical_device, format, &chain);

    return (features.optimalTilingFeatures &
            VK_FORMAT_FEATURE_2_STORAGE_READ_WITHOUT_FORMAT_BIT) != 0;
}

std::string GpuCapabilities::describe() const {
    std::ostringstream line;
    line << "device=" << std::quoted(properties.deviceName) << " type=" << deviceTypeName(properties.deviceType)
         << " api=" << apiVersionName(properties.apiVersion) << " vendor_id=" << hexValue(properties.vendorID)
         << " device_id=" << hexValue(properties.deviceID) << " driver_id=" << driver.driverID
         << " driver=" << std::quoted(driver.driverName)
         << " driver_info=" << std::quoted(driver.driverInfo) << " extensions=" << extensions.size()
         << " matrix_shapes=" << matrixShapeSummary() << " matrix_probe=" << matrixProbeRouteName()
         << " descriptor_sizes=" << descriptorSizeSummary()
         << " timestamp_period=" << properties.limits.timestampPeriod
         << " memory_types=" << memory.memoryTypeCount
         << " subgroup_size=" << subgroups.subgroupSize
         << " subgroup_range=" << subgroup_sizes.minSubgroupSize << '-' << subgroup_sizes.maxSubgroupSize
         << " amd_proprietary=" << (amdProprietaryDriver() ? "yes" : "no")
         << " rdna2=" << (rdna2Device() ? "yes" : "no")
         << " wave32=" << (supportsWave32Pipelines() ? "yes" : "no")
         << " runtime=" << (supportsRuntimeContract() ? "yes" : "no")
         << " int8=" << (supportsInt8Model() ? "yes" : "no")
         << " fp8=" << (supportsFp8Model() ? "yes" : "no");
    return line.str();
}

std::string joinBlockers(const std::vector<std::string>& blockers) {
    std::ostringstream joined;
    for (std::size_t index = 0; index < blockers.size(); ++index) {
        if (index != 0) {
            joined << "; ";
        }
        joined << blockers[index];
    }
    return joined.str();
}

// The loader hands the provider one device and its resolver; a row is satisfied
// when either of its spellings resolves. The rows live next to the resolver the
// runtime uses, so the provider cannot wave through a device whose runtime half
// would later reach for a function the contract never promised - the two used to
// keep separate name lists, this one and DeviceDispatch's literals.
std::vector<std::string> missingEntryPoints(VkDevice device,
                                            PFN_vkGetDeviceProcAddr get_device_proc_addr,
                                            DescriptorBackend backend) {
    return fsr4vk::missingEntryPointGroups(device, get_device_proc_addr, backend);
}

// The transport choice, decided from the snapshot the model choice came from so
// that a host which never implemented VK_EXT_descriptor_buffer still gets its
// models. The extension path wins when both are available: it is the one the
// shipped assets were generated and validated under, and it costs no pool, no
// 64 frame slots times 6 sets, and no descriptor update per recorded frame.
DescriptorBackend selectDescriptorBackend(const GpuCapabilities& capabilities) {
    if (capabilities.supportsDescriptorBuffers()) {
        return DescriptorBackend::DescriptorBuffers;
    }
    return DescriptorBackend::DescriptorSets;
}

// FSR4_DESCRIPTOR_BACKEND=descriptor-set|descriptor-buffer|sets|buffers. Only an
// exact spelling answers; anything else reports nothing so the operator sees a
// refusal naming the value instead of a silently different transport, which is
// how the two transports stay comparable on one device during a change of this
// size.
std::optional<DescriptorBackend> descriptorBackendOverride() {
    const char* value = std::getenv("FSR4_DESCRIPTOR_BACKEND");
    if (!value || !*value) {
        return std::nullopt;
    }
    if (std::strcmp(value, "descriptor-set") == 0 || std::strcmp(value, "sets") == 0) {
        return DescriptorBackend::DescriptorSets;
    }
    if (std::strcmp(value, "descriptor-buffer") == 0 || std::strcmp(value, "buffers") == 0) {
        return DescriptorBackend::DescriptorBuffers;
    }
    return std::nullopt;
}

std::optional<ModelChoice> parseModelChoice(const char* value) {
    if (!value || !*value || std::strcmp(value, "auto") == 0) {
        return ModelChoice::Auto;
    }
    if (std::strcmp(value, "int8") == 0 || std::strcmp(value, "1") == 0) {
        return ModelChoice::ForceInt8;
    }
    if (std::strcmp(value, "fp8") == 0 || std::strcmp(value, "2") == 0) {
        return ModelChoice::ForceFp8;
    }
    return std::nullopt;
}

const char* modelName(ModelChoice choice) {
    switch (choice) {
    case ModelChoice::ForceInt8:
        return "int8";
    case ModelChoice::ForceFp8:
        return "fp8";
    default:
        return "auto";
    }
}

// The ladder the provider and tools/cap-probe.cpp share. Reading both blocker
// lists here rather than one at the call site is deliberate: a caller that
// consulted only INT8 could not tell "fall back" from "give up", which is exactly
// the distinction a weak device's log line has to carry.
ModelResolution resolveModel(const GpuCapabilities& capabilities, ModelChoice requested) {
    const auto int8_blockers = capabilities.int8Blockers();
    const auto fp8_blockers = capabilities.fp8Blockers();

    if (requested == ModelChoice::ForceInt8) {
        if (int8_blockers.empty()) {
            return {ModelChoice::ForceInt8, "int8 contract satisfied"};
        }
        return {std::nullopt,
                "FSR4_VK_MODEL=int8 refused by this device: " +
                    joinBlockers(int8_blockers) +
                    "; fp8 was not considered, the request is strict"};
    }

    if (requested == ModelChoice::ForceFp8) {
        if (fp8_blockers.empty()) {
            return {ModelChoice::ForceFp8, "fp8 contract satisfied"};
        }
        return {std::nullopt,
                "FSR4_VK_MODEL=fp8 refused by this device: " +
                    joinBlockers(fp8_blockers) +
                    "; int8 was not considered, the request is strict"};
    }

    // Auto. INT8 first, unchanged for every device that can run it.
    if (int8_blockers.empty()) {
        return {ModelChoice::ForceInt8, "auto: int8 contract satisfied"};
    }
    if (fp8_blockers.empty()) {
        return {ModelChoice::ForceFp8,
                "auto: falling back to fp8 because int8 is refused: " +
                    joinBlockers(int8_blockers)};
    }
    return {std::nullopt,
            "neither model runs: int8 blocked (" + joinBlockers(int8_blockers) +
            ") fp8 blocked (" + joinBlockers(fp8_blockers) + ")"};
}

} // fsr4vk



