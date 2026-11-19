// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 fsr4-vk contributors.
// See LICENSE in the project root.

#include "framegeneration.hpp"

#include <cstring>
#include <string>
#include <vector>

namespace ffgvk {

const CreateBackendVkDesc* findBackend(const ffxApiHeader* header) {
    for (auto* current = header; current != nullptr; current = current->pNext) {
        if (current->type == kBackendVkDescType) {
            return reinterpret_cast<const CreateBackendVkDesc*>(current);
        }
    }
    return nullptr;
}

const CreateBackendVkInstanceDesc* findBackendInstance(const ffxApiHeader* header) {
    for (auto* current = header; current != nullptr; current = current->pNext) {
        if (current->type == kBackendVkInstanceDescType) {
            return reinterpret_cast<const CreateBackendVkInstanceDesc*>(current);
        }
    }
    return nullptr;
}

const CreateVersionOverride* findVersionOverride(const ffxApiHeader* header) {
    for (auto* current = header; current != nullptr; current = current->pNext) {
        if (current->type == kOverrideVersionDescType) {
            return reinterpret_cast<const CreateVersionOverride*>(current);
        }
    }
    return nullptr;
}

const CreateBackendVkEnabledFeaturesDesc* findEnabledFeatures(const ffxApiHeader* header) {
    for (auto* current = header; current != nullptr; current = current->pNext) {
        if (current->type == kEnabledFeaturesDescType) {
            return reinterpret_cast<const CreateBackendVkEnabledFeaturesDesc*>(current);
        }
    }
    return nullptr;
}

bool hasExtension(const GpuCapabilities& capabilities, const char* name) {
    for (const auto& extension : capabilities.enabledExtensions) {
        if (extension == name) {
            return true;
        }
    }
    return false;
}

// Identity + support snapshot of the caller's device. Physical-level core entry
// points are linked from the loader (the same route gpu-capabilities.cpp uses);
// the runtime never enables a feature itself, it only reads what the device offers
// and refuses if the caller has not enabled the Frame Generation contract.
// Reads the enabled feature set off the physical device into `capabilities`.
GpuCapabilities queryDeviceFeatures(GpuCapabilities capabilities, VkPhysicalDevice physical);

GpuCapabilities queryCapabilities(const CreateBackendVkDesc& backend,
                                  const CreateBackendVkEnabledFeaturesDesc* enabled) {
    GpuCapabilities capabilities;
    capabilities.device = backend.device;
    const VkPhysicalDevice physical = backend.physicalDevice;
    if (!physical) {
        return capabilities;
    }

    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(physical, &properties);
    capabilities.deviceName = properties.deviceName;
    capabilities.vendorId = properties.vendorID;
    capabilities.deviceId = properties.deviceID;
    capabilities.apiVersion = properties.apiVersion;
    capabilities.driverVersion = properties.driverVersion;
    capabilities.maxWorkGroupSize = properties.limits.maxComputeWorkGroupSize[0];

    // Subgroup support in this Vulkan revision is a props2 structure, not a
    // field on the base properties. Read the basic-operation flag the Frame
    // Generation subgroup contract needs, and the size-control limits.
    VkPhysicalDeviceProperties2 props2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    VkPhysicalDeviceSubgroupProperties subgroupProps{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
    VkPhysicalDeviceSubgroupSizeControlProperties sizeProps{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES};
    subgroupProps.pNext = &sizeProps;
    props2.pNext = &subgroupProps;
    vkGetPhysicalDeviceProperties2(physical, &props2);
    capabilities.subgroupOperations = subgroupProps.supportedOperations;
    capabilities.minSubgroupSize = sizeProps.minSubgroupSize;
    capabilities.maxSubgroupSize = sizeProps.maxSubgroupSize;

    uint32_t extensionCount = 0;
    if (vkEnumerateDeviceExtensionProperties(physical, nullptr, &extensionCount, nullptr) == VK_SUCCESS &&
        extensionCount != 0) {
        std::vector<VkExtensionProperties> table(extensionCount);
        if (vkEnumerateDeviceExtensionProperties(physical, nullptr, &extensionCount, table.data()) ==
            VK_SUCCESS) {
            for (const auto& extension : table) {
                capabilities.enabledExtensions.emplace_back(extension.extensionName);
            }
        }
    }
    // One physical features2 chain answers the whole Frame Generation execution
    // contract; drivers leave structures for extensions they do not implement
    // untouched, so those read back as unsupported.
    capabilities = queryDeviceFeatures(capabilities, physical);
    const auto supported = capabilities;
    if (enabled) {
        capabilities.shaderInt16 = supported.shaderInt16 && enabled->shaderInt16;
        capabilities.shaderInt64 = supported.shaderInt64 && enabled->shaderInt64;
        capabilities.shaderInt64Atomics =
            supported.shaderInt64Atomics && enabled->shaderInt64;
        capabilities.shaderFloat16 = supported.shaderFloat16 && enabled->shaderFloat16;
        capabilities.shaderInt8 = supported.shaderInt8 && enabled->shaderInt8;
        capabilities.storage8Bit = supported.storage8Bit && enabled->storage8Bit;
        capabilities.storage16Bit = supported.storage16Bit && enabled->storage16Bit;
        capabilities.bufferDeviceAddress =
            supported.bufferDeviceAddress && enabled->bufferDeviceAddress;
        capabilities.runtimeDescriptorArray =
            supported.runtimeDescriptorArray && enabled->runtimeDescriptorArray;
        capabilities.descriptorIndexing = capabilities.runtimeDescriptorArray;
        capabilities.cooperativeMatrix =
            supported.cooperativeMatrix && enabled->cooperativeMatrix;
        capabilities.vulkanMemoryModel =
            supported.vulkanMemoryModel && enabled->vulkanMemoryModel;
        capabilities.subgroupSizeControl =
            supported.subgroupSizeControl && enabled->subgroupSizeControl;
        capabilities.computeDerivativeGroupLinear =
            supported.computeDerivativeGroupLinear && enabled->computeDerivativeGroupLinear;
        capabilities.synchronization2 = supported.synchronization2 && enabled->synchronization2;
        capabilities.storageImageWriteWithoutFormat =
            supported.storageImageWriteWithoutFormat &&
            enabled->storageImageWriteWithoutFormat;
    } else {
        capabilities.shaderFloat16 = false;
        capabilities.shaderInt8 = false;
        capabilities.shaderInt16 = false;
        capabilities.shaderInt64 = false;
        capabilities.shaderInt64Atomics = false;
        capabilities.storage8Bit = false;
        capabilities.storage16Bit = false;
        capabilities.bufferDeviceAddress = false;
        capabilities.descriptorIndexing = false;
        capabilities.runtimeDescriptorArray = false;
        capabilities.storageImageDescriptorWithoutFormat = false;
        capabilities.cooperativeMatrix = false;
        capabilities.vulkanMemoryModel = false;
        capabilities.subgroupSizeControl = false;
        capabilities.computeDerivativeGroupLinear = false;
        capabilities.synchronization2 = false;
        capabilities.storageImageWriteWithoutFormat = false;
    }
    return capabilities;
}

GpuCapabilities queryDeviceFeatures(GpuCapabilities capabilities, VkPhysicalDevice physical) {
    VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    VkPhysicalDeviceShaderFloat16Int8Features float16Int8{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES};
    VkPhysicalDevice8BitStorageFeatures storage8{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES};
    VkPhysicalDevice16BitStorageFeatures storage16{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES};
    VkPhysicalDeviceBufferDeviceAddressFeatures bufferAddress{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES};
    VkPhysicalDeviceDescriptorIndexingFeatures descriptorIndexing{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES};
    VkPhysicalDeviceVulkanMemoryModelFeatures memoryModel{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_MEMORY_MODEL_FEATURES};
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR cooperative{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
    VkPhysicalDeviceSubgroupSizeControlFeatures subgroupControl{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES};
    VkPhysicalDeviceComputeShaderDerivativesFeaturesKHR derivatives{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COMPUTE_SHADER_DERIVATIVES_FEATURES_KHR};
    VkPhysicalDeviceVulkan13Features v13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};

    features.pNext = &float16Int8;
    float16Int8.pNext = &storage8;
    storage8.pNext = &storage16;
    storage16.pNext = &bufferAddress;
    bufferAddress.pNext = &descriptorIndexing;
    descriptorIndexing.pNext = &memoryModel;
    memoryModel.pNext = &cooperative;
    cooperative.pNext = &subgroupControl;
    subgroupControl.pNext = &derivatives;
    derivatives.pNext = &v13;

    vkGetPhysicalDeviceFeatures2(physical, &features);

    capabilities.shaderInt64 = features.features.shaderInt64 == VK_TRUE;
    // This revision exposes 64-bit atomics through the shaderInt64 feature (no
    // separate shaderInt64Atomics member); shaderInt64 support gates both.
    capabilities.shaderInt64Atomics = features.features.shaderInt64 == VK_TRUE;
    capabilities.storageImageWriteWithoutFormat =
        features.features.shaderStorageImageWriteWithoutFormat == VK_TRUE;
    capabilities.shaderInt16 = features.features.shaderInt16 == VK_TRUE;
    capabilities.shaderFloat16 = float16Int8.shaderFloat16 == VK_TRUE;
    capabilities.shaderInt8 = float16Int8.shaderInt8 == VK_TRUE;
    capabilities.storage8Bit = storage8.storageBuffer8BitAccess == VK_TRUE;
    capabilities.storage16Bit = storage16.storageBuffer16BitAccess == VK_TRUE;
    capabilities.bufferDeviceAddress = bufferAddress.bufferDeviceAddress == VK_TRUE;
    capabilities.runtimeDescriptorArray = descriptorIndexing.runtimeDescriptorArray == VK_TRUE;
    capabilities.descriptorIndexing = descriptorIndexing.runtimeDescriptorArray == VK_TRUE;
    capabilities.vulkanMemoryModel = memoryModel.vulkanMemoryModel == VK_TRUE;
    capabilities.cooperativeMatrix = cooperative.cooperativeMatrix == VK_TRUE;
    capabilities.subgroupSizeControl = subgroupControl.subgroupSizeControl == VK_TRUE;
    capabilities.computeDerivativeGroupLinear = derivatives.computeDerivativeGroupLinear == VK_TRUE;
    capabilities.synchronization2 = v13.synchronization2 == VK_TRUE;
    return capabilities;
}

// The Frame Generation execution contract, measured module-by-module against the
// pinned DLL (build/fg-capabilities/execution-contract.json). A feature the device
// supports but the caller has not enabled is still a blocker: the runtime may never
// enable it on the caller's behalf. Each blocker names the feature and how many
// modules demand it, so context creation can refuse before pipeline creation.
std::vector<std::string> executionContractBlockers(const GpuCapabilities& capabilities) {
    std::vector<std::string> blockers;

    auto require = [&](bool ok, const char* feature, const char* detail) {
        if (!ok) {
            blockers.push_back(std::string("missing enabled feature ") + feature + " (" +
                               detail + ")");
        }
    };
    require(capabilities.bufferDeviceAddress, "VK_KHR_buffer_device_address",
            "486/486 modules use PhysicalStorageBufferAddresses");
    require(capabilities.runtimeDescriptorArray, "runtimeDescriptorArray",
            "484 modules use Ext_descriptor_indexing RuntimeDescriptorArray");
    require(capabilities.cooperativeMatrix, "VK_KHR_cooperative_matrix",
            "156 ML Frame Generation neural modules use CooperativeMatrixKHR");
    require(capabilities.vulkanMemoryModel, "VK_KHR_vulkan_memory_model",
            "156 neural modules declare the VulkanMemoryModel capability");
    require(capabilities.shaderFloat16, "shaderFloat16", "174 modules use Float16");
    require(capabilities.shaderInt8, "shaderInt8", "156 neural modules use Int8");
    require(capabilities.storage8Bit, "VK_KHR_8bit_storage",
            "156 neural modules use StorageBuffer8BitAccess");
    require(capabilities.shaderInt16, "shaderInt16", "48 modules use Int16");
    require(capabilities.shaderInt64, "shaderInt64", "1 module uses Int64 and Int64Atomics");
    require(capabilities.storageImageWriteWithoutFormat, "shaderStorageImageWriteWithoutFormat",
            "82 modules write storage images without a declared format");
    if (!(capabilities.subgroupOperations & VK_SUBGROUP_FEATURE_BASIC_BIT)) {
        blockers.push_back("device reports no basic subgroup support "
                           "(146 modules use GroupNonUniform operations)");
    }
    return blockers;
}

} // namespace ffgvk
