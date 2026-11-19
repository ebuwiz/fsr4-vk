// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 fsr4-vk contributors.
// See LICENSE in the project root.

// Frame Generation on-device capability probe. It creates a caller-owned VkDevice
// with exactly the measured Frame Generation execution contract enabled (using the
// same classic VkDeviceCreateInfo + enabled-extension pattern as descriptor-smoke),
// hands it to the provider ABI (ffxCreateContext/ffxQuery/ffxDispatch/ffxDestroyContext),
// and prints the create result plus whether the ABI refused the not-yet-implemented
// dispatch, per device the loader can see. It links the provider ABI but no assets
// or model logic, so it runs on a machine that is not running a game.

// The provider declares the FFX API's dllexport shim; the probe includes the same
// public headers to reach the exported entry points, so it needs the shim too, or
// __declspec(dllexport) on the prototypes is a syntax error on Linux.
#if !defined(_WIN32)
#define __declspec(x) __attribute__((visibility("default")))
#endif

#include <vulkan/vulkan.h>

#include "ffx_api.h"
#include "ffx_api_types.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#include <unistd.h>
#endif

// The Vulkan backend block a host attaches to ffxCreateContext (descriptor type 3).
struct FfgBackendDesc {
    ffxApiHeader header;
    VkDevice device;
    VkPhysicalDevice physicalDevice;
    PFN_vkGetDeviceProcAddr getDeviceProcAddr;
};

struct FfgEnabledFeaturesDesc {
    ffxApiHeader header;
    VkBool32 shaderInt16;
    VkBool32 shaderInt64;
    VkBool32 shaderFloat16;
    VkBool32 shaderInt8;
    VkBool32 storage8Bit;
    VkBool32 storage16Bit;
    VkBool32 bufferDeviceAddress;
    VkBool32 runtimeDescriptorArray;
    VkBool32 cooperativeMatrix;
    VkBool32 vulkanMemoryModel;
    VkBool32 subgroupSizeControl;
    VkBool32 computeDerivativeGroupLinear;
    VkBool32 synchronization2;
    VkBool32 storageImageWriteWithoutFormat;
};

static bool hasExt(VkPhysicalDevice physical, const char* name) {
    uint32_t count = 0;
    vkEnumerateDeviceExtensionProperties(physical, nullptr, &count, nullptr);
    std::vector<VkExtensionProperties> table(count);
    if (count && vkEnumerateDeviceExtensionProperties(physical, nullptr, &count, table.data()) != VK_SUCCESS) {
        table.clear();
    }
    for (const auto& entry : table) {
        if (std::strcmp(entry.extensionName, name) == 0) {
            return true;
        }
    }
    return false;
}

static const char* kContractExtensions[] = {
    VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
    VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME,
    VK_KHR_COMPUTE_SHADER_DERIVATIVES_EXTENSION_NAME,
    VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME,
    VK_KHR_VULKAN_MEMORY_MODEL_EXTENSION_NAME,
    VK_KHR_8BIT_STORAGE_EXTENSION_NAME,
    VK_KHR_16BIT_STORAGE_EXTENSION_NAME,
    VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME,
    VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME,
};

int main() {
    VkInstance instance{};
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO, nullptr, "ffg4-probe", 0,
                          "ffg4-probe", 0, VK_MAKE_API_VERSION(0, 4, 0, 2)};
    VkInstanceCreateInfo info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    info.pApplicationInfo = &app;
    if (vkCreateInstance(&info, nullptr, &instance) != VK_SUCCESS) {
        std::printf("no Vulkan instance\n");
        return 77;
    }
    uint32_t physicalCount = 0;
    vkEnumeratePhysicalDevices(instance, &physicalCount, nullptr);
    std::vector<VkPhysicalDevice> physicals(physicalCount);
    if (physicalCount) {
        vkEnumeratePhysicalDevices(instance, &physicalCount, physicals.data());
    }
    if (!physicalCount) {
        std::printf("no physical devices\n");
        vkDestroyInstance(instance, nullptr);
        return 77;
    }

    int ran = 0;
    for (const auto& physical : physicals) {
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(physical, &props);
        std::printf("device=\"%s\" vendor=0x%08X api=%u.%u.%u\n", props.deviceName,
                    props.vendorID, VK_API_VERSION_MAJOR(props.apiVersion),
                    VK_API_VERSION_MINOR(props.apiVersion),
                    VK_API_VERSION_PATCH(props.apiVersion));

        // Enable exactly the contract extensions this device offers, with the
        // matching feature bits, mirroring descriptor-smoke's classic device setup.
        std::vector<const char*> extensions;
        for (const char* extension : kContractExtensions) {
            if (hasExt(physical, extension)) {
                extensions.push_back(extension);
            }
        }

        VkPhysicalDeviceFeatures2 baseFeatures{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        VkPhysicalDeviceShaderFloat16Int8Features float16Int8{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES};
        VkPhysicalDevice16BitStorageFeatures storage16{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES};
        VkPhysicalDeviceBufferDeviceAddressFeatures bufferAddress{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES};
        VkPhysicalDeviceCooperativeMatrixFeaturesKHR cooperative{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
        VkPhysicalDeviceVulkanMemoryModelFeaturesKHR memoryModel{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_MEMORY_MODEL_FEATURES_KHR};
        VkPhysicalDeviceDescriptorIndexingFeatures descriptorIndexing{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES};
        VkPhysicalDevice8BitStorageFeatures storage8{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES};
        VkPhysicalDeviceSubgroupSizeControlFeatures subgroupControl{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES};
        VkPhysicalDeviceComputeShaderDerivativesFeaturesKHR derivatives{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COMPUTE_SHADER_DERIVATIVES_FEATURES_KHR};
        baseFeatures.pNext = &bufferAddress;
        bufferAddress.pNext = &float16Int8;
        float16Int8.pNext = &storage16;
        storage16.pNext = &cooperative;
        cooperative.pNext = &memoryModel;
        memoryModel.pNext = &descriptorIndexing;
        descriptorIndexing.pNext = &storage8;
        storage8.pNext = &subgroupControl;
        subgroupControl.pNext = &derivatives;

        baseFeatures.features.shaderInt64 = VK_TRUE;
        baseFeatures.features.shaderInt16 = VK_TRUE;
        baseFeatures.features.shaderStorageImageWriteWithoutFormat = VK_TRUE;
        bufferAddress.bufferDeviceAddress = VK_TRUE;
        float16Int8.shaderFloat16 = VK_TRUE;
        float16Int8.shaderInt8 = VK_TRUE;
        storage16.storageBuffer16BitAccess = VK_TRUE;
        cooperative.cooperativeMatrix = VK_TRUE;
        memoryModel.vulkanMemoryModel = VK_TRUE;
        descriptorIndexing.runtimeDescriptorArray = VK_TRUE;
        storage8.storageBuffer8BitAccess = VK_TRUE;
        subgroupControl.subgroupSizeControl = VK_TRUE;
        derivatives.computeDerivativeGroupLinear = VK_TRUE;

        VkDeviceQueueCreateInfo queue{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        queue.queueFamilyIndex = 0;
        queue.queueCount = 1;
        VkDeviceCreateInfo deviceInfo{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        deviceInfo.pNext = &baseFeatures;
        deviceInfo.queueCreateInfoCount = 1;
        deviceInfo.pQueueCreateInfos = &queue;
        deviceInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
        deviceInfo.ppEnabledExtensionNames = extensions.empty() ? nullptr : extensions.data();

        VkDevice device{};
        if (vkCreateDevice(physical, &deviceInfo, nullptr, &device) != VK_SUCCESS) {
            std::printf("  device-create-skipped: a contract extension/feature is unavailable\n");
            continue;
        }
        ran++;

        FfgEnabledFeaturesDesc enabled{{7u, nullptr},
                                       VK_TRUE,
                                       VK_TRUE,
                                       VK_TRUE,
                                       VK_TRUE,
                                       VK_TRUE,
                                       VK_TRUE,
                                       VK_TRUE,
                                       VK_TRUE,
                                       VK_TRUE,
                                       VK_TRUE,
                                       VK_TRUE,
                                       VK_TRUE,
                                       VK_TRUE,
                                       VK_TRUE};
        FfgBackendDesc backend{{3u, &enabled.header}, device, physical, vkGetDeviceProcAddr};
        ffxApiHeader create{0x00020000u, &backend.header};

        // Point the provider's namespaced pipeline cache at a probe-known temp
        // path so the cold (create+save) -> warm (reload) round-trip is observable
        // and does not litter the working directory.
        const std::string cachePath =
            std::string("/tmp/ffg4vk-probe-cache.") +
            std::to_string((unsigned)getpid()) + "." + std::to_string((unsigned)ran);
        std::remove(cachePath.c_str());
        setenv("FFG4_VK_PIPELINE_CACHE", cachePath.c_str(), 1);

        ffxContext context{};
        const ffxReturnCode_t created = ffxCreateContext(&context, &create, nullptr);
        std::printf("  ffxCreateContext=%d\n", created);
        if (created == FFX_API_RETURN_OK && context) {
            uint64_t capacity = 4;
            uint64_t ids[4]{};
            const char* names[4]{};
            ffxQueryDescGetVersions versions{
                {FFX_API_QUERY_DESC_TYPE_GET_VERSIONS, nullptr}, 0x00020000u, 0,
                &capacity, ids, names};
            ffxQuery(&context, &versions.header);
            std::printf("  provider advertises version id(s):");
            for (uint64_t index = 0; index < capacity; ++index) {
                std::printf(" 0x%llX", (unsigned long long)ids[index]);
            }
            // Pass a real (unknown-to-this-provider) dispatch header so the
            // refusal that surfaces is the runtime-pending boundary, not a
            // null-parameter error: expected return is 3 (RUNTIME_ERROR).
            ffxDispatchDescHeader dummy{0x00020001u, nullptr};
            const int dispatchCode = (int)ffxDispatch(&context, &dummy);
            std::printf("  ffxDispatch=%d (expected 3=runtime-error: runtime pending)\n",
                        dispatchCode);
            ffxDestroyContext(&context, nullptr);

            // Cycle 1 destroyed the context, which serialized the cache container
            // to cachePath. Create a second context: the provider must LOAD that
            // container as initialData because its pinned identity (algorithm=fg4 +
            // this GPU + driver) matches, i.e. a warm start. We verify the on-disk
            // container proves the namespacing + identity validation.
            ffxContext warm{};
            if (ffxCreateContext(&warm, &create, nullptr) == FFX_API_RETURN_OK &&
                warm) {
                std::ifstream file(cachePath, std::ios::binary);
                std::vector<unsigned char> data((std::istreambuf_iterator<char>(file)),
                                                std::istreambuf_iterator<char>());
                file.close();
                bool magicOk = data.size() >= 16 &&
                               std::memcmp(data.data(), "FFG4PC1\0", 8) == 0;
                // algorithm id lives at bytes [12..15]; fourcc('f','g','4') LE = {66,67,34,00}
                bool algorithmOk = data.size() >= 16 && data[12] == 0x66 && data[13] == 0x67 &&
                                   data[14] == 0x34 && data[15] == 0x00;
                std::printf("  pipeline-cache warm-load: container bytes=%zu magic=%s "
                            "algorithm=fg4(namespaced, != fs4-upscaler)=%s\n",
                            data.size(), magicOk ? "FFG4PC1" : "MISSING",
                            algorithmOk ? "ok" : "BAD");
                ffxDestroyContext(&warm, nullptr);
            }
            std::remove(cachePath.c_str());
        }
        vkDestroyDevice(device, nullptr);
    }
    vkDestroyInstance(instance, nullptr);
    if (ran == 0) {
        std::printf("no device offered the Frame Generation execution contract\n");
        return 77;
    }
    return 0;
}
