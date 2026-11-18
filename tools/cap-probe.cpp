// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 fsr4-vk contributors.
// See LICENSE in the project root.

// Capability probe for the runtime's device detection. It prints the vendor
// profile and the per-model blockers for every Vulkan device the loader can
// see, which is how the gates in src/runtime/gpu-capabilities.hpp get checked
// on real hardware without loading the provider.
//
//   g++ -std=c++23 -I src -I third_party/Vulkan-Headers/include -o /tmp/cap-probe
//       tools/cap-probe.cpp src/runtime/gpu-capabilities.cpp -ldl
//       /usr/lib/libvulkan.so.1 && /tmp/cap-probe
//
// Production has no VkInstance of its own: the host hands the provider a device,
// its physical device and a device-level resolver. vkGetDeviceProcAddr excludes
// physical-device-level commands by spec, and a loader such as the Mesa one exports
// no VK_KHR_cooperative_matrix entry point at all, so the tile query is unreachable
// unless the host also sends the instance that created the device - the optional
// type-4 block the provider reads in src/provider/provider.hpp. This harness plays
// both sides: it first probes without that block (the non-cooperating host, tiles
// unprobed) and then with it (the cooperating host, tiles confirmed), which is
// exactly the pair of behaviors the tri-state gate has to keep apart. The cache is
// cleared at the end because the devices enumerated here die with the instance.

#include "runtime/gpu-capabilities.hpp"

#include <vulkan/vulkan.h>

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace {

void report(const char* label, const fsr4vk::GpuCapabilities& capabilities) {
    std::cout << "  " << label << " int8=" << std::boolalpha << capabilities.supportsInt8Model()
              << " fp8=" << capabilities.supportsFp8Model() << '\n';
    for (const auto& blocker : capabilities.int8Blockers()) {
        std::cout << "  " << label << " int8 blocker: " << blocker << '\n';
    }
    for (const auto& blocker : capabilities.fp8Blockers()) {
        std::cout << "  " << label << " fp8 blocker: " << blocker << '\n';
    }
}

} // namespace

int main() {
    VkApplicationInfo application{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    application.apiVersion = VK_API_VERSION_1_4;

    // Both extensions have to be enabled at instance creation for the tile list
    // to mean anything, exactly as the app does through the provider chain.
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR cooperative{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
    cooperative.cooperativeMatrix = VK_TRUE;
    VkPhysicalDeviceShaderFloat8FeaturesEXT float8{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT};
    float8.shaderFloat8 = VK_TRUE;
    float8.shaderFloat8CooperativeMatrix = VK_TRUE;
    cooperative.pNext = &float8;

    VkInstanceCreateInfo create{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    create.pApplicationInfo = &application;
    create.pNext = &cooperative;

    VkInstance instance = VK_NULL_HANDLE;
    if (vkCreateInstance(&create, nullptr, &instance) != VK_SUCCESS) {
        create.pNext = nullptr;
        if (vkCreateInstance(&create, nullptr, &instance) != VK_SUCCESS) {
            std::cout << "no Vulkan instance available\n";
            return 1;
        }
        std::cout << "note: cooperative matrix and float8 could not be enabled\n";
    }

    std::uint32_t count = 0;
    if (vkEnumeratePhysicalDevices(instance, &count, nullptr) != VK_SUCCESS) {
        vkDestroyInstance(instance, nullptr);
        return 1;
    }

    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(instance, &count, devices.data());

    // What a cooperating host sends: the instance that created these devices plus
    // its instance-level resolver. The provider reads the same pair from the
    // optional type-4 block in the create-context chain.
    const fsr4vk::InstanceResolver resolver{instance, vkGetInstanceProcAddr};

    for (const auto device : devices) {
        const auto without_block =
            fsr4vk::detectCached(device).value_or(fsr4vk::GpuCapabilities{});
        std::cout << without_block.describe() << '\n';
        report("no instance block", without_block);

        // Same handle, resolver supplied: the cache refines the unprobed snapshot
        // instead of keeping the first answer, which is what lets the runtime -
        // it has no instance of its own - read the tiles the provider confirmed.
        const auto with_block =
            fsr4vk::detectCached(device, resolver).value_or(fsr4vk::GpuCapabilities{});
        std::cout << with_block.describe() << '\n';
        report("with instance block", with_block);

        // A later call with nothing supplied must reuse the refined snapshot,
        // which is the sharing the provider and the runtime depend on.
        const auto reused = fsr4vk::detectCached(device).value_or(fsr4vk::GpuCapabilities{});
        const bool first_answered = without_block.matrixShapesProbed();
        const bool second_answered = with_block.matrixShapesProbed();
        std::cout << "  refined=" << (!first_answered && second_answered)
                  << " cached_reuse=" << (reused.describe() == with_block.describe())
                  << " wmma=" << with_block.supportsModelMatrixShapes()
                  << " wave32=" << with_block.supportsWave32Pipelines()
                  << " null_handle_has_snapshot=" << fsr4vk::detectCached(nullptr).has_value()
                  << '\n';
    }

    // The devices enumerated here die with the instance, and the cache keys
    // itself on those handles, so it is dropped before the instance goes away.
    fsr4vk::clearCapabilityCache();
    vkDestroyInstance(instance, nullptr);
    return 0;
}
