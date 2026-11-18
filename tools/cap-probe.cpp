// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 fsr4-vk contributors.
// See LICENSE in the project root.

// Capability probe for the runtime's device detection. It prints the vendor
// profile and the per-model blockers for every Vulkan device the loader can
// see, which is how the gates in src/runtime/gpu-capabilities.hpp get checked
// on real hardware without loading the provider.
//
//   g++ -std=c++20 -I src -I third_party/Vulkan-Headers/include -o /tmp/cap-probe \
//       tools/cap-probe.cpp src/runtime/gpu-capabilities.cpp -ldl \
//       /usr/lib/libvulkan.so.1 && /tmp/cap-probe
//
// The provider never receives a VkInstance, so production leaves the cooperative
// matrix shapes unprobed and judges FP8 on the extension and feature bits alone.
// A harness like this one owns its instance, so it can enable the extensions,
// ask for the tiles itself and fill the public fields; the "with shapes" block
// below shows how a test does that and what changes once the tiles are known.

#include "runtime/gpu-capabilities.hpp"

#include <vulkan/vulkan.h>

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace {

// The shape list belongs to VK_KHR_cooperative_matrix and is resolved through
// the instance that created the device, so it is reached with GetInstanceProcAddr
// rather than a link-time reference the host loader might not export.
std::vector<VkCooperativeMatrixPropertiesKHR> queryShapes(VkInstance instance,
                                                          VkPhysicalDevice device) {
    const auto get_shapes = reinterpret_cast<PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR>(
        vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR"));
    if (!get_shapes) {
        return {};
    }

    uint32_t count = 0;
    if (get_shapes(device, &count, nullptr) != VK_SUCCESS) {
        return {};
    }

    std::vector<VkCooperativeMatrixPropertiesKHR> shapes(
        count, VkCooperativeMatrixPropertiesKHR{VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR});
    if (get_shapes(device, &count, shapes.data()) != VK_SUCCESS) {
        return {};
    }

    shapes.resize(count);
    return shapes;
}

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

    for (const auto device : devices) {
        auto capabilities = fsr4vk::detect(device);
        std::cout << capabilities.describe() << '\n';

        // The provider's view: whatever the module answered by itself.
        report("production", capabilities);

        // The harness view: fill the tile list, then re-run the same predicates.
        capabilities.matrix_shapes = queryShapes(instance, device);
        capabilities.matrix_shapes_probed = !capabilities.matrix_shapes.empty();
        std::cout << "  tiles=" << capabilities.matrix_shapes.size()
                  << " float16_16x16x16=" << capabilities.supportsModelMatrixShapes() << '\n';
        report("with shapes", capabilities);
    }

    vkDestroyInstance(instance, nullptr);
    return 0;
}
