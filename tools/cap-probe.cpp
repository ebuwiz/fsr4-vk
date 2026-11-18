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
#include "runtime/device-dispatch.hpp"

#include <vulkan/vulkan.h>

#include <cstdint>
#include <iostream>
#include <string>
#include <string_view>
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

// A resolver that never reaches a driver, used to test the entry-point contract
// itself. The VkDevice token selects which loader is being imitated: 1 answers
// only the KHR spellings of the promoted pairs, 2 answers everything except one
// descriptor row. Nothing is ever called through the returned pointer, so the
// stub cannot crash on a bogus address - it only reports whether a name exists.
PFN_vkVoidFunction stubDeviceProcAddr(VkDevice device, const char* name) {
    const auto token = reinterpret_cast<uintptr_t>(device);
    if (token == 0) {
        return nullptr;
    }

    const std::string_view requested(name);
    const bool khr_only = token == 1;
    const bool missing_one_row = token == 2;

    if (missing_one_row && requested == "vkGetDescriptorSetLayoutSizeEXT") {
        return nullptr;
    }

    const bool promoted_pair = requested == "vkCmdPipelineBarrier2" ||
                               requested == "vkCmdWriteTimestamp2" ||
                               requested == "vkGetBufferDeviceAddress";
    if (khr_only && promoted_pair) {
        return nullptr;
    }

    return reinterpret_cast<PFN_vkVoidFunction>(static_cast<uintptr_t>(1));
}

} // namespace

int main() {
    VkApplicationInfo application{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    application.apiVersion = VK_API_VERSION_1_4;

    // Both extensions have to be enabled at instance creation for the tile list
    // to mean anything, exactly as the app does through the provider chain.
    // Descriptor byte sizes only mean anything when VK_EXT_descriptor_buffer is
    // enabled on the instance that created the device, which is what the app's
    // own create-context chain does; asking without it is the unreported case.
    VkPhysicalDeviceDescriptorBufferFeaturesEXT buffer{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_BUFFER_FEATURES_EXT};
    buffer.descriptorBuffer = VK_TRUE;

    VkPhysicalDeviceCooperativeMatrixFeaturesKHR cooperative{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
    cooperative.cooperativeMatrix = VK_TRUE;
    VkPhysicalDeviceShaderFloat8FeaturesEXT float8{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT};
    float8.shaderFloat8 = VK_TRUE;
    float8.shaderFloat8CooperativeMatrix = VK_TRUE;
    cooperative.pNext = &float8;
    buffer.pNext = &cooperative;

    VkInstanceCreateInfo create{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    create.pApplicationInfo = &application;
    create.pNext = &buffer;

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

        // The Phase 7 facts come from the same snapshot the models were chosen
        // with: the descriptor byte sizes the runtime's sets are sized from, the
        // timestamp scale the GPU profiler converts with, and the per-format
        // read-back verdict that decides whether the overlay may blend against
        // the app's own output. Printing them here proves no consumer has to
        // ask the driver again for the gating half of those decisions.
        std::cout << "  descriptor_sizes=" << with_block.descriptorSizeSummary()
                  << " timestamp_period_ns=" << with_block.properties.limits.timestampPeriod
                  << " blend_bgra8="
                  << fsr4vk::canReadStorageImage(device, with_block, VK_FORMAT_B8G8R8A8_UNORM)
                  << " blend_rgba16f="
                  << fsr4vk::canReadStorageImage(device, with_block,
                                                 VK_FORMAT_R16G16B16A16_SFLOAT)
                  << '\n';

        // The provider is handed a device plus its resolver, so play that side
        // too: a device created from this instance drives the same table the
        // provider's check walks and DeviceDispatch binds, which is what the
        // runtime's [entry-points] line reports on real hardware.
        // Enabling an extension on the device is what exposes its entry points
        // through vkGetDeviceProcAddr: with a bare create-info the harness device
        // answered only the three promoted pairs, and DeviceDispatch refused it
        // by naming the five descriptor rows - the loud version of the silent
        // null-pointer path this used to leave to createDescriptorLayouts.
        // Ask for exactly what this device reports, which is a fact the snapshot
        // already holds; the matrix extension is absent on Intel and stays out.
        std::vector<const char*> enable_names;
        for (const char* name : {VK_EXT_DESCRIPTOR_BUFFER_EXTENSION_NAME,
                                 VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME}) {
            if (with_block.hasExtension(name)) {
                enable_names.push_back(name);
            }
        }

        VkDeviceCreateInfo device_info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        device_info.enabledExtensionCount = static_cast<uint32_t>(enable_names.size());
        device_info.ppEnabledExtensionNames = enable_names.empty() ? nullptr : enable_names.data();

        VkDevice vk_device = VK_NULL_HANDLE;
        if (vkCreateDevice(device, &device_info, nullptr, &vk_device) == VK_SUCCESS) {
            const auto missing = fsr4vk::missingEntryPoints(vk_device, vkGetDeviceProcAddr);
            try {
                const fsr4vk::DeviceDispatch real_dispatch(
                    vk_device, vkGetDeviceProcAddr, with_block.properties.apiVersion);
                std::cout << "  entry_points " << real_dispatch.describe()
                          << " provider_missing=" << missing.size() << '\n';
            } catch (const std::exception& error) {
                std::cout << "  entry_points refused: " << error.what() << '\n';
            }
            vkDestroyDevice(vk_device, nullptr);
        } else {
            std::cout << "  note: this loader refuses vkCreateDevice, entry points unprobed\n";
        }
    }

    // One table, two consumers. The provider's refusal list and the runtime's
    // typed binding have to name the same rows, which is the whole point of
    // moving the list next to the resolver. These stub resolvers imitate the
    // two situations a loader on this machine may never show: a driver that
    // implements only the KHR spellings of the promoted pairs, and a device that
    // is one descriptor row short. Neither reaches a driver at all.
    {
        const auto khr_only = reinterpret_cast<VkDevice>(static_cast<uintptr_t>(1));
        const fsr4vk::DeviceDispatch khr_dispatch(
            khr_only, stubDeviceProcAddr, VK_API_VERSION_1_4);
        std::cout << "  khr_only_loader " << khr_dispatch.describe()
                  << " provider_missing="
                  << fsr4vk::missingEntryPoints(khr_only, stubDeviceProcAddr).size()
                  << '\n';
    }

    {
        const auto short_row = reinterpret_cast<VkDevice>(static_cast<uintptr_t>(2));
        const auto provider_missing =
            fsr4vk::missingEntryPoints(short_row, stubDeviceProcAddr);
        std::string dispatch_says;
        try {
            const fsr4vk::DeviceDispatch short_dispatch(
                short_row, stubDeviceProcAddr, VK_API_VERSION_1_4);
            dispatch_says = "resolved anyway " + short_dispatch.describe();
        } catch (const std::exception& error) {
            dispatch_says = error.what();
        }
        const bool same_row = provider_missing.size() == 1 &&
                              dispatch_says.find(provider_missing[0]) != std::string::npos;
        std::cout << "  one_row_short provider="
                  << fsr4vk::joinBlockers(provider_missing) << " dispatch=" << dispatch_says
                  << " same_row=" << same_row << '\n';
    }

    // The devices enumerated here die with the instance, and the cache keys
    // itself on those handles, so it is dropped before the instance goes away.
    fsr4vk::clearCapabilityCache();
    vkDestroyInstance(instance, nullptr);
    return 0;
}
