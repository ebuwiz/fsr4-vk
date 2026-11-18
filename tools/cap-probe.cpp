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

#include <cstddef>
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

    // The rung the provider would pick, printed next to the blocker lists because
    // the two only mean something together: a device that answers int8=false is
    // either a fallback to fp8 or a refusal, and which one it is decides whether
    // the app gets an upscaler at all.
    const auto resolved = fsr4vk::resolveModel(capabilities, fsr4vk::ModelChoice::Auto);
    std::cout << "  " << label << " auto_ladder="
              << (resolved.chosen ? fsr4vk::modelName(*resolved.chosen) : "none")
              << " reason=" << resolved.reason << '\n';
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

// A snapshot that satisfies every line the shared runtime contract and both model
// contracts check, so the only thing left to vary is the rung under test. Fabricated
// rather than probed: a device that refuses INT8's own SPIR-V capabilities is what
// the ladder exists for, and this machine has none, so the ladder has to be checked
// without waiting for that hardware.
fsr4vk::GpuCapabilities syntheticHost() {
    fsr4vk::GpuCapabilities host{};

    host.properties.apiVersion = VK_API_VERSION_1_3;
    host.core.shaderInt16 = VK_TRUE;
    host.core.shaderStorageImageReadWithoutFormat = VK_TRUE;
    host.core.shaderStorageImageWriteWithoutFormat = VK_TRUE;
    host.float16_int8.shaderFloat16 = VK_TRUE;
    host.float16_int8.shaderInt8 = VK_TRUE;
    host.storage8.storageBuffer8BitAccess = VK_TRUE;
    host.buffer_device_address.bufferDeviceAddress = VK_TRUE;
    host.descriptor_indexing.runtimeDescriptorArray = VK_TRUE;
    host.subgroups.supportedStages = VK_SHADER_STAGE_COMPUTE_BIT;
    host.subgroups.supportedOperations = VK_SUBGROUP_FEATURE_QUAD_BIT;
    host.subgroup_size_control.subgroupSizeControl = VK_TRUE;
    host.subgroup_size_control.computeFullSubgroups = VK_TRUE;
    host.memory_model.vulkanMemoryModel = VK_TRUE;
    host.cooperative.cooperativeMatrixSupportedStages = VK_SHADER_STAGE_COMPUTE_BIT;
    host.float_controls.shaderDenormPreserveFloat16 = VK_TRUE;
    host.subgroup_sizes.requiredSubgroupSizeStages = VK_SHADER_STAGE_COMPUTE_BIT;

    // The two lines that are the INT8 model's own SPIR-V contract.
    host.vulkan13.shaderIntegerDotProduct = VK_TRUE;
    host.compute_derivatives.computeDerivativeGroupLinear = VK_TRUE;

    // And the FP8 model's: extension names plus the feature bits they promote.
    host.shader_float8.shaderFloat8 = VK_TRUE;
    host.shader_float8.shaderFloat8CooperativeMatrix = VK_TRUE;
    host.cooperative_matrix.cooperativeMatrix = VK_TRUE;
    host.extensions = {{VK_EXT_SHADER_FLOAT8_EXTENSION_NAME, 1},
                       {VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME, 1}};
    // matrix_probe_route stays Unprobed on purpose: an unprobed device is judged
    // on extension and feature bits alone, so nothing here can look FP8-capable
    // for a reason this fixture did not set.
    return host;
}

// One ladder case: the rung the shared resolver would pick, and a substring its
// reason has to carry. A disagreement counts against the probe's exit code, because
// picking the wrong rung silently is worse than a device that refuses loudly.
std::size_t ladder_failures = 0;

void checkLadder(const char* case_name,
                 const fsr4vk::ModelResolution& resolved,
                 const std::string& expected_model,
                 const std::string& expected_reason) {
    const std::string actual =
        resolved.chosen ? fsr4vk::modelName(*resolved.chosen) : "none";
    const bool model_ok = actual == expected_model;
    const bool reason_ok = resolved.reason.find(expected_reason) != std::string::npos;
    std::cout << "  synthetic_ladder case=" << case_name << " verdict=" << actual
              << " expected=" << expected_model << " ok=" << std::boolalpha
              << (model_ok && reason_ok) << " reason=" << resolved.reason << '\n';
    if (!model_ok || !reason_ok) {
        ++ladder_failures;
    }
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

        // Phase 9 evidence: which descriptor transports this host can actually run,
        // and the three SPIR-V capabilities the shipped modules declare now that
        // the capability contract checks them. A host that never implemented
        // VK_EXT_descriptor_buffer should still answer sets=yes, which is the
        // whole point of the fallback; the gate columns show whether a model
        // request would be refused for a capability its own shaders use.
        std::cout << "  descriptor_transports buffers="
                  << (with_block.supportsDescriptorBuffers() ? "yes" : "no")
                  << " sets=" << (with_block.supportsDescriptorSets() ? "yes" : "no")
                  << " dot_product="
                  << (with_block.vulkan13.shaderIntegerDotProduct ? "yes" : "no")
                  << " derivative_group_linear="
                  << (with_block.compute_derivatives.computeDerivativeGroupLinear ? "yes" : "no")
                  << " float_controls_api="
                  << (with_block.properties.apiVersion >= VK_API_VERSION_1_2 ? "yes" : "no")
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
            // Both transports against the same device: the buffers contract is
            // all eight rows, the sets contract is the three promoted core pairs,
            // and a host that never implemented VK_EXT_descriptor_buffer should
            // answer the second one cleanly instead of refusing five functions
            // it was never going to call.
            for (const auto backend : {fsr4vk::DescriptorBackend::DescriptorBuffers,
                                       fsr4vk::DescriptorBackend::DescriptorSets}) {
                const auto missing =
                    fsr4vk::missingEntryPoints(vk_device, vkGetDeviceProcAddr, backend);
                try {
                    const fsr4vk::DeviceDispatch real_dispatch(
                        vk_device,
                        vkGetDeviceProcAddr,
                        with_block.properties.apiVersion,
                        backend);
                    std::cout << "  entry_points " << real_dispatch.describe()
                              << " provider_missing=" << missing.size() << '\n';
                } catch (const std::exception& error) {
                    std::cout << "  entry_points refused: " << error.what() << '\n';
                }
            }
            vkDestroyDevice(vk_device, nullptr);
        } else {
            std::cout << "  note: this loader refuses vkCreateDevice, entry points unprobed\n";
        }
    }

    // One table, two consumers, both transports. The provider's refusal list and
    // the runtime's typed binding have to name the same rows, which is the whole
    // point of moving the list next to the resolver. These stub resolvers imitate
    // the two situations a loader on this machine may never show: a driver that
    // implements only the KHR spellings of the promoted pairs, and a device that
    // is one descriptor row short. Neither reaches a driver at all.
    {
        const auto khr_only = reinterpret_cast<VkDevice>(static_cast<uintptr_t>(1));
        for (const auto backend : {fsr4vk::DescriptorBackend::DescriptorBuffers,
                                   fsr4vk::DescriptorBackend::DescriptorSets}) {
            const fsr4vk::DeviceDispatch khr_dispatch(
                khr_only, stubDeviceProcAddr, VK_API_VERSION_1_4, backend);
            std::cout << "  khr_only_loader " << khr_dispatch.describe()
                      << " provider_missing="
                      << fsr4vk::missingEntryPoints(khr_only, stubDeviceProcAddr, backend).size()
                      << '\n';
        }
    }

    {
        const auto short_row = reinterpret_cast<VkDevice>(static_cast<uintptr_t>(2));
        // The stub device is short of one descriptor-buffer row. Under the buffers
        // contract both consumers must name that same row; under the sets contract
        // neither may mention it, which is the measurable half of "a host without
        // the extension is not refused for functions it would never call".
        for (const auto backend : {fsr4vk::DescriptorBackend::DescriptorBuffers,
                                   fsr4vk::DescriptorBackend::DescriptorSets}) {
            const auto provider_missing =
                fsr4vk::missingEntryPoints(short_row, stubDeviceProcAddr, backend);
            std::string dispatch_says;
            try {
                const fsr4vk::DeviceDispatch short_dispatch(
                    short_row, stubDeviceProcAddr, VK_API_VERSION_1_4, backend);
                dispatch_says = "resolved anyway " + short_dispatch.describe();
            } catch (const std::exception& error) {
                dispatch_says = error.what();
            }
            const bool same_row = provider_missing.size() == 1 &&
                                  dispatch_says.find(provider_missing[0]) != std::string::npos;
            const bool row_dropped =
                backend == fsr4vk::DescriptorBackend::DescriptorSets && provider_missing.empty();
            std::cout << "  one_row_short "
                      << fsr4vk::descriptorBackendName(backend) << " provider="
                      << fsr4vk::joinBlockers(provider_missing) << " dispatch=" << dispatch_says
                      << " same_row=" << same_row << " row_dropped=" << row_dropped << '\n';
        }
    }

    // The model ladder, checked on snapshots this machine cannot be expected to
    // have: a host that runs INT8, a host whose driver refuses INT8's own SPIR-V
    // contract (what a low-entry AMD APU looks like), and a host that refuses both.
    // Before this phase the second one produced no upscaler at all, so the case
    // that matters most is the one that needs no hardware to notice.
    {
        const auto both_ok = syntheticHost();
        checkLadder("both_ok_auto",
                    fsr4vk::resolveModel(both_ok, fsr4vk::ModelChoice::Auto), "int8",
                    "int8 contract satisfied");
        checkLadder("both_ok_explicit_fp8",
                    fsr4vk::resolveModel(both_ok, fsr4vk::ModelChoice::ForceFp8), "fp8",
                    "fp8 contract satisfied");

        auto int8_refused = syntheticHost();
        int8_refused.vulkan13.shaderIntegerDotProduct = VK_FALSE;
        checkLadder("dot_product_missing_auto",
                    fsr4vk::resolveModel(int8_refused, fsr4vk::ModelChoice::Auto), "fp8",
                    "falling back to fp8");
        checkLadder("dot_product_missing_explicit_int8",
                    fsr4vk::resolveModel(int8_refused, fsr4vk::ModelChoice::ForceInt8), "none",
                    "the request is strict");
        checkLadder("dot_product_missing_explicit_fp8",
                    fsr4vk::resolveModel(int8_refused, fsr4vk::ModelChoice::ForceFp8), "fp8",
                    "fp8 contract satisfied");

        auto derivative_missing = syntheticHost();
        derivative_missing.compute_derivatives.computeDerivativeGroupLinear = VK_FALSE;
        checkLadder("derivative_group_missing_auto",
                    fsr4vk::resolveModel(derivative_missing, fsr4vk::ModelChoice::Auto), "fp8",
                    "computeDerivativeGroupLinear");

        auto neither_runs = syntheticHost();
        neither_runs.vulkan13.shaderIntegerDotProduct = VK_FALSE;
        neither_runs.shader_float8.shaderFloat8 = VK_FALSE;
        checkLadder("neither_runs_auto",
                    fsr4vk::resolveModel(neither_runs, fsr4vk::ModelChoice::Auto), "none",
                    "neither model runs");
        checkLadder("neither_runs_lists_both_rungs",
                    fsr4vk::resolveModel(neither_runs, fsr4vk::ModelChoice::Auto), "none",
                    "fp8 blocked (");
        std::cout << "  synthetic_ladder failures=" << ladder_failures << '\n';
    }

    // The devices enumerated here die with the instance, and the cache keys
    // itself on those handles, so it is dropped before the instance goes away.
    fsr4vk::clearCapabilityCache();
    vkDestroyInstance(instance, nullptr);
    return ladder_failures == 0 ? 0 : 1;
}
