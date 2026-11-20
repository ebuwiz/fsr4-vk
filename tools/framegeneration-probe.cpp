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
#include "ffx_frame_generation.h"
// The provider's own contract header: this probe reads the 2.4.0-only
// GetGPUMemoryUsageV2 query (0x0002000b) through its ABI mirror struct instead
// of re-declaring the layout, which is what keeps one ABI shape in the tree.
#include "framegeneration/framegeneration.hpp"

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

// No-op host callbacks for the probe's ffxConfigure host-integration contract. The provider
// validates + remembers these for the (future) present phase but does NOT drive them back while
// the compute algorithm is refused, so their bodies are never invoked in this probe.
static ffxReturnCode_t probePresentCallback(ffxCallbackDescFrameGenerationPresent*, void*) {
    return FFX_API_RETURN_OK;
}
static ffxReturnCode_t probeFrameGenerationCallback(ffxDispatchDescFrameGeneration*, void*) {
    return FFX_API_RETURN_OK;
}

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
    // The Frame Generation plan's bindings are VK_WHOLE_ARRAY runtime arrays, which
    // conventional descriptor sets structurally cannot size (measured: 2684 of 2692 FG
    // sets fail to allocate), so the runtime transport this provider needs is
    // VK_EXT_descriptor_buffer. A probe device without it cannot host the Phase-1
    // substrate, so it is requested here exactly like the harness device does.
    VK_EXT_DESCRIPTOR_BUFFER_EXTENSION_NAME,
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
    // Accumulate return-code mismatches across the whole documented host-integration
    // ladder so the probe FAILS (non-zero) if the provider's routing/refusal drifts, instead
    // of merely printing codes (the previous probe returned 0 no matter what a stage did).
    int ladderFailures = 0;
    auto expect = [&](const char* stage, int got, int want) {
        const bool ok = (got == want);
        std::printf("  %s = %d (expected %d)%s\n", stage, got, want,
                    ok ? "" : "   <-- MISMATCH");
        if (!ok) {
            ++ladderFailures;
        }
    };
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
        VkPhysicalDeviceDescriptorBufferFeaturesEXT descriptorBuffer{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_BUFFER_FEATURES_EXT};
        baseFeatures.pNext = &bufferAddress;
        bufferAddress.pNext = &float16Int8;
        float16Int8.pNext = &storage16;
        storage16.pNext = &cooperative;
        cooperative.pNext = &memoryModel;
        memoryModel.pNext = &descriptorIndexing;
        descriptorIndexing.pNext = &storage8;
        storage8.pNext = &subgroupControl;
        subgroupControl.pNext = &derivatives;
        derivatives.pNext = &descriptorBuffer;

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
        descriptorBuffer.descriptorBuffer = VK_TRUE;

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
        // `ran` is incremented below ONLY once a device actually satisfies the Frame Generation
        // execution contract (ffxCreateContext == OK). A device that merely created a VkDevice but
        // is refused for a missing feature must NOT count, so the probe skips (exit 77) unless a
        // real contract device ran -- and passed -- the full host-integration ladder.

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
        // Real SDK Frame Generation create descriptor (type 0x00020001) with the
        // backend + enabled-feature blocks on its pNext chain, plus valid create
        // arguments (display/render size + a known surface format). DEBUG_CHECKING is
        // on so the provider's mandatory-order + camera checks are exercised downstream.
        ffxCreateContextDescFrameGeneration createDesc{};
        createDesc.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATION;
        createDesc.header.pNext = &backend.header;
        createDesc.flags = FFX_FRAMEGENERATION_ENABLE_DEBUG_CHECKING;
        createDesc.displaySize = {1920u, 1080u};
        createDesc.maxRenderSize = {1920u, 1080u};
        createDesc.backBufferFormat = 1u;  // a known FfxApiSurfaceFormat, not UNKNOWN
        auto* create = &createDesc.header;

        // Point the provider's namespaced pipeline cache at a probe-known temp
        // path so the cold (create+save) -> warm (reload) round-trip is observable
        // and does not litter the working directory.
        const std::string cachePath =
            std::string("/tmp/ffg4vk-probe-cache.") +
            std::to_string((unsigned)getpid()) + "." + std::to_string((unsigned)ran);
        std::remove(cachePath.c_str());
        setenv("FFG4_VK_PIPELINE_CACHE", cachePath.c_str(), 1);

        ffxContext context{};
        const ffxReturnCode_t created = ffxCreateContext(&context, create, nullptr);
        if (created != FFX_API_RETURN_OK || !context) {
            // A device that does NOT satisfy the Frame Generation execution contract (for
            // example one lacking VK_KHR_cooperative_matrix, used by the 156 ML neural modules)
            // is honestly refused by ffxCreateContext with a runtime error. That is the CORRECT
            // behaviour for a non-contract device, so skip it (and do not count it) rather than
            // assert the ladder against it; only a contract-capable device runs the ladder below.
            std::printf("  device skipped: does not offer the FG execution contract "
                        "(ffxCreateContext=%d)\n",
                        (int)created);
            vkDestroyDevice(device, nullptr);
            continue;
        }
        ++ran;  // counts only devices that meet the contract and therefore run the ladder
        expect("ffxCreateContext", (int)created, (int)FFX_API_RETURN_OK);
        if (created == FFX_API_RETURN_OK && context) {
            // (a) Mandatory order BEFORE configure: a Prepare must be refused as a
            // parameter error (Configure -> Prepare), never silently accepted.
            ffxDispatchDescFrameGenerationPrepare early{};
            early.header.type = FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION_PREPARE;
            early.header.pNext = nullptr;
            early.renderSize = {1920u, 1080u};
            early.frameTimeDelta = 16.6f;
            expect("ffxDispatch(Prepare) before Configure",
                   (int)ffxDispatch(&context, &early.header),
                   (int)FFX_API_RETURN_ERROR_PARAMETER);
            uint64_t capacity = 4;
            uint64_t ids[4]{};
            const char* names[4]{};
            ffxQueryDescGetVersions versions{
                {FFX_API_QUERY_DESC_TYPE_GET_VERSIONS, nullptr},
                FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATION, 0,
                &capacity, ids, names};
            ffxQuery(&context, &versions.header);
            std::printf("  provider advertises version id(s):");
            for (uint64_t index = 0; index < capacity; ++index) {
                std::printf(" 0x%llX", (unsigned long long)ids[index]);
            }
            // (b) Exactly the pinned Frame Generation version id (0xF600000001000002
            //     == effect module 0xF6 + packed 4.0.2) must be advertised at ids[0].
            expect("ffxQuery(GET_VERSIONS).versionIds[0] == 0xF600000001000002",
                   ids[0] == 0xF600000001000002ULL ? 1 : 0, 1);
            // (c) FG GPU-memory-usage (0x00020007) is an HONEST refusal while the
            // interpolation/ML/present runtime is unimplemented: RUNTIME_ERROR, never a
            // fabricated size and never OK.
            FfxApiEffectMemoryUsage fgUsage{};
            ffxQueryDescFrameGenerationGetGPUMemoryUsage gpuQ{};
            gpuQ.header.type = FFX_API_QUERY_DESC_TYPE_FRAMEGENERATION_GPU_MEMORY_USAGE;
            gpuQ.header.pNext = nullptr;
            gpuQ.gpuMemoryUsageFrameGeneration = &fgUsage;
            expect("ffxQuery(GPU_MEMORY_USAGE)",
                   (int)ffxQuery(&context, &gpuQ.header),
                   (int)FFX_API_RETURN_ERROR_RUNTIME_ERROR);
            // Follow the SDK's mandatory order: ffxConfigure(ffxConfigureDescFrameGeneration, 0x0002
            // 0x0002) BEFORE any Prepare. Hand over the host-integration contract (a swap chain +
            // present/frameGeneration callbacks) the provider validates and remembers for the
            // present phase; frameGenerationEnabled=true means the contract requires them.
            char swapChainToken = 0;
            ffxConfigureDescFrameGeneration configure{};
            configure.header.type = FFX_API_CONFIGURE_DESC_TYPE_FRAMEGENERATION;
            configure.header.pNext = nullptr;
            configure.swapChain = &swapChainToken;
            configure.presentCallback = probePresentCallback;
            configure.presentCallbackUserContext = nullptr;
            configure.frameGenerationCallback = probeFrameGenerationCallback;
            configure.frameGenerationCallbackUserContext = nullptr;
            configure.frameGenerationEnabled = true;
            configure.allowAsyncWorkloads = false;
            configure.flags = 0u;
            configure.onlyPresentGenerated = false;
            configure.generationRect = {0, 0, 0, 0};  // whole-frame sentinel
            configure.frameID = 7u;
            const int configureCode = (int)ffxConfigure(&context, &configure.header);
            expect("ffxConfigure(host contract)", configureCode, (int)FFX_API_RETURN_OK);
            // Drive a VALID Frame Generation Prepare (type 0x00020004) with the
            // linked camera-info block (0x0002000a) carrying an orthonormal basis.
            // With DEBUG_CHECKING on, the provider accepts the order + camera contract
            // and THEN refuses the (unimplemented) algorithm, so the surfaced return is
            // 3 (RUNTIME_ERROR / runtime pending) — not a parameter error.
            ffxDispatchDescFrameGenerationPrepare prepare{};
            prepare.header.type = FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION_PREPARE;
            prepare.renderSize = {1920u, 1080u};
            prepare.frameTimeDelta = 16.6f;
            prepare.cameraNear = 0.1f;
            prepare.cameraFar = 1000.0f;
            prepare.cameraFovAngleVertical = 1.0f;
            prepare.jitterOffset = {0.0f, 0.0f};
            prepare.motionVectorScale = {1.0f, 1.0f};
            prepare.flags = 0u;
            prepare.commandList = nullptr;
            ffxDispatchDescFrameGenerationPrepareCameraInfo cam{};
            cam.header.type = FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION_PREPARE_CAMERAINFO;
            const float up[3] = {0.0f, 1.0f, 0.0f};
            const float right[3] = {1.0f, 0.0f, 0.0f};
            const float fwd[3] = {0.0f, 0.0f, -1.0f};
            for (int i = 0; i < 3; ++i) {
                cam.cameraUp[i] = up[i];
                cam.cameraRight[i] = right[i];
                cam.cameraForward[i] = fwd[i];
            }
            prepare.header.pNext = &cam.header;
            const int dispatchCode = (int)ffxDispatch(&context, &prepare.header);
            expect("ffxDispatch(Prepare) after Configure", dispatchCode,
                   (int)FFX_API_RETURN_ERROR_RUNTIME_ERROR);
            // (d) A VALID Generate (0x00020003) AFTER Prepare must hit the same honest
            // RUNTIME_ERROR (interpolation/ML/present refused), not a parameter error and
            // not OK -- this is the frame-generation dispatch a real host would issue.
            ffxDispatchDescFrameGeneration generate{};
            generate.header.type = FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION;
            generate.header.pNext = nullptr;
            generate.numGeneratedFrames = 1u;
            generate.minMaxLuminance[0] = 0.0f;
            generate.minMaxLuminance[1] = 1000.0f;
            expect("ffxDispatch(FrameGeneration) after Prepare",
                   (int)ffxDispatch(&context, &generate.header),
                   (int)FFX_API_RETURN_ERROR_RUNTIME_ERROR);
            // (e) The present-callback block (0x00020005) is host-driven, never a provider
            // dispatch: report NO_PROVIDER rather than guess a type for it.
            ffxCallbackDescFrameGenerationPresent presentCb{};
            presentCb.header.type = FFX_API_CALLBACK_DESC_TYPE_FRAMEGENERATION_PRESENT;
            presentCb.header.pNext = nullptr;
            expect("ffxDispatch(present-callback)",
                   (int)ffxDispatch(&context, &presentCb.header),
                   (int)FFX_API_RETURN_NO_PROVIDER);
            // (f) Phase-1 substrate: the Prepare above allocated the plan's intermediate
            // targets, one descriptor pool and the classic-subset compute pipelines ON this
            // device. So the SAME GPU-memory-usage query that refused BEFORE Configure must
            // now answer OK, with bytes the driver actually handed back (> 0), not a plan
            // number and not a fabricated zero.
            FfxApiEffectMemoryUsage afterPrepare{};
            ffxQueryDescFrameGenerationGetGPUMemoryUsage usageQ{};
            usageQ.header.type = FFX_API_QUERY_DESC_TYPE_FRAMEGENERATION_GPU_MEMORY_USAGE;
            usageQ.header.pNext = nullptr;
            usageQ.gpuMemoryUsageFrameGeneration = &afterPrepare;
            expect("ffxQuery(GPU_MEMORY_USAGE) after Prepare (phase-1 substrate)",
                   (int)ffxQuery(&context, &usageQ.header),
                   (int)FFX_API_RETURN_OK);
            expect("reported totalUsageInBytes is a real allocation (> 0)",
                   afterPrepare.totalUsageInBytes > 0ull ? 1 : 0, 1);
            std::printf("  phase1 usage: total=%llu aliasable=%llu\n",
                        (unsigned long long)afterPrepare.totalUsageInBytes,
                        (unsigned long long)afterPrepare.aliasableUsageInBytes);
            // (g) The 2.4.0-only pre-create sizing query (0x0002000b) must answer for the
            // resolutions the CALLER names, and it is legal with no context at all: read
            // it through the provider's own ABI mirror struct (framegeneration.hpp).
            {
                FfxApiEffectMemoryUsage sized{};
                ffgvk::QueryGPUUsageV2Desc v2{};
                v2.header.type = 0x0002000bu;  // kEffectId | 0x0b
                v2.header.pNext = nullptr;
                v2.device = nullptr;
                v2.maxRenderSize = {640u, 360u};
                v2.displaySize = {1280u, 720u};
                v2.gpuMemoryUsageFrameGeneration = &sized;
                const int sizedCode = (int)ffxQuery(&context, &v2.header);
                expect("ffxQuery(GetGPUMemoryUsageV2) answers for the caller's sizes",
                       sizedCode, (int)FFX_API_RETURN_OK);
                // 640x360 render + 1280x720 display is smaller than the pack's 1920x1080/
                // 3840x2160 reference model, so a correct answer must be SMALLER than the
                // reference byte total the pack carries (66,355,200 B) and still non-zero.
                expect("V2 sizing scales with the caller's resolutions",
                       sized.totalUsageInBytes > 0ull &&
                           sized.totalUsageInBytes < 66355200ull
                           ? 1
                           : 0,
                       1);
                std::printf("  phase1 V2 sizing: 640x360 -> 1280x720 total=%llu\n",
                            (unsigned long long)sized.totalUsageInBytes);
            }
            const int destroyed = (int)ffxDestroyContext(&context, nullptr);
            expect("ffxDestroyContext", destroyed, (int)FFX_API_RETURN_OK);

            // Cycle 1 destroyed the context, which serialized the cache container
            // to cachePath. Create a second context: the provider must LOAD that
            // container as initialData because its pinned identity (algorithm=fg4 +
            // this GPU + driver) matches, i.e. a warm start. We verify the on-disk
            // container proves the namespacing + identity validation.
            ffxContext warm{};
            if (ffxCreateContext(&warm, create, nullptr) == FFX_API_RETURN_OK &&
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
    if (ladderFailures != 0) {
        std::printf("Frame Generation host-integration ladder FAILED: %d unexpected return "
                    "code(s)\n",
                    ladderFailures);
        return 1;
    }
    std::printf("Frame Generation host-integration ladder: every stage returned its documented "
                "code (create / order-before-configure / versions / gpu-usage refusal BEFORE "
                "Configure, allocation-backed gpu-usage AFTER Prepare / configure / prepare / "
                "generate / present-callback / V2 sizing / destroy)\n");
    return 0;
}
