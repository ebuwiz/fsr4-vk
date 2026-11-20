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
#include "framegeneration/fg-runtime.hpp"

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

// Which plan modules to actually dispatch, and over what canvas, comes from the COMMAND
// LINE (opt-in; the contract ladder never runs GPU work). --dispatch-one=N,
// --dispatch-orders=a,b,c and --dispatch-canvas=WxH are parsed by hand rather than with a
// parser library because this tree has none, and an unrecognised --dispatch-* value is
// reported loudly instead of being ignored (an ignored typo would silently mean "no
// dispatch at all", which looks like a passing measurement).
static void parseDispatchArgs(int argc, char** argv, std::vector<std::uint32_t>& orders,
                              std::uint32_t& canvasWidth, std::uint32_t& canvasHeight,
                              std::uint32_t& dispatchRepeat, std::uint32_t& sweepCount,
                              bool& badArg) {
    // Every prefix length comes from std::strlen OF THE PREFIX ITSELF. Hard-coding them is
    // how this function first shipped wrong: "--dispatch-orders=" is 18 characters, not 16,
    // so the comparison matched on a 16-byte prefix and then parsed the tail starting at
    // "s=0,1,2..." - the malformed-argument path, and (worse) "--dispatch-canvas=64x64"
    // parsed as "s=64x64", giving a canvas of 128x64 from a request for 64x64. Both bugs
    // produced plausible-looking output, which is why the tail must end at a NUL here.
    static constexpr const char* kOne = "--dispatch-one=";
    static constexpr const char* kOrders = "--dispatch-orders=";
    static constexpr const char* kCanvas = "--dispatch-canvas=";
    static constexpr const char* kRepeat = "--dispatch-repeat=";
    static constexpr const char* kSweep = "--dispatch-sweep=";
    static constexpr const char* kAny = "--dispatch";
    // Own digit parser rather than std::strtoul: strtoul wants a char** for its end pointer,
    // and these are const char* slices of argv - a const_cast would be the lazy version of
    // the bug above (it compiles, and hides where the tail actually ended).
    auto parseUint = [](const char* text) {
        struct Parsed {
            std::uint64_t value;
            const char* end;
            bool ok;
        };
        const char* p = text;
        std::uint64_t value = 0;
        while (*p >= '0' && *p <= '9') {
            value = value * 10ull + static_cast<std::uint64_t>(*p - '0');
            ++p;
        }
        return Parsed{value, p, p != text};
    };
    for (int i = 1; i < argc; ++i) {
        const char* arg = argv[i];
        if (arg[0] != '-' || arg[1] != '-') {
            continue;  // positional arguments are none of this probe's business
        }
        if (std::strncmp(arg, kOne, std::strlen(kOne)) == 0) {
            const auto parsed = parseUint(arg + std::strlen(kOne));
            if (!parsed.ok || *parsed.end != '\0') {
                badArg = true;
            } else {
                orders.push_back(static_cast<std::uint32_t>(parsed.value));
            }
        } else if (std::strncmp(arg, kOrders, std::strlen(kOrders)) == 0) {
            const char* p = arg + std::strlen(kOrders);
            if (*p == '\0') {
                badArg = true;  // --dispatch-orders= with no list is a typo, not "none"
            }
            while (*p != '\0') {
                const auto parsed = parseUint(p);
                if (!parsed.ok) {
                    badArg = true;
                    break;
                }
                orders.push_back(static_cast<std::uint32_t>(parsed.value));
                p = parsed.end;
                if (*p == ',') {
                    ++p;
                    if (*p == '\0') {
                        badArg = true;  // trailing comma: 0,1,2, is a typo for a list
                    }
                } else if (*p != '\0') {
                    badArg = true;
                    break;
                }
            }
        } else if (std::strncmp(arg, kCanvas, std::strlen(kCanvas)) == 0) {
            const char* text = arg + std::strlen(kCanvas);
            const char* at = std::strchr(text, 'x');
            if (at == nullptr) {
                const auto parsed = parseUint(text);
                if (!parsed.ok || *parsed.end != '\0') {
                    badArg = true;
                } else {
                    canvasWidth = static_cast<std::uint32_t>(parsed.value);
                    canvasHeight = canvasWidth;
                }
            } else {
                const auto first = parseUint(text);
                const auto second = parseUint(at + 1);
                if (!first.ok || first.end != at || !second.ok || *second.end != '\0') {
                    badArg = true;
                } else {
                    canvasWidth = static_cast<std::uint32_t>(first.value);
                    canvasHeight = static_cast<std::uint32_t>(second.value);
                }
            }
        } else if (std::strncmp(arg, kRepeat, std::strlen(kRepeat)) == 0) {
            const auto parsed = parseUint(arg + std::strlen(kRepeat));
            if (!parsed.ok || *parsed.end != '\0') {
                badArg = true;
            } else {
                dispatchRepeat = static_cast<std::uint32_t>(parsed.value);
            }
        } else if (std::strncmp(arg, kSweep, std::strlen(kSweep)) == 0) {
            const auto parsed = parseUint(arg + std::strlen(kSweep));
            if (!parsed.ok || *parsed.end != '\0') {
                badArg = true;
            } else {
                sweepCount = static_cast<std::uint32_t>(parsed.value);
            }
        } else if (std::strncmp(arg, kAny, std::strlen(kAny)) == 0) {
            badArg = true;  // a --dispatch flag this probe does not know
        }
    }
    // A plan-order above 473 or a canvas outside 16..8192 is a typo on the command line,
    // not a driver answer; rejecting them HERE names the argument instead of leaving the
    // provider to report something that reads like a measurement.
    for (const std::uint32_t order : orders) {
        if (order > 473u) {
            badArg = true;  // the plan carries 474 modules
        }
    }
    if (canvasWidth != 0u && (canvasWidth < 16u || canvasWidth > 8192u)) {
        badArg = true;
    }
    if (canvasHeight != 0u && (canvasHeight < 16u || canvasHeight > 8192u)) {
        badArg = true;
    }
    // Repeating the same pass 64 times inside one command buffer is already far past any
    // chain the plan describes (and any measurement worth making); above that the number
    // is a typo, and a typo here would read as a timing result.
    if (dispatchRepeat > 64u) {
        badArg = true;
    }
    // 92 is the classic subset the plan carries; a sweep past it would silently re-run the
    // ML chain's orders with no names bound to them, which is a different (and refused)
    // question, so the argument is rejected instead of the run being quietly partial.
    if (sweepCount > 92u) {
        badArg = true;
    }
}

int main(int argc, char** argv) {
    std::vector<std::uint32_t> dispatchOrders;
    std::uint32_t canvasWidth = 0u;
    std::uint32_t canvasHeight = 0u;
    std::uint32_t dispatchRepeat = 0u;
    std::uint32_t dispatchSweepCount = 0u;
    bool badDispatchArg = false;
    parseDispatchArgs(argc, argv, dispatchOrders, canvasWidth, canvasHeight, dispatchRepeat,
                      dispatchSweepCount, badDispatchArg);
    if (badDispatchArg) {
        std::printf("unrecognised or malformed --dispatch argument; this probe understands "
                    "--dispatch-one=N, --dispatch-orders=a,b,c and --dispatch-canvas=WxH\n");
        return 2;
    }
    // (h) The SWEEP MODE, in ONE process: --dispatch-sweep=N asks for plan orders
    // 0..N-1 (N<=92) with no command-line list at all. This exists because the alternative
    // - one process per module - was measured to be worse than it looks: every probe process
    // creates a VkInstance and a VkDevice on the host's real desktop GPU session (KDE/Wayland
    // here), so 92 processes tear that session down and re-create it 92 times. Inside one
    // process the device is created once, and a pass that leaves the device unusable
    // (contextLost) is reported as SKIPPED-REMAINING with the orders named, so the run still
    // answers "how far does the plumbing get on this driver" without touching the display again.
    if (dispatchOrders.empty()) {
        for (std::uint32_t order = 0u; order < dispatchSweepCount; ++order) {
            dispatchOrders.push_back(order);
        }
    }
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
            // (f2) The MEASURED binding chain: ONE real compute dispatch of ONE plan module
            // through the substrate built above, ON THIS device, asked for through the
            // repo-internal query 0x000200f0. Opt-in (--dispatch-one=N /
            // --dispatch-orders=a,b,c / --dispatch-canvas=WxH), because a faulting pass puts
            // the device into hard recovery and the contract ladder above must stay runnable
            // on a box that runs no GPU work at all. Every number printed is the driver's:
            // whether the pool-bind -> set-offsets -> bind -> dispatch -> submit -> wait ->
            // read-back commands of this VK_EXT_descriptor_buffer revision can be driven at
            // all, what one pass costs, and whether the storage image the pass was handed
            // changed bytes across it.
            std::uint64_t dispatchAttempted = 0;
            std::uint64_t dispatchRan = 0;
            std::uint64_t dispatchChanged = 0;
            // Ran AND touched something are different claims. "Ran" means the driver executed
            // the pass without faulting; "wrote something" means at least one of the writable
            // resources the pass was handed changed. A pass that runs but writes nothing under
            // zero-filled push constants/uniforms is a HINT about what it branches on, not a
            // failure of the plumbing - and conflating the two would hide that.
            std::uint64_t dispatchWroteSomething = 0;
            for (std::size_t index = 0; index < dispatchOrders.size(); ++index) {
                const std::uint32_t order = dispatchOrders[index];
                ffgvk::PlanRuntime::DispatchFacts measured{};
                ffgvk::ProbeDispatchOneDesc probeQ{};
                probeQ.header.type = ffgvk::kProbeDispatchOneType;
                probeQ.header.pNext = nullptr;
                probeQ.planOrder = order;
                probeQ.canvasWidth = canvasWidth;
                probeQ.canvasHeight = canvasHeight;
                probeQ.dispatchRepeat = dispatchRepeat;
                probeQ.factsOut = &measured;
                const int probeCode = (int)ffxQuery(&context, &probeQ.header);
                ++dispatchAttempted;
                std::printf(
                    "  dispatch order=%u code=%d attempted=%d bound=%d submitted=%d ran=%d "
                    "ms_record=%.2f ms_submit=%.2f ms_wait=%.2f ms=%.2f repeats=%u "
                    "barrier=%s "
                    "pool=%lluB descriptors=%lluB groups=%ux%ux%u workgroup=%ux%ux%u canvas=%ux%u "
                    "planMaxExtent=%ux%u texel=%uB readBack=%lluB sets=%u bindings=%u out_changed=%d "
                    "wrote_sampled=%d sb_nonzeroB=%llu ub_nonzeroB=%llu any_writable_changed=%d "
                    "readback=%d context_lost=%d before=%016llX after=%016llX basis=%s failure=%s\n",
                    order, probeCode, measured.attempted ? 1 : 0, measured.bound ? 1 : 0,
                    measured.submitted ? 1 : 0, measured.ok ? 1 : 0, measured.msRecord,
                    measured.msSubmit, measured.msWait, measured.ms, measured.repeats,
                    measured.barrierFlags.empty() ? "none" : measured.barrierFlags.c_str(),
                    (unsigned long long)measured.poolBytes,
                    (unsigned long long)measured.descriptorBytesWritten, measured.groups[0],
                    measured.groups[1], measured.groups[2], measured.workgroup[0],
                    measured.workgroup[1], measured.workgroup[2], measured.canvasWidth,
                    measured.canvasHeight, measured.planMaxWidth, measured.planMaxHeight,
                    measured.bytesPerTexel, (unsigned long long)measured.readBackBytes,
                    measured.sets, measured.bindings, measured.outputChanged ? 1 : 0,
                    measured.wroteSampledImage ? 1 : 0,
                    (unsigned long long)measured.nonZeroStorageBufferBytes,
                    (unsigned long long)measured.nonZeroUniformBufferBytes,
                    measured.anyWritableOutputChanged ? 1 : 0,
                    measured.readbackAvailable ? 1 : 0,
                    measured.contextLost ? 1 : 0, (unsigned long long)measured.checksumBefore,
                    (unsigned long long)measured.checksumAfter,
                    measured.note.empty() ? "none" : measured.note.c_str(),
                    measured.failure.empty() ? "none" : measured.failure.c_str());
                if (measured.ok) {
                    ++dispatchRan;
                    if (measured.outputChanged) {
                        ++dispatchChanged;
                    }
                    if (measured.anyWritableOutputChanged) {
                        ++dispatchWroteSomething;
                    }
                }
                // The device is unusable after a faulted CS or after a submit that answered
                // VK_ERROR_OTHER; the remaining orders in THIS process would all report
                // VK_ERROR_OTHER and be read as driver refusals. Stop, name the cause, and
                // say which orders were NOT measured - the answer is "how far the plumbing
                // gets in one device lifetime", which is exactly the question that does NOT
                // require re-creating the host's desktop GPU session per module.
                if (measured.contextLost) {
                    std::printf("  dispatch sweep stopped_after_order=%u cause=%s not_measured=%zu\n",
                                order, measured.failure.empty() ? "wait_failed" : "submit_or_wait_failed",
                                dispatchOrders.size() - index - 1u);
                    for (std::size_t rest = index + 1; rest < dispatchOrders.size(); ++rest) {
                        std::printf("  dispatch order=%u skipped=device_no_longer_usable_in_this_process\n",
                                    dispatchOrders[rest]);
                    }
                    break;
                }
            }
            if (!dispatchOrders.empty()) {
                std::printf("  dispatch summary: attempted=%llu ran=%llu image_changed=%llu "
                            "wrote_something=%llu of %zu requested\n",
                            (unsigned long long)dispatchAttempted, (unsigned long long)dispatchRan,
                            (unsigned long long)dispatchChanged,
                            (unsigned long long)dispatchWroteSomething, dispatchOrders.size());
                // WHY there is no ms_per_dispatch line: dividing ms_wait by the repeat count
                // was measured to be meaningless. One dispatch per device lifetime cost
                // 2003-2017 ms of wait on this host; FOUR dispatches of the same module inside
                // ONE command buffer cost 2042 ms total - so the ~2 s is a ONE-TIME first-
                // execution cost (driver-side materialisation), and the extra 3 dispatches added
                // ~30 ms. A reader given "510 ms per dispatch" would conclude the opposite.
                if (dispatchRepeat != 1u) {
                    std::printf(
                        "  dispatch timing note: ms_wait is NOT proportional to repeats; the "
                        "~2 s measured on this host is one-time first-execution cost, so per-dispatch "
                        "cost must be read as (ms_wait(repeat=N) - ms_wait(repeat=1)) / (N-1), never "
                        "as ms_wait / N. Measured here: repeats=4 -> 2042 ms total.\n");
                    // The same run also produced a driver hard-recovery WITH repeats>1 and
                    // NONE with repeats=1, which is the shape a missing VK_DEPENDENCY_BY_TRANSITION
                    // barrier between pool writes produces. Recorded as a finding, not a
                    // suggestion: the flag is absent from the pinned headers, so this driver
                    // revision cannot express the dependency the chain needs.
                    std::printf("  dispatch chain note: repeats>1 issues dispatches over ONE descriptor "
                                "pool; on this host repeats=4 was measured to trigger 'the CS has been "
                                "cancelled because the context is lost' while vkQueueWaitIdle still "
                                "returned VK_SUCCESS. vkCmdPipelineBarrier here can only be PLAIN "
                                "(VK_DEPENDENCY_BY_TRANSITION is not in these headers), so a multi-pass "
                                "chain cannot be declared correct against this SDK revision.\n");
                }
            }
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
