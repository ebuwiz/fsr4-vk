// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 fsr4-vk contributors.
// See LICENSE in the project root.

// Frame Generation provider ABI. This is the second independent provider family.
// It shares only the FFX generic C API shape with the upscaler (the same five
// entry points every FFX effect DLL exports, verified against this DLL's export
// table). It owns its own context type, version id and capability contract and
// touches no upscaler runtime code, assets or model logic, so the two .so/.dll
// outputs cannot accidentally become one provider.
//
// HONEST SCOPE NOTE (see README + audit report): this slice implements the provider
// ABI, caller-owned-device capability gating, version advertisement and the
// explicit host-integration boundary. The Frame Generation algorithm runtime (the
// optical-flow / SCD / dilate classic compute, the 382 ML-neural inference passes,
// and the swap-chain present blit measured as mainVS/mainPS/7 'main') is NOT yet
// implemented or image-quality-validated, so ffxDispatch refuses with a precise
// blocker instead of faking frame generation. This .so is therefore not a completed
// Frame Generation equivalent; it is the ABI + capability contract + host boundary
// that a validated runtime would plug into.
//
// Update (roadmap Phase 0, 2026-11-18): the missing precise piece is now the pass
// PLAN, not the ABI. tools/extract/frame-generation/plan.py reconstructs it from the
// pack's OWN modules.tsv/pipelines.tsv (474 compute modules: 382 compute-neural mlfi +
// 92 compute-classic; 1208 de-duplicated binding groups; an intermediate-target +
// byte-count model). That plan is machine-checked two ways: framegeneration-plan-test
// re-parses pipelines.tsv independently and must agree (GPU-free), and
// framegeneration-exec --expect-plan ties the DRIVER-backed binding census to it (on
// device, measured checked=474 missing=0 mismatch=0). ffxDispatch(Prepare|Frame
// Generation) STILL refuses: recording passes against that plan is Phase 1-2, and the
// GPU-memory-usage query (0x00020007) becomes honestly reportable in Phase 3 once the
// provider actually allocates the plan's intermediate targets.

#include "framegeneration.hpp"
#include "assets.hpp"
#include "framegeneration-checks.hpp"
#include "pipeline-cache.hpp"
#include "fg-runtime.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

namespace ffgprovider {

struct FrameGenerationContext {
    ffgvk::CreateBackendVkDesc backend{};
    ffgvk::GpuCapabilities capabilities;
    std::uint64_t versionId = ffgvk::kReferenceVersionId;

    // Frame Generation call-order + camera contract state (see ffx_frame_generation.h
    // mandatory order: Configure -> Dispatch(Prepare) -> Dispatch(Frame Generation)).
    // The algorithm itself is not implemented, but the provider tracks the order so the
    // documented debug checker can reject misuse, and remembers the create flags that
    // gate the camera/depth validation.
    bool debugChecking = false;   // FFX_FRAMEGENERATION_ENABLE_DEBUG_CHECKING
    bool depthInfinite = false;   // FFX_FRAMEGENERATION_ENABLE_DEPTH_INFINITE
    bool highDynamicRange = false;         // FFX_FRAMEGENERATION_ENABLE_HIGH_DYNAMIC_RANGE
    bool depthInverted = false;             // FFX_FRAMEGENERATION_ENABLE_DEPTH_INVERTED
    bool displayResMotionVectors = false;   // FFX_FRAMEGENERATION_ENABLE_DISPLAY_RESOLUTION_MOTION_VECTORS
    bool jitterCancellation = false;        // FFX_FRAMEGENERATION_ENABLE_MOTION_VECTORS_JITTER_CANCELLATION
    bool asyncWorkloadCreate = false;       // FFX_FRAMEGENERATION_ENABLE_ASYNC_WORKLOAD_SUPPORT
    bool versionLinked = false;             // ffxCreateContextDescFrameGenerationVersion linked at create
    bool configured = false;
    bool prepared = false;
    bool frameGenerationEnabled = true;   // from ffxConfigureDescFrameGeneration
    bool allowAsyncWorkloads = false;     // from ffxConfigureDescFrameGeneration

    // Phase-1 runtime substrate (see fg-runtime.hpp). Built lazily at Prepare from the
    // pack's OWN plan: it allocates the plan's intermediate targets as real device
    // memory, one descriptor pool sized by the driver's element costs, and a real
    // compute pipeline per module of the classic subset. It never dispatches, so the
    // algorithm refusal stands while GPU-memory-usage becomes reportable.
    ffgvk::PlanRuntime substrate;
    ffgvk::SubstrateFacts substrateFacts;
    std::string substrateNote;      // what happened, once attempted
    bool substrateAttempted = false;
    std::uint32_t renderWidth = 0;
    std::uint32_t renderHeight = 0;
    std::uint32_t displayWidth = 0;   // the display resolution the host declared
    std::uint32_t displayHeight = 0;

    // The host-integration contract captured at ffxConfigure (ffxConfigureDescFrameGeneration):
    // the swap chain + present/frameGeneration callbacks (and their user contexts) the runtime
    // would drive, plus the optional HUDLessColor, onlyPresentGenerated, generationRect and the
    // most recent frameID. Captured NOW so the (future) present phase can invoke them; the compute
    // algorithm is still refused, so nothing is driven back to the host yet.
    void* swapChain = nullptr;
    FfxApiPresentCallbackFunc presentCallback = nullptr;
    void* presentCallbackUserContext = nullptr;
    FfxApiFrameGenerationDispatchFunc frameGenerationCallback = nullptr;
    void* frameGenerationCallbackUserContext = nullptr;
    struct FfxApiResource hudLessColor{};
    bool onlyPresentGenerated = false;
    struct FfxApiRect2D generationRect{};
    std::uint64_t lastConfiguredFrameId = 0;

    // Optional create/configure FG blocks recognized at the host boundary (lens-distortion is
    // applied at present time; HUDLess selects a HUD-less back buffer). Stored for the (future)
    // present phase; the compute algorithm is still refused so nothing is rendered yet.
    struct FfxApiResource distortionField{};
    bool distortionRegistered = false;
    bool hudlessEnabled = false;
    std::uint32_t hudlessBackBufferFormat = 0u;

    // Namespaced, identity-validated Vulkan pipeline cache (see pipeline-cache.hpp).
    // Created on context-construct and serialized on context-destroy so a warm
    // start on the same effect/GPU/driver reuses the driver's compiled state.
    VkPipelineCache pipelineCache = VK_NULL_HANDLE;
    PFN_vkGetPipelineCacheData getPipelineCacheData = nullptr;
    PFN_vkDestroyPipelineCache destroyPipelineCache = nullptr;
    ffgvk::CacheIdentity cacheIdentity{};
    std::string cachePath;
    bool cacheActive = false;
};

static bool extensiveLogging() {
    const char* value = std::getenv("FFG4_VK_EXTENSIVE_LOG");
    return value && std::strcmp(value, "0") != 0;
}

static void logLine(const std::string& text) {
    std::fprintf(stderr, "[ffg4] %s\n", text.c_str());
}

// Reads the pack's own plan tables into a Plan. Shared by ensureSubstrate (device)
// and the GetGPUMemoryUsageV2 sizing query (device-free), so both sides answer from
// the SAME numbers plan.py measured rather than each holding their own copy.
static bool loadPlanModel(ffgplan::Plan& plan, std::string& reason) {
    try {
        const auto root = ffgassets::packRoot();
        if (!ffgassets::exists(root / "fg-plan.tsv") ||
            !ffgassets::exists(root / "pipelines.tsv") ||
            !ffgassets::exists(root / "fg-plan.json")) {
            reason = "pack carries no plan tables (fg-plan.tsv/pipelines.tsv/fg-plan.json)";
            return false;
        }
        ffgplan::parsePlanTsv(ffgassets::read(root / "fg-plan.tsv"), plan);
        ffgplan::parsePipelinesTsv(ffgassets::read(root / "pipelines.tsv"), plan);
        ffgplan::parsePlanJson(ffgassets::read(root / "fg-plan.json"), plan);
    } catch (const std::exception& error) {
        reason = std::string("plan read failed: ") + error.what();
        return false;
    }
    if (!plan.modules.empty()) {
        std::string why;
        if (!ffgplan::consistent(plan, why)) {
            reason = "pack plan tables disagree: " + why;
            return false;
        }
        return true;
    }
    reason = plan.reason.empty() ? "no plan rows parsed" : plan.reason;
    return false;
}

// Builds the Phase-1 substrate exactly once, and only where the host has declared
// the resolutions this effect will run at (Prepare). Reading the pack's own plan
// tables is what decides the module set, the binding shapes and the intermediate
// byte model; nothing in here is a constant copied from a tool.
static void ensureSubstrate(FrameGenerationContext* ctx) {
    if (ctx->substrateAttempted) {
        return;
    }
    ctx->substrateAttempted = true;
    // Lever, in the same read-once/parse-by-exact-spelling style as the rest: a host
    // that wants this provider to allocate NOTHING on its device sets FFG4_VK_SUBSTRATE=0,
    // and then GPU memory usage keeps refusing instead of reporting the substrate.
    if (const char* off = std::getenv("FFG4_VK_SUBSTRATE"); off && std::strcmp(off, "0") == 0) {
        ctx->substrateNote = "substrate disabled via FFG4_VK_SUBSTRATE=0";
        return;
    }
    ffgplan::Plan plan;
    std::string why;
    if (!loadPlanModel(plan, why)) {
        ctx->substrateNote = "plan unavailable: " + why;
        return;
    }
    std::string reason;
    ffgvk::SubstrateFacts facts;
    const bool built =
        ctx->substrate.build(ctx->backend, ctx->capabilities, plan,
                             ctx->renderWidth, ctx->renderHeight,
                             ctx->displayWidth, ctx->displayHeight,
                             /*includeNeuralSubset=*/false, ctx->pipelineCache,
                             facts, reason);
    ctx->substrateFacts = facts;
    if (!built) {
        ctx->substrateNote = "substrate refused: " + reason;
        return;
    }
    ctx->substrateNote = facts.usageReportable ? "substrate built (usage reportable)"
                                             : "substrate partial (usage not reportable)";
    logLine("phase1 substrate render=" + std::to_string(ctx->renderWidth) + "x" +
            std::to_string(ctx->renderHeight) + " display=" +
            std::to_string(ctx->displayWidth) + "x" + std::to_string(ctx->displayHeight) +
            " plan modules=" + std::to_string(facts.planModules) +
            " classic=" + std::to_string(facts.classicModules) +
            " neural=" + std::to_string(facts.neuralModules) +
            " groups=" + std::to_string(facts.bindingGroups) +
            " order=" + facts.orderSource +
            " intermediates=" + std::to_string(facts.targetsAllocated) + "/" +
            std::to_string(facts.targetBytes) + "B pool=" +
            std::to_string(facts.descriptorPoolBytes) + "B" +
            " pipelines=" + std::to_string(facts.pipelinesCreated) + "/" +
            std::to_string(facts.modulesAttempted) +
            " layouts=" + std::to_string(facts.setLayouts) + "/" +
            std::to_string(facts.pipelineLayouts) +
            " descriptors=" + std::to_string(facts.descriptorBindingsOk) + "/" +
            std::to_string(facts.descriptorBindings) +
            " backings=" + std::to_string(facts.scratchBackings) +
            (facts.descriptorCensusSkipped ? " census=skipped" : " census=run") +
            " usage=" +
            (facts.usageReportable
                 ? std::to_string(facts.totalUsageInBytes) + "B (aliasable " +
                   std::to_string(facts.aliasableUsageInBytes) + "B)"
                 : "not-reportable") +
            (facts.firstFailure.empty()
                 ? std::string("")
                 : std::string(" firstFailure=") + facts.firstFailure +
                   " module=" + facts.firstFailureModule));
}

static bool validAllocationCallbacks(const ffxAllocationCallbacks* callbacks) {
    return callbacks == nullptr ||
           (callbacks->alloc != nullptr && callbacks->dealloc != nullptr);
}

static void* allocate(const ffxAllocationCallbacks* callbacks, std::size_t size) {
    return callbacks ? callbacks->alloc(callbacks->pUserData, size)
                     : std::malloc(size);
}

static void deallocate(const ffxAllocationCallbacks* callbacks, void* memory) {
    if (callbacks) {
        callbacks->dealloc(callbacks->pUserData, memory);
    } else {
        std::free(memory);
    }
}

// Resolve one device-scope entry point through the caller-owned resolver only (never
// the global vk* wrappers), which is what keeps the provider bound to the host device.
static PFN_vkVoidFunction resolveDevice(const ffgvk::CreateBackendVkDesc& backend,
                                        const char* name) {
    return backend.getDeviceProcAddr ? backend.getDeviceProcAddr(backend.device, name)
                                     : nullptr;
}

static bool profiling() {
    const char* value = std::getenv("FFG4_VK_PROFILE");
    return value && std::strcmp(value, "0") != 0;
}

// Create the context's namespaced pipeline cache, seeding it from the on-disk
// container only when its pinned identity (algorithm + GPU + driver) still matches.
static void initCache(FrameGenerationContext* ctx) {
    const char* disabled = std::getenv("FFG4_VK_PIPELINE_CACHE_DISABLE");
    if (disabled && std::strcmp(disabled, "0") != 0) {
        if (extensiveLogging()) {
            logLine("pipeline cache disabled via FFG4_VK_PIPELINE_CACHE_DISABLE");
        }
        return;
    }
    auto* create = reinterpret_cast<PFN_vkCreatePipelineCache>(
        resolveDevice(ctx->backend, "vkCreatePipelineCache"));
    auto* getData = reinterpret_cast<PFN_vkGetPipelineCacheData>(
        resolveDevice(ctx->backend, "vkGetPipelineCacheData"));
    auto* destroy = reinterpret_cast<PFN_vkDestroyPipelineCache>(
        resolveDevice(ctx->backend, "vkDestroyPipelineCache"));
    if (!create || !getData || !destroy) {
        if (extensiveLogging()) {
            logLine("pipeline cache entry points not exported by the device; running uncached");
        }
        return;
    }
    ffgvk::CacheIdentity id{};
    id.deviceName = ctx->capabilities.deviceName;
    id.driverVersion = ctx->capabilities.driverVersion;
    id.apiVersion = ctx->capabilities.apiVersion;
    id.vendorId = ctx->capabilities.vendorId;
    id.deviceId = ctx->capabilities.deviceId;
    std::string path;
    const char* override = std::getenv("FFG4_VK_PIPELINE_CACHE");
    if (override && *override) {
        path = override;
    } else {
        path = ffgvk::defaultCachePath(id);
    }
    const auto blob = ffgvk::readCacheBlob(path, id);
    VkPipelineCacheCreateInfo info{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
    info.initialDataSize = blob.size();
    info.pInitialData = blob.empty() ? nullptr : blob.data();
    VkPipelineCache cache = VK_NULL_HANDLE;
    if (create(ctx->backend.device, &info, nullptr, &cache) != VK_SUCCESS) {
        if (extensiveLogging()) {
            logLine("pipeline cache create rejected by driver; running uncached");
        }
        return;
    }
    ctx->pipelineCache = cache;
    ctx->getPipelineCacheData = getData;
    ctx->destroyPipelineCache = destroy;
    ctx->cacheIdentity = id;
    ctx->cachePath = path;
    ctx->cacheActive = true;
    if (profiling() || extensiveLogging()) {
        logLine("pipeline cache created algorithm=frame-generation device=" + id.deviceName +
                " seedBytes=" + std::to_string(blob.size()) + " path=" + path);
    }
}

// Serialize the cache back into the identity-pinned container on teardown so the
// next same-effect/GPU/driver context reuses it (a warm start).
static void saveCache(FrameGenerationContext* ctx) {
    if (!ctx->cacheActive) {
        return;
    }
    std::size_t size = 0;
    if (ctx->getPipelineCacheData(ctx->backend.device, ctx->pipelineCache, &size,
                                  nullptr) == VK_SUCCESS && size) {
        std::vector<unsigned char> blob(size);
        if (ctx->getPipelineCacheData(ctx->backend.device, ctx->pipelineCache, &size,
                                      blob.data()) == VK_SUCCESS) {
            ffgvk::writeCacheBlob(ctx->cachePath, ctx->cacheIdentity, blob);
            if (profiling() || extensiveLogging()) {
                logLine("pipeline cache saved bytes=" + std::to_string(blob.size()) +
                        " path=" + ctx->cachePath);
            }
        }
    }
    if (ctx->destroyPipelineCache) {
        ctx->destroyPipelineCache(ctx->backend.device, ctx->pipelineCache, nullptr);
    }
    ctx->pipelineCache = VK_NULL_HANDLE;
    ctx->cacheActive = false;
}


ffxReturnCode_t createContext(ffxContext* context, ffxCreateContextDescHeader* header,
                              const ffxAllocationCallbacks* callbacks) {
    if (!context) {
        return FFX_API_RETURN_ERROR_PARAMETER;
    }
    *context = nullptr;
    if (!validAllocationCallbacks(callbacks)) {
        return FFX_API_RETURN_ERROR_PARAMETER;
    }
    if (!header) {
        return FFX_API_RETURN_NO_PROVIDER;
    }
    if (header->type != ffgvk::kCreateDescType) {
        return FFX_API_RETURN_NO_PROVIDER;
    }
    // The create descriptor is the SDK's ffxCreateContextDescFrameGeneration, so validate
    // its OWN arguments (a real display/render resolution + a known surface format) before
    // trusting the backend block on its pNext chain.
    if (!ffgcheck::createDescOk(reinterpret_cast<const ffxCreateContextDescFrameGeneration*>(
        header))
             .ok) {
        logLine("create refused: invalid ffxCreateContextDescFrameGeneration arguments");
        return FFX_API_RETURN_ERROR_PARAMETER;
    }

    const auto* backend = ffgvk::findBackend(header);
    if (!backend) {
        // No Vulkan backend block in this create chain: this (Vulkan) provider cannot
        // serve it. Unknown pNext blocks are forwarded past, never rejected.
        return FFX_API_RETURN_NO_PROVIDER;
    }
    if (!backend->device || !backend->physicalDevice || !backend->getDeviceProcAddr) {
        logLine("create refused: Vulkan backend block missing device/physical/resolver");
        return FFX_API_RETURN_ERROR_PARAMETER;
    }

    // Recognize the optional HUDLess create block (0x00020009) on this create chain: if present
    // its back-buffer format must be a real surface. Other unknown pNext blocks stay forwarded
    // past, never rejected (the modular FFX API forwards extensions this provider does not model).
    const ffxCreateContextDescFrameGenerationHudless* hudless = nullptr;
    for (const auto* cur = header; cur != nullptr; cur = cur->pNext) {
        if (cur->type == ffgvk::kHudlessCreateDescType) {
            hudless = reinterpret_cast<const ffxCreateContextDescFrameGenerationHudless*>(cur);
            break;
        }
    }
    if (hudless) {
        const auto hv = ffgcheck::hudlessCreateDescOk(hudless);
        if (!hv.ok) {
            logLine("create refused: " + hv.reason);
            return FFX_API_RETURN_ERROR_PARAMETER;
        }
    }
    // Honor version negotiation honestly (grounded by the FSRFG_Dx12 injector reference: a host selects
    // the algorithm version via ffxOverrideVersion.versionId from ffxQueryDescGetVersions; 0=non-ML, 1=ML).
    // This .so implements exactly ONE FG version (classic; the 382-pass ML compute is not present), so a
    // host pinning a DIFFERENT id is refused rather than silently substituted. No override = default classic.
    if (const auto* override = ffgvk::findVersionOverride(header);
        override != nullptr && override->versionId != ffgvk::kReferenceVersionId) {
        logLine("create refused: requested ffxOverrideVersion id is not the single classic FG version "
                "this provider implements (ML inference not provided; refusing to substitute)");
        return FFX_API_RETURN_ERROR_RUNTIME_ERROR;
    }

    const auto* enabled = ffgvk::findEnabledFeatures(header);
    if (!enabled) {
        logLine("create refused: missing host-declared enabled Vulkan feature block "
                "(descriptor type 7)");
        return FFX_API_RETURN_ERROR_PARAMETER;
    }
    const auto capabilities = ffgvk::queryCapabilities(*backend, enabled);
    const auto blockers = ffgvk::executionContractBlockers(capabilities);
    if (!blockers.empty()) {
        std::string reason;
        for (const auto& blocker : blockers) {
            if (!reason.empty()) {
                reason += "; ";
            }
            reason += blocker;
        }
        logLine("context refused: " + reason);
        // Refuse before pipeline creation, naming the exact enabled-feature gaps
        // the measured Frame Generation execution contract demands.
        return FFX_API_RETURN_ERROR_RUNTIME_ERROR;
    }

    void* raw = allocate(callbacks, sizeof(FrameGenerationContext));
    if (!raw) {
        return FFX_API_RETURN_ERROR_MEMORY;
    }
    auto* created = new (raw) FrameGenerationContext();
    created->backend = *backend;
    created->capabilities = capabilities;
    // Remember the create-time flags that gate the documented debug checker: the
    // depth-infinite bit changes how cameraNear/Far are validated, and the debug
    // bit turns on argument/order enforcement at dispatch.
    {
        const auto* fgCreate =
            reinterpret_cast<const ffxCreateContextDescFrameGeneration*>(header);
        created->debugChecking =
            (fgCreate->flags & FFX_FRAMEGENERATION_ENABLE_DEBUG_CHECKING) != 0u;
        created->depthInfinite =
            (fgCreate->flags & FFX_FRAMEGENERATION_ENABLE_DEPTH_INFINITE) != 0u;
        created->highDynamicRange =
            (fgCreate->flags & FFX_FRAMEGENERATION_ENABLE_HIGH_DYNAMIC_RANGE) != 0u;
        created->depthInverted =
            (fgCreate->flags & FFX_FRAMEGENERATION_ENABLE_DEPTH_INVERTED) != 0u;
        created->displayResMotionVectors =
            (fgCreate->flags & FFX_FRAMEGENERATION_ENABLE_DISPLAY_RESOLUTION_MOTION_VECTORS) != 0u;
        created->jitterCancellation =
            (fgCreate->flags & FFX_FRAMEGENERATION_ENABLE_MOTION_VECTORS_JITTER_CANCELLATION) != 0u;
        created->asyncWorkloadCreate =
            (fgCreate->flags & FFX_FRAMEGENERATION_ENABLE_ASYNC_WORKLOAD_SUPPORT) != 0u;
        // Recognize the 2.4.0 version-link block (0x0e) on the create chain (matched by type id only;
        // the older vendored header has no ffxCreateContextDescFrameGenerationVersion struct).
        for (const auto* cur = header; cur != nullptr; cur = cur->pNext) {
            if (cur->type == ffgvk::kCreateVersionDescType) {
                created->versionLinked = true;
                break;
            }
        }
        if (hudless) {
            created->hudlessEnabled = true;
            created->hudlessBackBufferFormat = hudless->hudlessBackBufferFormat;
        }
    }
    *context = created;
    try {
        ffgassets::initializeEmbedded();
        const auto root = ffgassets::packRoot();
        if (!ffgassets::exists(root / "manifest.json") ||
            !ffgassets::exists(root / "mlfi-model.bin") ||
            !ffgassets::exists(root / "mlfi-graph.json")) {
            throw std::runtime_error(
                "Frame Generation pack is incomplete (manifest/model/graph)");
        }
    } catch (const std::exception& error) {
        created->~FrameGenerationContext();
        deallocate(callbacks, created);
        *context = nullptr;
        logLine(std::string("create refused: ") + error.what());
        return FFX_API_RETURN_ERROR_RUNTIME_ERROR;
    }
    // Phase 6 (cache) groundwork: create the namespaced, identity-pinned pipeline
    // cache up front so any later pipeline creation (once the algorithm runtime
    // lands) reuses it. The provider owns the cache handle but NOT the device.
    initCache(created);
    if (extensiveLogging()) {
        logLine("context created device=" + capabilities.deviceName +
                " version=0xF600000001000002 capability-contract=met");
    }
    return FFX_API_RETURN_OK;
}

ffxReturnCode_t destroyContext(ffxContext* context,
                               const ffxAllocationCallbacks* callbacks) {
    if (!context || !*context) {
        return FFX_API_RETURN_OK;
    }
    if (!validAllocationCallbacks(callbacks)) {
        return FFX_API_RETURN_ERROR_PARAMETER;
    }
    auto* created = static_cast<FrameGenerationContext*>(*context);
    // Release the Phase-1 substrate (device memory, layouts, pipelines) BEFORE the
    // pipeline cache is serialized, so a warm start still finds the driver's compiled
    // state and the caller's device is left with nothing of ours still allocated on it.
    created->substrate.release();
    saveCache(created);
    created->~FrameGenerationContext();
    deallocate(callbacks, created);
    *context = nullptr;
    return FFX_API_RETURN_OK;
}

ffxReturnCode_t configure(ffxContext* context,
                          const ffxConfigureDescHeader* header) {
    if (!header) {
        return FFX_API_RETURN_ERROR_PARAMETER;
    }
    auto* created = (context && *context)
        ? static_cast<FrameGenerationContext*>(*context)
        : nullptr;
    // Walk the configure pNext chain. The SDK's Frame Generation configure block
    // (ffxConfigureDescFrameGeneration, type 0x00020002) carries the swap-chain, the
    // present + frame-generation callbacks, and the enable/async booleans; recognizing
    // it sets the mandatory "configure before prepare/generate" flag. Key-value and
    // other optional blocks are forwarded past, never rejected, so hosts using newer
    // configure shapes keep working.
    for (const auto* current = header; current != nullptr; current = current->pNext) {
        if (current->type == ffgvk::kConfigureDescType) {
            if (!created) {
                continue;
            }
            const auto* cfg =
                reinterpret_cast<const ffxConfigureDescFrameGeneration*>(current);
            const auto v = ffgcheck::configureDescOk(cfg);
            if (!v.ok) {
                logLine("configure refused: " + v.reason);
                return FFX_API_RETURN_ERROR_PARAMETER;
            }
            created->configured = true;
            created->frameGenerationEnabled = cfg->frameGenerationEnabled;
            created->allowAsyncWorkloads = cfg->allowAsyncWorkloads;
            // Remember the full host-integration contract so the (future) present phase can drive
            // these callbacks. Validated above (when enabled, the swap chain + both callbacks are
            // required), so this replaces the earlier debug-only callback presence check.
            created->swapChain = cfg->swapChain;
            created->presentCallback = cfg->presentCallback;
            created->presentCallbackUserContext = cfg->presentCallbackUserContext;
            created->frameGenerationCallback = cfg->frameGenerationCallback;
            created->frameGenerationCallbackUserContext = cfg->frameGenerationCallbackUserContext;
            created->hudLessColor = cfg->HUDLessColor;
            created->onlyPresentGenerated = cfg->onlyPresentGenerated;
            created->generationRect = cfg->generationRect;
            created->lastConfiguredFrameId = cfg->frameID;
            // The generation rect is where a host declares the DISPLAY resolution both
            // real and generated frames land on (a zero rect means "whole frame").
            // The Phase-1 substrate sizes the plan's upscale-following intermediates
            // from exactly this, and never from an assumed display size.
            if (cfg->generationRect.width > 0 && cfg->generationRect.height > 0) {
                created->displayWidth =
                    static_cast<std::uint32_t>(cfg->generationRect.width);
                created->displayHeight =
                    static_cast<std::uint32_t>(cfg->generationRect.height);
            }
        }
        // Register a lens-distortion field (0x00020008): validate + remember it for the (future)
        // present phase. A null resource is refused precisely, never silently ignored.
        if (current->type == ffgvk::kRegisterDistortionDescType) {
            if (!created) {
                continue;
            }
            const auto* d =
                reinterpret_cast<const ffxConfigureDescFrameGenerationRegisterDistortionFieldResource*>(
                    current);
            const auto dv = ffgcheck::registerDistortionOk(d);
            if (!dv.ok) {
                logLine("configure refused: " + dv.reason);
                return FFX_API_RETURN_ERROR_PARAMETER;
            }
            created->distortionField = d->distortionField;
            created->distortionRegistered = true;
        }
        // A Frame Generation key-value configure block (0x00020006): this SDK's FfxApiConfigureFrameGeneration
        // enum has NO defined keys, so there is nothing recognized to apply. Accept the block (the
        // modular API forbids rejecting a well-formed optional descriptor) rather than reject it.
        if (current->type == ffgvk::kConfigureKeyValueDescType) {
            continue;
        }
    }
    return FFX_API_RETURN_OK;
}

ffxReturnCode_t query(ffxContext* context, ffxQueryDescHeader* header) {
    if (!header) {
        return FFX_API_RETURN_ERROR_PARAMETER;
    }
    for (auto* current = header; current != nullptr; current = current->pNext) {
        if (current->type == ffgvk::kGetProviderVersionQueryType) {
            auto* version = reinterpret_cast<ffxQueryGetProviderVersion*>(current);
            version->versionId = (context && *context)
                ? static_cast<FrameGenerationContext*>(*context)->versionId
                : ffgvk::kReferenceVersionId;
            version->versionName = ffgvk::kComponentName;
            return FFX_API_RETURN_OK;
        }
        if (current->type == FFX_API_QUERY_DESC_TYPE_GET_VERSIONS) {
            // Advertise exactly this provider's one supported version id so a host
            // can pin it; it names no algorithm model family.
            auto* versions = reinterpret_cast<ffxQueryDescGetVersions*>(current);
            if (versions->createDescType != ffgvk::kCreateDescType) {
                return FFX_API_RETURN_NO_PROVIDER;
            }
            if (!versions->outputCount) {
                return FFX_API_RETURN_ERROR_PARAMETER;
            }
            const std::uint64_t capacity = *versions->outputCount;
            if (capacity > 0) {
                if (versions->versionIds) {
                    versions->versionIds[0] = ffgvk::kReferenceVersionId;
                }
                if (versions->versionNames) {
                    versions->versionNames[0] = ffgvk::kComponentName;
                }
            }
            *versions->outputCount = 1;
            return FFX_API_RETURN_OK;
        }
        if (current->type == ffgvk::kQueryGPUUsageType) {
            // The SDK exposes GPU memory usage for this effect (ffxQueryDescFrameGenerationGetGPUMemoryUsage,
            // type 0x00020007). With the Phase-1 substrate built, the numbers on the wire are the
            // allocations the driver actually handed back for this context: the plan's intermediates
            // plus its descriptor pool. Before that point (no Prepare yet, or a partial build) a
            // usage number would still be a fabrication, so the refusal stands.
            auto* usage =
                reinterpret_cast<ffxQueryDescFrameGenerationGetGPUMemoryUsage*>(current);
            auto* holder = (context && *context)
                               ? static_cast<FrameGenerationContext*>(*context)
                               : nullptr;
            if (holder && holder->substrateFacts.usageReportable &&
                usage->gpuMemoryUsageFrameGeneration) {
                usage->gpuMemoryUsageFrameGeneration->totalUsageInBytes =
                    holder->substrateFacts.totalUsageInBytes;
                usage->gpuMemoryUsageFrameGeneration->aliasableUsageInBytes =
                    holder->substrateFacts.aliasableUsageInBytes;
                logLine("gpu memory usage reported from allocations: intermediates=" +
                        std::to_string(holder->substrateFacts.targetBytes) +
                        "B descriptor-pool=" +
                        std::to_string(holder->substrateFacts.descriptorPoolBytes) +
                        "B (allocated targets=" +
                        std::to_string(holder->substrateFacts.targetsAllocated) +
                        " pipelines=" +
                        std::to_string(holder->substrateFacts.pipelinesCreated) + ")");
                return FFX_API_RETURN_OK;
            }
            logLine("query refused: Frame Generation GPU memory usage is not reportable "
                    "until the interpolation/ML/present runtime is implemented" +
                    (holder && !holder->substrateNote.empty()
                         ? std::string("; substrate state: ") + holder->substrateNote
                         : std::string(" (no substrate built yet: Prepare allocates it)")));
            return FFX_API_RETURN_ERROR_RUNTIME_ERROR;
        }
        if (current->type == ffgvk::kProbeDispatchOneType) {
            // Repo-internal measurement, never a shipped contract: run ONE real compute
            // dispatch of ONE plan module over the substrate THIS provider already built, on
            // the device the caller handed it, and return the driver's numbers through the
            // caller's own out struct. It exists so the bind-pool -> set-offsets -> bind ->
            // dispatch -> submit -> wait -> read-back chain can be MEASURED here rather than
            // argued about: whether this VK_EXT_descriptor_buffer revision can be driven at
            // all, how long a pass costs, and whether the storage image the pass was handed
            // changed bytes across it. If no substrate exists on this context it refuses
            // outright: measuring on some other device would say nothing about this one.
            const auto* probe = reinterpret_cast<const ffgvk::ProbeDispatchOneDesc*>(current);
            if (probe->factsOut == nullptr) {
                return FFX_API_RETURN_ERROR_PARAMETER;
            }
            auto* state = (context && *context) ? static_cast<FrameGenerationContext*>(*context) : nullptr;
            if (!state || !state->substrateAttempted) {
                logLine("dispatch measurement refused: this context has no substrate "
                        "(Prepare is what builds it; nothing is dispatched on a foreign device)");
                return FFX_API_RETURN_ERROR_RUNTIME_ERROR;
            }
            *probe->factsOut = ffgvk::PlanRuntime::DispatchFacts{};
            std::string why;
            const bool ran =
                state->substrate.dispatchOne(probe->planOrder, probe->canvasWidth, probe->canvasHeight,
                                             probe->dispatchRepeat, *probe->factsOut, why);
            auto& measured = *probe->factsOut;
            logLine("dispatch module=" + measured.module + " order=" + std::to_string(measured.order) +
                    " stage=" + measured.stage + " sets=" + std::to_string(measured.sets) +
                    " bindings=" + std::to_string(measured.bindings) +
                    " bound=" + (measured.bound ? "yes" : "no") +
                    " submitted=" + (measured.submitted ? "yes" : "no") +
                    " ran=" + (measured.ok ? "yes" : "no") +
                    // The four timings print as INTEGER MICROSECONDS, not std::to_string(double):
                    // the double overload drags __gnu_cxx::__to_xstring<..., std::size_t,
                    // std::va_list> into the .so's dynamic symbol table, which the ABI gate
                    // test_abi_exports correctly reports as C++ leakage (measured: one symbol,
                    // _ZN9__gnu_cxx12__to_xstringINSt7__...P13__va_list_tagEmSB_z). Integers use a
                    // helper the provider already exports cleanly, and 1 us resolution is finer than
                    // any of these numbers needs.
                    " ms_record_us=" + std::to_string((unsigned long long)(measured.msRecord * 1000.0)) +
                    " ms_submit_us=" + std::to_string((unsigned long long)(measured.msSubmit * 1000.0)) +
                    " ms_wait_us=" + std::to_string((unsigned long long)(measured.msWait * 1000.0)) +
                    " ms_total_us=" + std::to_string((unsigned long long)(measured.ms * 1000.0)) +
                    " repeats=" + std::to_string(measured.repeats) + " barrier=" + measured.barrierFlags +
                    " pool=" + std::to_string(measured.poolBytes) + "B descriptorBytes=" +
                    std::to_string(measured.descriptorBytesWritten) +
                    "B groups=" + std::to_string(measured.groups[0]) + "x" +
                    std::to_string(measured.groups[1]) + "x" + std::to_string(measured.groups[2]) +
                    " canvas=" + std::to_string(measured.canvasWidth) + "x" +
                    std::to_string(measured.canvasHeight) + " planMaxExtent=" +
                    std::to_string(measured.planMaxWidth) + "x" +
                    std::to_string(measured.planMaxHeight) + " texel=" +
                    std::to_string(measured.bytesPerTexel) + "B readBack=" +
                    std::to_string(measured.readBackBytes) + "B outChanged=" +
                    (measured.outputChanged ? "yes" : "no") +
                    " wroteSampledImage=" + (measured.wroteSampledImage ? "yes" : "no") +
                    " nonZeroStorageBufferBytes=" + std::to_string(measured.nonZeroStorageBufferBytes) +
                    " nonZeroUniformBufferBytes=" + std::to_string(measured.nonZeroUniformBufferBytes) +
                    " anyWritableChanged=" + (measured.anyWritableOutputChanged ? "yes" : "no") +
                    " readbackAvailable=" + (measured.readbackAvailable ? "yes" : "no") +
                    " contextLost=" + (measured.contextLost ? "yes" : "no") +
                    " basis=" + measured.note +
                    " reason=" + (measured.failure.empty() ? why : measured.failure));
            return ran ? FFX_API_RETURN_OK : FFX_API_RETURN_ERROR_RUNTIME_ERROR;
        }
        if (current->type == ffgvk::kQueryGPUUsageV2Type) {
            // ffxQueryDescFrameGenerationGetGPUMemoryUsageV2 (0x0b): the pre-context (NULL-context)
            // sizing query - the app FILLS device/maxRenderSize/displaySize/formats and asks how big
            // the effect WILL be, which is exactly what a host needs before it creates anything.
            // The answer is the pack's OWN byte model for the resolutions the caller named, and the
            // log says so in those words: this is a plan-model answer, not an allocation, and it
            // never substitutes a number for a device it was not given.
            const auto* v2 =
                reinterpret_cast<const ffgvk::QueryGPUUsageV2Desc*>(current);
            if (!v2->gpuMemoryUsageFrameGeneration) {
                return FFX_API_RETURN_ERROR_PARAMETER;
            }
            if (v2->maxRenderSize.width == 0u || v2->maxRenderSize.height == 0u) {
                logLine("query refused: GetGPUMemoryUsageV2 needs a non-zero maxRenderSize "
                        "from the caller");
                return FFX_API_RETURN_ERROR_PARAMETER;
            }
            ffgplan::Plan plan;
            std::string why;
            if (!loadPlanModel(plan, why)) {
                logLine("query refused: GetGPUMemoryUsageV2 has no plan model to size "
                        "the effect against: " + why);
                return FFX_API_RETURN_ERROR_RUNTIME_ERROR;
            }
            const auto sized = ffgplan::sizeTargets(plan.facts, v2->maxRenderSize.width,
                                                   v2->maxRenderSize.height,
                                                   v2->displaySize.width,
                                                   v2->displaySize.height);
            v2->gpuMemoryUsageFrameGeneration->totalUsageInBytes = sized.totalBytes;
            // Nothing is allocated for a pre-create sizing answer, so nothing may be
            // claimed aliasable.
            v2->gpuMemoryUsageFrameGeneration->aliasableUsageInBytes = 0u;
            logLine("gpu memory usage V2 basis=plan-model(no allocation yet) render=" +
                    std::to_string(v2->maxRenderSize.width) + "x" +
                    std::to_string(v2->maxRenderSize.height) +
                    " display=" + std::to_string(v2->displaySize.width) + "x" +
                    std::to_string(v2->displaySize.height) + " targets=" +
                    std::to_string(sized.entries.size()) + " total=" +
                    std::to_string(sized.totalBytes) + "B (V1 0x07 reports "
                    "allocations instead)");
            return FFX_API_RETURN_OK;
        }
    }
    // Unknown query descriptors are reported, never answered wrongly.
    return FFX_API_RETURN_ERROR_UNKNOWN_DESCTYPE;
}

ffxReturnCode_t dispatch(ffxContext* context, const ffxDispatchDescHeader* header) {
    if (!context || !*context || !header) {
        return FFX_API_RETURN_ERROR_PARAMETER;
    }
    auto* created = static_cast<FrameGenerationContext*>(*context);

    // ffxDispatchDescFrameGenerationPrepare (type 0x00020004). Validate the SDK's own
    // arguments; with FFX_FRAMEGENERATION_ENABLE_DEBUG_CHECKING set, also require the
    // linked camera-info block (0x0002000a) with a valid orthonormal basis and enforce
    // the frameTimeDelta-in-milliseconds expectation. This mirrors AMD's debug checker
    // exactly and is what Step 4's "mandatory order + camera validation" delivers.
    if (header->type == ffgvk::kDispatchPrepareType) {
        const auto* prepare =
            reinterpret_cast<const ffxDispatchDescFrameGenerationPrepare*>(header);
        const auto ord = ffgcheck::prepareOrderOk(created->configured);
        if (!ord.ok) {
            logLine("prepare refused: " + ord.reason);
            return FFX_API_RETURN_ERROR_PARAMETER;
        }
        const auto v = ffgcheck::prepareDescOk(prepare, created->depthInfinite,
                                               created->debugChecking);
        if (!v.ok) {
            logLine("prepare refused: " + v.reason);
            return FFX_API_RETURN_ERROR_PARAMETER;
        }
        if (created->debugChecking) {
            const auto* cam = ffgcheck::findCameraInfo(header);
            std::string reason;
            if (!ffgcheck::cameraBasisOk(cam, reason)) {
                logLine("prepare refused: " + reason);
                return FFX_API_RETURN_ERROR_PARAMETER;
            }
        }
        created->prepared = true;
        // Prepare is where the host declares the render resolution, so it is also where
        // the Phase-1 substrate is built (intermediates + descriptor pool + classic
        // pipelines). Building it here means the resources a later GPU-memory-usage
        // answer reports are ones the driver actually handed back.
        created->renderWidth = prepare->renderSize.width;
        created->renderHeight = prepare->renderSize.height;
        ensureSubstrate(created);
        // The Prepare pass records into the host command list AND launches the dilate/optical-
        // flow compute, and the generate pass runs the interpolation + ML graph + swap-chain
        // present — NONE of which is implemented or image-quality-validated in this slice.
        // So after ACCEPTING a well-formed, in-order, camera-valid Prepare, still refuse the
        // algorithm rather than fabricate one.
        logLine("prepare accepted (order + camera valid) but dispatch pending: the "
                "Frame Generation interpolation/ML/present runtime is not implemented in "
                "this provider slice; refusing to fake frame generation. See README "
                "host-integration boundary.");
        if (!created->substrateNote.empty()) {
            logLine("prepare substrate state: " + created->substrateNote);
        }
        return FFX_API_RETURN_ERROR_RUNTIME_ERROR;
    }

    // ffxDispatchDescFrameGeneration (type 0x00020003). Validate 1..3 generated frames
    // + luminance range; with the debug checker, require a preceding Prepare (mandatory
    // order). Then the same honest algorithm refusal.
    if (header->type == ffgvk::kDispatchGenerateType) {
        const auto* generate =
            reinterpret_cast<const ffxDispatchDescFrameGeneration*>(header);
        // Documented cancel path (grounded by the injector reference's DispatchCallback): numGeneratedFrames
        // == 0 means "no interpolation this frame" (e.g. host disabled FG mid-frame or duplicate frameID).
        // There is nothing to fabricate, so honestly return OK instead of refusing.
        if (generate->numGeneratedFrames == 0u) {
            logLine("generate: numGeneratedFrames==0 cancel (no interpolation requested this frame)");
            return FFX_API_RETURN_OK;
        }
        const auto v = ffgcheck::generateDescOk(generate);
        if (!v.ok) {
            logLine("generate refused: " + v.reason);
            return FFX_API_RETURN_ERROR_PARAMETER;
        }
        const auto ord = ffgcheck::generateOrderOk(created->configured, created->prepared);
        if (!ord.ok) {
            logLine("generate refused: " + ord.reason);
            return FFX_API_RETURN_ERROR_PARAMETER;
        }
        logLine("dispatch refused: Frame Generation interpolation + present runtime not "
                "implemented in this provider slice; refusing to fake frame generation. "
                "Host-integration boundary (swap-chain present / pacing) is documented "
                "in README.");
        return FFX_API_RETURN_ERROR_RUNTIME_ERROR;
    }

    // ffxDispatchDescFrameGenerationPrepareV2 (0x0c) SUPERSEDES Prepare(0x04)+CameraInfo(0x0a) in 2.4.0; its
    // camera basis is EMBEDDED. Read the host descriptor through ffgcheck::PrepareV2Desc (an ABI mirror of the
    // pinned 2.4.0 layout, machine-checked by check-test) and validate ORDER + arguments + the EMBEDDED basis
    // - the completion of this provider's former PrepareV2 TODO - then STILL refuse the algorithm honestly:
    // the interpolation/ML/present runtime is not implemented, so no frames are fabricated.
    if (header->type == ffgvk::kDispatchPrepareV2Type) {
        const auto ord = ffgcheck::prepareOrderOk(created->configured);
        if (!ord.ok) {
            logLine("prepareV2 refused: " + ord.reason);
            return FFX_API_RETURN_ERROR_PARAMETER;
        }
        const auto* v2 = reinterpret_cast<const ffgcheck::PrepareV2Desc*>(header);
        const auto pv = ffgcheck::prepareV2DescOk(v2, created->depthInfinite, created->debugChecking);
        if (!pv.ok) {
            logLine("prepareV2 refused: " + pv.reason);
            return FFX_API_RETURN_ERROR_PARAMETER;
        }
        created->prepared = true;
        // PrepareV2 carries no render extent of its own, so the substrate builds from
        // whatever resolutions this context already knows (Configure's generation rect
        // and/or an earlier Prepare), never from a made-up size.
        ensureSubstrate(created);
        logLine("prepareV2 accepted (order + arguments + embedded camera basis valid) but dispatch pending: "
                "the Frame Generation interpolation/ML/present runtime is not implemented in this provider "
                "slice; refusing to fake frame generation.");
        return FFX_API_RETURN_ERROR_RUNTIME_ERROR;
    }

    // Present-callback (0x00020005), key-value configure, camera-info outside a Prepare
    // chain, or any other descriptor is not a dispatch this provider can serve: report
    // no provider rather than guess a type.
    return FFX_API_RETURN_NO_PROVIDER;
}

} // namespace ffgprovider

extern "C" FFX_API_ENTRY ffxReturnCode_t ffxCreateContext(ffxContext* context,
                                                          ffxCreateContextDescHeader* header,
                                                          const ffxAllocationCallbacks* callbacks) {
    return ffgprovider::createContext(context, header, callbacks);
}

extern "C" FFX_API_ENTRY ffxReturnCode_t
ffxDestroyContext(ffxContext* context, const ffxAllocationCallbacks* callbacks) {
    return ffgprovider::destroyContext(context, callbacks);
}

extern "C" FFX_API_ENTRY ffxReturnCode_t ffxConfigure(ffxContext* context,
                                                      const ffxConfigureDescHeader* header) {
    return ffgprovider::configure(context, header);
}

extern "C" FFX_API_ENTRY ffxReturnCode_t ffxQuery(ffxContext* context,
                                                  ffxQueryDescHeader* header) {
    return ffgprovider::query(context, header);
}

extern "C" FFX_API_ENTRY ffxReturnCode_t ffxDispatch(ffxContext* context,
                                                     const ffxDispatchDescHeader* header) {
    return ffgprovider::dispatch(context, header);
}
