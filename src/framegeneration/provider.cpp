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

#include "framegeneration.hpp"
#include "pipeline-cache.hpp"

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
    if (header->type != ffgvk::kCreateContextDescType) {
        return FFX_API_RETURN_NO_PROVIDER;
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
    *context = created;
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
    // This slice owns no submitted GPU work, so there is no owned fence/event to
    // wait on. Teardown serializes the namespaced pipeline cache (so a warm start
    // reuses it), then destroys the cache handle and context only; the caller owns
    // the device and every resource it handed over. Never wait on unrelated device
    // work.
    saveCache(created);
    created->~FrameGenerationContext();
    deallocate(callbacks, created);
    *context = nullptr;
    return FFX_API_RETURN_OK;
}

ffxReturnCode_t configure(ffxContext* /*context*/,
                          const ffxConfigureDescHeader* header) {
    if (!header) {
        return FFX_API_RETURN_ERROR_PARAMETER;
    }
    // The generic global-debug configure block is accepted; unrecognized blocks are
    // forwarded past rather than rejected, so hosts sending newer optional blocks
    // keep working. (ffxConfigure on a null context also operates on global state
    // per the API, which for this provider is nothing.)
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
            if (versions->createDescType != ffgvk::kCreateContextDescType) {
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
    }
    // Unknown query descriptors are reported, never answered wrongly.
    return FFX_API_RETURN_ERROR_UNKNOWN_DESCTYPE;
}

ffxReturnCode_t dispatch(ffxContext* context, const ffxDispatchDescHeader* header) {
    if (!context || !*context || !header) {
        return FFX_API_RETURN_ERROR_PARAMETER;
    }
    if (header->type != ffgvk::kDispatchDescType) {
        return FFX_API_RETURN_NO_PROVIDER;
    }
    // The Frame Generation interpolation/prepare passes and the swap-chain present
    // (measured: 382 ML-neural inference modules, 92 classic compute passes, and a
    // mainVS/mainPS/7 'main' raster blit) are NOT implemented or validated in this
    // slice. Refuse with the exact boundary rather than fabricating a blend, a
    // motion-blur pass or a made-up optical-flow stand-in.
    logLine("dispatch refused: Frame Generation interpolation + present runtime not "
            "implemented in this provider slice; refusing to fake frame generation. "
            "Host-integration boundary (swap-chain present / pacing) is documented "
            "in README.");
    return FFX_API_RETURN_ERROR_RUNTIME_ERROR;
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
