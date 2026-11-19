// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 fsr4-vk contributors.
// See LICENSE in the project root.
//
// LD_PRELOAD this shared library while AMD's Frame Generation provider runs (its Vulkan DLL, or the DX12 DLL
// under vkd3d-proton). SPIR-V bytes arrive at vkCreateShaderModule (dump to file, map module handle -> file);
// vkCreateComputePipelines links each pipeline to its stage module's file; every vkCmdDispatch* emits
// (pipeline's SPIR-V file, threadgrid) in CALL ORDER -> capture.tsv. Enable with FSR4_VK_CAPTURE_DIR=<outdir>;
// unset means pure pass-through. Feed capture.tsv to capture-diff.py (hashes each file, matches to the pack plan).
// The ORDER is AMD's own vkCmdDispatch sequence - the only honest source of the 382-pass schedule; never invented.
//
//   g++ -std=c++20 -O2 -shared -fPIC -I third_party/Vulkan-Headers/include capture-shim.cpp -o libffxvk-capture.so
//   LD_PRELOAD=$PWD/libffxvk-capture.so FSR4_VK_CAPTURE_DIR=/tmp/fgcap <amd-sample>
#include <dlfcn.h>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vulkan/vulkan.h>
#include "capture-ledger.hpp"

namespace {
std::mutex gMutex;
ffcap::Ledger gLedger;
std::unordered_map<std::uintptr_t, std::string> gModuleSpv;
std::string gDir;
bool gActive = false;
bool gInit = false;

void ensureInit() {
    if (gInit) {
        return;
    }
    gInit = true;
    const char* dir = std::getenv("FSR4_VK_CAPTURE_DIR");
    if (dir && *dir) {
        gDir = dir;
        gActive = true;
    }
}

void dumpSpv(std::uintptr_t module, const void* code, std::size_t n) {
    if (!code || !n) {
        return;
    }
    char name[512];
    std::snprintf(name, sizeof(name), "%s/mod-%016lx.spv", gDir.c_str(), (unsigned long)module);
    std::FILE* f = std::fopen(name, "wb");
    if (!f) {
        return;
    }
    std::fwrite(code, 1, n, f);
    std::fclose(f);
    std::lock_guard<std::mutex> lock(gMutex);
    gModuleSpv.emplace(module, std::string(name));
}

struct FlushAtExit {
    ~FlushAtExit() {
        if (!gActive) {
            return;
        }
        std::lock_guard<std::mutex> lock(gMutex);
        const char* j = std::getenv("FSR4_VK_CAPTURE_JSON");
        if (j && std::strcmp(j, "0") != 0) {
            gLedger.write_json(gDir + "/capture.json");
        } else {
            gLedger.write_tsv(gDir + "/capture.tsv");
        }
    }
} gFlush;
} // anonymous namespace

extern "C" VkResult vkCreateShaderModule(VkDevice device, const VkShaderModuleCreateInfo* info,
                                         const VkAllocationCallbacks* alloc, VkShaderModule* outModule) {
    using Fn = decltype(&vkCreateShaderModule);
    auto fn = reinterpret_cast<Fn>(dlsym(RTLD_NEXT, "vkCreateShaderModule"));
    VkResult r = fn(device, info, alloc, outModule);
    ensureInit();
    if (gActive && r == VK_SUCCESS) {
        dumpSpv(reinterpret_cast<std::uintptr_t>(*outModule), info->pCode, info->codeSize);
    }
    return r;
}

extern "C" VkResult vkCreateComputePipelines(VkDevice device, VkPipelineCache cache, uint32_t count,
                                             const VkComputePipelineCreateInfo* infos,
                                             const VkAllocationCallbacks* alloc, VkPipeline* outPipelines) {
    using Fn = decltype(&vkCreateComputePipelines);
    auto fn = reinterpret_cast<Fn>(dlsym(RTLD_NEXT, "vkCreateComputePipelines"));
    VkResult r = fn(device, cache, count, infos, alloc, outPipelines);
    ensureInit();
    if (gActive && r == VK_SUCCESS) {
        std::lock_guard<std::mutex> lock(gMutex);
        for (uint32_t i = 0; i < count; ++i) {
            auto m = gModuleSpv.find(reinterpret_cast<std::uintptr_t>(infos[i].stage.module));
            if (m != gModuleSpv.end()) {
                gLedger.note_pipeline(reinterpret_cast<std::uintptr_t>(outPipelines[i]), m->second);
            }
        }
    }
    return r;
}

extern "C" void vkCmdBindPipeline(VkCommandBuffer cb, VkPipelineBindPoint bp, VkPipeline pipeline) {
    using Fn = decltype(&vkCmdBindPipeline);
    auto fn = reinterpret_cast<Fn>(dlsym(RTLD_NEXT, "vkCmdBindPipeline"));
    if (bp == VK_PIPELINE_BIND_POINT_COMPUTE) {
        std::lock_guard<std::mutex> lock(gMutex);
        gLedger.note_bind(reinterpret_cast<std::uintptr_t>(cb), reinterpret_cast<std::uintptr_t>(pipeline));
    }
    fn(cb, bp, pipeline);
}

extern "C" void vkCmdDispatch(VkCommandBuffer cb, uint32_t x, uint32_t y, uint32_t z) {
    using Fn = decltype(&vkCmdDispatch);
    auto fn = reinterpret_cast<Fn>(dlsym(RTLD_NEXT, "vkCmdDispatch"));
    ensureInit();
    if (gActive) {
        std::lock_guard<std::mutex> lock(gMutex);
        char d[48];
        std::snprintf(d, sizeof(d), "%u,%u,%u", x, y, z);
        gLedger.note_dispatch(reinterpret_cast<std::uintptr_t>(cb), d);
    }
    fn(cb, x, y, z);
}

extern "C" void vkCmdDispatchBase(VkCommandBuffer cb, uint32_t bx, uint32_t by, uint32_t bz, uint32_t x,
                                   uint32_t y, uint32_t z) {
    using Fn = decltype(&vkCmdDispatchBase);
    auto fn = reinterpret_cast<Fn>(dlsym(RTLD_NEXT, "vkCmdDispatchBase"));
    ensureInit();
    if (gActive) {
        std::lock_guard<std::mutex> lock(gMutex);
        char d[64];
        std::snprintf(d, sizeof(d), "%u,%u,%u,%u,%u,%u", bx, by, bz, x, y, z);
        gLedger.note_dispatch(reinterpret_cast<std::uintptr_t>(cb), d);
    }
    fn(cb, bx, by, bz, x, y, z);
}

extern "C" void vkCmdDispatchIndirect(VkCommandBuffer cb, VkBuffer buffer, VkDeviceSize offset) {
    using Fn = decltype(&vkCmdDispatchIndirect);
    auto fn = reinterpret_cast<Fn>(dlsym(RTLD_NEXT, "vkCmdDispatchIndirect"));
    ensureInit();
    if (gActive) {
        std::lock_guard<std::mutex> lock(gMutex);
        gLedger.note_dispatch(reinterpret_cast<std::uintptr_t>(cb), "");  // GPU-side args -> marker
    }
    fn(cb, buffer, offset);
}
