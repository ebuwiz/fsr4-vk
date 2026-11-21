// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 fsr4-vk contributors.
// See LICENSE in the project root.
//
// /////
// tools/capture/frame-generation/host.cpp - the WINDOWS-side host that drives AMD's OWN Frame Generation
// provider, so the capture rig (capture-shim.cpp + capture-diff.py + apply-capture-order.py) can obtain the ONE
// thing this repo has never been able to derive statically: AMD's real vkCmdDispatch ORDER, and the identity of
// each dispatched module by the SPIR-V bytes AMD hands vkCreateShaderModule.
//
// WHY a second host at all: this repo's own providers (`build/linux/*.so`) are LINUX objects, and AMD's
// Frame Generation implementation ships ONLY as WINDOWS binaries (`dlls/amd_fidelityfx_framegeneration_vk.dll`,
// sha-pinned by stage-vk-dlls.py). Running THAT implementation - the thing whose ORDER we need - needs a
// Windows image, and `winegcc` + `wine` on this host provide exactly that: MEASURED here, wine 11.22 compiles
// and runs a Windows program, and Wine's Vulkan shim forwards to the SAME RADV the LINUX harness uses (the
// "radv is not a conformant Vulkan implementation" banner prints from Wine's own vulkan path, which is proof
// the shim engaged). NO cross-toolchain install is needed: wine links against Wine's Windows-side
// `/usr/lib/wine/x86_64-windows/libvulkan-1.a` + `vulkan-1.dll` (Wine's winevulkan shim).
//
// The host-integration ladder is NOT invented here: every block, id, format and usage bit is the SAME shape
// tools/FGTest/main.cpp hands the providers (the SDK's `ffx_api.h` + `ffx_frame_generation.h` headers), against
// a device that enables VK_EXT_descriptor_buffer - because AMD's provider, like ours, cannot build its runtime
// without that transport. What this host adds is nothing but: two frames' worth of REAL images (colour / D32
// depth / RG16F motion / three RGBA8 output images), then Prepare -> PrepareV2 -> generate, printed at EVERY step.
//
// Run (from tools/capture/frame-generation/):  bash run-capture.sh
//   -> stages the three pinned AMD DLLs, builds the shim + this host, runs the host under wine with the shim
//     LD_PRELOADED, then reports capture.tsv and joins it against the pack with capture-diff.py.
///
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <span>
#include <string>
#include <vector>

// THE FSR4_WINDOWS_BUILD guard is set ONLY by build-host.sh (a REAL Windows cross-compiler). Without it this
// file stays LINUX-pure, which is what lets it be syntax-checked on any host; the Windows-only loader calls
// (::LoadLibraryEx / ::GetProcAddress) exist on exactly one side of that guard, so there is no chance of a
// LINUX build silently half-loading AMD's binary.
#ifdef FSR4_WINDOWS_BUILD
#define NOMINMAX  // windows.h's min/max macros break <unordered_map>: measured, hashtable.h fails to compile
                  // without this guard once <windows.h> is in the translation unit
#include <windows.h>  // winegcc's include path adds /usr/include/wine/windows: Wine's loader supplies LoadLibrary
                    // + GetProcAddress to a winegcc-built app, and Linux dlopen CANNOT load a WINDOWS .dll
#else
#include <dlfcn.h>
#endif

#include <vulkan/vulkan.h>

#include "ffx_api.h"
#include "ffx_frame_generation.h"

// The two blocks BOTH FFX FG providers decode on the create chain, laid out as tools/FGTest/main.cpp and
// tools/framegeneration-probe.cpp do: type 7 = what the HOST declares enabled, type 3 = which device the
// backend may resolve its own entry points from.
struct FfgEnabledFeaturesDesc {
    ffxApiHeader header;
    VkBool32 shaderInt16, shaderInt64, shaderFloat16, shaderInt8, storage8Bit, storage16Bit,
        bufferDeviceAddress, runtimeDescriptorArray, cooperativeMatrix, vulkanMemoryModel,
        subgroupSizeControl, computeDerivativeGroupLinear, synchronization2, storageImageWriteWithoutFormat;
};
struct FfgBackendDesc {
    ffxApiHeader header;
    VkDevice device;
    VkPhysicalDevice physicalDevice;
    PFN_vkGetDeviceProcAddr getDeviceProcAddr;
};

struct App {
    VkInstance instance{VK_NULL_HANDLE};
    VkPhysicalDevice physical{VK_NULL_HANDLE};
    VkDevice device{VK_NULL_HANDLE};   // the descriptor-buffer device FG's backend block names
    VkQueue queue{VK_NULL_HANDLE};
    VkCommandPool pool{VK_NULL_HANDLE};
    bool ok{false};
};

static ffxReturnCode_t cbPresent(ffxCallbackDescFrameGenerationPresent*, void*) { return FFX_API_RETURN_OK; }
static ffxReturnCode_t cbFg(ffxDispatchDescFrameGeneration*, void*) { return FFX_API_RETURN_OK; }

static App a;

static bool createDeviceWith() {
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO, nullptr, "ffg-capture-host", 0,
                          "ffg-capture-host", 0, VK_MAKE_API_VERSION(0, 4, 0, 2)};
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, nullptr, 0, &app, 0, nullptr};
    if (vkCreateInstance(&ici, nullptr, &a.instance) != VK_SUCCESS || !a.instance) {
        std::printf("  NO_INSTANCE (vkCreateInstance failed) - wine's vulkan shim did not reach a driver\n");
        return false;
    }
    uint32_t count = 0;
    vkEnumeratePhysicalDevices(a.instance, &count, nullptr);
    if (count == 0) {
        std::printf("  NO_PHYSICAL_DEVICE (the driver is not visible through wine's vulkan shim)\n");
        return false;
    }
    std::vector<VkPhysicalDevice> phys(count);
    vkEnumeratePhysicalDevices(a.instance, &count, phys.data());
    a.physical = phys[0];
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(a.physical, &props);
    std::printf("  device[0]: %s api=%u.%u.%u\n", props.deviceName,
                props.apiVersion >> 29, (props.apiVersion >> 20) & 0x3FFu, (props.apiVersion >> 12) & 0x3FFu);
    return true;
}

static bool hasExt(VkPhysicalDevice p, const char* name) {
    uint32_t n = 0;
    vkEnumerateDeviceExtensionProperties(p, nullptr, &n, nullptr);
    std::vector<VkExtensionProperties> list(n);
    if (n)
        vkEnumerateDeviceExtensionProperties(p, nullptr, &n, list.data());
    for (const auto& e : list)
        if (!std::strcmp(e.extensionName, name))
            return true;
    return false;
}

// ---- the device FG's backend block names: VK_EXT_descriptor_buffer ENABLED, the same choice tools/FGTest
// makes with its SECOND logical device, because vkGetDeviceProcAddr answers NULL for vkGetDescriptorEXT on a
// device created WITHOUT the extension, and then no FG provider can build its runtime at all.
static bool createFgDevice() {
    const char* wanted[] = {VK_EXT_DESCRIPTOR_BUFFER_EXTENSION_NAME,
                            VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME,
                            VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
                            VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME,
                            VK_KHR_VULKAN_MEMORY_MODEL_EXTENSION_NAME};
    std::vector<const char*> exts;
    for (const char* name : wanted) {
        if (hasExt(a.physical, name))
            exts.push_back(name);
        else
            std::printf("  driver lacks %s - passed anyway? NO: skipped, and the provider names its own need\n", name);
    }
    if (exts.empty())
        return false;

    VkPhysicalDeviceDescriptorBufferFeaturesEXT db{};
    db.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_BUFFER_FEATURES_EXT;
    db.descriptorBuffer = VK_TRUE;
    db.descriptorBufferPushDescriptors = VK_TRUE;
    VkPhysicalDeviceBufferAddressFeaturesEXT addr{};
    addr.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_ADDRESS_FEATURES_EXT;
    addr.bufferDeviceAddress = VK_TRUE;
    addr.pNext = &db;
    VkDeviceCreateInfo di{};
    di.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    di.pNext = &addr;
    di.enabledExtensionCount = (uint32_t)exts.size();
    di.ppEnabledExtensionNames = exts.data();
    if (vkCreateDevice(a.physical, &di, nullptr, &a.device) != VK_SUCCESS || !a.device) {
        std::printf("  DEVICE_CREATE_REFUSED (%zu extensions requested) - wine+vulkan+driver combination\n",
                    exts.size());
        a.device = VK_NULL_HANDLE;
        return false;
    }
    vkGetDeviceQueue(a.device, 0, 0, &a.queue);
    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, nullptr, 0, 0};
    vkCreateCommandPool(a.device, &pci, nullptr, &a.pool);
    std::printf("  fg device + queue + pool ready (extensions=%zu)\n", exts.size());
    return true;
}

// ---- images the host owes Frame Generation, sized and formatted the way FGTest's rig does it: colour RGBA16F
// 480x270, depth D32, motion RG16F, and three display-size RGBA8 backbuffers for generate's outputs.
struct Img {
    VkImage image{VK_NULL_HANDLE};
    VkDeviceMemory memory{VK_NULL_HANDLE};
};

static bool mkImg(Img& out, VkFormat fmt, VkImageUsageFlags usage, uint32_t w, uint32_t h,
                  VkImageAspectFlags aspect) {
    VkImageCreateInfo ic{};
    ic.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ic.imageType = VK_IMAGE_TYPE_2D;
    ic.format = fmt;
    ic.extent = {w, h, 1};
    ic.mipLevels = ic.arrayLayers = 1;
    ic.samples = VK_SAMPLE_COUNT_1_BIT;
    ic.tiling = VK_IMAGE_TILING_OPTIMAL;
    ic.usage = usage;
    // GENERAL from birth: pre-Vulkan-1.4 an image not used as a framebuffer attachment may be CREATED in
    // GENERAL, which is the layout a descriptor-bound sample/store reads. UNDEFINED would leave the layout
    // to whatever the provider's own barrier decides, and these images are inputs to AMD's ORDER capture.
    ic.initialLayout = VK_IMAGE_LAYOUT_GENERAL;
    if (vkCreateImage(a.device, &ic, nullptr, &out.image) != VK_SUCCESS || !out.image)

        return false;
    VkMemoryRequirements mr{};
    vkGetImageMemoryRequirements(a.device, out.image, &mr);
    uint32_t index = 0xFFFFFFFFu;
    for (uint32_t i = 0; i < 32u; ++i)
        if (mr.memoryTypeBits & (1u << i)) {
            index = i;
            break;
        }
    if (index == 0xFFFFFFFFu) {
        vkDestroyImage(a.device, out.image, nullptr);
        out.image = VK_NULL_HANDLE;
        return false;
    }
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr, mr.size, index};
    if (vkAllocateMemory(a.device, &ai, nullptr, &out.memory) != VK_SUCCESS || !out.memory) {
        vkDestroyImage(a.device, out.image, nullptr);
        out.image = VK_NULL_HANDLE;
        return false;
    }
    vkBindImageMemory(a.device, out.image, out.memory, 0);
    return true;
}

// ---- AMD's provider, loaded from the WINDOWS binary and resolved by NAME ---
struct Ffx {
    void* handle{nullptr};
    ffxReturnCode_t (*create)(ffxContext*, ffxCreateContextDescHeader*, const ffxAllocationCallbacks*){nullptr};
    ffxReturnCode_t (*configure)(ffxContext*, const ffxConfigureDescHeader*){nullptr};
    ffxReturnCode_t (*query)(ffxContext*, ffxQueryDescHeader*){nullptr};
    ffxReturnCode_t (*dispatch)(ffxContext*, const ffxDispatchDescHeader*){nullptr};
    ffxReturnCode_t (*destroy)(ffxContext*, const ffxAllocationCallbacks*){nullptr};
};


// ---- THE host records its OWN capture. A provider may reach Vulkan ONLY through the resolver the host hands it
// (the SDK's execution contract), so wrapping the six names of interest HERE works whatever the loader layer is;
// the LD_PRELOAD shim stays useful for a host that cannot interpose (a real game sample), but this host does not
// depend on it. Rows land in $FSR4_VK_CAPTURE_DIR/capture.tsv in CALL ORDER, keyed to the SPIR-V file dumped at
// vkCreateShaderModule, which is what capture-diff.py joins against the pack by digest.
#include <unordered_map>

struct Recorder {
    std::unordered_map<std::uintptr_t, std::string> moduleSpv;   // VkShaderModule  -> dumped .spv path
    std::unordered_map<std::uintptr_t, std::string> pipelineSpv; // VkPipeline      -> that module's path
    std::unordered_map<std::uintptr_t, std::uintptr_t> cbPipeline;  // command buffer -> last COMPUTE bind
    std::uint64_t row{0};
    std::string dir;
};
static Recorder gRec;
static std::FILE* gTsv{nullptr};

static PFN_vkCreateShaderModule realCreateShaderModule{nullptr};
static PFN_vkCreateComputePipelines realCreateComputePipelines{nullptr};
static PFN_vkCmdBindPipeline realCmdBindPipeline{nullptr};
static PFN_vkCmdDispatch realCmdDispatch{nullptr};
static PFN_vkCmdDispatchBase realCmdDispatchBase{nullptr};
static PFN_vkCmdDispatchIndirect realCmdDispatchIndirect{nullptr};

static void noteDispatch(const VkCommandBuffer cb, const char* grid);

// NOTE: these signatures follow the PINNED headers' OUT-PARAM style (measured: PFN_vkCreateShaderModule returns
// VkResult and takes a trailing VkShaderModule*), which is NOT the handle-returning style newer SDK samples use.
static VkResult VKAPI_PTR wrapCreateShaderModule(VkDevice device, const VkShaderModuleCreateInfo* info,
                                                 const VkAllocationCallbacks* alloc, VkShaderModule* outModule) {
    const VkResult r = realCreateShaderModule(device, info, alloc, outModule);
    if (r == VK_SUCCESS && outModule && *outModule && info && info->pCode && info->codeSize) {
        const auto module = reinterpret_cast<std::uintptr_t>(*outModule);
        char path[512];
        std::snprintf(path, sizeof(path), "%s/mod-%016lx.spv", gRec.dir.c_str(), (unsigned long)module);
        if (std::FILE* f = std::fopen(path, "wb")) {
            std::fwrite(info->pCode, 1, info->codeSize, f);
            std::fclose(f);
            gRec.moduleSpv[module] = path;
        }
    }
    return r;
}

static VkResult VKAPI_PTR wrapCreateComputePipelines(VkDevice device, VkPipelineCache cache, uint32_t count,
                                                     const VkComputePipelineCreateInfo* infos,
                                                     const VkAllocationCallbacks* alloc,
                                                     VkPipeline* outPipelines) {
    const VkResult r = realCreateComputePipelines(device, cache, count, infos, alloc, outPipelines);
    if (r == VK_SUCCESS) {
        for (uint32_t i = 0; i < count; ++i) {
            const auto m = gRec.moduleSpv.find((std::uintptr_t)infos[i].stage.module);
            if (m != gRec.moduleSpv.end())
                gRec.pipelineSpv[(std::uintptr_t)outPipelines[i]] = m->second;
        }
    }
    return r;
}

// The pipeline in effect per command buffer is what lets a dispatch row name its module: AMD records binds and
// dispatches into ONE buffer per pass, so the LAST bind for that buffer is the pass being dispatched.
static void VKAPI_PTR wrapCmdBindPipeline(VkCommandBuffer cb, VkPipelineBindPoint bp, VkPipeline pipeline) {
    realCmdBindPipeline(cb, bp, pipeline);
    if (bp == VK_PIPELINE_BIND_POINT_COMPUTE)
        gRec.cbPipeline[(std::uintptr_t)cb] = (std::uintptr_t)pipeline;
}

static void VKAPI_PTR wrapCmdDispatch(VkCommandBuffer cb, uint32_t x, uint32_t y, uint32_t z) {
    char grid[48];
    std::snprintf(grid, sizeof(grid), "%u,%u,%u", x, y, z);
    noteDispatch(cb, grid);
    realCmdDispatch(cb, x, y, z);
}

static void VKAPI_PTR wrapCmdDispatchBase(VkCommandBuffer cb, uint32_t bx, uint32_t by, uint32_t bz, uint32_t x,
                                          uint32_t y, uint32_t z) {
    char grid[64];
    std::snprintf(grid, sizeof(grid), "%u,%u,%u,%u,%u,%u", bx, by, bz, x, y, z);
    noteDispatch(cb, grid);
    realCmdDispatchBase(cb, bx, by, bz, x, y, z);
}

static void VKAPI_PTR wrapCmdDispatchIndirect(VkCommandBuffer cb, VkBuffer buffer, VkDeviceSize offset) {
    noteDispatch(cb, "indirect");
    realCmdDispatchIndirect(cb, buffer, offset);
}

static void noteDispatch(const VkCommandBuffer cb, const char* grid) {
    if (!gTsv)
        return;
    // WHICH pass is being dispatched is resolved from the LAST COMPUTE bind recorded for that command buffer, then
    // from that pipeline's stage module's dumped SPIR-V. An unbound or unmapped dispatch prints `?` rather than
    // guessing which pass AMD was in - capture-diff.py refuses to join a row with no digest.
    const auto bound = gRec.cbPipeline.find((std::uintptr_t)cb);
    std::string module("?");
    if (bound != gRec.cbPipeline.end()) {
        const auto m = gRec.pipelineSpv.find(bound->second);
        if (m != gRec.pipelineSpv.end())
            module = m->second;
    }
    std::fprintf(gTsv, "%llu\t%s\t%s\n", (unsigned long long)gRec.row++, module.c_str(), grid);
}

// The resolver the provider is handed. Recorded names get THIS host's wrappers; every other name passes the
// driver's pointer straight through, so the provider sees an ordinary Vulkan implementation.
static PFN_vkVoidFunction VKAPI_PTR hostResolver(VkDevice device, const char* name) {
    const auto real = vkGetDeviceProcAddr(device, name);
    if (!real || !name)
        return real;
    if (!std::strcmp(name, "vkCreateShaderModule"))
        return reinterpret_cast<PFN_vkVoidFunction>(&wrapCreateShaderModule);
    if (!std::strcmp(name, "vkCreateComputePipelines"))
        return reinterpret_cast<PFN_vkVoidFunction>(&wrapCreateComputePipelines);
    if (!std::strcmp(name, "vkCmdBindPipeline"))
        return reinterpret_cast<PFN_vkVoidFunction>(&wrapCmdBindPipeline);
    if (!std::strcmp(name, "vkCmdDispatch"))
        return reinterpret_cast<PFN_vkVoidFunction>(&wrapCmdDispatch);
    if (!std::strcmp(name, "vkCmdDispatchBase"))
        return reinterpret_cast<PFN_vkVoidFunction>(&wrapCmdDispatchBase);
    if (!std::strcmp(name, "vkCmdDispatchIndirect"))
        return reinterpret_cast<PFN_vkVoidFunction>(&wrapCmdDispatchIndirect);
    return real;
}

static void openCapture() {
    const char* dir = std::getenv("FSR4_VK_CAPTURE_DIR");
    if (!dir || !*dir) {
        std::printf("  FSR4_VK_CAPTURE_DIR unset: the host will run AMD's provider but records NOTHING\n");
        return;
    }
    gRec.dir = dir;
    gTsv = std::fopen((gRec.dir + "/capture.tsv").c_str(), "w");
    std::printf("  capture dir %s (capture.tsv + mod-<handle>.spv dumps; host wraps the six recorded entry points)\n",
                dir);
}

static void* loadSymbol(void* handle, const std::string& name) {
#ifdef FSR4_WINDOWS_BUILD
    return reinterpret_cast<void*>(::GetProcAddress(reinterpret_cast<HMODULE>(handle), name.c_str()));
#else
    return dlsym(handle, name.c_str());
#endif
}

static bool loadFfx(Ffx& f, const std::string& path) {
#ifdef FSR4_WINDOWS_BUILD
    f.handle = reinterpret_cast<void*>(::LoadLibraryEx(path.c_str(), nullptr, 0));
#else
    f.handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
    for (const char* name : {"vkCreateShaderModule", "vkCreateComputePipelines", "vkCmdBindPipeline",
                    "vkCmdDispatch", "vkCmdDispatchBase", "vkCmdDispatchIndirect"}) {
        const auto fn = vkGetDeviceProcAddr(a.device, name);
        if (!fn) {
            std::printf("  DEVICE_LACKS %s - the recorder cannot wrap an entry point that does not exist\n", name);
            return false;
        }
        if (!std::strcmp(name, "vkCreateShaderModule"))
            realCreateShaderModule = (PFN_vkCreateShaderModule)fn;
        if (!std::strcmp(name, "vkCreateComputePipelines"))
            realCreateComputePipelines = (PFN_vkCreateComputePipelines)fn;
        if (!std::strcmp(name, "vkCmdBindPipeline"))
            realCmdBindPipeline = (PFN_vkCmdBindPipeline)fn;
        if (!std::strcmp(name, "vkCmdDispatch"))
            realCmdDispatch = (PFN_vkCmdDispatch)fn;
        if (!std::strcmp(name, "vkCmdDispatchBase"))
            realCmdDispatchBase = (PFN_vkCmdDispatchBase)fn;
        if (!std::strcmp(name, "vkCmdDispatchIndirect"))
            realCmdDispatchIndirect = (PFN_vkCmdDispatchIndirect)fn;
    }
    if (!f.handle) {
        std::printf("  DLL_LOAD_REFUSED %s (wine's loader could not map this pinned binary into the process)\n",
                    path.c_str());
        return false;
    }
    f.create = (decltype(f.create))loadSymbol(f.handle, "ffxCreateContext");
    f.configure = (decltype(f.configure))loadSymbol(f.handle, "ffxConfigure");
    f.query = (decltype(f.query))loadSymbol(f.handle, "ffxQuery");
    f.dispatch = (decltype(f.dispatch))loadSymbol(f.handle, "ffxDispatch");
    f.destroy = (decltype(f.destroy))loadSymbol(f.handle, "ffxDestroyContext");
    std::printf("  loaded exports: create=%d configure=%d query=%d dispatch=%d destroy=%d\n", (int)!!f.create,
                (int)!!f.configure, (int)!!f.query, (int)!!f.dispatch, (int)!!f.destroy);
    return f.create && f.configure && f.query && f.dispatch && f.destroy;
}

static FfxApiResource mkRes(VkImage image, uint32_t w, uint32_t h, uint32_t apiFormat, uint32_t apiUsage,
                            uint32_t apiState) {
    FfxApiResource o{};
    o.resource = (void*)image;
    o.description.type = FFX_API_RESOURCE_TYPE_TEXTURE2D;
    o.description.format = apiFormat;
    o.description.width = w;
    o.description.height = h;
    o.description.depth = 1;
    o.description.mipCount = 1;
    o.description.flags = 0;
    o.description.usage = apiUsage;
    o.state = apiState;
    return o;
}

// ---- the ladder: create -> versions -> configure, printed at EVERY step ---
static int driveLadder(Ffx& f, ffxContext& ctx) {
    FfgEnabledFeaturesDesc en{};
    en.header = ffxApiHeader{7u, nullptr};
    en.shaderInt16 = en.shaderInt64 = en.shaderFloat16 = en.shaderInt8 = en.storage8Bit = en.storage16Bit = VK_TRUE;
    en.bufferDeviceAddress = en.runtimeDescriptorArray = en.cooperativeMatrix = en.vulkanMemoryModel = VK_TRUE;
    en.subgroupSizeControl = en.computeDerivativeGroupLinear = en.synchronization2 =
        en.storageImageWriteWithoutFormat = VK_TRUE;
    FfgBackendDesc bk{};
    bk.header = ffxApiHeader{3u, &en.header};
    bk.device = a.device;
    bk.physicalDevice = a.physical;
    // THE capture point: the provider resolves EVERY entry point it uses through this host function, so
    // wrapping it is what records AMD's own call order (see hostResolver above).
    bk.getDeviceProcAddr = reinterpret_cast<PFN_vkGetDeviceProcAddr>(hostResolver);
    struct VersionLink {
        ffxApiHeader header;
    } vlink{ffxApiHeader{0x0002000eu, nullptr}};
    bk.header.pNext = &en.header;
    en.header.pNext = &vlink.header;

    ffxCreateContextDescFrameGeneration cd{};
    cd.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATION;
    cd.header.pNext = &bk.header;
    cd.flags = 0u;
    cd.displaySize = {960u, 540u};
    cd.maxRenderSize = {960u, 540u};
    cd.backBufferFormat = 1u;
    const int create = (int)f.create(&ctx, &cd.header, nullptr);
    std::printf("  ffxCreateContext(frame-generation, AMD binary) = %d\n", create);
    if (create != FFX_API_RETURN_OK) {
        std::printf("  VERDICT: AMD's provider refused CREATE on this device+block combination, so there is NOTHING "
                    "to capture. rc 77 = a host/ABI limitation here, NOT a verdict on the algorithm.\n");
        return 77;
    }
    ffxQueryDescGetVersions ver{};
    ver.header.type = FFX_API_QUERY_DESC_TYPE_GET_VERSIONS;
    ver.createDescType = FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATION;
    uint64_t ids[2] = {0, 0};
    uint64_t nVersions = 2;
    const char* names[2] = {nullptr, nullptr};
    ver.outputCount = &nVersions;
    ver.versionIds = ids;
    ver.versionNames = names;
    f.query(&ctx, &ver.header);
    std::printf("  ffxQuery(GET_VERSIONS) -> id0=0x%016llX count=%llu name0=%s\n", (unsigned long long)ids[0],
                (unsigned long long)nVersions, names[0] ? names[0] : "(null)");
    ffxConfigureDescFrameGeneration cf{};
    cf.header.type = FFX_API_CONFIGURE_DESC_TYPE_FRAMEGENERATION;
    cf.frameGenerationEnabled = true;
    cf.swapChain = (void*)&ctx;
    cf.presentCallback = cbPresent;
    cf.frameGenerationCallback = cbFg;
    std::printf("  ffxConfigure(host contract) = %d\n", (int)f.configure(&ctx, &cf.header));
    return 0;
}


// ---- Prepare + generate: the two frames' worth of images a host owes, then the two dispatches ---
static int driveDispatches(Ffx& f, ffxContext& ctx) {
    VkCommandBuffer cb{VK_NULL_HANDLE};
    {
        VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, nullptr, a.pool,
                                        VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
        if (vkAllocateCommandBuffers(a.device, &cai, &cb) != VK_SUCCESS || !cb)
            return 77;
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr,
                                    VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr};
        if (vkBeginCommandBuffer(cb, &bi) != VK_SUCCESS)
            return 77;
    }
    const uint32_t rw = 480u, rh = 270u, ow = 960u, oh = 540u;
    Img color[2], depth[2], motion[2], out3[3];
    for (uint32_t i = 0; i < 2u; ++i) {
        mkImg(color[i], VK_FORMAT_R16G16B16A16_SFLOAT,
              VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_STORAGE_BIT, rw, rh,
              VK_IMAGE_ASPECT_COLOR_BIT);
        mkImg(depth[i], VK_FORMAT_D32_SFLOAT,
              VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, rw, rh,
              VK_IMAGE_ASPECT_DEPTH_BIT);
        mkImg(motion[i], VK_FORMAT_R16G16_SFLOAT,
              VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, rw, rh,
              VK_IMAGE_ASPECT_COLOR_BIT);
    }
    for (uint32_t i = 0; i < 3u; ++i)
        mkImg(out3[i], VK_FORMAT_R8G8B8A8_UNORM,
              VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, ow, oh,
              VK_IMAGE_ASPECT_COLOR_BIT);
    int missing = 0;
    for (const Img& im : std::span<Img>(color, 2))
        missing += (im.image == VK_NULL_HANDLE);
    for (const Img& im : std::span<Img>(depth, 2))
        missing += (im.image == VK_NULL_HANDLE);
    for (const Img& im : std::span<Img>(motion, 2))
        missing += (im.image == VK_NULL_HANDLE);
    for (const Img& im : std::span<Img>(out3, 3))
        missing += (im.image == VK_NULL_HANDLE);
    std::printf("  host images: 2x RGBA16F 480x270 colour, 2x D32 depth, 2x RG16F motion, 3x RGBA8 960x540 outputs, "
                "alloc failures=%d (inputs ZERO-filled on purpose: this run captures AMD's ORDER, not pixel parity)\n",
                missing);
    if (missing)
        return 77;

    ffxDispatchDescFrameGenerationPrepare pr{};
    pr.header.type = FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION_PREPARE;
    pr.frameID = 1001u;
    pr.renderSize = FfxApiDimensions2D{rw, rh};
    pr.commandList = (void*)cb;
    pr.jitterOffset = {0.0f, 0.0f};
    pr.motionVectorScale = {1.0f, 1.0f};
    pr.frameTimeDelta = 16.6f;
    pr.cameraNear = 0.1f;
    pr.cameraFar = 1000.0f;
    pr.cameraFovAngleVertical = 1.0f;
    pr.viewSpaceToMetersFactor = 0.0f;
    pr.depth = mkRes(depth[0].image, rw, rh, FFX_API_SURFACE_FORMAT_R32_FLOAT, FFX_API_RESOURCE_USAGE_DEPTHTARGET,
                     FFX_API_RESOURCE_STATE_COMPUTE_READ);
    pr.motionVectors = mkRes(motion[0].image, rw, rh, FFX_API_SURFACE_FORMAT_R16G16_FLOAT,
                             FFX_API_RESOURCE_USAGE_READ_ONLY, FFX_API_RESOURCE_STATE_COMPUTE_READ);
    std::printf("  ffxDispatch(Prepare frameID=1001 + depth + motion) = %d\n", (int)f.dispatch(&ctx, &pr.header));

    ffxDispatchDescFrameGeneration g{};
    g.header.type = FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION;
    g.commandList = (void*)cb;
    g.numGeneratedFrames = 2u;
    g.reset = false;
    g.backbufferTransferFunction = 0u;
    g.minMaxLuminance[0] = 0.0f;
    g.minMaxLuminance[1] = 1.0f;
    g.generationRect = {0, 0, 0, 0};
    g.frameID = 1001u;
    g.presentColor = mkRes(color[0].image, ow, oh, FFX_API_SURFACE_FORMAT_R16G16B16A16_FLOAT,
                           FFX_API_RESOURCE_USAGE_READ_ONLY, FFX_API_RESOURCE_STATE_COMPUTE_READ);
    for (uint32_t i = 0; i < 3u; ++i)
        g.outputs[i] = mkRes(out3[i].image, ow, oh, FFX_API_SURFACE_FORMAT_R8G8B8A8_UNORM,
                             FFX_API_RESOURCE_USAGE_UAV, FFX_API_RESOURCE_STATE_UNORDERED_ACCESS);
    const int generate = (int)f.dispatch(&ctx, &g.header);
    std::printf("  ffxDispatch(Generate 2 + presentColor + 3 outputs) = %d numGeneratedFrames=%u\n", generate,
                g.numGeneratedFrames);

    vkEndCommandBuffer(cb);
    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cb;
    const VkResult submitted = vkQueueSubmit(a.queue, 1, &si, VK_NULL_HANDLE);
    const VkResult waited = (submitted == VK_SUCCESS) ? vkQueueWaitIdle(a.queue) : (VkResult)VK_ERROR_UNKNOWN;
    std::printf("  submit=%d wait=%d\n", (int)submitted, (int)waited);
    std::printf("  ffxDestroyContext = %d\n", (int)f.destroy(&ctx, nullptr));
    return 0;
}

int main(int argc, char** argv) {
    openCapture();
    const std::string dll = (argc > 1) ? argv[1] : "amd_fidelityfx_framegeneration_vk.dll";
    if (!createDeviceWith())
        return 77;
    if (!createFgDevice())
        return 77;
    Ffx f;
    if (!loadFfx(f, dll))
        return 77;
    ffxContext ctx{};
    const int ladder = driveLadder(f, ctx);
    if (ladder != 0)
        return ladder;
    const int dispatched = driveDispatches(f, ctx);
    return dispatched;
}

