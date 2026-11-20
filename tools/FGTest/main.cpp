// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 fsr4-vk contributors.
// See LICENSE in the project root.
//
// tools/FGTest - a headless, self-contained animated-triangle Vulkan host that drives BOTH FSR4 providers
// (the upscaler amd_fidelityfx_upscaler_vk.so and Frame Generation amd_fidelityfx_framegeneration_vk.so) through
// their REAL, documented FFX modular-API calls. It answers "does it run?" per feature with printed codes:
//   * the moving triangle renders OFFSCREEN (no window) -> PPM frames, proving the graphics host runs;
//   * the UPSCALER provider is live: create + the deterministic ffxQuery helpers (ratio / render-resolution) AND
//     a REAL pixel upscale dispatch (`ffxDispatchDescUpscale`) over the app's own VkImages, submitted and read back
//     off the GPU -> 960x540 PPM frames whose bytes CHANGE per frame (proof the AMD compute ran, not a stub);
//   * the FRAME GENERATION provider is live: the full create->versions->memory->configure->prepare->generate
//     host-integration ladder, printing AMD's documented return codes.
// HUD: every frame that goes to the window (and, with --hud-dump, into the PPM proof frames) carries a
// readout painted by a hand-set 5x7 bitmap font, with NO text pipeline at all -- the window path already
// composes a CPU RGBA8 buffer and uploads it to SDL's software renderer, so the readout is blitted INTO
// those SAME bytes BEFORE the upload. Top-LEFT = the three resolutions that are actually in play, each read
// back off the rig or off SDL rather than assumed: the window extent, the app's own render extent, and the
// extent amd_fidelityfx_upscaler_vk rendered. Top-RIGHT = the clocks: measured FPS painted 4x with its
// frame-to-frame MS delta under it at 2x, then the upscaler cost, the frame-generation call cost, and the
// multiplier as GOT <n>X REQ <n>X -- so while generate is refused the screen keeps showing 1 X against the
// requested 2 X/4 X instead of implying an interpolation that never happened. `--hud-dump` also prints the
// two corners as ASCII, which is how the glyph table's legibility is checked without a display.
// HONEST LIMIT (never fabricated): Frame Generation's ffxDispatch(generate 1..3) returns
// FFX_API_RETURN_ERROR_RUNTIME_ERROR in this provider slice -- the interpolation/ML/present runtime is not
// implemented, so no interpolated frame is invented. Since Phase 1 the provider DOES allocate a real substrate
// (the pack plan's intermediates + descriptor pool + the 92 classic-subset compute pipelines) at Prepare, and
// then ffxQuery(GetGPUMemoryUsage) answers with those allocations - but ONLY on a host device that enabled
// VK_EXT_descriptor_buffer, which is what tools/framegeneration-probe drives. This harness deliberately keeps
// its device WITHOUT that transport: enabling it was measured to crash inside the driver's own GRAPHICS pipeline
// creation here (a SIGSEGV in libvulkan_radeon from setupPipeline), so FGTest shows the substrate refusing and
// naming the missing transport instead of pretending the numbers exist. This app is the moving-graphics host
// such a runtime would plug into; it does NOT fake interpolated frames. The image marshalling is NOT guessed: every
// resource layout, surface-format claim and usage bit follows the validators in this repo's own upscaler provider
// (`src/provider/provider.cpp`), which is why a refusal here is always the provider's own (RUNTIME_ERROR) and never
// a host-side PARAMETER error.
// THIS IS THE RUNNABLE MINI APP OF THE PHASE: `bash tools/FGTest/run.sh` opens the window for 30 seconds (default;
// `--seconds=N` to change, Esc/q or closing the window ends it early) and prints the measured lines as it goes.
// `run.sh --headless` (or exit code 77 from a host with no display/device) runs the same proof offscreen and prints
// the HUD corners as ASCII, which is how the readout is checked without ever putting pixels on a screen.

#if !defined(_WIN32)
#if !defined(__declspec)
#define __declspec(x) __attribute__((visibility("default")))
#endif
#endif

#include <vulkan/vulkan.h>
#include "ffx_api.h"
#include "ffx_upscale.h"
#include "ffx_frame_generation.h"

#include <dlfcn.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <cmath>
#include <chrono>
#include <thread>
#include <bit>  // std::bit_cast for the IEEE half -> float decode of the FSR FP16 output

// SDL2 drives the optional --window present mode (it manages the native window so
// Vulkan can take the surface, unlike the host's GLFW build). <vulkan/vulkan.h> is
// included first so SDL_vulkan.h's VkInstance/VkSurfaceKHR prototypes are visible.
#include <SDL.h>
#include <SDL_vulkan.h>

static constexpr uint32_t kWidth = 480;
static constexpr uint32_t kHeight = 270;
static constexpr uint32_t kFrames = 6;
// --window demo geometry: render at 1280x720, present into a 1440x1080 window.
static constexpr uint32_t kRenderW = 1280, kRenderH = 720;
static constexpr uint32_t kDispW = 1440, kDispH = 1080;

static SDL_Window* gWin = nullptr;  // created before VkInstance so SDL can drive WSI

// HUD: the readout is PAINTED by the hand-set 5x7 bitmap font further down this file, but the FSR proof path
// and the Frame Generation ladder both need to reach it, so the facts they fill and the two entry points the
// callers use are declared where they are first used. Every value in HudFacts is MEASURED in this process -
// extents read back off the rig SDL reports, wall-clock costs of the calls actually made - never assumed.
struct HudFacts {
    uint32_t winW = 0, winH = 0;   // the on-screen window extent SDL reports
    uint32_t rndW = 0, rndH = 0;   // the app's own offscreen render extent
    uint32_t upW = 0, upH = 0;     // what amd_fidelityfx_upscaler_vk rendered (0 == no dispatch)
    double frameMs = 0.0;          // presented frame to presented frame
    double upMs = 0.0;             // this frame's upscale dispatch + read-back cost
    double fgMs = 0.0;             // the frame-generation call cost
    int fps = 0;
    bool presented = false;        // false == no presented-frame cadence to time (headless proof path)
    int fgRequested = 1;           // what the host asked for (--fg=2x / 4x)
    int fgActual = 1;              // what the provider really delivered (1 == no interpolation)
    bool fgRefused = false;        // generate returned RUNTIME_ERROR, i.e. honest refusal
    // The counted FRAME numbers, filled by the window path's per-frame Prepare+generate attempts.
    long long fgAttempts = 0;      // generate calls made in this process
    long long fgGenerated = 0;     // frames the provider actually DELIVERED, summed over its answers
    int fgCode = 0xFFFF;           // the last generate return code (0 OK, 3 runtime, 1/6 parameter, ...)
    bool fgLive = false;           // false == no FG context in this session (provider .so absent/refused)
};
static HudFacts gHudFacts;
static bool gHudDump = false;  // --hud-dump: paint the HUD into PPM frames and print the ASCII crop
static bool gHudEchoLines = false;  // with --hud-dump: print each string actually painted, in paint order
static void drawHud(unsigned char* rgba, uint32_t W, uint32_t H, const HudFacts& f);
static void hudDumpAscii(const unsigned char* rgba, uint32_t W, uint32_t H, int x0, int y0, int w, int h,
                         int step);

struct Mat4 {
    float m[4][4];  // column-major
    Mat4() { std::memset(m, 0, sizeof(m)); m[0][0] = m[1][1] = m[2][2] = m[3][3] = 1.0f; }
};
static Mat4 mul(const Mat4& a, const Mat4& b) {
    Mat4 r;
    for (int c = 0; c < 4; ++c)
        for (int rw = 0; rw < 4; ++rw) {
            float s = 0.0f;
            for (int k = 0; k < 4; ++k) s += a.m[k][rw] * b.m[c][k];
            r.m[c][rw] = s;
        }
    return r;
}
static Mat4 translate(float x, float y, float z) {
    Mat4 r; r.m[3][0] = x; r.m[3][1] = y; r.m[3][2] = z; return r;
}
static Mat4 rotateZ(float a) {
    Mat4 r; const float c = std::cos(a), s = std::sin(a);
    r.m[0][0] = c; r.m[0][1] = s; r.m[1][0] = -s; r.m[1][1] = c; return r;
}
static Mat4 ortho(float l, float r, float t, float b, float n, float f) {
    Mat4 o;
    o.m[0][0] = 2.0f / (r - l); o.m[1][1] = 2.0f / (t - b); o.m[2][2] = 1.0f / (f - n); o.m[3][3] = 1.0f;
    o.m[3][0] = -(r + l) / (r - l); o.m[3][1] = -(t + b) / (t - b); o.m[3][2] = -n / (f - n);
    return o;
}

struct ShaderData {  // std140: three mat4, column-major; memcpy directly.
    float projection[16];
    float model[16];
    float view[16];
} ;
static void putMat(float* dst, const Mat4& a) {
    for (int c = 0; c < 4; ++c) for (int r = 0; r < 4; ++r) dst[c * 4 + r] = a.m[c][r];
}

struct VkApp {
    VkInstance instance{VK_NULL_HANDLE};
    VkPhysicalDevice physical{VK_NULL_HANDLE};
    VkDevice device{VK_NULL_HANDLE};
    VkQueue queue{VK_NULL_HANDLE};
    VkFormat colorFmt{VK_FORMAT_R8G8B8A8_UNORM};
    VkFormat depthFmt{VK_FORMAT_D32_SFLOAT};
    VkImage color{VK_NULL_HANDLE}, depth{VK_NULL_HANDLE};
    VkImageView colorView{VK_NULL_HANDLE}, depthView{VK_NULL_HANDLE};
    VkBuffer vertexBuf{VK_NULL_HANDLE}, indexBuf{VK_NULL_HANDLE};
    VkDeviceMemory vertexMem{VK_NULL_HANDLE}, indexMem{VK_NULL_HANDLE};
    VkBuffer ubo[kFrames]{}; VkDeviceMemory uboMem[kFrames]{}; uint8_t* uboMap[kFrames]{};
    VkDescriptorSet descSet[kFrames]{};
    VkDescriptorPool pool{VK_NULL_HANDLE};
    VkDescriptorSetLayout setLayout{VK_NULL_HANDLE};
    VkPipelineLayout pipeLayout{VK_NULL_HANDLE};
    VkPipeline pipeline{VK_NULL_HANDLE};
    VkCommandPool cmdPool{VK_NULL_HANDLE};
    // --- present (--window) members; unused in the headless path ---
    VkSurfaceKHR surface{VK_NULL_HANDLE};
    VkSwapchainKHR swap{VK_NULL_HANDLE};
    VkFormat presentFmt{VK_FORMAT_B8G8R8A8_UNORM};
    VkExtent2D surfExtent{};
    VkPipeline presentPipeline{VK_NULL_HANDLE};
    std::vector<VkImage> scColor, scDepth;
    std::vector<VkImageView> scColorView, scDepthView;
    std::vector<VkDeviceMemory> scDepthMem;
    uint32_t imageCount{0};
    uint32_t memTypeCoherent(uint32_t bits, VkMemoryPropertyFlags props);
};
static uint32_t memIndex(VkApp& a, uint32_t bits, VkMemoryPropertyFlags props) {
    VkPhysicalDeviceMemoryProperties mp{};
    vkGetPhysicalDeviceMemoryProperties(a.physical, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
        if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & props) == props) return i;
    return 0xFFFFFFFFu;
}
static std::vector<unsigned char> readSpv(const std::string& p) {
    FILE* f = std::fopen(p.c_str(), "rb");
    if (!f) return {};
    std::fseek(f, 0, SEEK_END); long n = std::ftell(f); std::fseek(f, 0, SEEK_SET);
    std::vector<unsigned char> v((size_t)n); if (n) fread(v.data(), 1, n, f); std::fclose(f); return v;
}
static VkShaderModule mkShader(VkApp& a, const std::string& path) {
    auto code = readSpv(path);
    VkShaderModuleCreateInfo ci{.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                .codeSize = code.size(), .pCode = (const uint32_t*)code.data()};
    VkShaderModule s{VK_NULL_HANDLE};
    if (vkCreateShaderModule(a.device, &ci, nullptr, &s) != VK_SUCCESS || s == VK_NULL_HANDLE) {
        std::fprintf(stderr, "could not load SPIR-V: %s\n", path.c_str());
    }
    return s;
}

// --- modular FFX provider ABI, loaded per-.so (both providers export the SAME ffx* symbols, so dlopen +
//     RTLD_LOCAL disambiguates them) ---
struct Provider {
    void* h{nullptr};
    ffxReturnCode_t (*create)(ffxContext*, ffxCreateContextDescHeader*, const ffxAllocationCallbacks*){nullptr};
    ffxReturnCode_t (*configure)(ffxContext*, const ffxConfigureDescHeader*){nullptr};
    ffxReturnCode_t (*query)(ffxContext*, ffxQueryDescHeader*){nullptr};
    ffxReturnCode_t (*dispatch)(ffxContext*, const ffxDispatchDescHeader*){nullptr};
    ffxReturnCode_t (*destroy)(ffxContext*, const ffxAllocationCallbacks*){nullptr};
};
static bool loadProvider(Provider& p, const std::string& path) {
    p.h = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!p.h) { std::fprintf(stderr, "dlopen failed %s: %s\n", path.c_str(), dlerror()); return false; }
    p.create    = (decltype(p.create))  dlsym(p.h, "ffxCreateContext");
    p.configure = (decltype(p.configure))dlsym(p.h, "ffxConfigure");
    p.query     = (decltype(p.query))   dlsym(p.h, "ffxQuery");
    p.dispatch  = (decltype(p.dispatch))dlsym(p.h, "ffxDispatch");
    p.destroy   = (decltype(p.destroy)) dlsym(p.h, "ffxDestroyContext");
    return p.create && p.configure && p.query && p.dispatch && p.destroy;
}

// The host-declared enabled-feature block (type 7) + Vulkan backend block (type 3), laid out exactly as
// framegeneration-probe.cpp so both FSR4 providers decode it identically.
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
struct FfgBackendInstanceDesc {
    ffxApiHeader header;
    VkInstance instance;
    PFN_vkGetInstanceProcAddr getInstanceProcAddr;
};
static const char* kContractExtensions[] = {
    VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME, VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME,
    VK_KHR_COMPUTE_SHADER_DERIVATIVES_EXTENSION_NAME, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME,
    VK_KHR_VULKAN_MEMORY_MODEL_EXTENSION_NAME, VK_KHR_8BIT_STORAGE_EXTENSION_NAME,
    VK_KHR_16BIT_STORAGE_EXTENSION_NAME, VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME,
    VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME,
};
static bool hasExt(VkPhysicalDevice p, const char* name) {
    uint32_t c = 0; vkEnumerateDeviceExtensionProperties(p, nullptr, &c, nullptr);
    std::vector<VkExtensionProperties> t(c);
    if (c) vkEnumerateDeviceExtensionProperties(p, nullptr, &c, t.data());
    for (auto& e : t) if (!std::strcmp(e.extensionName, name)) return true;
    return false;
}
static bool initVulkan(VkApp& a, bool windowMode) {
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO, nullptr, "FGTest", 0, "FGTest", 0,
                          VK_MAKE_API_VERSION(0, 4, 0, 2)};
    VkInstanceCreateInfo ii{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ii.pApplicationInfo = &app;
    std::vector<const char*> instExts;
    if (windowMode && gWin) {
        unsigned n = 0; SDL_Vulkan_GetInstanceExtensions(gWin, &n, nullptr);
        std::vector<const char*> req(n);
        if (n) SDL_Vulkan_GetInstanceExtensions(gWin, &n, req.data());
        instExts = req;
    }
    ii.enabledExtensionCount = (uint32_t)instExts.size();
    ii.ppEnabledExtensionNames = instExts.empty() ? nullptr : instExts.data();
    if (vkCreateInstance(&ii, nullptr, &a.instance) != VK_SUCCESS) return false;
    uint32_t pc = 0; vkEnumeratePhysicalDevices(a.instance, &pc, nullptr);
    std::vector<VkPhysicalDevice> ps(pc);
    if (pc) vkEnumeratePhysicalDevices(a.instance, &pc, ps.data());
    if (!pc) { std::fprintf(stderr, "no Vulkan device\n"); return false; }
    for (auto p : ps)
        if (hasExt(p, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME)) { a.physical = p; break; }
    if (!a.physical) a.physical = ps[0];
    std::vector<const char*> exts;
    for (const char* e : kContractExtensions) if (hasExt(a.physical, e)) exts.push_back(e);
    if (hasExt(a.physical, "VK_xr_swapchain")) exts.push_back("VK_xr_swapchain"); // required by the loader for vkCreateSwapchainKHR in window mode

    VkPhysicalDeviceFeatures2 base{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    VkPhysicalDeviceVulkan12Features v12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceVulkan13Features v13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    VkPhysicalDevice8BitStorageFeatures st8{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES};
    VkPhysicalDevice16BitStorageFeatures st16{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES};
    VkPhysicalDeviceShaderFloat16Int8Features f16i8{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES};
    VkPhysicalDeviceDescriptorIndexingFeatures descIdx{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES};
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR coop{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
    VkPhysicalDeviceSubgroupSizeControlFeatures ssc{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES};
    base.pNext = &v12; v12.pNext = &v13; v13.pNext = &st8; st8.pNext = &st16;
    st16.pNext = &f16i8; f16i8.pNext = &descIdx; descIdx.pNext = &coop; coop.pNext = &ssc;
    base.features.shaderInt16 = VK_TRUE; base.features.shaderInt64 = VK_TRUE;
    base.features.shaderStorageImageWriteWithoutFormat = VK_TRUE;
    v12.bufferDeviceAddress = VK_TRUE;
    v13.dynamicRendering = VK_TRUE; v13.synchronization2 = VK_TRUE;
    st8.storageBuffer8BitAccess = VK_TRUE;
    st16.storageBuffer16BitAccess = VK_TRUE;
    f16i8.shaderFloat16 = VK_TRUE; f16i8.shaderInt8 = VK_TRUE;
    descIdx.runtimeDescriptorArray = VK_TRUE;
    coop.cooperativeMatrix = VK_TRUE;
    ssc.subgroupSizeControl = VK_TRUE;
    VkDeviceQueueCreateInfo q{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    q.queueFamilyIndex = 0; q.queueCount = 1;
    VkDeviceCreateInfo di{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    di.pNext = &base; di.queueCreateInfoCount = 1; di.pQueueCreateInfos = &q;
    di.enabledExtensionCount = (uint32_t)exts.size();
    di.ppEnabledExtensionNames = exts.empty() ? nullptr : exts.data();
    if (vkCreateDevice(a.physical, &di, nullptr, &a.device) != VK_SUCCESS) return false;
    vkGetDeviceQueue(a.device, 0, 0, &a.queue);

    auto mkImg = [&](VkFormat fmt, VkImageUsageFlags usage, VkImage& img, VkImageView& view,
                     VkImageAspectFlags aspect) {
        VkImageCreateInfo ic{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ic.imageType = VK_IMAGE_TYPE_2D; ic.format = fmt;
        ic.extent = {kWidth, kHeight, 1}; ic.mipLevels = 1; ic.arrayLayers = 1;
        ic.samples = VK_SAMPLE_COUNT_1_BIT; ic.usage = usage; ic.tiling = VK_IMAGE_TILING_OPTIMAL;
        ic.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        vkCreateImage(a.device, &ic, nullptr, &img);
        VkMemoryRequirements mr{}; vkGetImageMemoryRequirements(a.device, img, &mr);
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr, mr.size,
            memIndex(a, mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)};
        VkDeviceMemory m{VK_NULL_HANDLE}; vkAllocateMemory(a.device, &ai, nullptr, &m);
        vkBindImageMemory(a.device, img, m, 0);
        VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vi.image = img; vi.viewType = VK_IMAGE_VIEW_TYPE_2D; vi.format = fmt;
        vi.subresourceRange = {aspect, 0, 1, 0, 1};
        vkCreateImageView(a.device, &vi, nullptr, &view);
    };
    mkImg(a.colorFmt, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
          a.color, a.colorView, VK_IMAGE_ASPECT_COLOR_BIT);
    mkImg(a.depthFmt, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
          a.depth, a.depthView, VK_IMAGE_ASPECT_DEPTH_BIT);
    return true;
}

static uint32_t gIndexCount = 3;
static VkCommandPool mkCmdPool(VkApp& a);
static VkBuffer mkBuffer(VkApp& a, VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags props,
                         VkDeviceMemory* memOut) {
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, nullptr, 0, size, usage, VK_SHARING_MODE_EXCLUSIVE};
    VkBuffer b{VK_NULL_HANDLE}; vkCreateBuffer(a.device, &bi, nullptr, &b);
    VkMemoryRequirements mr{}; vkGetBufferMemoryRequirements(a.device, b, &mr);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr, mr.size,
        memIndex(a, mr.memoryTypeBits, props)};
    vkAllocateMemory(a.device, &ai, nullptr, memOut);
    vkBindBufferMemory(a.device, b, *memOut, 0);
    return b;
}
static void setupGeom(VkApp& a) {
    struct V { float p[3]; float c[3]; };
    const V verts[3] = { {{0.6f,0.6f,0.0f},{1,0,0}}, {{-0.6f,0.6f,0.0f},{0,1,0}}, {{0.0f,-0.6f,0.0f},{0,0,1}} };
    const uint32_t idx[3] = {0,1,2};
    const VkDeviceSize vSize = sizeof(verts), iSize = sizeof(idx);
    VkDeviceMemory stageMem{VK_NULL_HANDLE};
    VkBuffer stage = mkBuffer(a, vSize + iSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &stageMem);
    uint8_t* mp{nullptr}; vkMapMemory(a.device, stageMem, 0, vSize + iSize, 0, (void**)&mp);
    std::memcpy(mp, verts, (size_t)vSize); std::memcpy(mp + vSize, idx, (size_t)iSize);
    vkUnmapMemory(a.device, stageMem);
    a.vertexBuf = mkBuffer(a, vSize, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &a.vertexMem);
    a.indexBuf = mkBuffer(a, iSize, VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &a.indexMem);
    a.cmdPool = mkCmdPool(a);
    VkCommandBuffer cc{VK_NULL_HANDLE};
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, nullptr, a.cmdPool,
        VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
    vkAllocateCommandBuffers(a.device, &cai, &cc);
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    vkBeginCommandBuffer(cc, &bi);
    VkBufferCopy r1{0, 0, (VkDeviceSize)vSize}, r2{(VkDeviceSize)vSize, 0, (VkDeviceSize)iSize};
    vkCmdCopyBuffer(cc, stage, a.vertexBuf, 1, &r1);
    vkCmdCopyBuffer(cc, stage, a.indexBuf, 1, &r2);
    vkEndCommandBuffer(cc);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr, 1, &cc, 0, nullptr};
    vkQueueSubmit(a.queue, 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(a.queue);
    vkDestroyBuffer(a.device, stage, nullptr); vkFreeMemory(a.device, stageMem, nullptr);
    vkFreeCommandBuffers(a.device, a.cmdPool, 1, &cc);
}
static VkCommandPool mkCmdPool(VkApp& a) {
    VkCommandPoolCreateInfo ci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, nullptr,
        VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, 0};
    VkCommandPool p{VK_NULL_HANDLE}; vkCreateCommandPool(a.device, &ci, nullptr, &p); return p;
}
static void setupUbo(VkApp& a) {
    VkDeviceSize n = sizeof(ShaderData);
    for (uint32_t i = 0; i < kFrames; ++i) {
        a.ubo[i] = mkBuffer(a, n, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &a.uboMem[i]);
        vkMapMemory(a.device, a.uboMem[i], 0, n, 0, (void**)&a.uboMap[i]);
    }
}
static void setupDescriptors(VkApp& a) {
    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, kFrames};
    VkDescriptorPoolCreateInfo pci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, nullptr,
        VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT, kFrames, 1, &ps};
    vkCreateDescriptorPool(a.device, &pci, nullptr, &a.pool);
    VkDescriptorSetLayoutBinding b{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0, 1, &b};
    vkCreateDescriptorSetLayout(a.device, &li, nullptr, &a.setLayout);
    for (uint32_t i = 0; i < kFrames; ++i) {
        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr, a.pool, 1,
            &a.setLayout};
        vkAllocateDescriptorSets(a.device, &ai, &a.descSet[i]);
        VkDescriptorBufferInfo bufi{a.ubo[i], 0, sizeof(ShaderData)};
        VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, a.descSet[i], 0, 0, 1,
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &bufi, nullptr};
        vkUpdateDescriptorSets(a.device, 1, &w, 0, nullptr);
    }
}
static void setupPipeline(VkApp& a, VkShaderModule vs, VkShaderModule fs, VkFormat cfmt, VkPipeline& outPipe) {
    VkPipelineLayoutCreateInfo li{.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1, .pSetLayouts = &a.setLayout};
    if (!a.pipeLayout) vkCreatePipelineLayout(a.device, &li, nullptr, &a.pipeLayout);
    VkVertexInputBindingDescription vb{.binding = 0, .stride = sizeof(float) * 6, .inputRate = VK_VERTEX_INPUT_RATE_VERTEX};
    VkVertexInputAttributeDescription va[2] = {
        {.location = 0, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = 0},
        {.location = 1, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = sizeof(float) * 3}};
    VkPipelineVertexInputStateCreateInfo vi{.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .vertexBindingDescriptionCount = 1, .pVertexBindingDescriptions = &vb,
        .vertexAttributeDescriptionCount = 2, .pVertexAttributeDescriptions = va};
    VkPipelineInputAssemblyStateCreateInfo ia{.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, .primitiveRestartEnable = VK_FALSE};
    VkPipelineViewportStateCreateInfo vp{.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .viewportCount = 1, .scissorCount = 1};
    VkPipelineRasterizationStateCreateInfo rs{.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .polygonMode = VK_POLYGON_MODE_FILL, .cullMode = VK_CULL_MODE_NONE,
        .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE, .lineWidth = 1.0f};
    VkPipelineMultisampleStateCreateInfo ms{.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT};
    VkPipelineColorBlendAttachmentState cba{.blendEnable = VK_FALSE, .colorWriteMask = 0xf};
    VkPipelineColorBlendStateCreateInfo cb{.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .attachmentCount = 1, .pAttachments = &cba};
    VkDynamicState dyn[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo ds{.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .dynamicStateCount = 2, .pDynamicStates = dyn};
    VkPipelineShaderStageCreateInfo st[2] = {
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, vs, "main", nullptr},
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, fs, "main", nullptr}};
    VkPipelineRenderingCreateInfo rr{.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO, .colorAttachmentCount = 1,
        .pColorAttachmentFormats = &cfmt, .depthAttachmentFormat = a.depthFmt};
    VkGraphicsPipelineCreateInfo gci{.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .pNext = &rr, .stageCount = 2,
        .pStages = st, .pVertexInputState = &vi, .pInputAssemblyState = &ia, .pViewportState = &vp,
        .pRasterizationState = &rs, .pMultisampleState = &ms, .pColorBlendState = &cb, .pDynamicState = &ds,
        .layout = a.pipeLayout};
    vkCreateGraphicsPipelines(a.device, VK_NULL_HANDLE, 1, &gci, nullptr, &outPipe);
    vkDestroyShaderModule(a.device, vs, nullptr);
    vkDestroyShaderModule(a.device, fs, nullptr);
}

static ffxReturnCode_t cbPresent(ffxCallbackDescFrameGenerationPresent*, void*) { return FFX_API_RETURN_OK; }
static ffxReturnCode_t cbFg(ffxDispatchDescFrameGeneration*, void*) { return FFX_API_RETURN_OK; }

// Fill the host-declared enabled-feature block (type 7) + Vulkan backend block (type 3) that BOTH FSR4
// providers require on their create chain (same layout framegeneration-probe.cpp uses).
static void buildContract(VkApp& a, FfgEnabledFeaturesDesc& en, FfgBackendDesc& bk) {
    en = FfgEnabledFeaturesDesc{};
    en.header = ffxApiHeader{7u, nullptr};
    en.shaderInt16 = en.shaderInt64 = en.shaderFloat16 = en.shaderInt8 = en.storage8Bit = en.storage16Bit = VK_TRUE;
    en.bufferDeviceAddress = en.runtimeDescriptorArray = en.cooperativeMatrix = en.vulkanMemoryModel = VK_TRUE;
    en.subgroupSizeControl = en.computeDerivativeGroupLinear = en.synchronization2 = en.storageImageWriteWithoutFormat = VK_TRUE;
    bk = FfgBackendDesc{};
    bk.header = ffxApiHeader{3u, &en.header};
    bk.device = a.device; bk.physicalDevice = a.physical; bk.getDeviceProcAddr = vkGetDeviceProcAddr;
}

static void writePPM(const std::string& path, uint32_t w, uint32_t h, const uint8_t* rgba) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return;
    std::fprintf(f, "P6\n%u %u\n255\n", w, h);
    for (uint32_t i = 0; i < w * h; ++i) std::fwrite(rgba + (size_t)i * 4, 1, 3, f);
    std::fclose(f);
}
// Shared triangle recording: begins a command buffer, renders the moving/rotating triangle into the CALLER's
// colour/depth images, and returns the STILL-OPEN buffer so each caller can append its own tail (an AMD FSR4
// upscale dispatch that records compute work, or an image read-back) before ending + submitting.
// `f` picks the UBO/descriptor slot (mod kFrames); `phase` is the animation clock, kept SEPARATE so the offscreen
// PPM loop (phase = frame index) and the live window loop (phase = elapsed seconds * 2, the --present clock)
// both animate at a sane rate.
static VkCommandBuffer recordTriangle(VkApp& a, uint32_t f, double phase, VkImage colorImg, VkImageView colorView,
                                      VkImage depthImg, VkImageView depthView, VkPipeline pipe, uint32_t rw,
                                      uint32_t rh) {
    const float p = (float)phase;
    Mat4 model = mul(translate(0.55f * std::sinf(p), 0.25f * std::cosf(p * 0.8f), 0.0f),
                     rotateZ(p * 0.7f));
    Mat4 proj = ortho(-1.6f, 1.6f, -0.9f, 0.9f, -1.0f, 1.0f);
    Mat4 view;
    ShaderData d;
    putMat(d.projection, proj); putMat(d.model, model); putMat(d.view, view);
    std::memcpy(a.uboMap[f % kFrames], &d, sizeof(d));

    VkCommandBuffer cc{VK_NULL_HANDLE};
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, nullptr, a.cmdPool,
        VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
    vkAllocateCommandBuffers(a.device, &cai, &cc);
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr, 0, nullptr};
    vkBeginCommandBuffer(cc, &bi);

    VkImageMemoryBarrier toAttach[2] = {
        {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, .srcAccessMask = 0,
         .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
         .newLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL, .image = colorImg,
         .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}},
        {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, .srcAccessMask = 0,
         .dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
         .newLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL, .image = depthImg,
         .subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1}}};
    vkCmdPipelineBarrier(cc, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_DEPENDENCY_BY_REGION_BIT, 0, nullptr, 0, nullptr, 2, toAttach);

    VkClearValue cval; cval.color = {0.10f, 0.10f, 0.20f, 1.0f};
    VkRenderingAttachmentInfo colorA{.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, .imageView = colorView,
        .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE, .clearValue = cval};
    VkClearValue dval; dval.depthStencil = {1.0f, 0};
    VkRenderingAttachmentInfo depthA{.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, .imageView = depthView,
        .imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE, .clearValue = dval};
    VkRenderingInfo ri{.sType = VK_STRUCTURE_TYPE_RENDERING_INFO, .renderArea = {0, 0, rw, rh},
        .layerCount = 1, .colorAttachmentCount = 1, .pColorAttachments = &colorA, .pDepthAttachment = &depthA};
    vkCmdBeginRendering(cc, &ri);
    VkViewport vpt{0.0f, 0.0f, (float)rw, (float)rh, 0.0f, 1.0f};
    vkCmdSetViewport(cc, 0, 1, &vpt);
    VkRect2D sc{0, 0, rw, rh};
    vkCmdSetScissor(cc, 0, 1, &sc);
    vkCmdBindDescriptorSets(cc, VK_PIPELINE_BIND_POINT_GRAPHICS, a.pipeLayout, 0, 1, &a.descSet[f % kFrames], 0, nullptr);
    vkCmdBindPipeline(cc, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
    VkDeviceSize off[1] = {0};
    vkCmdBindVertexBuffers(cc, 0, 1, &a.vertexBuf, off);
    vkCmdBindIndexBuffer(cc, a.indexBuf, 0, VK_INDEX_TYPE_UINT32);
    vkCmdDrawIndexed(cc, gIndexCount, 1, 0, 0, 0);
    vkCmdEndRendering(cc);
    return cc;
}

// OFFSCREEN render of one frame + read-back into the CPU host buffer (RGBA8) used for PPM output.
static void renderFrame(VkApp& a, uint32_t f, double phase, std::vector<unsigned char>& host) {
    VkCommandBuffer cc = recordTriangle(a, f, phase, a.color, a.colorView, a.depth, a.depthView, a.pipeline,
                                       kWidth, kHeight);


    VkImageMemoryBarrier toTransfer{.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL, .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        .image = a.color, .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    vkCmdPipelineBarrier(cc, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_DEPENDENCY_BY_REGION_BIT, 0, nullptr, 0, nullptr, 1, &toTransfer);

    VkDeviceSize n = (VkDeviceSize)kWidth * kHeight * 4;
    VkDeviceMemory rbMem{VK_NULL_HANDLE};
    VkBuffer rb = mkBuffer(a, n, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &rbMem);
    VkBufferImageCopy cp{}; cp.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}; cp.imageExtent = {kWidth, kHeight, 1};
    vkCmdCopyImageToBuffer(cc, a.color, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, rb, 1, &cp);
    vkEndCommandBuffer(cc);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr, 1, &cc, 0, nullptr};
    vkQueueSubmit(a.queue, 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(a.queue);
    uint8_t* map{nullptr}; vkMapMemory(a.device, rbMem, 0, n, 0, (void**)&map);
    host.assign(map, map + n);
    vkUnmapMemory(a.device, rbMem);
    vkDestroyBuffer(a.device, rb, nullptr); vkFreeMemory(a.device, rbMem, nullptr);
    vkFreeCommandBuffers(a.device, a.cmdPool, 1, &cc);
}






// ================================================================================================
// REAL AMD FSR4 IMAGE DISPATCH (not just create+query): feed the upscaler the app's OWN GPU images.
// Ground truth read from this repo's upscaler provider (src/provider/provider.cpp -> the .so under test):
//   * ffxDispatch RECORDS compute work into the HOST command list and never submits (the host submits).
//   * FfxApiResource.resource decodes as a bare VkImage (reinterpret_cast<VkImage>(resource.resource));
//     validate_resource needs: handle!=0, type==TEXTURE2D, description.width/height >= ACTIVE size,
//     description.format == the role's claimed format, description.state == the role's state, depth/stencil
//     usage bits as derived, and FFX_API_RESOURCE_USAGE_UAV when the state is UNORDERED_ACCESS.
//   * inputColorVkFormat accepts ONLY R16G16B16A16_FLOAT / B10G11R11 / RGB9E5 -- the old --present code
//     claimed R8G8B8A8_UNORM for colour, so the provider REJECTED it with PARAMETER and FSR never ran.
//   * the depth role MUST CLAIM FFX_API_SURFACE_FORMAT_R32_FLOAT; with FFX_API_RESOURCE_USAGE_DEPTHTARGET it
//     binds VK_FORMAT_D32_SFLOAT + DEPTH aspect -> the app's REAL rendered depth attachment can be handed
//     over untouched (no synthetic depth).
//   * motion = R16G16_FLOAT (VK_FORMAT_R16G16_SFLOAT); output = R16G16B16A16_FLOAT; AUTO_EXPOSURE lets
//     exposure stay null. Layouts follow the repo's own on-device gate tools/descriptor-smoke.cpp:
//     inputs SHADER_READ_ONLY_OPTIMAL, output GENERAL, then GENERAL -> TRANSFER_SRC_OPTIMAL to read back.
// ================================================================================================
static constexpr uint32_t kFsrOw = 960, kFsrOh = 540;  // 480x270 render -> 960x540 AMD FSR upscale
struct FsrRig {
    uint32_t rw{}, rh{}, ow{}, oh{};
    VkImage color{}, depth{}, motion{}, out{};
    VkImageView colorV{}, depthV{}, motionV{}, outV{};
    VkDeviceMemory colorM{}, depthM{}, motionM{}, outM{};
    VkPipeline pipeline{VK_NULL_HANDLE};
    VkBuffer rb{VK_NULL_HANDLE}; VkDeviceMemory rbM{VK_NULL_HANDLE};  // FP16 output read-back
    uint8_t* rbMap{nullptr};
    bool live{false};
};
static void mkImg(VkApp& a, uint32_t w, uint32_t h, VkFormat fmt, VkImageUsageFlags usage, VkImageAspectFlags aspect,
                  VkImage& im, VkImageView& vw, VkDeviceMemory& mem) {
    VkImageCreateInfo ic{.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D, .format = fmt,
        .extent = {w, h, 1}, .mipLevels = 1, .arrayLayers = 1, .samples = VK_SAMPLE_COUNT_1_BIT, .usage = usage,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
    if (vkCreateImage(a.device, &ic, nullptr, &im) != VK_SUCCESS || !im) return;
    VkMemoryRequirements mr{}; vkGetImageMemoryRequirements(a.device, im, &mr);
    const uint32_t mi = memIndex(a, mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (mi == 0xFFFFFFFFu) {  // no compatible memory type: destroy the image and let the caller report it
        vkDestroyImage(a.device, im, nullptr); im = VK_NULL_HANDLE; return;
    }
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr, mr.size, mi};
    if (vkAllocateMemory(a.device, &ai, nullptr, &mem) != VK_SUCCESS || !mem) {
        vkDestroyImage(a.device, im, nullptr); im = VK_NULL_HANDLE; return;
    }
    vkBindImageMemory(a.device, im, mem, 0);
    VkImageViewCreateInfo vi{.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = im,
        .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = fmt, .subresourceRange = {aspect, 0, 1, 0, 1}};
    if (vkCreateImageView(a.device, &vi, nullptr, &vw) != VK_SUCCESS) return;
}
// FSR4 rig teardown (images, views, pipeline, read-back buffer).
static void teardownFsrRig(VkApp& a, FsrRig& r) {
    if (r.rbMap && r.rbM) vkUnmapMemory(a.device, r.rbM);
    if (r.rb) vkDestroyBuffer(a.device, r.rb, nullptr);
    if (r.rbM) vkFreeMemory(a.device, r.rbM, nullptr);
    if (r.pipeline) vkDestroyPipeline(a.device, r.pipeline, nullptr);
    for (auto* v : {&r.colorV, &r.depthV, &r.motionV, &r.outV}) if (*v) vkDestroyImageView(a.device, *v, nullptr);
    for (auto* im : {&r.color, &r.depth, &r.motion, &r.out}) if (*im) vkDestroyImage(a.device, *im, nullptr);
    for (auto* m : {&r.colorM, &r.depthM, &r.motionM, &r.outM}) if (*m) vkFreeMemory(a.device, *m, nullptr);
    r = FsrRig{};
}
static bool setupFsrRig(VkApp& a, FsrRig& r, const std::string& vertSpv, const std::string& fragSpv, uint32_t rw,
                       uint32_t rh, uint32_t ow, uint32_t oh) {
    r.rw = rw; r.rh = rh; r.ow = ow; r.oh = oh;
    mkImg(a, rw, rh, VK_FORMAT_R16G16B16A16_SFLOAT,
          VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
          VK_IMAGE_ASPECT_COLOR_BIT, r.color, r.colorV, r.colorM);
    mkImg(a, rw, rh, VK_FORMAT_D32_SFLOAT,
          VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_DEPTH_BIT,
          r.depth, r.depthV, r.depthM);
    mkImg(a, rw, rh, VK_FORMAT_R16G16_SFLOAT, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
          VK_IMAGE_ASPECT_COLOR_BIT, r.motion, r.motionV, r.motionM);
    mkImg(a, ow, oh, VK_FORMAT_R16G16B16A16_SFLOAT,
          VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
              VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_IMAGE_ASPECT_COLOR_BIT, r.out, r.outV, r.outM);
    if (!r.color || !r.depth || !r.motion || !r.out || !r.colorV || !r.depthV || !r.motionV || !r.outV) {
        teardownFsrRig(a, r); return false;
    }
    auto vs = mkShader(a, vertSpv), fs = mkShader(a, fragSpv);
    if (vs == VK_NULL_HANDLE || fs == VK_NULL_HANDLE) { teardownFsrRig(a, r); return false; }
    // setupPipeline OWNS the shader-module teardown (it destroys vs/fs itself), so do NOT destroy them again here.
    setupPipeline(a, vs, fs, VK_FORMAT_R16G16B16A16_SFLOAT, r.pipeline);  // FP16 colour + D32 depth attachment
    if (r.pipeline == VK_NULL_HANDLE) { teardownFsrRig(a, r); return false; }
    r.rb = mkBuffer(a, (VkDeviceSize)ow * oh * 8u, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &r.rbM);
    if (!r.rb || !r.rbM) { teardownFsrRig(a, r); return false; }
    // Map the staging buffer ONCE for the whole session: per-frame map/unmap on a multi-MB buffer is what made
    // the read-back cost ~350ms/frame on this driver.
    if (vkMapMemory(a.device, r.rbM, 0, (VkDeviceSize)ow * oh * 8u, 0, (void**)&r.rbMap) != VK_SUCCESS) {
        teardownFsrRig(a, r); return false;
    }
    // One-time pass: motion vectors = (0,0) [this host supplies no motion-vector pass; the printed result says
    // so, it is not presented as real G-buffer data] and the FP16 output cleared, then both are parked in the
    // layouts the FSR model expects (descriptor-smoke.cpp): inputs SHADER_READ_ONLY_OPTIMAL, output GENERAL.
    VkCommandBuffer cc{VK_NULL_HANDLE};
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, nullptr, a.cmdPool,
        VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
    vkAllocateCommandBuffers(a.device, &cai, &cc);
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr, 0, nullptr};
    vkBeginCommandBuffer(cc, &bi);
    VkImageMemoryBarrier toDst[2] = {
        {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, .srcAccessMask = 0,
         .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT, .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
         .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, .image = r.motion,
         .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}},
        {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, .srcAccessMask = 0,
         .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT, .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
         .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, .image = r.out,
         .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}}};
    vkCmdPipelineBarrier(cc, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_DEPENDENCY_BY_REGION_BIT, 0, nullptr, 0, nullptr, 2, toDst);
    VkClearColorValue zero{}; zero.float32[0] = zero.float32[1] = zero.float32[2] = zero.float32[3] = 0.0f;
    VkImageSubresourceRange cRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdClearColorImage(cc, r.motion, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &zero, 1, &cRange);
    vkCmdClearColorImage(cc, r.out, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &zero, 1, &cRange);
    VkImageMemoryBarrier toUse[2] = {
        {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
         .dstAccessMask = VK_ACCESS_SHADER_READ_BIT, .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
         .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, .image = r.motion, .subresourceRange = cRange},
        {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
         .dstAccessMask = (VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT),
         .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
         .newLayout = VK_IMAGE_LAYOUT_GENERAL, .image = r.out, .subresourceRange = cRange}};
    vkCmdPipelineBarrier(cc, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_DEPENDENCY_BY_REGION_BIT, 0, nullptr, 0, nullptr, 2, toUse);
    vkEndCommandBuffer(cc);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr, 1, &cc, 0, nullptr};
    vkQueueSubmit(a.queue, 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(a.queue);
    vkFreeCommandBuffers(a.device, a.cmdPool, 1, &cc);
    return true;
}
// Exact IEEE half -> float. A half's exponent bias is 15, so the float exponent is (e - 15 + 127) = e + 112;
// subnormal halves (exp 0) are normalized first, which lands on 113 - s for s left-shifts.
static float halfToFloat(uint16_t h) {
    const uint32_t bits = h;
    const uint32_t sign = (bits & 0x8000u) << 16;
    const int32_t exp = (int32_t)((bits >> 10) & 0x1Fu);
    uint32_t man = bits & 0x3FFu;
    if (exp == 0) {
        if (man == 0) return std::bit_cast<float, uint32_t>(sign);
        int32_t s = 0;  // normalize the subnormal mantissa
        while ((man & 0x400u) == 0) { man <<= 1; ++s; }
        man &= 0x3FFu;
        return std::bit_cast<float, uint32_t>(sign | ((uint32_t)(113 - s) << 23) | (man << 13));
    }
    if (exp == 30) return std::bit_cast<float, uint32_t>(sign | 0x7F800000u | (man << 13));  // inf / nan
    return std::bit_cast<float, uint32_t>(sign | ((uint32_t)(exp + 112) << 23) | (man << 13));
}
static uint8_t toByte(float v) { return (uint8_t)(v <= 0.0f ? 0 : (v >= 1.0f ? 255 : (int)(v * 255.0f + 0.5f))); }
// Marshals one app-owned VkImage into the FFX modular-API resource ABI the way this repo's provider decodes
// it: the handle itself (reinterpret_cast<VkImage>(resource)), a TEXTURE2D description carrying the BACKING
// size, the role's claimed API surface format + usage bits, and the role's state.
static FfxApiResource mkRes(VkImage img, uint32_t w, uint32_t h, uint32_t apiFmt, uint32_t apiUsage, uint32_t apiState) {
    FfxApiResource o{};
    o.resource = (void*)img;
    o.description.type = FFX_API_RESOURCE_TYPE_TEXTURE2D;
    o.description.format = apiFmt;
    o.description.width = w; o.description.height = h; o.description.depth = 1;
    o.description.mipCount = 1; o.description.flags = 0; o.description.usage = apiUsage;
    o.state = apiState;
    return o;
}
// ONE REAL AMD FSR4 upscale pass: render the triangle into the rig's FP16 colour + D32 depth, hand the provider
// {colour, the REAL rendered depth (DEPTHTARGET), motion, FP16 output} plus its command list, submit the recorded
// work, then copy the upscaled FP16 image back to CPU and convert to RGBA8. rc==0 together with a non-black
// output whose checksum CHANGES between frames is the evidence the AMD compute actually ran (nothing is faked).
static ffxReturnCode_t fsrUpscaleFrame(VkApp& a, Provider& up, ffxContext& ctx, FsrRig& r, uint32_t slot,
                                       double phase, std::vector<unsigned char>& outRGBA, uint64_t* checksum,
                                       double* tOut = nullptr) {  // tOut[0..3]: dispatch GPU, read-back GPU, checksum pass, half-decode pass
    const auto tStart = std::chrono::steady_clock::now();
    VkCommandBuffer cc = recordTriangle(a, slot, phase, r.color, r.colorV, r.depth, r.depthV, r.pipeline, r.rw, r.rh);
    // Attachments -> the SHADER_READ_ONLY layout the FSR model samples its inputs in.
    VkImageMemoryBarrier toRead[2] = {
        {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
         .srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
         .oldLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL, .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
         .image = r.color, .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}},
        {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
         .srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
         .oldLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL, .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
         .image = r.depth, .subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1}}};
    vkCmdPipelineBarrier(cc, VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_DEPENDENCY_BY_REGION_BIT, 0, nullptr, 0, nullptr, 2, toRead);

    ffxDispatchDescUpscale u{};
    u.header.type = FFX_API_DISPATCH_DESC_TYPE_UPSCALE;
    u.commandList = (void*)cc;
    u.color = mkRes(r.color, r.rw, r.rh, FFX_API_SURFACE_FORMAT_R16G16B16A16_FLOAT, FFX_API_RESOURCE_USAGE_READ_ONLY,
        FFX_API_RESOURCE_STATE_COMPUTE_READ);
    u.depth = mkRes(r.depth, r.rw, r.rh, FFX_API_SURFACE_FORMAT_R32_FLOAT, FFX_API_RESOURCE_USAGE_DEPTHTARGET,
        FFX_API_RESOURCE_STATE_COMPUTE_READ);
    u.motionVectors = mkRes(r.motion, r.rw, r.rh, FFX_API_SURFACE_FORMAT_R16G16_FLOAT,
        FFX_API_RESOURCE_USAGE_READ_ONLY, FFX_API_RESOURCE_STATE_COMPUTE_READ);
    u.output = mkRes(r.out, r.ow, r.oh, FFX_API_SURFACE_FORMAT_R16G16B16A16_FLOAT, FFX_API_RESOURCE_USAGE_UAV,
        FFX_API_RESOURCE_STATE_UNORDERED_ACCESS);
    u.exposure.resource = nullptr; u.reactive.resource = nullptr;
    u.transparencyAndComposition.resource = nullptr;
    u.renderSize = {r.rw, r.rh}; u.upscaleSize = {r.ow, r.oh};
    u.jitterOffset = {0.0f, 0.0f}; u.motionVectorScale = {1.0f, 1.0f};
    u.enableSharpening = true; u.sharpness = 0.2f; u.preExposure = 1.0f; u.reset = false;
    u.frameTimeDelta = 16.6f; u.cameraNear = 0.1f; u.cameraFar = 1000.0f; u.cameraFovAngleVertical = 1.0f;
    u.viewSpaceToMetersFactor = 0.0f; u.flags = 0;
    const ffxReturnCode_t rc = up.dispatch(&ctx, &u.header);
    vkEndCommandBuffer(cc);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr, 1, &cc, 0, nullptr};
    vkQueueSubmit(a.queue, 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(a.queue);
    if (rc != FFX_API_RETURN_OK) { vkFreeCommandBuffers(a.device, a.cmdPool, 1, &cc); return rc; }
    const auto tAfterDispatch = std::chrono::steady_clock::now();
    if (tOut) tOut[0] = std::chrono::duration<double, std::milli>(tAfterDispatch - tStart).count();
    // Read the upscaled FP16 result back. No layout transition: the model leaves the output in GENERAL and
    // vkCmdCopyImageToBuffer may read GENERAL directly, so a per-frame GENERAL->TRANSFER_SRC flip (which makes the
    // driver re-tile a multi-MB image every frame) is avoided; only an access-memory barrier is needed.
    VkCommandBuffer rb{VK_NULL_HANDLE};
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, nullptr, a.cmdPool,
        VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
    vkAllocateCommandBuffers(a.device, &cai, &rb);
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr, 0, nullptr};
    vkBeginCommandBuffer(rb, &bi);
    VkImageMemoryBarrier outSrc{.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = (VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT), .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_GENERAL, .newLayout = VK_IMAGE_LAYOUT_GENERAL, .image = r.out,
        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    vkCmdPipelineBarrier(rb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_DEPENDENCY_BY_REGION_BIT, 0, nullptr, 0, nullptr, 1, &outSrc);
    VkBufferImageCopy cp{}; cp.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}; cp.imageExtent = {r.ow, r.oh, 1};
    vkCmdCopyImageToBuffer(rb, r.out, VK_IMAGE_LAYOUT_GENERAL, r.rb, 1, &cp);
    vkEndCommandBuffer(rb);
    VkSubmitInfo si2{VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr, 1, &rb, 0, nullptr};
    vkQueueSubmit(a.queue, 1, &si2, VK_NULL_HANDLE);
    vkQueueWaitIdle(a.queue);
    const VkDeviceSize n = (VkDeviceSize)r.ow * r.oh * 8u;
    uint8_t* map = r.rbMap;  // mapped once in setupFsrRig; HOST_COHERENT so the copy is already visible here
    outRGBA.assign((size_t)r.ow * r.oh * 4u, 0u);
    // Reading the driver's HOST_VISIBLE mapping byte-by-byte costs ~250ms for 4MB on this RADV build, while a
    // single bulk copy is cheap - so pull the staging buffer into a local host buffer ONCE and checksum + FP16->
    // RGBA8 decode from that (this is what kept the window at 2 fps before).
    std::vector<uint8_t> cpu((size_t)n);
    std::memcpy(cpu.data(), map, (size_t)n);
    const auto tAfterMemcpy = std::chrono::steady_clock::now();
    uint64_t sum = 0;
    if (checksum)  // the checksum walk is only needed by the PPM proof; the live window path skips it
        for (size_t i = 0; i < cpu.size(); ++i) sum += (uint64_t)cpu[i] * (i + 1u);
    const auto tAfterChecksum = std::chrono::steady_clock::now();
    const uint16_t* px = (const uint16_t*)cpu.data();
    float vmin = 1e30f, vmax = -1e30f, vsum = 0.0f;
    for (uint32_t i = 0; i < r.ow * r.oh; ++i) {
        const float cr = halfToFloat(px[i * 4u + 0u]), cg = halfToFloat(px[i * 4u + 1u]), cb = halfToFloat(px[i * 4u + 2u]);
        if (cr < vmin) vmin = cr;
        if (cr > vmax) vmax = cr;
        vsum += cr;
        outRGBA[(size_t)i * 4u + 0u] = toByte(cr);
        outRGBA[(size_t)i * 4u + 1u] = toByte(cg);
        outRGBA[(size_t)i * 4u + 2u] = toByte(cb);
        outRGBA[(size_t)i * 4u + 3u] = 255u;
    }
    const auto tAfterConvert = std::chrono::steady_clock::now();
    vkFreeCommandBuffers(a.device, a.cmdPool, 1, &rb);
    if (checksum) *checksum = sum;
    if (tOut) {
        tOut[0] = std::chrono::duration<double, std::milli>(tAfterDispatch - tStart).count();
        tOut[1] = std::chrono::duration<double, std::milli>(tAfterMemcpy - tAfterDispatch).count();
        tOut[2] = std::chrono::duration<double, std::milli>(tAfterChecksum - tAfterMemcpy).count();
        tOut[3] = std::chrono::duration<double, std::milli>(tAfterConvert - tAfterChecksum).count();
    }
    std::printf("    FP16 output R channel: min=%.4f max=%.4f mean=%.4f\n", (double)vmin, (double)vmax, (double)(vsum / (double)(r.ow * r.oh)));
    return rc;
}
// Upscaler context sized for the rig. create() needs the enabled-features(7) + backend(3) + instance(4) chain
// and FSR4_VK_ASSET_ROOT for the model sidecars; AUTO_EXPOSURE is what lets the exposure resource stay null.
static ffxReturnCode_t createUpscaleSized(Provider& p, VkApp& a, ffxContext& ctx, uint32_t rw, uint32_t rh,
                                         uint32_t ow, uint32_t oh) {
    FfgEnabledFeaturesDesc en; FfgBackendDesc bk; FfgBackendInstanceDesc inst;
    buildContract(a, en, bk);
    inst.header = ffxApiHeader{4u, &en.header};
    inst.instance = a.instance; inst.getInstanceProcAddr = vkGetInstanceProcAddr;
    bk.header.pNext = &inst.header;
    ffxCreateContextDescUpscale cd{};
    cd.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
    cd.header.pNext = &bk.header;
    cd.flags = FFX_UPSCALE_ENABLE_AUTO_EXPOSURE;
    cd.maxRenderSize = {rw, rh};
    cd.maxUpscaleSize = {ow, oh};
    return p.create(&ctx, &cd.header, nullptr);
}
// The FSR host: the 4 provider-visible images + their pipeline, and the live upscale context. Created ONCE,
// reused by the PPM proof and by the window loop (the provider keeps temporal state between dispatches, so one
// context across frames is what FSR's accumulation/reactive design expects).
struct FsrHost {
    FsrRig rig;
    ffxContext ctx{};
    Provider* up{nullptr};
    bool live{false};
};
static bool initFsrHost(VkApp& a, Provider& up, FsrHost& h, const std::string& outDir) {
    std::printf("\n=== FSR4 UPSCALER IMAGE DISPATCH: app-owned VkImages %ux%u -> %ux%u ===\n", kWidth, kHeight,
                kFsrOw, kFsrOh);
    if (!setupFsrRig(a, h.rig, outDir + "/triangle.vert.spv", outDir + "/triangle.frag.spv", kWidth, kHeight,
                    kFsrOw, kFsrOh)) {
        std::printf("  RIG: FP16 colour / D32 depth / RG16F motion / FP16 output could not be created -> no proof\n");
        return false;
    }
    const ffxReturnCode_t cc = createUpscaleSized(up, a, h.ctx, kWidth, kHeight, kFsrOw, kFsrOh);
    std::printf("  ffxCreateContext(upscale) = %d (0=OK)\n", (int)cc);
    if (cc != FFX_API_RETURN_OK) { teardownFsrRig(a, h.rig); return false; }
    h.up = &up;
    h.live = true;
    return true;
}
static void teardownFsrHost(VkApp& a, FsrHost& h) {
    if (h.live && h.up) h.up->destroy(&h.ctx, nullptr);
    teardownFsrRig(a, h.rig);
    h.live = false;
}
// 3 frames of the MOVING triangle through the REAL AMD FSR4 compute; the per-frame checksum proves the output
// BYTES actually changed (a refused/no-op dispatch would leave them identical), and the PPMs are evidence on disk.
static bool runFsrProof(VkApp& a, FsrHost& h, const std::string& outDir, int frames) {
    std::vector<unsigned char> img;
    uint64_t prev = 0; int okN = 0, changed = 0, nonblack = 0;
    for (int f = 0; f < frames; ++f) {
        uint64_t sum = 0;
        double parts[4] = {0.0, 0.0, 0.0, 0.0};
        const ffxReturnCode_t rc = fsrUpscaleFrame(a, *h.up, h.ctx, h.rig, (uint32_t)f, (double)f * 0.9, img, &sum, parts);
        uint32_t nb = 0;
        for (size_t i = 0; i + 2 < img.size(); i += 4)
            if (img[i] || img[i + 1] || img[i + 2]) ++nb;
        const bool differs = (f == 0) ? (sum != 0ull) : (sum != prev);
        if (rc == FFX_API_RETURN_OK) ++okN;
        if (differs) ++changed;
        if (nb) ++nonblack;
        std::printf("  frame %d: ffxDispatch(upscale) = %d  out=%ux%u nonblack_px=%u checksum=%llu [%s] fsr=%.1fms copy=%.1fms checksum=%.1fms half=%.1fms\n", f, (int)rc,
                    kFsrOw, kFsrOh, nb, (unsigned long long)sum, differs ? "output CHANGED" : "output IDENTICAL",
                    parts[0], parts[1], parts[2], parts[3]);
        if (gHudDump) {
            // Same HUD the window gets, painted into the proof frames AFTER the checksum was
            // taken, so the readout can be verified headlessly (the ASCII crop below is the
            // legibility check) without touching what the per-frame proof measures.
            gHudFacts.rndW = h.rig.rw;
            gHudFacts.rndH = h.rig.rh;
            gHudFacts.upW = h.rig.ow;
            gHudFacts.upH = h.rig.oh;
            gHudFacts.winW = kDispW;
            gHudFacts.winH = kDispH;  // the display extent this host targets (SDL window size)
            gHudFacts.upMs = parts[0] + parts[1];
            // No presented frames on this path: leave frameMs/fps alone so drawHud prints
            // NO PRESENTED instead of a derived pseudo-frame-rate.
            drawHud(img.data(), kFsrOw, kFsrOh, gHudFacts);
        }
        if (nb) writePPM(outDir + "/fsr_" + std::to_string((uint32_t)f) + ".ppm", kFsrOw, kFsrOh, img.data());
        prev = sum;
        if (gHudDump && f + 1 == frames) {
            std::printf("HUD CROP, TOP-LEFT (the three resolutions; glyphs sampled back to 5x7):\n");
            hudDumpAscii(img.data(), kFsrOw, kFsrOh, 6, 6, 420, 60, 2);
            std::printf("HUD CROP, TOP-RIGHT (clocks: FPS at 4x, its deltas at 2x):\n");
            hudDumpAscii(img.data(), kFsrOw, kFsrOh, (int)kFsrOw - 300, 6, 300, 150, 2);
        }
    }
    const bool ran = (okN == frames) && (changed == frames) && (nonblack == frames);
    std::printf("  VERDICT: rc=OK %d/%d, output-changed %d/%d, non-black %d/%d -> AMD FSR4 compute %s\n", okN, frames,
                changed, frames, nonblack, frames, ran ? "RAN (proved: pixels + checksums vary per frame)" : "NOT PROVEN");
    return ran;
}
// ---------- HUD: a hand-set 5x7 bitmap font, painted into the host-side RGBA8 buffer ----------
// Deliberately low-level ("assembly-like"): no text pipeline, no glyph atlas, no shader change. The --window
// path already composes a CPU buffer (the FSR read-back, or the offscreen render) and uploads it to SDL's
// SOFTWARE renderer, so the readout is painted into those SAME bytes BEFORE the upload and therefore survives
// the software blit. Any character with no glyph here renders as a hollow box, so a typo in a label shows up
// on screen instead of silently dropping a word.
struct Glyph {
    char ch;
    const char* rows[7];  // 7 rows of 5 characters each; '1' == an inked pixel
};
static const Glyph kGlyphs[] = {
    {' ', {"00000", "00000", "00000", "00000", "00000", "00000", "00000"}},
    {':', {"00000", "00100", "00100", "00000", "00100", "00100", "00000"}},
    {'-', {"00000", "00000", "01110", "01110", "01110", "00000", "00000"}},
    {'.', {"00000", "00000", "00000", "00000", "00000", "00100", "00100"}},
    {'/', {"00011", "00011", "01111", "00110", "11100", "11000", "11000"}},
    {'%', {"11001", "00001", "00001", "01110", "10001", "10001", "00011"}},
    {'(', {"00110", "00011", "00001", "00001", "00001", "00011", "00110"}},
    {')', {"01100", "11000", "10000", "10000", "10000", "11000", "01100"}},
    {'0', {"01110", "11011", "10001", "10001", "10001", "11011", "01110"}},
    {'1', {"01110", "00100", "00100", "00100", "00100", "00100", "01110"}},
    {'2', {"01110", "00111", "00011", "01111", "11100", "11000", "01111"}},
    {'3', {"01110", "11011", "10001", "01110", "10001", "11011", "01110"}},
    {'4', {"11011", "10001", "11011", "11011", "01001", "00101", "00001"}},
    {'5', {"01110", "11110", "10010", "00010", "00110", "11100", "01000"}},
    {'6', {"01110", "11011", "10001", "00111", "00111", "00111", "01111"}},
    {'7', {"01110", "11011", "10001", "01001", "01001", "00001", "00001"}},
    {'8', {"01110", "11011", "10001", "11111", "10001", "11011", "01110"}},
    {'9', {"01111", "11011", "10001", "11110", "01000", "00100", "00000"}},
    {'A', {"01110", "00100", "00100", "01110", "01110", "10001", "10001"}},
    {'B', {"11010", "11101", "10110", "10011", "11011", "11101", "11010"}},
    {'C', {"01110", "11000", "10000", "10000", "10000", "11000", "01110"}},
    {'D', {"11100", "10010", "10001", "10001", "10001", "10010", "11100"}},
    {'E', {"11111", "10000", "10000", "11110", "10000", "10000", "11111"}},
    {'F', {"11110", "11110", "10000", "11110", "10000", "10000", "10000"}},
    {'G', {"01110", "11000", "10000", "10011", "11110", "11000", "01110"}},
    {'I', {"01110", "00100", "00100", "00100", "00100", "00100", "01110"}},
    {'J', {"01111", "01000", "01000", "01000", "01010", "00110", "00010"}},
    {'K', {"11000", "11010", "10110", "10011", "10110", "11010", "11000"}},
    {'L', {"10000", "10000", "10000", "10000", "10000", "11100", "11100"}},
    {'M', {"11001", "11001", "10101", "10101", "10011", "10011", "10001"}},
    {'N', {"11001", "11001", "10101", "10011", "10011", "10011", "10001"}},
    {'O', {"01110", "11011", "10001", "10001", "10001", "11011", "01110"}},
    {'P', {"11110", "10010", "10010", "11110", "10000", "10000", "10000"}},
    {'Q', {"01110", "11011", "10001", "10001", "10011", "11011", "00111"}},
    {'R', {"11110", "10010", "10010", "11110", "10100", "10010", "10001"}},
    {'S', {"01110", "11000", "11000", "01110", "00011", "00011", "01110"}},
    {'T', {"11111", "11111", "00110", "00100", "00100", "00100", "00100"}},
    {'U', {"11001", "10001", "10001", "10001", "10001", "10011", "01100"}},
    {'V', {"11001", "11001", "10101", "10101", "10011", "10011", "01010"}},
    {'W', {"10001", "10001", "10101", "10101", "11011", "11011", "01010"}},
    {'X', {"10001", "10001", "01010", "00100", "01010", "10001", "10001"}},
    {'Y', {"11001", "01001", "01001", "01110", "00110", "00110", "00110"}},
    {'Z', {"11111", "11111", "01110", "00110", "00011", "11110", "11111"}},
};
static const Glyph kUnknownGlyph = {'?', {"00110", "11111", "10001", "01110", "01110", "10001", "00111"}};

static const Glyph& glyphFor(char ch) {
    const char upper = (ch >= 'a' && ch <= 'z') ? (char)(ch - 'a' + 'A') : ch;
    for (const auto& g : kGlyphs)
        if (g.ch == upper) return g;
    return kUnknownGlyph;
}

// One line of the HUD: what the labels/values are, and where they land. Kept as data so the
// window overlay, the PPM overlay and the ASCII self-check all draw the exact same thing.
// (HudFacts + gHudFacts + gHudDump are declared up top, next to the callers that fill them.)
static void hudPixel(unsigned char* rgba, uint32_t W, uint32_t H, int x, int y,
                     unsigned char r, unsigned char g, unsigned char b) {
    if (x < 0 || y < 0 || x >= (int)W || y >= (int)H) return;
    const size_t o = ((size_t)y * (size_t)W + (size_t)x) * 4u;
    rgba[o] = r; rgba[o + 1] = g; rgba[o + 2] = b; rgba[o + 3] = 255;
}

static int hudWidth(const char* text, int scale) {
    const int n = (int)std::strlen(text);
    return n > 0 ? n * (5 * scale + scale) - scale : 0;
}

static void hudText(unsigned char* rgba, uint32_t W, uint32_t H, int x, int y, int scale,
                    unsigned char r, unsigned char g, unsigned char b, const char* text) {
    const int advance = 5 * scale + scale;  // one blank cell between glyphs keeps words readable
    for (int ci = 0; text[ci] != '\0'; ++ci) {
        const Glyph& gl = glyphFor(text[ci]);
        const int bx = x + ci * advance;
        for (int row = 0; row < 7; ++row)
            for (int col = 0; col < 5; ++col) {
                if (gl.rows[row][col] != '1') continue;
                for (int sy = 0; sy < scale; ++sy)
                    for (int sx = 0; sx < scale; ++sx)
                        hudPixel(rgba, W, H, bx + col * scale + sx, y + row * scale + sy, r, g, b);
            }
    }
}

static void hudShade(unsigned char* rgba, uint32_t W, uint32_t H, int x0, int y0, int x1, int y1) {
    for (int y = y0; y <= y1; ++y)
        for (int x = x0; x <= x1; ++x) {
            if (x < 0 || y < 0 || x >= (int)W || y >= (int)H) continue;
            const size_t o = ((size_t)y * (size_t)W + (size_t)x) * 4u;
            rgba[o] = (unsigned char)(rgba[o] / 2);
            rgba[o + 1] = (unsigned char)(rgba[o + 1] / 2);
            rgba[o + 2] = (unsigned char)(rgba[o + 2] / 2);
            rgba[o + 3] = 255;
        }
}

// One label/value line, either left-aligned from a margin or right-aligned against it, with a
// darkened bar behind it: over a moving triangle a bare glyph would flicker into the image.
static void hudLine(unsigned char* rgba, uint32_t W, uint32_t H, int y, int margin, bool right,
                    int scale, const char* text, unsigned char r, unsigned char g, unsigned char b) {
    const int w = hudWidth(text, scale);
    const int h = 7 * scale;
    const int x = right ? (int)W - margin - w : margin;
    hudShade(rgba, W, H, x - 3, y - 3, x + w + 3, y + h + 3);
    hudText(rgba, W, H, x, y, scale, r, g, b, text);
    if (gHudEchoLines)
        std::printf("  painted %-32s side=%s scale=%d box=%dx%d at %d,%d\n", text,
                    right ? "right" : "left", scale, w, h, x, y);
}

// Paint the whole readout. Left = the three resolutions this host actually works with (window
// extent SDL reports, the app's own render extent, and the extent amd_fidelityfx_upscaler_vk
// rendered). Right = the two clocks: presented-frame delta + FPS (FPS painted larger, as asked),
// then the per-effect costs and the frame-generation multiplier the provider really delivered -
// which stays 1X + REFUSED while generate is refused, instead of echoing the requested 2X/4X.
static void drawHud(unsigned char* rgba, uint32_t W, uint32_t H, const HudFacts& f) {
    char line[96];
    const int margin = 8;
    // --- top-left: the resolutions, as allocated, never as assumed ---
    int y = margin;
    std::snprintf(line, sizeof line, "WIN: %ux%u", f.winW, f.winH);
    hudLine(rgba, W, H, y, margin, false, 2, line, 255, 255, 255);
    y += 7 * 2 + 6;
    std::snprintf(line, sizeof line, "RENDER: %ux%u", f.rndW, f.rndH);
    hudLine(rgba, W, H, y, margin, false, 2, line, 255, 255, 255);
    y += 7 * 2 + 6;
    if (f.upW > 0u && f.upH > 0u)
        std::snprintf(line, sizeof line, "UPSCALER VK: %ux%u", f.upW, f.upH);
    else
        std::snprintf(line, sizeof line, "UPSCALER VK: NO DISPATCH");
    hudLine(rgba, W, H, y, margin, false, 2, line, 120, 220, 255);
    // --- top-right: the clocks (FPS is the big one, its delta below it) ---
    // A frame clock only EXISTS where frames are actually presented: the headless proof path has
    // no present cadence, so it prints NO PRESENTED there rather than a number that would look
    // like a frame rate it never measured.
    y = margin;
    if (f.presented)
        std::snprintf(line, sizeof line, "%d FPS", f.fps < 0 ? 0 : f.fps);
    else
        std::snprintf(line, sizeof line, "FPS NO PRESENTED");
    hudLine(rgba, W, H, y, margin, true, 4, line, 120, 255, 120);
    y += 7 * 4 + 6;
    if (f.presented)
        std::snprintf(line, sizeof line, "%.1f MS FRAME", f.frameMs);
    else
        std::snprintf(line, sizeof line, "FRAME NO PRESENTED");
    hudLine(rgba, W, H, y, margin, true, 2, line, 255, 255, 255);
    y += 7 * 2 + 5;
    std::snprintf(line, sizeof line, "%.1f MS UPSCALER", f.upMs);
    hudLine(rgba, W, H, y, margin, true, 2, line, 120, 220, 255);
    y += 7 * 2 + 5;
    if (f.fgRefused)
        std::snprintf(line, sizeof line, "%.1f MS FGMEM REFUSED", f.fgMs);
    else
        std::snprintf(line, sizeof line, "%.1f MS FGMEM", f.fgMs);
    hudLine(rgba, W, H, y, margin, true, 2, line, 255, 200, 90);
    y += 7 * 2 + 5;
    std::snprintf(line, sizeof line, "GOT %d X REQ %d X", f.fgActual, f.fgRequested);
    hudLine(rgba, W, H, y, margin, true, 2, line, 255, 200, 90);
    // THE NUMBER BEING ASKED FOR: frames the provider PRODUCED vs the host frames it was ASKED to
    // interpolate, plus the provider's own last return code. Today that reads 0/N CODE 3 (RUNTIME_ERROR:
    // refusal, no fake frames). The moment the classic chain lands it becomes e.g. 552/552 CODE 0 with NO
    // change to this code path, which is the whole point of counting the provider's ANSWER instead of the
    // multiplier the host requested.
    y += 7 * 2 + 5;
    if (f.fgLive)
        std::snprintf(line, sizeof line, "FGEN %lld/%lld CODE %d", f.fgGenerated, f.fgAttempts, f.fgCode);
    else
        std::snprintf(line, sizeof line, "FGEN NO CONTEXT");
    hudLine(rgba, W, H, y, margin, true, 2, line, 150, 255, 150);
}

// Self-check, no display needed: re-paint the HUD region of a buffer and print it as ASCII, so
// the glyph table's legibility is verifiable in a terminal (and in CI) rather than by eye.
// `step` samples every Nth pixel: at a 2x glyph scale that collapses each glyph back to its
// 5x7 bitmap, which is what makes the letters readable in the dump.
static void hudDumpAscii(const unsigned char* rgba, uint32_t W, uint32_t H, int x0, int y0,
                         int w, int h, int step) {
    if (step < 1) step = 1;
    for (int y = y0; y < y0 + h && y < (int)H; y += step) {
        std::string row;
        for (int x = x0; x < x0 + w && x < (int)W; x += step) {
            const size_t o = ((size_t)y * (size_t)W + (size_t)x) * 4u;
            // Ink is bright over a darkened bar, so a luminance test separates the two cleanly.
            const int lum = (int)rgba[o] + (int)rgba[o + 1] + (int)rgba[o + 2];
            row += (lum > 300 ? '#' : (lum > 120 ? '+' : '.'));
        }
        std::printf("%s\n", row.c_str());
    }
}

// ffxDispatchDescFrameGenerationPrepareV2 (0x0002000c) is absent from the vendored header this tool compiles
// against. The repo's FG provider reads that host descriptor through ffgcheck::PrepareV2Desc - an ABI mirror of
// the pinned 2.4.0 layout whose size is machine-checked by tools/framegeneration-check-test - so the HOST side
// needs the same mirror: the Prepare(0x04) prefix (`reset` replacing the unused `unused_reset`) PLUS the camera
// basis EMBEDDED in the descriptor, which SUPERSEDES the linked CameraInfo(0x0a) block.
struct FfgPrepareV2Desc {
    ffxDispatchDescHeader header;
    uint64_t frameID;
    uint32_t flags;
    void* commandList;
    FfxApiDimensions2D renderSize;
    FfxApiFloatCoords2D jitterOffset;
    FfxApiFloatCoords2D motionVectorScale;
    float frameTimeDelta;
    bool reset;
    float cameraNear;
    float cameraFar;
    float cameraFovAngleVertical;
    float viewSpaceToMetersFactor;
    FfxApiResource depth;
    FfxApiResource motionVectors;
    float cameraPosition[3];
    float cameraUp[3];
    float cameraRight[3];
    float cameraForward[3];
};
static constexpr uint32_t kFgPrepareV2Type = 0x0002000cu;




static void driveUpscaler(Provider& p, VkApp& a) {
    std::printf("\n=== FSR4 UPSCALER provider (create + query helpers) ===\n");
    FfgEnabledFeaturesDesc en; FfgBackendDesc bk; FfgBackendInstanceDesc inst; buildContract(a, en, bk);
    // Attach the optional instance(4) block so the upscaler can confirm its cooperative-matrix
    // tiles / device entry-point groups (the runtime does this; without it create() -> RUNTIME_ERROR).
    inst.header = ffxApiHeader{4u, &en.header};
    inst.instance = a.instance; inst.getInstanceProcAddr = vkGetInstanceProcAddr;
    bk.header.pNext = &inst.header;
    ffxCreateContextDescUpscale cd{};
    cd.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
    cd.header.pNext = &bk.header;
    cd.flags = FFX_UPSCALE_ENABLE_DEBUG_CHECKING;
    cd.maxRenderSize = {kWidth, kHeight};
    cd.maxUpscaleSize = {kWidth * 3u, kHeight * 3u};
    ffxContext ctx{};
    int create = (int)p.create(&ctx, &cd.header, nullptr);
    std::printf("  ffxCreateContext(upscale) = %d\n", create);
    if (create == FFX_API_RETURN_OK) {
        ffxQueryDescUpscaleGetUpscaleRatioFromQualityMode q{};
        q.header.type = FFX_API_QUERY_DESC_TYPE_UPSCALE_GETUPSCALERATIOFROMQUALITYMODE;
        q.qualityMode = FFX_UPSCALE_QUALITY_MODE_BALANCED; float ratio = 0.0f; q.pOutUpscaleRatio = &ratio;
        p.query(&ctx, &q.header);
        std::printf("  ffxQuery(GET_UPSCALE_RATIO: BALANCED) -> ratio=%.2f (expect 1.70)\n", ratio);
        ffxQueryDescUpscaleGetRenderResolutionFromQualityMode rr{};
        rr.header.type = FFX_API_QUERY_DESC_TYPE_UPSCALE_GETRENDERRESOLUTIONFROMQUALITYMODE;
        rr.displayWidth = kWidth * 3u; rr.displayHeight = kHeight * 3u; rr.qualityMode = FFX_UPSCALE_QUALITY_MODE_BALANCED;
        uint32_t rw = 0, rh = 0; rr.pOutRenderWidth = &rw; rr.pOutRenderHeight = &rh;
        p.query(&ctx, &rr.header);
        std::printf("  ffxQuery(GET_RENDER_RESOLUTION) -> render %ux%u\n", rw, rh);
        std::printf("  (create OK; the deterministic query helpers answered here. The PIXEL upscale DISPATCH is\n");
        std::printf("   proven by the FSR4 UPSCALER IMAGE DISPATCH section above: real ffxDispatch(upscale) over\n");
        std::printf("   app-owned VkImages, submitted, with the FP16 result read back off the GPU.)\n");
    } else {
        std::printf("  create=%d: unexpected (device should meet the contract); the modular upscaler create\n", create);
        std::printf("   + cooperative-matrix-tile negotiation (instance-level extension enable + resolver) that\n");
        std::printf("   descriptor-smoke performs through the runtime. Honest, not faked; the upscaler's real on-\n");
        std::printf("   GPU run is proven by the green descriptor-smoke (render->submit->read-back, byte-identical).\n");
    }
    p.destroy(&ctx, nullptr);
}

// --- FRAME GENERATION provider: the documented host-integration ladder, printed codes. When the FSR rig is
// live this host also hands FG the REAL images it owes per frame (depth, motion vectors, present colour, back
// buffers) so a refusal can only be the provider's own, never a host-side parameter error.
static void driveFrameGeneration(Provider& p, VkApp& a, const FsrRig* rig) {
    std::printf("\n=== FSR4 FRAME GENERATION provider (host-integration ladder) ===\n");
    FfgEnabledFeaturesDesc en; FfgBackendDesc bk; buildContract(a, en, bk);
    struct VersionLink { ffxCreateContextDescHeader header; } vlink{ffxApiHeader{0x0002000eu, nullptr}};
    bk.header.pNext = &en.header; en.header.pNext = &vlink.header;
    ffxCreateContextDescFrameGeneration cd{};
    cd.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATION;
    cd.header.pNext = &bk.header;
    cd.flags = FFX_FRAMEGENERATION_ENABLE_DEBUG_CHECKING;
    // With the FSR rig live, this host's shape is: GAME render 480x270 -> AMD FSR upscale -> PRESENT 960x540,
    // so FG's display/backbuffer size is the upscaled size while Prepare's renderSize stays the game size.
    const uint32_t dspW = rig ? rig->ow : kWidth, dspH = rig ? rig->oh : kHeight;
    cd.displaySize = {dspW, dspH};
    cd.maxRenderSize = {dspW, dspH};
    cd.backBufferFormat = 1u;  // R8G8B8A8_UNORM-style surface format id (non-UNKNOWN)
    ffxContext ctx{};
    int create = (int)p.create(&ctx, &cd.header, nullptr);
    std::printf("  ffxCreateContext(frame-generation) = %d (0=OK if device meets contract)\n", create);
    if (create == FFX_API_RETURN_OK) {
        ffxQueryDescGetVersions ver{};
        ver.header.type = FFX_API_QUERY_DESC_TYPE_GET_VERSIONS;
        ver.createDescType = FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATION;
        uint64_t ids[2] = {0, 0}; uint64_t cnt = 2; const char* names[2] = {nullptr, nullptr};
        ver.outputCount = &cnt; ver.versionIds = ids; ver.versionNames = names;
        p.query(&ctx, &ver.header);
        std::printf("  ffxQuery(GET_VERSIONS) -> ids[0]=0x%016llX count=%llu\n", (unsigned long long)ids[0],
                    (unsigned long long)cnt);
        ffxQueryDescFrameGenerationGetGPUMemoryUsage mem{};
        mem.header.type = FFX_API_QUERY_DESC_TYPE_FRAMEGENERATION_GPU_MEMORY_USAGE;
        std::printf("  ffxQuery(GET_GPUMEMORYUSAGE) = %d (expect 3=RUNTIME_ERROR: not faked)\n",
                    (int)p.query(&ctx, &mem.header));
        ffxConfigureDescFrameGeneration cf{};
        cf.header.type = FFX_API_CONFIGURE_DESC_TYPE_FRAMEGENERATION;
        cf.frameGenerationEnabled = true;
        cf.swapChain = (void*)&ctx; cf.presentCallback = cbPresent; cf.frameGenerationCallback = cbFg;
        std::printf("  ffxConfigure(host contract) = %d (expect 0=OK)\n", (int)p.configure(&ctx, &cf.header));
        // A live command buffer is part of the host contract on BOTH FG descriptors (the provider would RECORD
        // its dilate / optical-flow compute here). Ended + discarded below, since while it refuses the algorithm
        // nothing is recorded into it.
        VkCommandBuffer cc{VK_NULL_HANDLE};
        {
            VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, nullptr, a.cmdPool,
                VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
            vkAllocateCommandBuffers(a.device, &cai, &cc);
            VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr, 0, nullptr};
            vkBeginCommandBuffer(cc, &bi);
        }
        const auto expectFg = [](const char* what, int got, int want, const char* why) {
            std::printf("  %s = %d (expect %d: %s)%s\n", what, got, want, why,
                        (got == want) ? "" : "   [MISMATCH -> HOST contract wrong, not a provider refusal]");
        };
        // (a) ORDER GUARD: generate BEFORE Prepare must come back PARAMETER (1), proving the provider enforces
        // Configure -> Prepare -> generate instead of accepting any order.
        {
            ffxDispatchDescFrameGeneration early{};
            early.header.type = FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION;
            early.commandList = (void*)cc;
            early.numGeneratedFrames = 2u;
            early.minMaxLuminance[0] = 0.0f; early.minMaxLuminance[1] = 1.0f;
            expectFg("ffxDispatch(Generate) BEFORE Prepare", (int)p.dispatch(&ctx, &early.header),
                     (int)FFX_API_RETURN_ERROR_PARAMETER, "order enforced");
        }
        // (b) PREPARE WITH THE IMAGES A HOST OWES PER FRAME - this app's REAL rendered D32 depth (claimed with the
        // depth role's R32_FLOAT format + DEPTHTARGET usage, the bits a provider binds as VK_FORMAT_D32_SFLOAT +
        // DEPTH aspect) and the REAL RG16F motion vectors. Before this change NO image was handed over at all.
        ffxDispatchDescFrameGenerationPrepare pr{};
        pr.header.type = FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION_PREPARE;
        pr.frameID = 1001u;
        pr.renderSize = rig ? FfxApiDimensions2D{rig->rw, rig->rh} : FfxApiDimensions2D{kWidth, kHeight};
        pr.commandList = (void*)cc;
        pr.jitterOffset = {0.5f, -0.5f}; pr.motionVectorScale = {1.0f, 1.0f};
        pr.frameTimeDelta = 16.6f; pr.cameraNear = 0.1f; pr.cameraFar = 1000.0f;
        pr.cameraFovAngleVertical = 1.0f; pr.viewSpaceToMetersFactor = 0.0f;
        if (rig) {
            pr.depth = mkRes(rig->depth, rig->rw, rig->rh, FFX_API_SURFACE_FORMAT_R32_FLOAT,
                             FFX_API_RESOURCE_USAGE_DEPTHTARGET, FFX_API_RESOURCE_STATE_COMPUTE_READ);
            pr.motionVectors = mkRes(rig->motion, rig->rw, rig->rh, FFX_API_SURFACE_FORMAT_R16G16_FLOAT,
                                     FFX_API_RESOURCE_USAGE_READ_ONLY, FFX_API_RESOURCE_STATE_COMPUTE_READ);
        }
        ffxDispatchDescFrameGenerationPrepareCameraInfo cam{};
        cam.header.type = 0x0002000au;  // V1 camera-info block (probe path); orthonormal basis
        cam.cameraPosition[0] = 0.0f; cam.cameraPosition[1] = 0.0f; cam.cameraPosition[2] = 0.0f;
        cam.cameraUp[0] = 0.0f; cam.cameraUp[1] = 1.0f; cam.cameraUp[2] = 0.0f;
        cam.cameraRight[0] = 1.0f; cam.cameraRight[1] = 0.0f; cam.cameraRight[2] = 0.0f;
        cam.cameraForward[0] = 0.0f; cam.cameraForward[1] = 0.0f; cam.cameraForward[2] = -1.0f;
        pr.header.pNext = &cam.header;
        expectFg("ffxDispatch(Prepare + REAL depth/motion + basis)", (int)p.dispatch(&ctx, &pr.header),
                 (int)FFX_API_RETURN_ERROR_RUNTIME_ERROR, "accepted then refused: no ML/interp runtime in slice");
        // (c) PREPAREV2 (0x0002000c, the pinned 2.4.0 shape): the SAME real images, with the camera basis EMBEDDED
        // in the descriptor instead of linked as CameraInfo(0x0a) - which is what the provider's PrepareV2 branch
        // validates (order + arguments + embedded orthonormal basis) before its honest algorithm refusal.
        {
            FfgPrepareV2Desc v2{};
            v2.header.type = kFgPrepareV2Type;
            v2.frameID = 1001u; v2.flags = 0u; v2.commandList = (void*)cc;
            v2.renderSize = pr.renderSize;
            v2.jitterOffset = {0.5f, -0.5f}; v2.motionVectorScale = {1.0f, 1.0f};
            v2.frameTimeDelta = 16.6f; v2.reset = false;
            v2.cameraNear = 0.1f; v2.cameraFar = 1000.0f; v2.cameraFovAngleVertical = 1.0f;
            v2.viewSpaceToMetersFactor = 0.0f;
            if (rig) { v2.depth = pr.depth; v2.motionVectors = pr.motionVectors; }
            v2.cameraPosition[0] = v2.cameraPosition[1] = v2.cameraPosition[2] = 0.0f;
            v2.cameraUp[0] = 0.0f; v2.cameraUp[1] = 1.0f; v2.cameraUp[2] = 0.0f;
            v2.cameraRight[0] = 1.0f; v2.cameraRight[1] = 0.0f; v2.cameraRight[2] = 0.0f;
            v2.cameraForward[0] = 0.0f; v2.cameraForward[1] = 0.0f; v2.cameraForward[2] = -1.0f;
            expectFg("ffxDispatch(PrepareV2, embedded basis)", (int)p.dispatch(&ctx, &v2.header),
                     (int)FFX_API_RETURN_ERROR_RUNTIME_ERROR, "accepted then refused (runtime absent)");
        }
        // (d) GENERATE WITH THE IMAGES: presentColor = the FSR-upscaled FP16 colour this host would show, and
        // outputs[0..numGeneratedFrames-1] = REAL display-size backbuffer images allocated by the host (the pinned
        // effect produces at most 3 interpolated frames per base frame, hence 3 backbuffers here).
        VkImage bb[3]{}; VkImageView bbV[3]{}; VkDeviceMemory bbM[3]{};
        if (rig)
            for (uint32_t i = 0; i < 3u; ++i)
                mkImg(a, rig->ow, rig->oh, VK_FORMAT_R8G8B8A8_UNORM,
                      VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_IMAGE_ASPECT_COLOR_BIT,
                      bb[i], bbV[i], bbM[i]);
        ffxDispatchDescFrameGeneration g{};
        g.header.type = FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION;
        g.commandList = (void*)cc;
        g.numGeneratedFrames = 2u;
        g.reset = false; g.backbufferTransferFunction = 0u;
        g.minMaxLuminance[0] = 0.0f; g.minMaxLuminance[1] = 1.0f;
        g.generationRect = {0, 0, 0, 0};  // whole-frame sentinel
        g.frameID = 1001u;
        if (rig) {
            g.presentColor = mkRes(rig->out, rig->ow, rig->oh, FFX_API_SURFACE_FORMAT_R16G16B16A16_FLOAT,
                                   FFX_API_RESOURCE_USAGE_READ_ONLY, FFX_API_RESOURCE_STATE_COMPUTE_READ);
            for (uint32_t i = 0; i < 3u; ++i)
                g.outputs[i] = mkRes(bb[i], rig->ow, rig->oh, FFX_API_SURFACE_FORMAT_R8G8B8A8_UNORM,
                                     FFX_API_RESOURCE_USAGE_UAV, FFX_API_RESOURCE_STATE_UNORDERED_ACCESS);
        }
        // Time the frame-generation call itself so the HUD's FGMEM line is a measurement made in
        // this process, and record what the provider ACTUALLY delivered: 1X with an honest refusal,
        // never the 2X the host asked for -- that is the point of showing the multiplier at all.
        const auto tFg0 = std::chrono::steady_clock::now();
        const int fgGenerateCode = (int)p.dispatch(&ctx, &g.header);
        gHudFacts.fgMs =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tFg0).count();
        gHudFacts.fgRefused = (fgGenerateCode == (int)FFX_API_RETURN_ERROR_RUNTIME_ERROR);
        gHudFacts.fgActual = gHudFacts.fgRefused ? 1 : (1 + (int)g.numGeneratedFrames);
        std::printf("  frame-generation call cost: %.3fms%s (this is what the HUD FGMEM line prints)\n",
                    gHudFacts.fgMs, gHudFacts.fgRefused ? " REFUSED" : "");
        expectFg("ffxDispatch(Generate 2 + presentColor + outputs)", fgGenerateCode,
                 (int)FFX_API_RETURN_ERROR_RUNTIME_ERROR, "NO fake frames; runtime absent");
        // (e) The documented CANCEL branch: numGeneratedFrames == 0 means "no interpolation this frame", so there
        // is nothing to fabricate and the honest answer the provider gives is OK (0).
        {
            ffxDispatchDescFrameGeneration cancel = g;
            cancel.numGeneratedFrames = 0u;
            expectFg("ffxDispatch(Generate cancel, 0 frames)", (int)p.dispatch(&ctx, &cancel.header),
                     (int)FFX_API_RETURN_OK, "documented cancel path");
        }
        if (rig)
            for (uint32_t i = 0; i < 3u; ++i) {
                if (bbV[i]) vkDestroyImageView(a.device, bbV[i], nullptr);
                if (bb[i]) vkDestroyImage(a.device, bb[i], nullptr);
                if (bbM[i]) vkFreeMemory(a.device, bbM[i], nullptr);
            }
        vkEndCommandBuffer(cc);
        vkFreeCommandBuffers(a.device, a.cmdPool, 1, &cc);
        p.destroy(&ctx, nullptr);
    } else {
        std::printf("  (device did not meet the FG execution contract; create refused=%d)\n", create);
    }
}

// ============================================================================
// OPTIONAL VISIBLE WINDOW (--window): presents the SAME animated-triangle host
// to a real Wayland/X11 window through core Vulkan WSI (GLFW window surface ->
// VkSurfaceKHR -> swapchain -> present). HONEST about Frame Generation: AMD's
// provider refuses ffxDispatch(generate) here (RUNTIME_ERROR, "refusing to fake
// frame generation") and the FSR pixel-dispatch is not hand-rolled, so NO
// interpolated/generated frames are fabricated. The log reports the real base
// present plus interpolated=0 / generated=0 with the refusal reason.
static uint32_t findDeviceMemoryType(VkApp& a, VkMemoryRequirements req, VkMemoryPropertyFlags want) {
    VkPhysicalDeviceMemoryProperties mp{};
    vkGetPhysicalDeviceMemoryProperties(a.physical, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
        if ((req.memoryTypeBits & (1u << i)) &&
            (mp.memoryTypes[i].propertyFlags & want) == want) return i;
    return 0xFFFFFFFFu;
}
// Live upscaler context for the window path (create only; AUTO_EXPOSURE so no 1x1 exposure
// texture is required; sized for render 1280x720 -> upscale 1440x1080).
static ffxReturnCode_t createUpscale(Provider& p, VkApp& a, ffxContext& ctx) {
    FfgEnabledFeaturesDesc en; FfgBackendDesc bk; FfgBackendInstanceDesc inst; buildContract(a, en, bk);
    inst.header = ffxApiHeader{4u, &en.header}; inst.instance = a.instance; inst.getInstanceProcAddr = vkGetInstanceProcAddr;
    bk.header.pNext = &inst.header;
    ffxCreateContextDescUpscale cd{}; cd.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE; cd.header.pNext = &bk.header;
    cd.flags = FFX_UPSCALE_ENABLE_AUTO_EXPOSURE; cd.maxRenderSize = {kRenderW, kRenderH}; cd.maxUpscaleSize = {kDispW, kDispH};
    return p.create(&ctx, &cd.header, nullptr);
}
// Live Frame Generation context (create; the provider's generate stays refused here).
static ffxReturnCode_t createFg(Provider& p, VkApp& a, ffxContext& ctx) {
    FfgEnabledFeaturesDesc en; FfgBackendDesc bk; buildContract(a, en, bk);
    struct VersionLink { ffxCreateContextDescHeader header; } vlink{ffxApiHeader{0x0002000eu, nullptr}};
    bk.header.pNext = &en.header; en.header.pNext = &vlink.header;
    ffxCreateContextDescFrameGeneration cd{};
    cd.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATION; cd.header.pNext = &bk.header;
    cd.flags = FFX_FRAMEGENERATION_ENABLE_DEBUG_CHECKING;
    cd.displaySize = {kDispW, kDispH}; cd.maxRenderSize = {kDispW, kDispH}; cd.backBufferFormat = 1u;
    return p.create(&ctx, &cd.header, nullptr);
}
static void runPresent(VkApp& a, Provider& up, ffxContext& upCtx, bool upLive, Provider& fg, ffxContext& fgCtx,
                       bool fgLive, const std::string& vertSpv, const std::string& fragSpv, VkShaderModule vsP, VkShaderModule fsP,
                       int seconds, int fgMult, SDL_Window* win);


// Keep an SDL window on screen WITHOUT touching the WSI swapchain path. On this host the (software-RADV)
// driver returns a NULL vkCreateSwapchainKHR, which the loader masks with VK_SUCCESS -> the app would then
// SIGSEGV. Holding the window without presenting is the safe, crash-free way to show the tool on such hosts.
static void holdWindow(int seconds, SDL_Window* win) {
    if (!win) return;
    SDL_ShowWindow(win);
    auto t0 = std::chrono::steady_clock::now();
    while (true) {
        SDL_Event e;
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_QUIT) return;
            if (e.type == SDL_WINDOWEVENT && e.window.event == SDL_WINDOWEVENT_CLOSE) return;
        }
        double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (el >= (double)seconds) return;
        SDL_Delay(16);
    }
}

// VISIBLE WINDOW on hosts whose Vulkan driver cannot present into a window (on this Wayland / Mesa
// software-RADV host vkCreateSwapchainKHR resolves to a NULL driver function, so the --present path SIGSEGVs).
// Instead: render the LIVE animated triangle OFFSCREEN through the app's OWN headless VkInstance (renderFrame
// submits to the real GPU and reads the colour image back into a CPU RGBA8 buffer whose bytes already match
// SDL_PIXELFORMAT_RGBA8888), then blit that buffer to the SDL window with SDL's SOFTWARE renderer -- no GL
// context, no WSI swapchain, therefore crash-free. The HUD readout (resolutions top-left, clocks top-right)
// is painted INTO that buffer every frame by the 5x7 bitmap font above, and the same numbers are also printed
// once per second.
// HONEST LIMIT: the 480x270 -> window-size scaling here is SDL's software scaler, NOT AMD's FSR; the REAL AMD
// upscaler provider is still created/driven in this same process (runHeadless above, -> PPM), and the true GPU
// FSR blit to the window is the --present path, which only runs where the driver exposes a working swapchain.
// Paint the readout into the RGBA host buffer BEFORE the SDL upload, so it survives the software blit.

// --- Frame Generation driven EVERY frame of the window, so "how many frames did amd_fidelityfx_framegeneration_vk
// actually produce?" is a NUMBER ON SCREEN instead of a sentence in prose.
//
// The contract shape is the SDK's own: create (host backend + enabled-feature blocks), Configure
// (frameGenerationEnabled + present/generate callbacks), then Prepare and generate PAIRED through frameID -
// ffx_provider_fsr3framegeneration.cpp pins MAX_QUEUED_FRAMES = 2, so the frameID that Prepared is the one
// generate must name. Prepare is therefore re-issued per frame with a fresh frameID rather than once, and the
// images handed over are the app's REAL ones: the FSR rig's D32 depth and RG16F motion vectors, its FP16
// present colour, and three real R8G8B8A8 backbuffers at the upscaled extent (3 == MAX_QUEUED_FRAMES + 1, the
// SDK's own arithmetic).
//
// Counting rule: generate returning OK with numGeneratedFrames = N means the provider PRODUCED N frames.
// RUNTIME_ERROR (this provider slice's honest state) produces ZERO and the HUD shows 0 - never the multiplier
// the host asked for.
struct FgWindowRig {
    Provider* p{nullptr};
    ffxContext ctx{};
    bool live{false};
    int createCode{0xFFFF}, configureCode{0xFFFF}, prepareCode{0xFFFF}, generateCode{0xFFFF};
    VkImage bb[3]{};
    VkImageView bbV[3]{};
    VkDeviceMemory bbM[3]{};
    uint64_t frameID{1001ull};
    long long attempts{0};    // generate calls made
    long long generated{0};   // frames the provider actually delivered, summed
    long long okCalls{0};     // generate calls that returned OK
    double msTotal{0.0};      // wall cost of those calls (what the HUD's FGMEM line prints)
    bool cmdOk{false};        // a command buffer could be allocated + begun (the provider records INTO one)
};

static bool initFgWindowRig(VkApp& a, Provider& p, FsrRig* rig, FgWindowRig& r, int mult) {
    (void)mult;
    r.p = &p;
    FfgEnabledFeaturesDesc en;
    FfgBackendDesc bk;
    buildContract(a, en, bk);
    struct VersionLink {
        ffxCreateContextDescHeader header;
    } vlink{ffxApiHeader{0x0002000eu, nullptr}};
    bk.header.pNext = &en.header;
    en.header.pNext = &vlink.header;
    ffxCreateContextDescFrameGeneration cd{};
    cd.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATION;
    cd.header.pNext = &bk.header;
    cd.flags = FFX_FRAMEGENERATION_ENABLE_DEBUG_CHECKING;
    // Host shape here: GAME render (rig->rw x rh) -> AMD FSR upscale -> PRESENT (rig->ow x oh). FG is told the
    // UPSCALED extent as its display/backbuffer size while Prepare keeps naming the GAME extent.
    const uint32_t dspW = rig ? rig->ow : kWidth;
    const uint32_t dspH = rig ? rig->oh : kHeight;
    cd.displaySize = {dspW, dspH};
    cd.maxRenderSize = {dspW, dspH};
    cd.backBufferFormat = 1u;  // the non-UNKNOWN surface-format id the contract asks for
    r.createCode = (int)p.create(&r.ctx, &cd.header, nullptr);
    std::printf("  window FG ffxCreateContext = %d (0=OK)\n", r.createCode);
    if (r.createCode != FFX_API_RETURN_OK) return false;
    ffxConfigureDescFrameGeneration cf{};
    cf.header.type = FFX_API_CONFIGURE_DESC_TYPE_FRAMEGENERATION;
    cf.frameGenerationEnabled = true;
    cf.swapChain = (void*)&r.ctx;
    cf.presentCallback = cbPresent;
    cf.frameGenerationCallback = cbFg;
    r.configureCode = (int)p.configure(&r.ctx, &cf.header);
    std::printf("  window FG ffxConfigure = %d (0=OK)\n", r.configureCode);
    if (r.configureCode != FFX_API_RETURN_OK) {
        p.destroy(&r.ctx, nullptr);
        r.ctx = {};
        return false;
    }
    if (rig) {
        for (uint32_t i = 0; i < 3u; ++i) {
            mkImg(a, rig->ow, rig->oh, VK_FORMAT_R8G8B8A8_UNORM,
                  VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_IMAGE_ASPECT_COLOR_BIT,
                  r.bb[i], r.bbV[i], r.bbM[i]);
            if (!r.bb[i]) {
                std::printf("  window FG: backbuffer %u unavailable -> generate is called WITHOUT that output\n", i);
                break;
            }
        }
    }
    r.live = true;
    return true;
}

// ONE host frame of Frame Generation: Prepare(frameID) then generate(frameID), the way the SDK pairs them.
// The provider RECORDS compute into the HOST command list and never submits, so the host ends + submits +
// waits here even when a refused call recorded nothing - that is the shape the real runtime will need, and
// running it now is what keeps the difference between "refused" and "unimplemented plumbing" honest.
static void fgFrameAttempt(FgWindowRig& r, VkApp& a, FsrRig* rig, int mult) {
    if (!r.live || !r.p) return;
    VkCommandBuffer cb{VK_NULL_HANDLE};
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, nullptr, a.cmdPool,
                                   VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1u};
    if (vkAllocateCommandBuffers(a.device, &ai, &cb) != VK_SUCCESS || !cb) {
        r.cmdOk = false;
        return;
    }
    r.cmdOk = true;
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr, 0u, nullptr};
    if (vkBeginCommandBuffer(cb, &bi) != VK_SUCCESS) {
        vkFreeCommandBuffers(a.device, a.cmdPool, 1u, &cb);
        return;
    }
    ffxDispatchDescFrameGenerationPrepare pr{};
    pr.header.type = FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION_PREPARE;
    pr.frameID = r.frameID;
    pr.renderSize = rig ? FfxApiDimensions2D{rig->rw, rig->rh} : FfxApiDimensions2D{kWidth, kHeight};
    pr.commandList = (void*)cb;
    pr.jitterOffset = {0.5f, -0.5f};
    pr.motionVectorScale = {1.0f, 1.0f};
    pr.frameTimeDelta = 16.6f;
    pr.cameraNear = 0.1f;
    pr.cameraFar = 1000.0f;
    pr.cameraFovAngleVertical = 1.0f;
    pr.viewSpaceToMetersFactor = 0.0f;
    if (rig) {
        pr.depth = mkRes(rig->depth, rig->rw, rig->rh, FFX_API_SURFACE_FORMAT_R32_FLOAT,
                         FFX_API_RESOURCE_USAGE_DEPTHTARGET, FFX_API_RESOURCE_STATE_COMPUTE_READ);
        pr.motionVectors = mkRes(rig->motion, rig->rw, rig->rh, FFX_API_SURFACE_FORMAT_R16G16_FLOAT,
                                 FFX_API_RESOURCE_USAGE_READ_ONLY, FFX_API_RESOURCE_STATE_COMPUTE_READ);
    }
    ffxDispatchDescFrameGenerationPrepareCameraInfo cam{};
    cam.header.type = 0x0002000au;  // V1 camera-info block (linked, not embedded)
    cam.cameraPosition[0] = 0.0f; cam.cameraPosition[1] = 0.0f; cam.cameraPosition[2] = 0.0f;
    cam.cameraUp[0] = 0.0f; cam.cameraUp[1] = 1.0f; cam.cameraUp[2] = 0.0f;
    cam.cameraRight[0] = 1.0f; cam.cameraRight[1] = 0.0f; cam.cameraRight[2] = 0.0f;
    cam.cameraForward[0] = 0.0f; cam.cameraForward[1] = 0.0f; cam.cameraForward[2] = -1.0f;
    pr.header.pNext = &cam.header;
    r.prepareCode = (int)r.p->dispatch(&r.ctx, &pr.header);
    ffxDispatchDescFrameGeneration g{};
    g.header.type = FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION;
    g.commandList = (void*)cb;
    // numGeneratedFrames is what the host ASKS for; the provider may answer with fewer or refuse entirely,
    // and only its ANSWER is summed into r.generated below.
    g.numGeneratedFrames = (mult > 1) ? (uint32_t)(mult - 1) : 0u;
    g.reset = false;
    g.backbufferTransferFunction = 0u;
    g.minMaxLuminance[0] = 0.0f;
    g.minMaxLuminance[1] = 1.0f;
    g.generationRect = {0, 0, 0, 0};  // whole-frame sentinel: the provider's own fallback rule
    g.frameID = r.frameID;
    if (rig) {
        g.presentColor = mkRes(rig->out, rig->ow, rig->oh, FFX_API_SURFACE_FORMAT_R16G16B16A16_FLOAT,
                               FFX_API_RESOURCE_USAGE_READ_ONLY, FFX_API_RESOURCE_STATE_COMPUTE_READ);
        for (uint32_t i = 0; i < 3u; ++i) {
            if (r.bb[i]) {
                g.outputs[i] = mkRes(r.bb[i], rig->ow, rig->oh, FFX_API_SURFACE_FORMAT_R8G8B8A8_UNORM,
                                     FFX_API_RESOURCE_USAGE_UAV, FFX_API_RESOURCE_STATE_UNORDERED_ACCESS);
            }
        }
    }
    const auto t0 = std::chrono::steady_clock::now();
    r.generateCode = (int)r.p->dispatch(&r.ctx, &g.header);
    r.msTotal += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    ++r.attempts;
    if (r.generateCode == FFX_API_RETURN_OK) {
        ++r.okCalls;
        r.generated += (long long)g.numGeneratedFrames;  // the ANSWER, re-read after the call, is what counts
    }
    if (vkEndCommandBuffer(cb) == VK_SUCCESS && a.queue) {
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0u, nullptr, nullptr, 1u, &cb, 0u, nullptr};
        if (vkQueueSubmit(a.queue, 1u, &si, VK_NULL_HANDLE) == VK_SUCCESS) vkQueueWaitIdle(a.queue);
    }
    vkFreeCommandBuffers(a.device, a.cmdPool, 1u, &cb);
    ++r.frameID;
}

static void teardownFgWindowRig(VkApp& a, FgWindowRig& r) {
    if (r.live && r.p) r.p->destroy(&r.ctx, nullptr);
    for (uint32_t i = 0; i < 3u; ++i) {
        if (r.bbV[i]) vkDestroyImageView(a.device, r.bbV[i], nullptr);
        if (r.bb[i]) vkDestroyImage(a.device, r.bb[i], nullptr);
        if (r.bbM[i]) vkFreeMemory(a.device, r.bbM[i], nullptr);
    }
}

// Blit loop. Returns false (caller then falls back to holdWindow) if SDL cannot hand us a software renderer
// or texture, so a host without a working SDL video path still behaves as before.
static bool showWindowSdl(VkApp& a, Provider* up, FsrHost* fsr, Provider* fg, int fgMult, SDL_Window* win,
                          int seconds) {
    if (!win) return false;
    // When the REAL AMD FSR4 upscale host is live, the window shows the UPSCALED GPU result (960x540 read back
    // from the provider's FP16 output image); otherwise it shows the native 480x270 offscreen render.
    const bool useFsr = (fsr != nullptr) && fsr->live && (up != nullptr);
    // The FG rig the WINDOW drives every frame, so the frame counts are measured HERE rather than taken from
    // the headless ladder. Its images are the FSR rig's own, which is why Prepare can only be refused by the
    // provider's own rule and never by a host-side parameter error.
    FsrRig* rigForFg = (fsr && fsr->live) ? &fsr->rig : nullptr;
    FgWindowRig fgRig;
    if (fg != nullptr) {
        if (!initFgWindowRig(a, *fg, rigForFg, fgRig, fgMult)) {
            std::printf("  window FG rig NOT live: create=%d configure=%d (HUD will print FGEN NO CONTEXT)\n",
                        fgRig.createCode, fgRig.configureCode);
        }
    } else {
        std::printf("  window FG rig NOT live: no frame-generation provider loaded (FGEN NO CONTEXT)\n");
    }
    const uint32_t texW = useFsr ? kFsrOw : kWidth, texH = useFsr ? kFsrOh : kHeight;
    SDL_ShowWindow(win);
    SDL_Surface* target = SDL_GetWindowSurface(win);
    SDL_Renderer* rend = SDL_CreateSoftwareRenderer(target);
    if (!rend) {
        std::printf("NO_RENDERER: SDL_CreateSoftwareRenderer failed (%s)\n", SDL_GetError());
        return false;
    }
    // SDL2 signature (SDL_renderer.h): SDL_CreateTexture takes the renderer. RGBA8888 == our buffer byte order,
    // so the read-back buffer uploads without a channel swap.
    SDL_Texture* tex = SDL_CreateTexture(rend, SDL_PIXELFORMAT_RGBA8888, SDL_TEXTUREACCESS_STATIC, (int)texW, (int)texH);
    if (!tex) {
        std::printf("NO_TEXTURE: SDL_CreateTexture failed (%s)\n", SDL_GetError());
        SDL_DestroyRenderer(rend);
        return false;
    }
    std::vector<unsigned char> host((size_t)texW * texH * 4u);
    const SDL_Rect full{0, 0, (int)texW, (int)texH};  // SDL2 wants an explicit rect (SDL3 allows NULL)
    const auto t0 = std::chrono::steady_clock::now();
    auto tPrevFrame = t0;  // the previous PRESENTED frame, for the measured frame-to-frame delta
    uint32_t frame = 0, lastFpsFrame = 0;
    double lastFpsT = 0.0;
    int fps = 0, lastSec = -1, printedSec = -1;
    while (true) {
        const double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (el >= (double)seconds) break;
        SDL_Event e;
        bool done = false;
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_QUIT) done = true;
            if (e.type == SDL_WINDOWEVENT && e.window.event == SDL_WINDOWEVENT_CLOSE) done = true;
            if (e.type == SDL_KEYDOWN &&
                (e.key.keysym.sym == SDLK_ESCAPE || e.key.keysym.sym == SDLK_q)) done = true;
        }
        if (done) break;
        // FPS over the current whole second: reset the counter window when the second ticks.
        if ((int)el != lastSec) { lastSec = (int)el; lastFpsFrame = frame; lastFpsT = el; }
        if (el - lastFpsT > 0.001)
            fps = (int)((double)(frame - lastFpsFrame) / (el - lastFpsT));
        // Frame-to-frame wall time, measured off the PRESENTED frames (not the 16.6ms SDL_Delay
        // implies): this is what the HUD's "FRAME" line and the FPS readout come from.
        {
            const auto now = std::chrono::steady_clock::now();
            if (frame > 0u)
                gHudFacts.frameMs =
                    std::chrono::duration<double, std::milli>(now - tPrevFrame).count();
            tPrevFrame = now;
        }
        double fsrMs = 0.0, fsrMsTotal = 0.0, fsrMsN = 0.0;
        if (useFsr) {
            const auto tA = std::chrono::steady_clock::now();
            const ffxReturnCode_t d = fsrUpscaleFrame(a, *up, fsr->ctx, fsr->rig, frame, el * 2.0, host, nullptr);
            fsrMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tA).count();
            fsrMsTotal += fsrMs; fsrMsN += 1.0;
            // The extents the HUD prints are the ones THIS rig allocated, read back off the rig.
            gHudFacts.upMs = fsrMs;
            gHudFacts.rndW = fsr->rig.rw;
            gHudFacts.rndH = fsr->rig.rh;
            gHudFacts.upW = fsr->rig.ow;
            gHudFacts.upH = fsr->rig.oh;
            if (frame == 0u) std::printf("  window frame 0: ffxDispatch(upscale) = %d (0=OK) %.1fms\n", (int)d, fsrMs);
            if (d != FFX_API_RETURN_OK) std::printf("  FSR dispatch rc=%d on frame %u -> window keeps blipping what is there\n", (int)d, frame);
        } else {
            renderFrame(a, frame, el * 2.0, host);  // same phase clock the --present path uses (el * 2)
            gHudFacts.rndW = kWidth;
            gHudFacts.rndH = kHeight;
            gHudFacts.upW = 0u;
            gHudFacts.upH = 0u;
            gHudFacts.upMs = 0.0;
        }
        // ONE host frame of Frame Generation, counted: Prepare(frameID) + generate(frameID) with the rig's
        // real depth/motion/present-colour and three real backbuffers. r.generated sums the provider's ANSWER
        // (numGeneratedFrames as written back), so this line can only rise when the provider really produces
        // frames - a refusal keeps it at 0 no matter what --fg= asked for.
        if (fgRig.live) {
            fgFrameAttempt(fgRig, a, rigForFg, fgMult);
            gHudFacts.fgLive = true;
            gHudFacts.fgAttempts = fgRig.attempts;
            gHudFacts.fgGenerated = fgRig.generated;
            gHudFacts.fgCode = fgRig.generateCode;
            gHudFacts.fgMs = fgRig.attempts ? fgRig.msTotal / (double)fgRig.attempts : 0.0;
            gHudFacts.fgRefused = (fgRig.generateCode == (int)FFX_API_RETURN_ERROR_RUNTIME_ERROR);
            // Delivered multiplier from the DELIVERED frames: 1 + (frames produced / host frames attempted).
            gHudFacts.fgActual =
                fgRig.attempts ? (int)(1 + (fgRig.generated / (double)fgRig.attempts)) : 1;
        }
        // The window is RESIZABLE, so its extent is re-queried every frame and printed as-is; the
        // HUD is painted into the SAME buffer SDL uploads, so it scales/blits with the image it labels.
        int ww = 0, hh = 0;
        SDL_GetWindowSize(win, &ww, &hh);
        if (ww < 1) ww = 1;
        if (hh < 1) hh = 1;
        gHudFacts.winW = (uint32_t)ww;
        gHudFacts.winH = (uint32_t)hh;
        gHudFacts.fps = fps;
        gHudFacts.presented = true;
        drawHud(host.data(), texW, texH, gHudFacts);
        if (SDL_UpdateTexture(tex, &full, host.data(), (int)texW * 4) != 0)
            std::printf("BLIT_WARN: SDL_UpdateTexture failed (%s)\n", SDL_GetError());
        const SDL_FRect dst{0.0f, 0.0f, (float)ww, (float)hh};
        SDL_RenderClear(rend);
        SDL_RenderCopyF(rend, tex, &full, &dst);  // SDL2's name for the SDL3 SDL_RenderTexture
        SDL_RenderFlush(rend);
        SDL_UpdateWindowSurface(win);             // window surface -> compositor (no WSI involved)
        if (printedSec != (int)el) {
            printedSec = (int)el;
            std::printf("  t=%ds frames=%u fps=%d [live triangle: FSR GPU -> CPU readback -> SDL software blit]\n",
                        printedSec, frame, fps);
            std::printf("    FSR dispatch cost: %.1fms avg over %.0f dispatches\n",
                        fsrMsN ? fsrMsTotal / fsrMsN : 0.0, fsrMsN);
            // The live Frame Generation counters, printed beside the FSR cost so the two are never confused:
            // FSR upscaled EVERY frame (dispatches == frames), while FG's frame count is whatever the provider
            // ANSWERED - 0 while it refuses, and it is printed as 0 rather than as the requested multiplier.
            if (fgRig.live) {
                std::printf("    FG: frames_generated=%lld host_frames_asked=%lld last_generate_code=%d (%.3fms avg)\n",
                            fgRig.generated, fgRig.attempts, fgRig.generateCode,
                            fgRig.attempts ? fgRig.msTotal / (double)fgRig.attempts : 0.0);
            }
        }
        SDL_Delay(16);
        ++frame;
    }
    std::printf("  SDL software blit summary: %u frames, last fps=%d (WSI swapchain NOT used)\n", frame, fps);
    // The end-of-run Frame Generation tally. Read it as: the host asked for `fgMult`-X on `attempts` frames and
    // the provider DELIVERED `generated` frames. FSR's own upscale count is `frame` dispatches (all OK), so
    // "frames generated" here means INTERPOLATED frames specifically - never conflated with the upscale proof.
    if (fgRig.live) {
        const double avgMs = fgRig.attempts ? fgRig.msTotal / (double)fgRig.attempts : 0.0;
        std::printf("  FRAME GENERATION tally: host_frames=%lld generated_frames=%lld ok_calls=%lld "
                    "last_generate=%d prepare=%d avg_cost=%.3fms -> delivered %dX of requested %dX\n",
                    fgRig.attempts, fgRig.generated, fgRig.okCalls, fgRig.generateCode, fgRig.prepareCode, avgMs,
                    fgRig.attempts ? (int)(1 + (fgRig.generated / (double)fgRig.attempts)) : 1,
                    (fgMult > 1 ? fgMult : 1));
    } else {
        std::printf("  FRAME GENERATION tally: NO CONTEXT (provider create/configure refused) generated_frames=0\n");
    }
    teardownFgWindowRig(a, fgRig);
    SDL_DestroyTexture(tex);
    SDL_DestroyRenderer(rend);
    return true;
}

int main(int argc, char** argv) {
    bool windowMode = false; int windowSec = 30; int fgMult = 0; bool wantPresent = false; bool hudDump = false;
    std::vector<std::string> pos;
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        if (s == "--window") { windowMode = true; continue; }
        if (s.rfind("--window=", 0) == 0) { windowMode = true; windowSec = std::stoi(s.substr(9)); continue; }
        if (s.rfind("--fg=", 0) == 0) { std::string v = s.substr(5);
            fgMult = (v=="1x")?1:(v=="2x")?2:(v=="3x")?3:(v=="4x")?4:std::stoi(v); continue; }
        if (s == "--present") { wantPresent = true; continue; }
        if (s == "--hud-dump") { hudDump = true; continue; }
        pos.push_back(s);
    }
    // The HUD's frame-generation lines show BOTH halves of the story: what this host REQUESTED
    // (--fg=2x / 4x) and what the provider DELIVERED, so a refusal is visible on screen rather
    // than hidden behind a multiplier copied from the request.
    gHudDump = hudDump;
    gHudEchoLines = hudDump;
    gHudFacts.fgRequested = (fgMult >= 1) ? fgMult : 1;
    const char* env = std::getenv("FSR4_FGTEST_OUTDIR");
    std::string outDir = env ? env : ".";
    auto P = [&](size_t k, const std::string& d) { return pos.size() > k ? pos[k] : d; };
    std::string upSo    = P(0, outDir + "/amd_fidelityfx_upscaler_vk.so");
    std::string fgSo    = P(1, outDir + "/amd_fidelityfx_framegeneration_vk.so");
    std::string vertSpv = P(2, outDir + "/triangle.vert.spv");
    std::string fragSpv = P(3, outDir + "/triangle.frag.spv");
    std::string pVertSpv = outDir + "/present.vert.spv";
    std::string pFragSpv = outDir + "/present.frag.spv";

    bool gpuPresent = windowMode && wantPresent;
    if (windowMode) {
        if (SDL_Init(SDL_INIT_VIDEO) != 0) { std::printf("NO_WINDOW: SDL_Init failed (%s)\n", SDL_GetError()); return 77; }
        // Only --present asks SDL for a Vulkan window (SDL_WINDOW_VULKAN). A plain --window opens a normal
        // window: on this host SDL_Vulkan + the software-RADV fallback crash during window creation, so we
        // keep the decorative window separate from the app's own headless VkInstance that runs the REAL FSR.
        SDL_WindowFlags fl = (SDL_WindowFlags)(SDL_WINDOW_RESIZABLE | (gpuPresent ? (SDL_WindowFlags)SDL_WINDOW_VULKAN : (SDL_WindowFlags)0));
        gWin = SDL_CreateWindow("fsr4-vk mini host: 480x270 -> FSR 960x540 -> window 1440x1080", 80, 40,
                                (int)kDispW, (int)kDispH, fl);
        if (!gWin) { std::printf("NO_WINDOW: SDL_CreateWindow failed (%s)\n", SDL_GetError()); SDL_Quit(); return 77; }
        SDL_ShowWindow(gWin);
    }
    VkApp a;
    if (!initVulkan(a, gpuPresent)) { std::printf("NO_DEVICE\n"); if (windowMode) SDL_Quit(); return 77; }
    VkShaderModule vs = mkShader(a, vertSpv), fs = mkShader(a, fragSpv);
    if (vs == VK_NULL_HANDLE || fs == VK_NULL_HANDLE) { std::printf("NO_SHADERS: triangle SPIR-V missing\n"); if (windowMode) SDL_Quit(); return 2; }
    setupGeom(a); setupUbo(a); setupDescriptors(a); setupPipeline(a, vs, fs, a.colorFmt, a.pipeline);
    Provider up, fg;
    bool upOk = loadProvider(up, upSo), fgOk = loadProvider(fg, fgSo);

    // The REAL AMD FSR4 upscale host (rig images + a live upscale context, 480x270 -> 960x540). Built ONCE so
    // the PPM proof and the live window share one context: the provider keeps FSR's temporal state
    // (accumulation / reactive) between dispatches, exactly as a game host would.
    FsrHost fsr;
    const bool fsrLive = upOk && initFsrHost(a, up, fsr, outDir);
    bool fsrRan = false;

    int written = 0;
    auto runHeadless = [&]() {
        if (upOk) driveUpscaler(up, a); else std::printf("\n  upscaler provider .so not loaded: %s\n", upSo.c_str());
        if (fgOk) driveFrameGeneration(fg, a, fsrLive ? &fsr.rig : nullptr); else std::printf("\n  frame-generation provider .so not loaded: %s\n", fgSo.c_str());
        std::printf("=== MOVING TRIANGLE: rendering %u offscreen frames (%ux%u) ===\n", kFrames, kWidth, kHeight);
        std::vector<unsigned char> host;
        for (uint32_t f = 0; f < kFrames; ++f) {
            renderFrame(a, f, (double)f, host);
            bool nonblack = false;
            for (size_t i = 0; i + 2 < host.size(); i += 4)
                if (host[i] || host[i + 1] || host[i + 2]) { nonblack = true; break; }
            if (nonblack) { writePPM(outDir + "/frame_" + std::to_string(f) + ".ppm", kWidth, kHeight, host.data()); ++written; }
        }
        std::printf("  frames rendered non-black: %d/%u (moving triangle drawn offscreen)\n", written, kFrames);
        // The proof that matters for the upscaler: an ACTUAL ffxDispatch(FFX_API_DISPATCH_DESC_TYPE_UPSCALE)
        // over app-owned images, submitted, with the result read back off the GPU.
        if (fsrLive) { if (runFsrProof(a, fsr, outDir, 3)) fsrRan = true; }
        else std::printf("\n  FSR IMAGE DISPATCH PROOF: SKIPPED (no upscaler context on this host)\n");
    };
    if (windowMode && wantPresent) {
        std::printf("\n=== WINDOWED PRESENT: render %ux%u -> real AMD FSR upscale -> present %ux%u (%ds, --fg=%dx) ===\n",
                    kRenderW, kRenderH, kDispW, kDispH, windowSec, fgMult);
        ffxContext upCtx{}, fgCtx{};
        ffxReturnCode_t ucc = upOk ? createUpscale(up, a, upCtx) : (ffxReturnCode_t)0xFFFF;
        ffxReturnCode_t gcc = fgOk ? createFg(fg, a, fgCtx) : (ffxReturnCode_t)0xFFFF;
        std::printf("  upscaler live create = %d (0=OK)\n", (int)ucc);
        std::printf("  frame-generation live create = %d (0=OK)\n", (int)gcc);
        VkShaderModule pv = mkShader(a, pVertSpv), pf = mkShader(a, pFragSpv);
        if (pv == VK_NULL_HANDLE || pf == VK_NULL_HANDLE) {
            std::printf("NO_PRESENT_SHADERS: present.vert/frag.spv missing (run build.sh)\n");
            if (upOk && ucc == FFX_API_RETURN_OK) up.destroy(&upCtx, nullptr);
            if (fgOk && gcc == FFX_API_RETURN_OK) fg.destroy(&fgCtx, nullptr);
            SDL_DestroyWindow(gWin); SDL_Quit(); gWin = nullptr; return 2;
        }
        runPresent(a, up, upCtx, ucc == FFX_API_RETURN_OK, fg, fgCtx, gcc == FFX_API_RETURN_OK,
                   vertSpv, fragSpv, pv, pf, windowSec, fgMult, gWin);
        if (upOk && ucc == FFX_API_RETURN_OK) up.destroy(&upCtx, nullptr);
        if (fgOk && gcc == FFX_API_RETURN_OK) fg.destroy(&fgCtx, nullptr);
        SDL_DestroyWindow(gWin); SDL_Quit(); gWin = nullptr;
        written = 1;
    } else if (windowMode) {
        // SAFE WINDOW (default for --window on hosts whose driver lacks vkCreateSwapchainKHR): open the 1440x1080
        // SDL window, run the REAL AMD FSR upscale headlessly (-> PPM), then show the LIVE animated triangle in
        // that window by blitting each offscreen GPU readback through SDL's SOFTWARE renderer (FPS readout baked
        // in, no WSI swapchain touched). Use --present to attempt the real Vulkan-WSI FSR blit on a host whose
        // GPU driver implements vkCreateSwapchainKHR.
        std::printf("\n=== WINDOW: opening %ux%u SDL window (WSI present OFF: driver lacks vkCreateSwapchainKHR) ===\n",
                    kDispW, kDispH);
        runHeadless();
        std::printf("\nWSI present NOT used on this host; the REAL AMD FSR upscale ran offscreen -> PPM above. The SDL window\n"
                    "stays open for %ds. It now shows the LIVE moving triangle (GPU render -> CPU readback -> SDL\n"
                    "software scaler %ux%u -> window size) with a top-right FPS readout; ESC/Q or window-close ends it.\n"
                    "Add --present for the real GPU FSR blit on a host whose driver implements vkCreateSwapchainKHR.\n",
                    windowSec, kWidth, kHeight);
        // If SDL cannot give us a software renderer/texture, keep the previous blank-window behaviour.
        // The FG provider is driven HERE too (per frame, counted), so the window shows the frame-generation
        // tally live instead of only the headless ladder's one-shot codes.
        if (!showWindowSdl(a, fsrLive ? &up : nullptr, fsrLive ? &fsr : nullptr, fgOk ? &fg : nullptr, fgMult,
                          gWin, windowSec))
            holdWindow(windowSec, gWin);
        SDL_DestroyWindow(gWin); SDL_Quit(); gWin = nullptr;
    } else {
        runHeadless();
    }
    if (fsr.live) teardownFsrHost(a, fsr);
    std::printf("\n=== RESULT ===\n  triangle host render: %s | AMD FSR4 upscale dispatch: %s | FG provider: driven "
                "with real per-frame images (see codes)%s\n",
                written > 0 ? "RUN" : "NO-FRAME", fsrRan ? "RUN (960x540 output proved)" : "NOT PROVEN",
                windowMode ? "  [windowed, WSI present OFF]" : "");
    return 0;
}

static void runPresent(VkApp& a, Provider& up, ffxContext& upCtx, bool upLive, Provider& fg, ffxContext& fgCtx,
                       bool fgLive, const std::string& vertSpv, const std::string& fragSpv, VkShaderModule vsP, VkShaderModule fsP,
                       int seconds, int fgMult, SDL_Window* win) {
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    if (SDL_Vulkan_CreateSurface(win, a.instance, &surface) != SDL_TRUE || !surface) {
        std::printf("NO_SURFACE: SDL_Vulkan_CreateSurface failed\n"); return; }
    a.surface = surface;
    VkSurfaceCapabilitiesKHR cap{};
    if (vkGetPhysicalDeviceSurfaceCapabilitiesKHR(a.physical, surface, &cap) != VK_SUCCESS || !cap.supportedUsageFlags) {
        std::printf("NO_CAPS: surface unsupported\n"); return; }
    VkExtent2D ext{ kDispW, kDispH };
    if (cap.currentExtent.width != 0xFFFFFFFFu) ext = cap.currentExtent;
    if (ext.width  > cap.maxImageExtent.width)  ext.width  = cap.maxImageExtent.width;
    if (ext.height > cap.maxImageExtent.height) ext.height = cap.maxImageExtent.height;
    if (ext.width > cap.maxImageExtent.width) ext.width = cap.maxImageExtent.width;
    if (ext.height > cap.maxImageExtent.height) ext.height = cap.maxImageExtent.height;
    a.surfExtent = ext;
    uint32_t nf = 0; vkGetPhysicalDeviceSurfaceFormatsKHR(a.physical, surface, &nf, nullptr);
    std::vector<VkSurfaceFormatKHR> fmts(nf);
    if (nf) vkGetPhysicalDeviceSurfaceFormatsKHR(a.physical, surface, &nf, fmts.data());
    VkSurfaceFormatKHR sf{VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR};
    if (nf == 1 && fmts[0].format == VK_FORMAT_UNDEFINED) sf = fmts[0];
    else if (nf) sf = fmts[0];
    a.presentFmt = sf.format;
    uint32_t icount = cap.minImageCount < 2u ? 2u : cap.minImageCount; if (icount > 3u) icount = 3u;
    a.imageCount = icount;
    // If the driver/loader exposes no present mode, the WSI path is unavailable on this host: bail cleanly
    // BEFORE creating a swapchain (the loader would mask a NULL driver function with a bogus handle -> SIGSEGV).
    uint32_t nModes = 0;
    if (vkGetPhysicalDeviceSurfacePresentModesKHR(a.physical, surface, &nModes, nullptr) != VK_SUCCESS || nModes == 0) {
        std::printf("NO_PRESENT_MODES: driver exposes no WSI present path on this host; skipping present.\n");
        vkDestroySurfaceKHR(a.instance, a.surface, nullptr); a.surface = VK_NULL_HANDLE; return; }
    VkSwapchainCreateInfoKHR sci{}; sci.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    sci.surface = surface; sci.minImageCount = icount; sci.imageFormat = sf.format; sci.imageColorSpace = sf.colorSpace;
    sci.imageExtent = ext; sci.imageArrayLayers = 1; sci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    sci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    sci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR; sci.presentMode = VK_PRESENT_MODE_FIFO_KHR;
    sci.clipped = VK_TRUE; sci.oldSwapchain = VK_NULL_HANDLE;
    if (vkCreateSwapchainKHR(a.device, &sci, nullptr, &a.swap) != VK_SUCCESS || !a.swap) {
        std::printf("NO_SWAPCHAIN: vkCreateSwapchainKHR failed\n"); return; }
    uint32_t nn = 0; vkGetSwapchainImagesKHR(a.device, a.swap, &nn, nullptr);
    std::vector<VkImage> imgs(nn); if (nn) vkGetSwapchainImagesKHR(a.device, a.swap, &nn, imgs.data());
    if (nn < 1) { std::printf("NO_SWAPCHAIN_IMAGES: driver present path unavailable (loader NULL)\n");
        vkDestroySwapchainKHR(a.device, a.swap, nullptr); a.swap = VK_NULL_HANDLE;
        vkDestroySurfaceKHR(a.instance, a.surface, nullptr); a.surface = VK_NULL_HANDLE; return; }
    a.scColor = imgs; a.scColorView.resize(nn); a.scDepth.resize(nn); a.scDepthView.resize(nn); a.scDepthMem.resize(nn);
    for (uint32_t i = 0; i < nn; ++i) {
        VkImageViewCreateInfo ci{}; ci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO; ci.image = imgs[i];
        ci.viewType = VK_IMAGE_VIEW_TYPE_2D; ci.format = sf.format; ci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCreateImageView(a.device, &ci, nullptr, &a.scColorView[i]);
        VkImageCreateInfo di{}; di.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO; di.imageType = VK_IMAGE_TYPE_2D;
        di.format = VK_FORMAT_D32_SFLOAT; di.extent = {ext.width, ext.height, 1}; di.mipLevels = 1; di.arrayLayers = 1;
        di.samples = VK_SAMPLE_COUNT_1_BIT; di.usage = VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
        di.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        vkCreateImage(a.device, &di, nullptr, &a.scDepth[i]);
        VkMemoryRequirements mr{}; vkGetImageMemoryRequirements(a.device, a.scDepth[i], &mr);
        uint32_t mi = findDeviceMemoryType(a, mr, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr, mr.size, mi};
        vkAllocateMemory(a.device, &ai, nullptr, &a.scDepthMem[i]);
        vkBindImageMemory(a.device, a.scDepth[i], a.scDepthMem[i], 0);
        VkImageViewCreateInfo dv{}; dv.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO; dv.image = a.scDepth[i];
        dv.viewType = VK_IMAGE_VIEW_TYPE_2D; dv.format = VK_FORMAT_D32_SFLOAT; dv.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
        vkCreateImageView(a.device, &dv, nullptr, &a.scDepthView[i]);
    }
    auto mkFmt = [&](VkFormat f, VkExtent2D e, VkImageUsageFlags u, VkImage& im, VkImageView& vw,
                     VkDeviceMemory& mem, VkImageAspectFlags asp) {
        VkImageCreateInfo ic{}; ic.sType=VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO; ic.imageType=VK_IMAGE_TYPE_2D; ic.format=f;
        ic.extent={e.width,e.height,1}; ic.mipLevels=1; ic.arrayLayers=1; ic.samples=VK_SAMPLE_COUNT_1_BIT; ic.usage=u; ic.initialLayout=VK_IMAGE_LAYOUT_UNDEFINED;
        vkCreateImage(a.device,&ic,nullptr,&im);
        VkMemoryRequirements mr{}; vkGetImageMemoryRequirements(a.device,im,&mr);
        uint32_t mi=findDeviceMemoryType(a,mr,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,nullptr,mr.size,mi};
        vkAllocateMemory(a.device,&ai,nullptr,&mem); vkBindImageMemory(a.device,im,mem,0);
        VkImageViewCreateInfo vci{}; vci.sType=VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO; vci.image=im; vci.viewType=VK_IMAGE_VIEW_TYPE_2D; vci.format=f;
        vci.subresourceRange={asp,0,1,0,1}; vkCreateImageView(a.device,&vci,nullptr,&vw);
    };
    VkImage colR{VK_NULL_HANDLE},triDepth{VK_NULL_HANDLE},depR{VK_NULL_HANDLE},mvR{VK_NULL_HANDLE},outU{VK_NULL_HANDLE};
    VkImageView colRV{VK_NULL_HANDLE},triDepthV{VK_NULL_HANDLE},depRV{VK_NULL_HANDLE},mvRV{VK_NULL_HANDLE},outUV{VK_NULL_HANDLE};
    VkDeviceMemory colRM{VK_NULL_HANDLE},triDepthM{VK_NULL_HANDLE},depRM{VK_NULL_HANDLE},mvRM{VK_NULL_HANDLE},outUM{VK_NULL_HANDLE};
    // Formats + usages mirror the PROVEN headless rig: the upscaler's inputColorVkFormat accepts ONLY
    // R16G16B16A16_FLOAT / B10G11R11 / RGB9E5 (the old RGBA8 colour claim is why this dispatch returned PARAMETER
    // and never ran), and the depth role must CLAIM R32_FLOAT + the DEPTHTARGET usage bit so the provider binds the
    // app's REAL D32_SFLOAT depth ATTACHMENT with the DEPTH aspect.
    mkFmt(VK_FORMAT_R16G16B16A16_SFLOAT,{kRenderW,kRenderH},VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT|VK_IMAGE_USAGE_SAMPLED_BIT,colR,colRV,colRM,VK_IMAGE_ASPECT_COLOR_BIT);
    mkFmt(VK_FORMAT_D32_SFLOAT,{kRenderW,kRenderH},VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,triDepth,triDepthV,triDepthM,VK_IMAGE_ASPECT_DEPTH_BIT);
    mkFmt(VK_FORMAT_R32_SFLOAT,{kRenderW,kRenderH},VK_IMAGE_USAGE_SAMPLED_BIT|VK_IMAGE_USAGE_TRANSFER_DST_BIT,depR,depRV,depRM,VK_IMAGE_ASPECT_COLOR_BIT);
    mkFmt(VK_FORMAT_R16G16_SFLOAT,{kRenderW,kRenderH},VK_IMAGE_USAGE_SAMPLED_BIT|VK_IMAGE_USAGE_TRANSFER_DST_BIT,mvR,mvRV,mvRM,VK_IMAGE_ASPECT_COLOR_BIT);
    mkFmt(VK_FORMAT_R16G16B16A16_SFLOAT,ext,VK_IMAGE_USAGE_STORAGE_BIT|VK_IMAGE_USAGE_SAMPLED_BIT|VK_IMAGE_USAGE_TRANSFER_DST_BIT,outU,outUV,outUM,VK_IMAGE_ASPECT_COLOR_BIT);
    if (!colR || !triDepth || !mvR || !outU) { std::printf("NO_RIG_IMAGES: FSR input images could not be created\n"); return; }
    // Triangle pipeline bound to the FP16 colour + D32 depth attachment pair. Fresh SPIR-V modules are loaded here
    // because setupPipeline OWNS their destruction (the handles main created were already retired with it).
    VkPipeline renderPipe{VK_NULL_HANDLE};
    {
        auto v2 = mkShader(a, vertSpv), f2 = mkShader(a, fragSpv);
        if (v2 != VK_NULL_HANDLE && f2 != VK_NULL_HANDLE) setupPipeline(a, v2, f2, VK_FORMAT_R16G16B16A16_SFLOAT, renderPipe);
    }
    if (renderPipe == VK_NULL_HANDLE) { std::printf("NO_RENDER_PIPELINE: FSR rig pipeline could not be created\n"); return; }
    // One-time layout + clear pass (mirrors setupFsrRig): motion vectors = (0,0) here, since this host supplies no
    // motion-vector pass, and both inputs end up SHADER_READ_ONLY while the storage output ends up GENERAL.
    {
        VkCommandBuffer cc{VK_NULL_HANDLE};
        VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,nullptr,a.cmdPool,VK_COMMAND_BUFFER_LEVEL_PRIMARY,1};
        vkAllocateCommandBuffers(a.device,&cai,&cc);
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,nullptr,0,nullptr};
        vkBeginCommandBuffer(cc,&bi);
        VkImageSubresourceRange cRange{VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1};
        VkImageMemoryBarrier toDst[2] = {
            {.sType=VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,.srcAccessMask=0,.dstAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT,
             .oldLayout=VK_IMAGE_LAYOUT_UNDEFINED,.newLayout=VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,.image=mvR,.subresourceRange=cRange},
            {.sType=VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,.srcAccessMask=0,.dstAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT,
             .oldLayout=VK_IMAGE_LAYOUT_UNDEFINED,.newLayout=VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,.image=outU,.subresourceRange=cRange}};
        vkCmdPipelineBarrier(cc,VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_DEPENDENCY_BY_REGION_BIT,0,nullptr,0,nullptr,2,toDst);
        VkClearColorValue zero{}; zero.float32[0]=zero.float32[1]=zero.float32[2]=zero.float32[3]=0.0f;
        vkCmdClearColorImage(cc,mvR,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,&zero,1,&cRange);
        vkCmdClearColorImage(cc,outU,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,&zero,1,&cRange);
        VkImageMemoryBarrier toUse[2] = {
            {.sType=VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT,.dstAccessMask=VK_ACCESS_SHADER_READ_BIT,
             .oldLayout=VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,.newLayout=VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,.image=mvR,.subresourceRange=cRange},
            {.sType=VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT,
             .dstAccessMask=(VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT),.oldLayout=VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
             .newLayout=VK_IMAGE_LAYOUT_GENERAL,.image=outU,.subresourceRange=cRange}};
        vkCmdPipelineBarrier(cc,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_DEPENDENCY_BY_REGION_BIT,0,nullptr,0,nullptr,2,toUse);
        vkEndCommandBuffer(cc);
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO,nullptr,0,nullptr,nullptr,1,&cc,0,nullptr};
        vkQueueSubmit(a.queue,1,&si,VK_NULL_HANDLE); vkQueueWaitIdle(a.queue); vkFreeCommandBuffers(a.device,a.cmdPool,1,&cc);
    }
    // present pipeline: fragment samples a FP16 texture (binding 0) + 16-byte push constants
    VkDescriptorSetLayout samplerSet{VK_NULL_HANDLE};
    { VkDescriptorSetLayoutBinding b{.binding=0,.descriptorType=VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,.descriptorCount=1,.stageFlags=VK_SHADER_STAGE_FRAGMENT_BIT};
      VkDescriptorSetLayoutCreateInfo li{.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,.bindingCount=1,.pBindings=&b};
      vkCreateDescriptorSetLayout(a.device,&li,nullptr,&samplerSet); }
    VkPushConstantRange pcRange{.stageFlags=VK_SHADER_STAGE_FRAGMENT_BIT,.offset=0,.size=16};
    VkPipelineLayout presentLayout{VK_NULL_HANDLE};
    { VkPipelineLayoutCreateInfo li{.sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,.setLayoutCount=1,.pSetLayouts=&samplerSet,.pushConstantRangeCount=1,.pPushConstantRanges=&pcRange};
      vkCreatePipelineLayout(a.device,&li,nullptr,&presentLayout); }
    VkPipeline presentPipe{VK_NULL_HANDLE};
    {
        VkPipelineVertexInputStateCreateInfo vi{.sType=VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        VkPipelineInputAssemblyStateCreateInfo ia{.sType=VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,.topology=VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,.primitiveRestartEnable=VK_FALSE};
        VkPipelineViewportStateCreateInfo vp{.sType=VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,.viewportCount=1,.scissorCount=1};
        VkPipelineRasterizationStateCreateInfo rs{.sType=VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,.polygonMode=VK_POLYGON_MODE_FILL,.cullMode=VK_CULL_MODE_NONE,.frontFace=VK_FRONT_FACE_COUNTER_CLOCKWISE,.lineWidth=1.0f};
        VkPipelineMultisampleStateCreateInfo ms{.sType=VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,.rasterizationSamples=VK_SAMPLE_COUNT_1_BIT};
        VkPipelineColorBlendAttachmentState cba{.blendEnable=VK_FALSE,.colorWriteMask=0xf};
        VkPipelineColorBlendStateCreateInfo cb{.sType=VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,.attachmentCount=1,.pAttachments=&cba};
        VkDynamicState dyn[2]={VK_DYNAMIC_STATE_VIEWPORT,VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo ds{.sType=VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,.dynamicStateCount=2,.pDynamicStates=dyn};
        VkPipelineShaderStageCreateInfo st[2] = {
          {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,nullptr,0,VK_SHADER_STAGE_VERTEX_BIT,vsP,"main",nullptr},
          {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,nullptr,0,VK_SHADER_STAGE_FRAGMENT_BIT,fsP,"main",nullptr}};
        VkPipelineRenderingCreateInfo rr{.sType=VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,.colorAttachmentCount=1,.pColorAttachmentFormats=&a.presentFmt};
        VkGraphicsPipelineCreateInfo gci{.sType=VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,.pNext=&rr,.stageCount=2,.pStages=st,.pVertexInputState=&vi,.pInputAssemblyState=&ia,.pViewportState=&vp,.pRasterizationState=&rs,.pMultisampleState=&ms,.pColorBlendState=&cb,.pDynamicState=&ds,.layout=presentLayout};
        vkCreateGraphicsPipelines(a.device,VK_NULL_HANDLE,1,&gci,nullptr,&presentPipe);
    }
    VkDescriptorPool pPool{VK_NULL_HANDLE};
    { VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,4};
      VkDescriptorPoolCreateInfo pi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,nullptr,0,1,1,&ps};
      vkCreateDescriptorPool(a.device,&pi,nullptr,&pPool); }
    VkDescriptorSet descP{VK_NULL_HANDLE};
    { VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,nullptr,pPool,1,&samplerSet};
      vkAllocateDescriptorSets(a.device,&ai,&descP);
      VkDescriptorImageInfo ii{VK_NULL_HANDLE,outUV,VK_IMAGE_LAYOUT_GENERAL};
      VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,nullptr,descP,0,0,1,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,&ii,nullptr,nullptr};
      vkUpdateDescriptorSets(a.device,1,&w,0,nullptr); }
    std::vector<VkFence> fences(nn, VK_NULL_HANDLE);
    for (uint32_t i = 0; i < nn; ++i) { VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,nullptr,VK_FENCE_CREATE_SIGNALED_BIT}; vkCreateFence(a.device,&fi,nullptr,&fences[i]); }
    VkPhysicalDeviceProperties p{}; vkGetPhysicalDeviceProperties(a.physical, &p);
    std::printf("\n=== WINDOWED PRESENT: 3D triangle 1280x720 -> REAL AMD FSR upscale -> %ux%u (%ds, --fg=%dx) ===\n",
                (unsigned)ext.width, (unsigned)ext.height, seconds, fgMult);
    std::printf("  device: %s | fmt=%d FIFO swapchain=%u | upscaler live=%d fg live=%d\n",
                p.deviceName, (int)sf.format, nn, upLive?1:0, fgLive?1:0);
    ffxReturnCode_t fgCfg = (ffxReturnCode_t)0xFFFF;
    if (fgLive) {
        ffxConfigureDescFrameGeneration cf{}; cf.header.type = FFX_API_CONFIGURE_DESC_TYPE_FRAMEGENERATION;
        cf.frameGenerationEnabled = true; cf.swapChain = (void*)&fgCtx; cf.presentCallback = cbPresent; cf.frameGenerationCallback = cbFg;
        fgCfg = fg.configure(&fgCtx, &cf.header);
        std::printf("  ffxConfigure(frame-generation) = %d\n", (int)fgCfg);
    }
    auto t0 = std::chrono::steady_clock::now();
    uint32_t frame = 0; uint64_t fgFrameId = 1; int fgGen = 0; double lastFpsT = 0; uint32_t lastFpsFrame = 0; int fps = 0;
    bool showWin = false;
    while (true) {
        auto now = std::chrono::steady_clock::now();
        double el = std::chrono::duration<double>(now - t0).count();
        if (el >= (double)seconds) break;
        if (showWin && (frame % 30u) == 0u) { SDL_RaiseWindow(win); showWin = false; }
        // FPS: whole seconds so far
        if (el >= 1.0) fps = (int)((frame - lastFpsFrame) / (el - lastFpsT));
        uint32_t i = 0; VkSemaphore acq{VK_NULL_HANDLE};
        { VkSemaphoreCreateInfo si{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO}; vkCreateSemaphore(a.device, &si, nullptr, &acq); }
        vkAcquireNextImageKHR(a.device, a.swap, UINT64_MAX, acq, VK_NULL_HANDLE, &i);
        float phase = (float)el * 2.0f;
        Mat4 model = mul(translate(0.55f * std::sinf(phase), 0.25f * std::cosf(phase * 0.8f), 0.0f), rotateZ(phase * 0.7f));
        Mat4 proj = ortho(-1.6f, 1.6f, -0.9f, 0.9f, -1.0f, 1.0f); Mat4 view; ShaderData d;
        putMat(d.projection, proj); putMat(d.model, model); putMat(d.view, view);
        std::memcpy(a.uboMap[i % kFrames], &d, sizeof(d));
        vkWaitForFences(a.device, 1, &fences[i], VK_TRUE, UINT64_MAX); vkResetFences(a.device, 1, &fences[i]);
        { VkCommandBuffer cc{VK_NULL_HANDLE}; VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,nullptr,a.cmdPool,VK_COMMAND_BUFFER_LEVEL_PRIMARY,1}; vkAllocateCommandBuffers(a.device,&cai,&cc);
          VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,nullptr,0,nullptr}; vkBeginCommandBuffer(cc,&bi);
          VkClearValue cval; cval.color={0.10f,0.10f,0.20f,1.0f}; VkClearValue dval; dval.depthStencil={1.0f,0};
          VkRenderingAttachmentInfo cA{}; cA.sType=VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO; cA.imageView=colRV; cA.imageLayout=VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL; cA.loadOp=VK_ATTACHMENT_LOAD_OP_CLEAR; cA.storeOp=VK_ATTACHMENT_STORE_OP_STORE; cA.clearValue=cval;
          VkRenderingAttachmentInfo dA{}; dA.sType=VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO; dA.imageView=triDepthV; dA.imageLayout=VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL; dA.loadOp=VK_ATTACHMENT_LOAD_OP_CLEAR; dA.storeOp=VK_ATTACHMENT_STORE_OP_DONT_CARE; dA.clearValue=dval;
          VkRenderingInfo rr{}; rr.sType=VK_STRUCTURE_TYPE_RENDERING_INFO; rr.renderArea={ {0,0}, {kRenderW,kRenderH} }; rr.layerCount=1; rr.colorAttachmentCount=1; rr.pColorAttachments=&cA; rr.pDepthAttachment=&dA;
          vkCmdBeginRendering(cc,&rr);
          VkViewport vpt{0.0f,0.0f,(float)kRenderW,(float)kRenderH,0.0f,1.0f}; vkCmdSetViewport(cc,0,1,&vpt);
          VkRect2D sc{ {0,0}, {kRenderW,kRenderH} }; vkCmdSetScissor(cc,0,1,&sc);
          vkCmdBindDescriptorSets(cc,VK_PIPELINE_BIND_POINT_GRAPHICS,a.pipeLayout,0,1,&a.descSet[i%kFrames],0,nullptr);
          vkCmdBindPipeline(cc,VK_PIPELINE_BIND_POINT_GRAPHICS,renderPipe);
          VkDeviceSize off[1]={0}; vkCmdBindVertexBuffers(cc,0,1,&a.vertexBuf,off); vkCmdBindIndexBuffer(cc,a.indexBuf,0,VK_INDEX_TYPE_UINT32);
          vkCmdDrawIndexed(cc,gIndexCount,1,0,0,0); vkCmdEndRendering(cc);
          // Both render outputs move to SHADER_READ_ONLY (the layouts the FSR model samples its inputs in, as in
          // the proven rig): colour with the COLOR aspect, the REAL depth attachment with the DEPTH aspect.
          VkImageMemoryBarrier toRead[2] = {
              {.sType=VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,.srcAccessMask=VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,.dstAccessMask=VK_ACCESS_SHADER_READ_BIT,.oldLayout=VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,.newLayout=VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,.image=colR,.subresourceRange={VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1}},
              {.sType=VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,.srcAccessMask=VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,.dstAccessMask=VK_ACCESS_SHADER_READ_BIT,.oldLayout=VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,.newLayout=VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,.image=triDepth,.subresourceRange={VK_IMAGE_ASPECT_DEPTH_BIT,0,1,0,1}}};
          vkCmdPipelineBarrier(cc,VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_DEPENDENCY_BY_REGION_BIT,0,nullptr,0,nullptr,2,toRead);
          vkEndCommandBuffer(cc);
          VkSubmitInfo s{}; s.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO; s.commandBufferCount=1; s.pCommandBuffers=&cc; vkQueueSubmit(a.queue,1,&s,VK_NULL_HANDLE); vkFreeCommandBuffers(a.device,a.cmdPool,1,&cc); }
        if (upLive) {
            VkCommandBuffer cc{VK_NULL_HANDLE}; VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,nullptr,a.cmdPool,VK_COMMAND_BUFFER_LEVEL_PRIMARY,1}; vkAllocateCommandBuffers(a.device,&cai,&cc);
            VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,nullptr,0,nullptr}; vkBeginCommandBuffer(cc,&bi);
            ffxDispatchDescUpscale u{}; u.header.type=FFX_API_DISPATCH_DESC_TYPE_UPSCALE; u.commandList=(void*)cc;
            u.color=mkRes(colR,kRenderW,kRenderH,FFX_API_SURFACE_FORMAT_R16G16B16A16_FLOAT,FFX_API_RESOURCE_USAGE_READ_ONLY,FFX_API_RESOURCE_STATE_COMPUTE_READ);
            u.depth=mkRes(triDepth,kRenderW,kRenderH,FFX_API_SURFACE_FORMAT_R32_FLOAT,FFX_API_RESOURCE_USAGE_DEPTHTARGET,FFX_API_RESOURCE_STATE_COMPUTE_READ);
            u.motionVectors=mkRes(mvR,kRenderW,kRenderH,FFX_API_SURFACE_FORMAT_R16G16_FLOAT,FFX_API_RESOURCE_USAGE_READ_ONLY,FFX_API_RESOURCE_STATE_COMPUTE_READ);
            u.output=mkRes(outU,ext.width,ext.height,FFX_API_SURFACE_FORMAT_R16G16B16A16_FLOAT,FFX_API_RESOURCE_USAGE_UAV,FFX_API_RESOURCE_STATE_UNORDERED_ACCESS);
            u.exposure.resource=nullptr; u.reactive.resource=nullptr; u.transparencyAndComposition.resource=nullptr;
            u.renderSize={kRenderW,kRenderH}; u.upscaleSize={ext.width,ext.height};
            u.jitterOffset={0.0f,0.0f}; u.motionVectorScale={1.0f,1.0f};
            u.enableSharpening=true; u.sharpness=0.2f; u.preExposure=1.0f; u.reset=false;
            u.frameTimeDelta=16.6f; u.cameraNear=0.1f; u.cameraFar=1000.0f; u.cameraFovAngleVertical=1.0f; u.viewSpaceToMetersFactor=0.0f; u.flags=0;
            ffxReturnCode_t ur=up.dispatch(&upCtx,&u.header);
            vkEndCommandBuffer(cc);
            VkSubmitInfo s{}; s.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO; s.commandBufferCount=1; s.pCommandBuffers=&cc;
            vkQueueSubmit(a.queue,1,&s,VK_NULL_HANDLE); vkFreeCommandBuffers(a.device,a.cmdPool,1,&cc);
            if (ur != FFX_API_RETURN_OK) { std::printf("  [FSR dispatch rc=%d -> falling back to native present]\n",(int)ur); upLive=false; }
        }
        if (fgLive && fgCfg == FFX_API_RETURN_OK) {
            // FG per-frame host contract WITH the images: Prepare receives this frame's REAL depth (DEPTHTARGET)
            // + motion vectors; generate receives the FSR-upscaled colour as presentColor. `outputs[]` stay null
            // because the swapchain's own images are not STORAGE-usable backbuffers and the (still absent) runtime
            // would drive presentation itself - that boundary is documented in README, not faked here.
            VkCommandBuffer fc{VK_NULL_HANDLE}; VkCommandBufferAllocateInfo fai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,nullptr,a.cmdPool,VK_COMMAND_BUFFER_LEVEL_PRIMARY,1}; vkAllocateCommandBuffers(a.device,&fai,&fc);
            VkCommandBufferBeginInfo fbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,nullptr,0,nullptr};
            vkBeginCommandBuffer(fc,&fbi);
            ffxDispatchDescFrameGenerationPrepare pr{}; pr.header.type=FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION_PREPARE;
            pr.frameID=fgFrameId; pr.renderSize={kRenderW,kRenderH}; pr.jitterOffset={0.0f,0.0f}; pr.motionVectorScale={1.0f,1.0f};
            pr.frameTimeDelta=16.6f; pr.cameraNear=0.1f; pr.cameraFar=1000.0f; pr.cameraFovAngleVertical=1.0f;
            pr.commandList=(void*)fc;
            pr.depth=mkRes(triDepth,kRenderW,kRenderH,FFX_API_SURFACE_FORMAT_R32_FLOAT,FFX_API_RESOURCE_USAGE_DEPTHTARGET,FFX_API_RESOURCE_STATE_COMPUTE_READ);
            pr.motionVectors=mkRes(mvR,kRenderW,kRenderH,FFX_API_SURFACE_FORMAT_R16G16_FLOAT,FFX_API_RESOURCE_USAGE_READ_ONLY,FFX_API_RESOURCE_STATE_COMPUTE_READ);
            ffxReturnCode_t qr=fg.dispatch(&fgCtx,&pr.header);
            const uint32_t genFrames = (uint32_t)(fgMult < 1 ? 1 : (fgMult > 3 ? 3 : fgMult));  // 1..3: one base frame
            ffxDispatchDescFrameGeneration gg{}; gg.header.type=FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION; gg.numGeneratedFrames=genFrames; gg.minMaxLuminance[0]=0.0f; gg.minMaxLuminance[1]=1.0f;
            gg.commandList=(void*)fc; gg.frameID=fgFrameId; gg.generationRect={0,0,0,0};
            gg.presentColor=mkRes(outU,ext.width,ext.height,FFX_API_SURFACE_FORMAT_R16G16B16A16_FLOAT,FFX_API_RESOURCE_USAGE_READ_ONLY,FFX_API_RESOURCE_STATE_COMPUTE_READ);
            ffxReturnCode_t gr=fg.dispatch(&fgCtx,&gg.header);
            vkEndCommandBuffer(fc);
            VkSubmitInfo fs2{VK_STRUCTURE_TYPE_SUBMIT_INFO,nullptr,0,nullptr,nullptr,1,&fc,0,nullptr};
            vkQueueSubmit(a.queue,1,&fs2,VK_NULL_HANDLE); vkFreeCommandBuffers(a.device,a.cmdPool,1,&fc);
            if (gr == FFX_API_RETURN_OK) fgGen += (int)genFrames;
            if (fgFrameId == 2u) {
                std::printf("  [FG] prepare=%d generate=%d (3 = provider refuses its ML/interp runtime WITH images handed over; 4x is not representable, asked for %dx -> %u)\n",
                            (int)qr,(int)gr,fgMult,genFrames);
            }
            ++fgFrameId;
        }
        { int fpsNow = (el > 0.3) ? (int)(frame / el) : 0;
          VkCommandBuffer cc{VK_NULL_HANDLE}; VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,nullptr,a.cmdPool,VK_COMMAND_BUFFER_LEVEL_PRIMARY,1}; vkAllocateCommandBuffers(a.device,&cai,&cc);
          VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,nullptr,0,nullptr}; vkBeginCommandBuffer(cc,&bi);
          VkImageMemoryBarrier imb{.sType=VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT,.dstAccessMask=VK_ACCESS_SHADER_READ_BIT,.oldLayout=VK_IMAGE_LAYOUT_GENERAL,.newLayout=VK_IMAGE_LAYOUT_GENERAL,.image=outU,.subresourceRange={VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1}};
          vkCmdPipelineBarrier(cc,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,VK_DEPENDENCY_BY_REGION_BIT,0,nullptr,0,nullptr,1,&imb);
          VkClearValue cval; cval.color={0.0f,0.0f,0.0f,1.0f};
          VkRenderingAttachmentInfo cA{}; cA.sType=VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO; cA.imageView=a.scColorView[i]; cA.imageLayout=VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL; cA.loadOp=VK_ATTACHMENT_LOAD_OP_CLEAR; cA.storeOp=VK_ATTACHMENT_STORE_OP_STORE; cA.clearValue=cval;
          VkRenderingInfo rr{}; rr.sType=VK_STRUCTURE_TYPE_RENDERING_INFO; rr.renderArea={ {0,0}, ext }; rr.layerCount=1; rr.colorAttachmentCount=1; rr.pColorAttachments=&cA;
          vkCmdBeginRendering(cc,&rr);
          VkViewport vpt{0.0f,0.0f,(float)ext.width,(float)ext.height,0.0f,1.0f}; vkCmdSetViewport(cc,0,1,&vpt);
          VkRect2D sc{ {0,0}, ext }; vkCmdSetScissor(cc,0,1,&sc);
          uint32_t pcvals[4]={(uint32_t)(fpsNow<0?0:fpsNow),(uint32_t)ext.width,(uint32_t)ext.height,4u};
          vkCmdPushConstants(cc,presentLayout,VK_SHADER_STAGE_FRAGMENT_BIT,0,16,pcvals);
          vkCmdBindPipeline(cc,VK_PIPELINE_BIND_POINT_GRAPHICS,presentPipe);
          vkCmdBindDescriptorSets(cc,VK_PIPELINE_BIND_POINT_GRAPHICS,presentLayout,0,1,&descP,0,nullptr);
          vkCmdDraw(cc,3,1,0,0);
          vkCmdEndRendering(cc);
          vkEndCommandBuffer(cc);
          VkSubmitInfo s{}; s.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO; s.waitSemaphoreCount=1; s.pWaitSemaphores=&acq; s.commandBufferCount=1; s.pCommandBuffers=&cc;
          vkQueueSubmit(a.queue,1,&s,fences[i]);
          VkPresentInfoKHR pi{}; pi.sType=VK_STRUCTURE_TYPE_PRESENT_INFO_KHR; pi.swapchainCount=1; pi.pSwapchains=&a.swap; pi.pImageIndices=&i;
          vkQueuePresentKHR(a.queue,&pi);
          vkFreeCommandBuffers(a.device,a.cmdPool,1,&cc); }
        std::printf("  frame %4u: present=FSR(%ux%u) t=%.2fs fps=%d  interpolated=0 generated=%d [FG req=%dx]\n",
                    frame, ext.width, ext.height, el, (el>0.3?(int)(frame/el):0), fgGen, fgMult);
        vkDestroySemaphore(a.device, acq, nullptr);
        if (!showWin) { SDL_RaiseWindow(win); showWin = true; }
        ++frame;
        std::this_thread::sleep_for(std::chrono::milliseconds(16));
    }
    std::printf("\n  WINDOW SUMMARY: presented=%u  generated=%d (FG requested %dx)  upscaler=%s\n",
                frame, fgGen, fgMult, upLive ? "REAL AMD FSR dispatch" : "native fallback (FSR create/dispatch unavailable)");
    (void)lastFpsT; (void)lastFpsFrame; (void)fps;
    for (uint32_t i = 0; i < nn; ++i) {
        vkDestroyFence(a.device, fences[i], nullptr);
        vkDestroyImageView(a.device, a.scColorView[i], nullptr);
        vkDestroyImageView(a.device, a.scDepthView[i], nullptr);
        vkDestroyImage(a.device, a.scDepth[i], nullptr);
        vkFreeMemory(a.device, a.scDepthMem[i], nullptr);
    }
    // free the render/output textures + present pipeline
    for (auto* v : {&colRV,&triDepthV,&depRV,&mvRV,&outUV}) if (*v) vkDestroyImageView(a.device,*v,nullptr);
    for (auto* im : {&colR,&triDepth,&depR,&mvR,&outU}) if (*im) vkDestroyImage(a.device,*im,nullptr);
    for (auto* m : {&colRM,&triDepthM,&depRM,&mvRM,&outUM}) if (*m) vkFreeMemory(a.device,*m,nullptr);
    if (descP) {} // part of pPool
    if (pPool) vkDestroyDescriptorPool(a.device,pPool,nullptr);
    if (presentPipe) vkDestroyPipeline(a.device,presentPipe,nullptr);
    if (presentLayout) vkDestroyPipelineLayout(a.device,presentLayout,nullptr);
    if (samplerSet) vkDestroyDescriptorSetLayout(a.device,samplerSet,nullptr);
    vkDestroySwapchainKHR(a.device, a.swap, nullptr); a.swap = VK_NULL_HANDLE;
    vkDestroySurfaceKHR(a.instance, a.surface, nullptr); a.surface = VK_NULL_HANDLE;
}




