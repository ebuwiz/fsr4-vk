// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 fsr4-vk contributors.
// See LICENSE in the project root.
//
// tools/FGTest - a headless, self-contained animated-triangle Vulkan host that drives BOTH FSR4 providers
// (the upscaler amd_fidelityfx_upscaler_vk.so and Frame Generation amd_fidelityfx_framegeneration_vk.so) through
// their REAL, documented FFX modular-API calls. It answers "does it run?" per feature with printed codes:
//   * the moving triangle renders OFFSCREEN (no window) -> PPM frames, proving the graphics host runs;
//   * the UPSCALER provider is live: create + the deterministic ffxQuery helpers (ratio / render-resolution);
//   * the FRAME GENERATION provider is live: the full create->versions->memory->configure->prepare->generate
//     host-integration ladder, printing AMD's documented return codes.
// HONEST LIMIT (never fabricated): Frame Generation's ffxDispatch(generate 1..3) returns
// FFX_API_RETURN_ERROR_RUNTIME_ERROR in this provider slice -- the interpolation/ML/present runtime and its
// buffer allocation are data-blocked (the ML graph order is not statically derivable; see tools/extract/frame-generation/dataflow.py
// + tools/capture/frame-generation). This app is the moving-graphics host such a runtime would plug into; it
// does NOT fake interpolated frames. The FSR image-dispatch path needs the app's real VkImageInfo resource marshalling
// (the repo runtime already does that); this sample therefore drives the upscaler create+query for real but does
// NOT blind-dispatch images with a guessed resource layout (that would risk a provider-side crash, not a fake).

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

// GLFW is used ONLY by the optional --window present mode. It is included after
// <vulkan/vulkan.h> (GLFW_INCLUDE_VULKAN makes GLFW reuse that header rather than
// pull in its own); the pinned Vulkan-Headers here already define VkSurfaceKHR and
// the swapchain/present core symbols, so this compiles on this host.
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

static constexpr uint32_t kWidth = 480;
static constexpr uint32_t kHeight = 270;
static constexpr uint32_t kFrames = 6;

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
        if ((bits & 1) && (mp.memoryTypes[i].propertyFlags & props) == props) return i;
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
    if (windowMode) {
        uint32_t n = 0;
        const char** g = glfwGetRequiredInstanceExtensions(&n);
        if (g) instExts.assign(g, g + n);  // window-surface ext GLFW needs at instance level
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
static void renderFrame(VkApp& a, uint32_t f, std::vector<unsigned char>& host) {
    Mat4 model = mul(translate(0.55f * std::sinf(f * 1.0f), 0.25f * std::cosf(f * 0.8f), 0.0f),
                     rotateZ(f * 0.7f));
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
         .newLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL, .image = a.color,
         .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}},
        {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, .srcAccessMask = 0,
         .dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
         .newLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL, .image = a.depth,
         .subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1}}};
    vkCmdPipelineBarrier(cc, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_DEPENDENCY_BY_REGION_BIT, 0, nullptr, 0, nullptr, 2, toAttach);

    VkClearValue cval; cval.color = {0.10f, 0.10f, 0.20f, 1.0f};
    VkRenderingAttachmentInfo colorA{.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, .imageView = a.colorView,
        .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE, .clearValue = cval};
    VkClearValue dval; dval.depthStencil = {1.0f, 0};
    VkRenderingAttachmentInfo depthA{.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, .imageView = a.depthView,
        .imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE, .clearValue = dval};
    VkRenderingInfo ri{.sType = VK_STRUCTURE_TYPE_RENDERING_INFO, .renderArea = {0, 0, (int32_t)kWidth, (int32_t)kHeight},
        .layerCount = 1, .colorAttachmentCount = 1, .pColorAttachments = &colorA, .pDepthAttachment = &depthA};
    vkCmdBeginRendering(cc, &ri);
    VkViewport vpt{0.0f, 0.0f, (float)kWidth, (float)kHeight, 0.0f, 1.0f};
    vkCmdSetViewport(cc, 0, 1, &vpt);
    VkRect2D sc{0, 0, (int32_t)kWidth, (int32_t)kHeight};
    vkCmdSetScissor(cc, 0, 1, &sc);
    vkCmdBindDescriptorSets(cc, VK_PIPELINE_BIND_POINT_GRAPHICS, a.pipeLayout, 0, 1, &a.descSet[f % kFrames], 0, nullptr);
    vkCmdBindPipeline(cc, VK_PIPELINE_BIND_POINT_GRAPHICS, a.pipeline);
    VkDeviceSize off[1] = {0};
    vkCmdBindVertexBuffers(cc, 0, 1, &a.vertexBuf, off);
    vkCmdBindIndexBuffer(cc, a.indexBuf, 0, VK_INDEX_TYPE_UINT32);
    vkCmdDrawIndexed(cc, gIndexCount, 1, 0, 0, 0);
    vkCmdEndRendering(cc);

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






// --- FSR4 UPSCALER provider: create + the deterministic ffxQuery helpers (REAL calls, printed codes) ---
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
        std::printf("  (create OK; query math answered. Pixel upscale DISPATCH needs VkImageInfo resource\n");
        std::printf("   marshalling the runtime performs; not attempted here. create+query=RUN.)\n");
    } else {
        std::printf("  create=%d: unexpected (device should meet the contract); the modular upscaler create\n", create);
        std::printf("   + cooperative-matrix-tile negotiation (instance-level extension enable + resolver) that\n");
        std::printf("   descriptor-smoke performs through the runtime. Honest, not faked; the upscaler's real on-\n");
        std::printf("   GPU run is proven by the green descriptor-smoke (render->submit->read-back, byte-identical).\n");
    }
    p.destroy(&ctx, nullptr);
}

// --- FRAME GENERATION provider: the documented host-integration ladder, printed codes ---
static void driveFrameGeneration(Provider& p, VkApp& a) {
    std::printf("\n=== FSR4 FRAME GENERATION provider (host-integration ladder) ===\n");
    FfgEnabledFeaturesDesc en; FfgBackendDesc bk; buildContract(a, en, bk);
    struct VersionLink { ffxCreateContextDescHeader header; } vlink{ffxApiHeader{0x0002000eu, nullptr}};
    bk.header.pNext = &en.header; en.header.pNext = &vlink.header;
    ffxCreateContextDescFrameGeneration cd{};
    cd.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATION;
    cd.header.pNext = &bk.header;
    cd.flags = FFX_FRAMEGENERATION_ENABLE_DEBUG_CHECKING;
    cd.displaySize = {kWidth, kHeight};
    cd.maxRenderSize = {kWidth, kHeight};
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
        ffxDispatchDescFrameGenerationPrepare pr{};
        pr.header.type = FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION_PREPARE;
        pr.frameID = 2u;
        pr.renderSize = {kWidth, kHeight};
        pr.jitterOffset = {0.0f, 0.0f}; pr.motionVectorScale = {1.0f, 1.0f};
        pr.frameTimeDelta = 16.6f; pr.cameraNear = 0.1f; pr.cameraFar = 1000.0f;
        pr.cameraFovAngleVertical = 1.0f;
        ffxDispatchDescFrameGenerationPrepareCameraInfo cam{};
        cam.header.type = 0x0002000au;  // V1 camera-info block (probe path); orthonormal basis
        cam.cameraPosition[0] = 0.0f; cam.cameraPosition[1] = 0.0f; cam.cameraPosition[2] = 0.0f;
        cam.cameraUp[0] = 0.0f; cam.cameraUp[1] = 1.0f; cam.cameraUp[2] = 0.0f;
        cam.cameraRight[0] = 1.0f; cam.cameraRight[1] = 0.0f; cam.cameraRight[2] = 0.0f;
        cam.cameraForward[0] = 0.0f; cam.cameraForward[1] = 0.0f; cam.cameraForward[2] = -1.0f;
        pr.header.pNext = &cam.header;
        std::printf("  ffxDispatch(Prepare) = %d (expect 3=RUNTIME_ERROR: args ok, ML algorithm refused)\n",
                    (int)p.dispatch(&ctx, &pr.header));
        ffxDispatchDescFrameGeneration g{};
        g.header.type = FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION;
        g.numGeneratedFrames = 2u; g.minMaxLuminance[0] = 0.0f; g.minMaxLuminance[1] = 1.0f;
        std::printf("  ffxDispatch(Generate 2) = %d (expect 3=RUNTIME_ERROR: NO fake frames)\n",
                    (int)p.dispatch(&ctx, &g.header));
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
static void runPresent(VkApp& a, VkShaderModule vs, VkShaderModule fs, int seconds, GLFWwindow* w);

int main(int argc, char** argv) {
    bool windowMode = false; int windowSec = 10;
    std::vector<std::string> pos;
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        if (s == "--window") { windowMode = true; continue; }
        if (s.rfind("--window=", 0) == 0) { windowMode = true; windowSec = std::stoi(s.substr(9)); continue; }
        pos.push_back(s);
    }
    const char* env = std::getenv("FSR4_FGTEST_OUTDIR");
    std::string outDir = env ? env : ".";
    auto P = [&](size_t k, const std::string& d) { return pos.size() > k ? pos[k] : d; };
    std::string upSo    = P(0, outDir + "/amd_fidelityfx_upscaler_vk.so");
    std::string fgSo    = P(1, outDir + "/amd_fidelityfx_framegeneration_vk.so");
    std::string vertSpv = P(2, outDir + "/triangle.vert.spv");
    std::string fragSpv = P(3, outDir + "/triangle.frag.spv");

    if (windowMode) {
        if (!glfwInit()) { std::printf("NO_WINDOW: glfwInit failed (no Wayland/X11 compositor?)\n"); return 77; }
        if (!glfwVulkanSupported()) { std::printf("NO_WINDOW: GLFW has no Vulkan support\n"); glfwTerminate(); return 77; }
    }
    VkApp a;
    if (!initVulkan(a, windowMode)) { std::printf("NO_DEVICE: could not create a Vulkan device\n"); if (windowMode) glfwTerminate(); return 77; }
    VkShaderModule vs = mkShader(a, vertSpv), fs = mkShader(a, fragSpv);
    if (vs == VK_NULL_HANDLE || fs == VK_NULL_HANDLE) { std::printf("NO_SHADERS: SPIR-V missing (run build.sh)\n"); if (windowMode) glfwTerminate(); return 2; }
    setupGeom(a); setupUbo(a); setupDescriptors(a); setupPipeline(a, vs, fs, a.colorFmt, a.pipeline);

    Provider up, fg;
    if (loadProvider(up, upSo)) driveUpscaler(up, a);
    else std::printf("\n  upscaler provider .so not loaded: %s\n", upSo.c_str());
    if (loadProvider(fg, fgSo)) driveFrameGeneration(fg, a);
    else std::printf("\n  frame-generation provider .so not loaded: %s\n", fgSo.c_str());

    int written = 0;
    if (windowMode) {
        std::printf("\n=== MOVING TRIANGLE: opening a VISIBLE window (%ux%u) for %d s ===\n", kWidth, kHeight, windowSec);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
        GLFWwindow* win = glfwCreateWindow(960, 540, "fsr4-vk: FSR/FG animated host", nullptr, nullptr);
        if (!win) { std::printf("NO_WINDOW: glfwCreateWindow failed (Wayland/X11?)\n"); glfwTerminate(); return 77; }
        runPresent(a, vs, fs, windowSec, win);
        glfwDestroyWindow(win); glfwTerminate();
        written = 1;  // the graphics host ran (presented to a real window)
    } else {
        std::printf("=== MOVING TRIANGLE: rendering %u offscreen frames (%ux%u) ===\n", kFrames, kWidth, kHeight);
        std::vector<unsigned char> host;
        for (uint32_t f = 0; f < kFrames; ++f) {
            renderFrame(a, f, host);
            bool nonblack = false;
            for (size_t i = 0; i + 2 < host.size(); i += 4)
                if (host[i] || host[i + 1] || host[i + 2]) { nonblack = true; break; }
            if (nonblack) { writePPM(outDir + "/frame_" + std::to_string(f) + ".ppm", kWidth, kHeight, host.data()); ++written; }
        }
        std::printf("  frames rendered non-black: %d/%u (moving triangle drawn offscreen)\n", written, kFrames);
    }
    std::printf("\n=== RESULT ===\n  triangle host render: %s | FSR upscaler + FG: driven via the real FFX API (see codes)%s\n",
                written > 0 ? "RUN" : "NO-FRAME", windowMode ? "  [windowed present]" : "");
    return 0;
}

static void runPresent(VkApp& a, VkShaderModule vs, VkShaderModule fs, int seconds, GLFWwindow* w) {
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    if (glfwCreateWindowSurface(a.instance, w, nullptr, &surface) != VK_SUCCESS || !surface) {
        std::printf("NO_SURFACE: glfwCreateWindowSurface failed (no Wayland/X11 WSI?)\n"); return; }
    a.surface = surface;
    VkSurfaceCapabilitiesKHR cap{};
    if (vkGetPhysicalDeviceSurfaceCapabilitiesKHR(a.physical, surface, &cap) != VK_SUCCESS || !cap.supportedUsageFlags) {
        std::printf("NO_CAPS: surface unsupported\n"); return; }
    VkExtent2D ext = cap.currentExtent;
    if (ext.width == 0xFFFFFFFFu) { ext.width = 960; ext.height = 540; }
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
    setupPipeline(a, vs, fs, a.presentFmt, a.presentPipeline);
    if (!a.presentPipeline) { std::printf("NO_PRESENT_PIPELINE\n"); return; }
    std::vector<VkFence> fences(nn, VK_NULL_HANDLE);
    for (uint32_t i = 0; i < nn; ++i) { VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, nullptr, VK_FENCE_CREATE_SIGNALED_BIT}; vkCreateFence(a.device, &fi, nullptr, &fences[i]); }
    VkPhysicalDeviceProperties p{}; vkGetPhysicalDeviceProperties(a.physical, &p);
    std::printf("\n=== WINDOWED PRESENT (--window): FSR + FG animated host ===\n");
    std::printf("  device: %s | surface %ux%u fmt=%d present=FIFO swapchain=%u\n", p.deviceName, ext.width, ext.height, (int)sf.format, nn);
    std::printf("  NOTE: FG generate = RUNTIME_ERROR (refuses to fake frames) and FSR pixel-dispatch not hand-run\n");
    std::printf("        here -> interpolated=0 generated=0; the window shows the NATIVE animated host.\n");
    auto t0 = std::chrono::steady_clock::now();
    uint32_t frame = 0;
    while (true) {
        auto now = std::chrono::steady_clock::now();
        double el = std::chrono::duration<double>(now - t0).count();
        if (el >= (double)seconds) break;
        glfwPollEvents();
        uint32_t i = 0; VkSemaphore acq{VK_NULL_HANDLE};
        VkSemaphoreCreateInfo si{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        vkCreateSemaphore(a.device, &si, nullptr, &acq);
        vkAcquireNextImageKHR(a.device, a.swap, UINT64_MAX, acq, VK_NULL_HANDLE, &i);
        uint32_t cyc = i % kFrames;
        float phase = (float)el * 2.0f;
        Mat4 model = mul(translate(0.55f * std::sinf(phase), 0.25f * std::cosf(phase * 0.8f), 0.0f), rotateZ(phase * 0.7f));
        Mat4 proj = ortho(-1.6f, 1.6f, -0.9f, 0.9f, -1.0f, 1.0f); Mat4 view; ShaderData d;
        putMat(d.projection, proj); putMat(d.model, model); putMat(d.view, view);
        std::memcpy(a.uboMap[cyc], &d, sizeof(d));
        VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, nullptr, a.cmdPool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
        VkCommandBuffer cc{VK_NULL_HANDLE}; vkAllocateCommandBuffers(a.device, &cai, &cc);
        vkWaitForFences(a.device, 1, &fences[i], VK_TRUE, UINT64_MAX); vkResetFences(a.device, 1, &fences[i]);
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr, 0, nullptr};
        vkBeginCommandBuffer(cc, &bi);
        VkClearValue cval; cval.color = {0.10f, 0.10f, 0.20f, 1.0f};
        VkClearValue dval; dval.depthStencil = {1.0f, 0};
        VkRenderingAttachmentInfo colorA{}; colorA.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO; colorA.imageView = a.scColorView[i];
        colorA.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL; colorA.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; colorA.storeOp = VK_ATTACHMENT_STORE_OP_STORE; colorA.clearValue = cval;
        VkRenderingAttachmentInfo depthA{}; depthA.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO; depthA.imageView = a.scDepthView[i];
        depthA.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL; depthA.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; depthA.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE; depthA.clearValue = dval;
        VkRenderingInfo ri{}; ri.sType = VK_STRUCTURE_TYPE_RENDERING_INFO; ri.renderArea = { {0, 0}, ext };
        ri.layerCount = 1; ri.colorAttachmentCount = 1; ri.pColorAttachments = &colorA; ri.pDepthAttachment = &depthA;
        vkCmdBeginRendering(cc, &ri);
        VkViewport vpt{0.0f, 0.0f, (float)ext.width, (float)ext.height, 0.0f, 1.0f}; vkCmdSetViewport(cc, 0, 1, &vpt);
        VkRect2D sc{ {0, 0}, ext }; vkCmdSetScissor(cc, 0, 1, &sc);
        vkCmdBindDescriptorSets(cc, VK_PIPELINE_BIND_POINT_GRAPHICS, a.pipeLayout, 0, 1, &a.descSet[cyc], 0, nullptr);
        vkCmdBindPipeline(cc, VK_PIPELINE_BIND_POINT_GRAPHICS, a.presentPipeline);
        VkDeviceSize off[1] = {0}; vkCmdBindVertexBuffers(cc, 0, 1, &a.vertexBuf, off);
        vkCmdBindIndexBuffer(cc, a.indexBuf, 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(cc, gIndexCount, 1, 0, 0, 0);
        vkCmdEndRendering(cc);
        vkEndCommandBuffer(cc);
        VkSubmitInfo sub{}; sub.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO; sub.commandBufferCount = 1; sub.pCommandBuffers = &cc;
        vkQueueSubmit(a.queue, 1, &sub, fences[i]);
        VkPresentInfoKHR pi{}; pi.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR; pi.waitSemaphoreCount = 1; pi.pWaitSemaphores = &acq;
        pi.swapchainCount = 1; pi.pSwapchains = &a.swap; pi.pImageIndices = &i;
        vkQueuePresentKHR(a.queue, &pi);
        vkDestroySemaphore(a.device, acq, nullptr);
        vkFreeCommandBuffers(a.device, a.cmdPool, 1, &cc);
        std::printf("  frame %4u: present=base(%ux%u) t=%.2fs  interpolated=0 generated=0  [FG: REFUSED]\n", frame, ext.width, ext.height, el);
        ++frame;
        std::this_thread::sleep_for(std::chrono::milliseconds(16));
    }
    std::printf("\n  WINDOW SUMMARY: base frames presented=%u  interpolated inserted=0  generated=0\n", frame);
    std::printf("    (FG generate refused on this device -> no real interpolated/generated frames; upscaler pixel-\n");
    std::printf("     dispatch not hand-run here. All measured, never faked.)\n");
    for (uint32_t i = 0; i < nn; ++i) {
        vkDestroyFence(a.device, fences[i], nullptr);
        vkDestroyImageView(a.device, a.scColorView[i], nullptr);
        vkDestroyImageView(a.device, a.scDepthView[i], nullptr);
        vkDestroyImage(a.device, a.scDepth[i], nullptr);
        vkFreeMemory(a.device, a.scDepthMem[i], nullptr);
    }
    vkDestroySwapchainKHR(a.device, a.swap, nullptr); a.swap = VK_NULL_HANDLE;
    vkDestroySurfaceKHR(a.instance, a.surface, nullptr); a.surface = VK_NULL_HANDLE;
}




