// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 fsr4-vk contributors.
// See LICENSE in the project root.
//
// Functional end-to-end selftest for capture-shim.cpp. It is NOT a capture of AMD's schedule (never fed to
// apply-capture-order.py); it drives a KNOWN synthetic compute dispatch ORDER through a real Vulkan device while
// the shim is LD_PRELOADED, then capture-shim-selftest.py asserts the shim reproduced that exact order + each
// pipeline's SPIR-V identity. This proves the shim FUNCTIONS (not merely compiles) on a real driver, which is what
// gates a future live AMD Frame Generation capture.
//
//   g++ -std=c++20 capture-shim-selftest.cpp -o selftest -lvulkan
//   LD_PRELOAD=libffxvk-capture.so FSR4_VK_CAPTURE_DIR=<dir> selftest a.spv b.spv c.spv
// Creates 3 distinct compute pipelines, records dispatches in order P0,P1,P2,P1,P0 (plus one dispatch on a
// never-bound command buffer that MUST be ignored), then exits WITHOUT submit (recording happens at record time).
#include <vulkan/vulkan.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>

static VkShaderModule mkMod(VkDevice d, const char* path) {
    FILE* f = std::fopen(path, "rb");
    if (!f) return VK_NULL_HANDLE;
    std::fseek(f, 0, SEEK_END); long n = std::ftell(f); std::fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> b((size_t)n);
    if (n) std::fread(b.data(), 1, n, f);
    std::fclose(f);
    VkShaderModuleCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    ci.codeSize = b.size(); ci.pCode = (const uint32_t*)b.data();
    VkShaderModule m = VK_NULL_HANDLE;
    if (vkCreateShaderModule(d, &ci, nullptr, &m) != VK_SUCCESS || !m) return VK_NULL_HANDLE;
    return m;
}

int main(int argc, char** argv) {
    if (argc < 4) { std::printf("usage: selftest a.spv b.spv c.spv\n"); return 2; }
    VkInstance in = VK_NULL_HANDLE;
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO, nullptr, "capself", 0, "capself", 0,
                          VK_MAKE_API_VERSION(0, 1, 3, 0)};
    VkInstanceCreateInfo ii{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; ii.pApplicationInfo = &app;
    if (vkCreateInstance(&ii, nullptr, &in) != VK_SUCCESS) { std::printf("NO_INSTANCE\n"); return 77; }
    uint32_t pc = 0; vkEnumeratePhysicalDevices(in, &pc, nullptr);
    if (!pc) { std::printf("NO_DEVICE\n"); return 77; }
    std::vector<VkPhysicalDevice> ps(pc); vkEnumeratePhysicalDevices(in, &pc, ps.data());
    VkPhysicalDevice ph = ps[0];
    VkDevice dev = VK_NULL_HANDLE;
    VkDeviceQueueCreateInfo q{}; q.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO; q.queueFamilyIndex = 0; q.queueCount = 1;
    VkDeviceCreateInfo di{}; di.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO; di.queueCreateInfoCount = 1; di.pQueueCreateInfos = &q;
    if (vkCreateDevice(ph, &di, nullptr, &dev) != VK_SUCCESS) { std::printf("NO_DEVICE\n"); return 77; }
    std::fprintf(stderr, "device_ok\n");

    VkPipelineLayout emptyLayout = VK_NULL_HANDLE;
    { VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
      vkCreatePipelineLayout(dev, &li, nullptr, &emptyLayout); }

    VkPipeline pl[3] = {VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE};
    for (int i = 0; i < 3; ++i) {
        VkShaderModule m = mkMod(dev, argv[i + 1]);
        if (!m) { std::fprintf(stderr, "module %d NULL\n", i); return 2; }
        VkPipelineShaderStageCreateInfo st{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0u,
                                            VK_SHADER_STAGE_COMPUTE_BIT, m, "main", nullptr};
        VkComputePipelineCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        ci.stage = st; ci.layout = emptyLayout;
        ci.basePipelineHandle = VK_NULL_HANDLE; ci.basePipelineIndex = -1;
        VkResult pr = vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &ci, nullptr, &pl[i]);
        std::fprintf(stderr, "pipeline %d rc=%d handle=%p\n", i, (int)pr, (void*)pl[i]);
        if (pr != VK_SUCCESS || !pl[i]) { std::fprintf(stderr, "PIPELINE_FAIL\n"); return 3; }
    }
    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandPoolCreateInfo pci{}; pci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    vkCreateCommandPool(dev, &pci, nullptr, &pool);
    VkCommandBufferAllocateInfo ai{}; ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool = pool; ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; ai.commandBufferCount = 1;
    VkCommandBuffer cb = VK_NULL_HANDLE; vkAllocateCommandBuffers(dev, &ai, &cb);
    VkCommandBufferBeginInfo bi{}; bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(cb, &bi);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl[0]); vkCmdDispatch(cb, 3, 4, 5);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl[1]); vkCmdDispatch(cb, 6, 1, 1);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl[2]); vkCmdDispatch(cb, 1, 1, 1);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl[1]); vkCmdDispatch(cb, 2, 2, 2);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl[0]); vkCmdDispatch(cb, 9, 8, 7);
    // dispatch on a NEVER-bound command buffer -> the ledger must IGNORE it (keeps the trace honest)
    VkCommandBuffer cb2 = VK_NULL_HANDLE; vkAllocateCommandBuffers(dev, &ai, &cb2);
    vkBeginCommandBuffer(cb2, &bi); vkCmdDispatch(cb2, 42, 42, 42);
    vkEndCommandBuffer(cb); vkEndCommandBuffer(cb2);
    std::printf("selftest_program_done\n");
    return 0;
}
