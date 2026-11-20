// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 fsr4-vk contributors.
// See LICENSE in the project root.

// Frame Generation Phase-1 runtime substrate (roadmap Phase 1, and the precondition
// that Phase 3's GPU-memory-usage honesty has always been waiting for).
//
// Built from the pack's OWN plan and NOTHING else:
//   * the plan's intermediate render targets, as real VkDeviceMemory allocations sized
//     to the HOST's render/display resolutions (the byte model plan.py derived);
//   * one descriptor pool sized by the DRIVER's own per-descriptor-element costs;
//   * a VkDescriptorSetLayout per declared set and a VkPipelineLayout (+ the pack's
//     32-byte push range) per module, then a real VkComputePipeline per module of the
//     selected subset, fed the pack's own SPIR-V through the context's namespaced
//     "fg4" pipeline cache;
//   * the descriptor-buffer bytes for every binding of every built module, serialized
//     with vkGetDescriptorEXT at the driver's measured element size.
//
// What it explicitly does NOT do: vkCmdDispatch. Nothing runs and no frame is produced,
// so ffxDispatch(Prepare|generate|PrepareV2) keeps refusing with RUNTIME_ERROR while a
// GPU memory-usage number can still be reported, because those resources really are
// allocated. A byte count stops being a fabrication exactly when it stops being a
// plan.py number and becomes an allocation the driver handed back.
//
// Every entry point is resolved through the caller's CreateBackendVkDesc::getDeviceProcAddr
// (never a global vk* wrapper), and every feature has to be enabled on the caller's device
// already - the execution contract refuses otherwise, and this runtime never enables
// anything on the caller's behalf.

#pragma once

#include "fg-plan.hpp"
#include "framegeneration.hpp"

#include <cstdint>
#include <string>
#include <vector>

#include <vulkan/vulkan.h>

namespace ffgvk {

// Measured outcome of one substrate build. Every number here is something a driver
// or the pack reported; the provider prints them and answers ffxQuery with two of them.
struct SubstrateFacts {
    // Plan side (what was read, before any device call)
    std::uint32_t planModules = 0;
    std::uint32_t classicModules = 0;
    std::uint32_t neuralModules = 0;
    std::uint32_t bindingGroups = 0;
    std::uint32_t unresolvedModules = 0;
    std::string orderSource;
    std::uint32_t renderWidth = 0;
    std::uint32_t renderHeight = 0;
    std::uint32_t displayWidth = 0;
    std::uint32_t displayHeight = 0;

    // Intermediates
    std::uint32_t targetsAllocated = 0;
    std::uint64_t targetBytes = 0;
    bool intermediatesComplete = false;  // every role the plan asked for, really allocated

    // Descriptor pool
    std::uint64_t descriptorPoolBytes = 0;
    bool poolAllocated = false;
    std::uint32_t scratchBackings = 0;
    bool descriptorCensusSkipped = false;
    std::uint32_t descriptorBindings = 0;
    std::uint32_t descriptorBindingsOk = 0;
    std::uint64_t descriptorBytesSerialized = 0;

    // Layouts / pipelines
    std::uint32_t setLayouts = 0;
    std::uint32_t pipelineLayouts = 0;
    std::uint32_t modulesAttempted = 0;
    std::uint32_t pipelinesCreated = 0;
    std::uint32_t pipelineFailures = 0;
    std::uint32_t modulesSkippedUnresolved = 0;
    std::uint32_t shaderModulesDestroyed = 0;

    // Reported usage. Valid only when intermediatesComplete && poolAllocated; anything
    // else keeps the query refused rather than reporting a partial number as if it were
    // the whole effect's footprint.
    std::uint64_t totalUsageInBytes = 0;
    std::uint64_t aliasableUsageInBytes = 0;
    bool usageReportable = false;

    std::string firstFailure;
    std::string firstFailureModule;
};

class PlanRuntime {
public:
    ~PlanRuntime();

    // Builds the substrate. Returns false (naming why in `reason`) when the plan
    // could not be read, when the caller's device exports none of the entry points
    // this needs, or when an allocation failed. `facts` keeps whatever the driver did
    // report, so a partial build is visible instead of being smoothed over.
    bool build(const CreateBackendVkDesc& backend,
               const GpuCapabilities& capabilities,
               const ffgplan::Plan& plan,
               std::uint32_t renderWidth,
               std::uint32_t renderHeight,
               std::uint32_t displayWidth,
               std::uint32_t displayHeight,
               bool includeNeuralSubset,
               VkPipelineCache pipelineCache,
               SubstrateFacts& facts,
               std::string& reason);

    // Releases everything this runtime created (pipelines, layouts, shader modules,
    // device memory). destroyContext calls it BEFORE serializing the pipeline cache,
    // so a warm start still reuses the driver's compiled state.
    void release();

private:
    struct Target {
        std::string role;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkDeviceAddress address = 0;
        std::uint64_t bytes = 0;
        std::uint32_t width = 0;
        std::uint32_t height = 0;
    };
    struct Layout {
        std::vector<VkDescriptorSetLayout> sets;
    };
    struct Built {
        std::string name;
        VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
        VkPipeline pipeline = VK_NULL_HANDLE;
    };

    std::vector<Target> targets_;
    std::vector<Layout> layouts_;
    std::vector<Built> built_;
    std::vector<VkDeviceMemory> allocations_;
    VkDeviceMemory pool_ = VK_NULL_HANDLE;

    // Census backing objects. vkGetDescriptorEXT dereferences the target it is asked to
    // describe, so a null imageView / zero address is not a polite probe but a driver
    // crash (measured: SIGSEGV inside libvulkan_radeon). The census therefore describes
    // REAL, deliberately tiny resources - 256 B buffers, one 4x4 sampled image, one 4x4
    // storage image, one sampler - that no pipeline is ever bound with and no work is
    // ever submitted against. They prove each binding KIND is populatable through the
    // descriptor-buffer transport and nothing more; release() destroys them first.
    VkBuffer scratchStorageBuffer_ = VK_NULL_HANDLE;
    VkBuffer scratchUniformBuffer_ = VK_NULL_HANDLE;
    VkImage scratchSampled_ = VK_NULL_HANDLE;
    VkImage scratchStorage_ = VK_NULL_HANDLE;
    VkImageView scratchSampledView_ = VK_NULL_HANDLE;
    VkImageView scratchStorageView_ = VK_NULL_HANDLE;
    VkSampler scratchSampler_ = VK_NULL_HANDLE;
    std::vector<VkDeviceMemory> scratchMemory_;
    VkDeviceAddress scratchStorageAddress_ = 0;
    VkDeviceAddress scratchUniformAddress_ = 0;
    bool scratchReady_ = false;

    PFN_vkCreateBuffer createBuffer_ = nullptr;
    PFN_vkDestroyBuffer destroyBuffer_ = nullptr;
    PFN_vkGetBufferDeviceAddress getBufferDeviceAddress_ = nullptr;
    PFN_vkGetBufferMemoryRequirements bufferMemoryRequirements_ = nullptr;
    PFN_vkBindBufferMemory bindBufferMemory_ = nullptr;
    PFN_vkCreateImage createImage_ = nullptr;
    PFN_vkDestroyImage destroyImage_ = nullptr;
    PFN_vkGetImageMemoryRequirements imageMemoryRequirements_ = nullptr;
    PFN_vkBindImageMemory bindImageMemory_ = nullptr;
    PFN_vkCreateImageView createImageView_ = nullptr;
    PFN_vkDestroyImageView destroyImageView_ = nullptr;
    PFN_vkCreateSampler createSampler_ = nullptr;
    PFN_vkDestroySampler destroySampler_ = nullptr;

    // Device-scope entry points, resolved once inside build().
    PFN_vkAllocateMemory allocateMemory_ = nullptr;
    PFN_vkFreeMemory freeMemory_ = nullptr;
    PFN_vkCreateDescriptorSetLayout createSetLayout_ = nullptr;
    PFN_vkDestroyDescriptorSetLayout destroySetLayout_ = nullptr;
    PFN_vkCreatePipelineLayout createPipelineLayout_ = nullptr;
    PFN_vkDestroyPipelineLayout destroyPipelineLayout_ = nullptr;
    PFN_vkCreateShaderModule createShaderModule_ = nullptr;
    PFN_vkDestroyShaderModule destroyShaderModule_ = nullptr;
    PFN_vkCreateComputePipelines createComputePipelines_ = nullptr;
    PFN_vkDestroyPipeline destroyPipeline_ = nullptr;
    PFN_vkGetDescriptorEXT getDescriptor_ = nullptr;

    VkDevice device_ = VK_NULL_HANDLE;
};

}  // namespace ffgvk

