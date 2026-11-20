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
// What the SHIPPED dispatch path still does not do: drive the interpolation/ML algorithm,
// so ffxDispatch(Prepare|generate|PrepareV2) keeps refusing with RUNTIME_ERROR and no frame
// is produced. What the runtime CAN now do is dispatchOne(): ONE real compute pass of ONE
// plan module, bound to the resources listed above and submitted to a queue, so the bind
// -> offset -> pipeline -> dispatch -> submit -> wait chain is MEASURED rather than assumed.
// A GPU memory-usage number is reportable because those resources really are allocated,
// and a dispatch is only believable for the same reason.
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
    // Measured outcome of ONE real dispatch. The difference between a substrate that
    // exists and a runtime that runs is exactly this: a pass bound to resources the plan
    // named, submitted to a queue, waited on, and checked. Every number here is the
    // driver's - the VkResult, the milliseconds between submit and idle, and whether
    // the storage image handed to the pass changed byte-for-byte across it.
    struct DispatchFacts {
        bool attempted = false;
        bool bound = false;
        bool submitted = false;
        bool ok = false;
        std::string module;
        std::string stage;
        std::string entry;
        std::uint32_t order = 0;
        std::uint32_t sets = 0;
        std::uint32_t bindings = 0;
        std::array<std::uint32_t, 3> workgroup{};
        std::array<std::uint32_t, 3> groups{};
        // How many times the SAME dispatch was issued inside this one command buffer (the
        // repeat experiment), and which dependency flags its barriers actually used.
        std::uint32_t repeats = 0;
        std::string barrierFlags;
        std::uint64_t poolBytes = 0;
        std::uint64_t descriptorBytesWritten = 0;
        std::uint64_t checksumBefore = 0;
        std::uint64_t checksumAfter = 0;
        bool outputChanged = false;
        // One number (ms) hides WHERE the time goes, and on this host the first measured
        // passes cost ~2000 ms for a 16x8 group dispatch, which is not GPU work. Split so
        // the reading is unambiguous: RECORD = driver-side command construction (begin..end,
        // including the inline vkCmdUpdateBuffer uploads), SUBMIT = queueing, WAIT = the
        // GPU actually running the pass and the read-backs. A 2 s in RECORD is a driver
        // cost that has nothing to do with the effect's frame time; a 2 s in WAIT is a real
        // shader cost. Measured numbers for all three are reported per pass.
        double msRecord = 0.0;
        double msSubmit = 0.0;
        double msWait = 0.0;
        double ms = 0.0;
        // A GPUVM fault / "context is lost. This context is guilty of a hard recovery" is
        // reported by the driver only in its stderr; this flag is how the CALLER learns the
        // device is dead and that further dispatches in this process are worthless. Measured:
        // after one faulting pass every later vkQueueSubmit on the same device returns
        // VK_ERROR_OTHER, which would otherwise read as "the driver cannot do this at all".
        bool contextLost = false;
        // The canvas the pass was actually run over (a dispatchOne parameter), next to the
        // plan's own largest allocated intermediate, so a cheap small-canvas measurement
        // can never be mistaken for a full-extent one and vice versa.
        std::uint32_t canvasWidth = 0;
        std::uint32_t canvasHeight = 0;
        std::uint32_t planMaxWidth = 0;
        std::uint32_t planMaxHeight = 0;
        // The format ACTUALLY bound at that canvas extent, in bytes per texel, and the read
        // -back size it therefore implies. Getting these wrong is not a wrong number but an
        // out-of-bounds vkCmdCopyImageToBuffer (measured: GPUVM fault), so both are reported
        // for the same reason the pool size is.
        std::uint32_t bytesPerTexel = 0;
        std::uint64_t readBackBytes = 0;
        // Which of the resources the pass was HANDED wrote anything at all. The first version
        // of this measurement watched only the storage image, and EVERY pass then reported
        // out_changed=no - a claim about one resource, not about the pass. The census is
        // therefore widened to the whole writable set: the storage image (outputChanged
        // below), the sampled image (a pass writing into its own INPUT is worth seeing), and
        // both buffer backs a STORAGE_BUFFER/UNIFORM_BUFFER slot points at. The backs were
        // zeroed by the SAME command buffer immediately before the pass, so a non-zero byte
        // afterwards is proof of a write with no before-copy needed.
        bool wroteSampledImage = false;
        std::uint64_t nonZeroStorageBufferBytes = 0;
        std::uint64_t nonZeroUniformBufferBytes = 0;
        bool anyWritableOutputChanged = false;
        // false here means the READ-BACK ITSELF was unavailable (no host-visible memory type,
        // or vkMapMemory refused) - which must never be read as "the pass wrote nothing".
        bool readbackAvailable = false;
        // How the pool was sized (the driver's own answer, or a computed fallback for
        // whole-array layouts) and what the pass was given in its push range. Both are
        // things a reader of the measurement needs in order to distrust the right parts.
        std::string note;
        std::string failure;
    };

    // Dispatch one built module by its PLAN ORDER (0..91 for the classic subset), over an
    // explicit CANVAS extent. A small canvas is not a shortcut: a 3840x2160 FP32 image is
    // 132 MB and its two read-backs 264 MB, and "faults only at full extent" is a different
    // finding from "faults everywhere". canvasWidth/Height of 0 asks for the default 128x128.
    //
    // dispatchRepeat issues that SAME dispatch N times inside ONE command buffer, separated
    // by vkCmdPipelineBarrier. It exists to answer a question a single dispatch cannot: a
    // measured ~2000 ms WAIT for an 8x8x1 group grid is not credible as shader cost, so
    // whether that time is PER-DISPATCH (which would make a 92-pass chain impossible to run
    // in a frame budget) or ONE-TIME first-execution cost (driver-side shader materialisation)
    // decides what the runtime has to be designed around. repeats is reported with the timing
    // split so the number can be divided correctly.
    bool dispatchOne(std::uint32_t planOrder, std::uint32_t canvasWidth, std::uint32_t canvasHeight,
                     std::uint32_t dispatchRepeat, DispatchFacts& facts, std::string& reason);

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
        std::string stage;
        std::string entry;
        std::uint32_t order = 0;        // the plan's own recorded execution order
        std::array<std::uint32_t, 3> workgroup{};
        std::uint32_t pushBytes = 0;  // the plan's own push range, needed again at dispatch time
        std::vector<ffgplan::BindingGroup> groups;
        std::vector<VkDescriptorSetLayout> sets;   // the plan's own per-set layouts, for a later bind
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

    // The descriptor-buffer CONSUMPTION surface, measured against THIS host's driver:
    // it implements VK_EXT_descriptor_buffer at spec version 1 (the 0.14.2 draft generation),
    // so binding a pool is vkCmdBindDescriptorBuffersEXT + vkCmdSetDescriptorBufferOffsetsEXT
    // and NOT the final-revision vkCmdPushPipelineDataEXT / vkCmdBindDescriptorRangesEXT, which
    // resolve to NULL in both instance and device scope here. These are resolved per call
    // (never assumed), and a missing one makes dispatchOne refuse with the name it lacked.
    PFN_vkGetDescriptorSetLayoutSizeEXT setLayoutSize_ = nullptr;
    PFN_vkGetDescriptorSetLayoutBindingOffsetEXT layoutBindingOffset_ = nullptr;
    // Kept so a later call can resolve the consumption commands through the CALLER's own
    // resolver (the route that works on this host: those commands exist at instance scope
    // and are the same ones the provider already resolves for vkGetDescriptorEXT).
    PFN_vkGetDeviceProcAddr resolve_ = nullptr;
    PFN_vkGetDeviceQueue getQueue_ = nullptr;
    PFN_vkCreateCommandPool createCommandPool_ = nullptr;
    PFN_vkAllocateCommandBuffers allocateCommandBuffers_ = nullptr;
    PFN_vkFreeCommandBuffers freeCommandBuffers_ = nullptr;
    PFN_vkCmdBindDescriptorBuffersEXT bindDescriptorBuffers_ = nullptr;
    PFN_vkCmdSetDescriptorBufferOffsetsEXT setDescriptorBufferOffsets_ = nullptr;
    PFN_vkCmdBindPipeline bindPipeline_ = nullptr;
    PFN_vkCmdDispatch dispatch_ = nullptr;
    PFN_vkCmdPipelineBarrier pipelineBarrier_ = nullptr;
    PFN_vkCmdClearColorImage clearImage_ = nullptr;
    PFN_vkCmdUpdateBuffer updateBuffer_ = nullptr;
    PFN_vkCmdCopyImageToBuffer copyImageToBuffer_ = nullptr;
    // The widened write census: reading back the buffer backs a STORAGE_BUFFER/UNIFORM_BUFFER
    // slot points at needs this, and "out_changed=no" for the storage IMAGE alone does not
    // mean the pass wrote nothing - it can have written into a buffer instead.
    PFN_vkCmdCopyBuffer copyBuffer_ = nullptr;
    PFN_vkCmdPushConstants pushConstants_ = nullptr;
    PFN_vkBeginCommandBuffer beginCommandBuffer_ = nullptr;
    PFN_vkEndCommandBuffer endCommandBuffer_ = nullptr;
    PFN_vkQueueSubmit queueSubmit_ = nullptr;
    PFN_vkQueueWaitIdle queueWaitIdle_ = nullptr;
    PFN_vkMapMemory mapMemory_ = nullptr;
    PFN_vkUnmapMemory unmapMemory_ = nullptr;
    PFN_vkDestroyCommandPool destroyCommandPool_ = nullptr;

    // The driver's own element sizes, kept from build() so a dispatch serializes descriptors
    // at the size the driver actually reported rather than a restated constant.
    bool sizesReady_ = false;
    VkDeviceSize elemStorageBuffer_ = 0;
    VkDeviceSize elemUniformBuffer_ = 0;
    VkDeviceSize elemCombinedImageSampler_ = 0;
    VkDeviceSize elemStorageImage_ = 0;
    VkDeviceSize elemSampler_ = 0;

    // Read in build() from the caller's physical device; dispatch uses it only to pick a
    // memory type the device really offers (and to verify a host mapping by DOING it).
    VkPhysicalDeviceMemoryProperties memoryProperties_{};
    bool haveMemoryProperties_ = false;

    VkDevice device_ = VK_NULL_HANDLE;
};

// Repo-internal measurement query, id 0x000200f0 - deliberately far outside the SDK's
// numbering, and the provider rejects unknown ids, so it cannot shadow a real ABI block.
// The provider answers it by running ONE real compute dispatch of ONE plan module on the
// substrate the provider itself built, and handing back the driver's numbers. That is how
// the bind-pool -> set-offsets -> bind-pipeline -> dispatch -> submit -> wait -> read-back
// chain gets measured on the device the provider chose, instead of on a second harness
// device that would prove nothing about this one. Not part of any shipped contract: the
// generate path still refuses until the algorithm exists.
constexpr ffxStructType_t kProbeDispatchOneType = kEffectId | 0xf0;
struct ProbeDispatchOneDesc {
    ffxQueryDescHeader header;
    std::uint32_t planOrder = 0u;  // which plan module to run (0..91 for the classic subset)
    // The canvas to run it over (0 = the default 128x128). Reported back through
    // DispatchFacts::canvasWidth/Height so the printed extent is unambiguous.
    std::uint32_t canvasWidth = 0u;
    std::uint32_t canvasHeight = 0u;
    // N copies of the SAME dispatch inside ONE command buffer (0 = one). Timing question this
    // answers, see PlanRuntime::dispatchOne's comment: per-dispatch cost or one-time cost.
    std::uint32_t dispatchRepeat = 0u;
    PlanRuntime::DispatchFacts* factsOut =
        nullptr;  // OUT: every number the driver reported for that one pass
};

}  // namespace ffgvk

