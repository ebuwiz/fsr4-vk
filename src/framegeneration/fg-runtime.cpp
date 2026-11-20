// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 fsr4-vk contributors.
// See LICENSE in the project root.

#include "fg-runtime.hpp"

#include <algorithm>
#include <cstring>
#include <chrono>
#include <cstdint>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "assets.hpp"

namespace ffgvk {

namespace {

// The descriptor kinds the plan's grammar can name. A type the driver reports no
// size for is recorded as a hard blocker rather than guessed at: serializing a
// descriptor of the wrong size silently corrupts the pool.
struct DescriptorSizes {
    bool available = false;
    VkDeviceSize storageBuffer = 0;
    VkDeviceSize uniformBuffer = 0;
    VkDeviceSize combinedImageSampler = 0;
    VkDeviceSize storageImage = 0;
    VkDeviceSize sampler = 0;
    VkDeviceSize alignment = 0;
    VkDeviceSize pool = 0;
};

VkDeviceSize sizeFor(const DescriptorSizes& sizes, VkDescriptorType type) {
    switch (type) {
        case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER: return sizes.storageBuffer;
        case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER: return sizes.uniformBuffer;
        case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER: return sizes.combinedImageSampler;
        case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE: return sizes.storageImage;
        case VK_DESCRIPTOR_TYPE_SAMPLER: return sizes.sampler;
        default: return 0;
    }
}

const char* kindName(VkDescriptorType type) {
    switch (type) {
        case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER: return "SB";
        case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER: return "UB";
        case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER: return "CIS";
        case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE: return "SI";
        case VK_DESCRIPTOR_TYPE_SAMPLER: return "SMP";
        default: return "U";
    }
}

bool allSizesUsable(const DescriptorSizes& sizes) {
    return sizes.available && sizes.storageBuffer > 0 && sizes.uniformBuffer > 0 &&
           sizes.combinedImageSampler > 0 && sizes.storageImage > 0 && sizes.sampler > 0;
}

}  // namespace

// Which modules a Phase-1 build covers. The classic interpolation subset is the
// default; the neural (mlfi) subset is opt-in per call, so a device can report a
// substrate for the half that is not gated on the closed ML schedule without
// pretending the other half is anywhere near runnable.
bool selected(const std::string& stage, bool includeNeural) {
    if (stage == "compute-classic") {
        return true;
    }
    return includeNeural && stage == "compute-neural";
}

// Driver result spelled out, so a failure in the log names the VkResult instead of
// leaving a bare integer that every reader has to look up.
const char* vkResultName(VkResult result) {
    switch (result) {
        case VK_SUCCESS: return "VK_SUCCESS";
        case VK_ERROR_OUT_OF_HOST_MEMORY: return "VK_ERROR_OUT_OF_HOST_MEMORY";
        case VK_ERROR_OUT_OF_DEVICE_MEMORY: return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
        case VK_ERROR_TOO_MANY_OBJECTS: return "VK_ERROR_TOO_MANY_OBJECTS";
        case VK_ERROR_LAYER_NOT_PRESENT: return "VK_ERROR_LAYER_NOT_PRESENT";
        case VK_ERROR_MEMORY_MAP_FAILED: return "VK_ERROR_MEMORY_MAP_FAILED";
        case VK_ERROR_INCOMPATIBLE_DRIVER: return "VK_ERROR_INCOMPATIBLE_DRIVER";
        default: return "VK_ERROR_OTHER";
    }
}

PlanRuntime::~PlanRuntime() {
    release();
}

void PlanRuntime::release() {
    // Census backing objects first: they are leaf resources, and destroying them
    // before the pipelines keeps the teardown order independent of anything else.
    if (scratchSampledView_ && destroyImageView_) {
        destroyImageView_(device_, scratchSampledView_, nullptr);
        scratchSampledView_ = VK_NULL_HANDLE;
    }
    if (scratchStorageView_ && destroyImageView_) {
        destroyImageView_(device_, scratchStorageView_, nullptr);
        scratchStorageView_ = VK_NULL_HANDLE;
    }
    if (scratchSampler_ && destroySampler_) {
        destroySampler_(device_, scratchSampler_, nullptr);
        scratchSampler_ = VK_NULL_HANDLE;
    }
    if (scratchSampled_ && destroyImage_) {
        destroyImage_(device_, scratchSampled_, nullptr);
        scratchSampled_ = VK_NULL_HANDLE;
    }
    if (scratchStorage_ && destroyImage_) {
        destroyImage_(device_, scratchStorage_, nullptr);
        scratchStorage_ = VK_NULL_HANDLE;
    }
    if (scratchStorageBuffer_ && destroyBuffer_) {
        destroyBuffer_(device_, scratchStorageBuffer_, nullptr);
        scratchStorageBuffer_ = VK_NULL_HANDLE;
    }
    if (scratchUniformBuffer_ && destroyBuffer_) {
        destroyBuffer_(device_, scratchUniformBuffer_, nullptr);
        scratchUniformBuffer_ = VK_NULL_HANDLE;
    }
    for (auto memory : scratchMemory_) {
        if (memory && freeMemory_) {
            freeMemory_(device_, memory, nullptr);
        }
    }
    scratchMemory_.clear();
    scratchReady_ = false;
    // Pipelines first, then their layouts and set layouts, then the device memory.
    // The caller owns the device; nothing this runtime created may outlive it.
    for (auto& entry : built_) {
        if (entry.pipeline && destroyPipeline_) {
            destroyPipeline_(device_, entry.pipeline, nullptr);
            entry.pipeline = VK_NULL_HANDLE;
        }
        if (entry.pipelineLayout && destroyPipelineLayout_) {
            destroyPipelineLayout_(device_, entry.pipelineLayout, nullptr);
            entry.pipelineLayout = VK_NULL_HANDLE;
        }
    }
    built_.clear();
    for (auto& layout : layouts_) {
        for (auto& set : layout.sets) {
            if (set && destroySetLayout_) {
                destroySetLayout_(device_, set, nullptr);
                set = VK_NULL_HANDLE;
            }
        }
    }
    layouts_.clear();
    for (auto memory : allocations_) {
        if (memory && freeMemory_) {
            freeMemory_(device_, memory, nullptr);
        }
    }
    allocations_.clear();
    pool_ = VK_NULL_HANDLE;
    targets_.clear();
    device_ = VK_NULL_HANDLE;
}

bool PlanRuntime::build(const CreateBackendVkDesc& backend,
                        const GpuCapabilities& capabilities,
                        const ffgplan::Plan& plan,
                        std::uint32_t renderWidth,
                        std::uint32_t renderHeight,
                        std::uint32_t displayWidth,
                        std::uint32_t displayHeight,
                        bool includeNeuralSubset,
                        VkPipelineCache pipelineCache,
                        SubstrateFacts& facts,
                        std::string& reason) {
    reason.clear();
    facts = SubstrateFacts{};

    // ---------- what the pack's own tables say, counted not assumed ----------
    for (const auto& module : plan.modules) {
        if (module.stage == "compute-classic") {
            ++facts.classicModules;
        } else if (module.stage == "compute-neural") {
            ++facts.neuralModules;
        }
        if (!module.resolved) {
            ++facts.unresolvedModules;
        }
        facts.bindingGroups += static_cast<std::uint32_t>(module.groups.size());
    }
    facts.planModules = static_cast<std::uint32_t>(plan.modules.size());
    facts.orderSource = plan.facts.orderSource;
    facts.renderWidth = renderWidth;
    facts.renderHeight = renderHeight;
    facts.displayWidth = displayWidth;
    facts.displayHeight = displayHeight;

    // ---------- entry points: only the caller's resolver is ever asked ----------
    std::vector<std::string> missing;
    auto take = [&](const char* name, PFN_vkVoidFunction& slot) {
        slot = backend.getDeviceProcAddr
                   ? backend.getDeviceProcAddr(backend.device, name)
                   : nullptr;
        if (!slot) {
            missing.push_back(name);
        }
    };
    PFN_vkVoidFunction allocateRaw = nullptr;
    PFN_vkVoidFunction freeRaw = nullptr;
    PFN_vkVoidFunction setLayoutRaw = nullptr;
    PFN_vkVoidFunction destroySetRaw = nullptr;
    PFN_vkVoidFunction layoutRaw = nullptr;
    PFN_vkVoidFunction destroyLayoutRaw = nullptr;
    PFN_vkVoidFunction shaderRaw = nullptr;
    PFN_vkVoidFunction destroyShaderRaw = nullptr;
    PFN_vkVoidFunction pipelinesRaw = nullptr;
    PFN_vkVoidFunction destroyPipelinesRaw = nullptr;
    PFN_vkVoidFunction descriptorRaw = nullptr;
    take("vkAllocateMemory", allocateRaw);
    take("vkFreeMemory", freeRaw);
    take("vkCreateDescriptorSetLayout", setLayoutRaw);
    take("vkDestroyDescriptorSetLayout", destroySetRaw);
    take("vkCreatePipelineLayout", layoutRaw);
    take("vkDestroyPipelineLayout", destroyLayoutRaw);
    take("vkCreateShaderModule", shaderRaw);
    take("vkDestroyShaderModule", destroyShaderRaw);
    take("vkCreateComputePipelines", pipelinesRaw);
    take("vkDestroyPipeline", destroyPipelinesRaw);
    take("vkGetDescriptorEXT", descriptorRaw);
    if (!missing.empty()) {
        std::string joined;
        for (const auto& name : missing) {
            if (!joined.empty()) {
                joined += ", ";
            }
            joined += name;
        }
        reason = "caller device exports none of: " + joined +
                 " (the Frame Generation plan's whole-array bindings need the "
                 "VK_EXT_descriptor_buffer transport enabled on the host device)";
        facts.firstFailure = reason;
        return false;
    }
    device_ = backend.device;
    resolve_ = reinterpret_cast<PFN_vkGetDeviceProcAddr>(backend.getDeviceProcAddr);
    allocateMemory_ = reinterpret_cast<PFN_vkAllocateMemory>(allocateRaw);
    freeMemory_ = reinterpret_cast<PFN_vkFreeMemory>(freeRaw);
    createSetLayout_ = reinterpret_cast<PFN_vkCreateDescriptorSetLayout>(setLayoutRaw);
    destroySetLayout_ =
        reinterpret_cast<PFN_vkDestroyDescriptorSetLayout>(destroySetRaw);
    createPipelineLayout_ = reinterpret_cast<PFN_vkCreatePipelineLayout>(layoutRaw);
    destroyPipelineLayout_ =
        reinterpret_cast<PFN_vkDestroyPipelineLayout>(destroyLayoutRaw);
    createShaderModule_ = reinterpret_cast<PFN_vkCreateShaderModule>(shaderRaw);
    destroyShaderModule_ = reinterpret_cast<PFN_vkDestroyShaderModule>(destroyShaderRaw);
    createComputePipelines_ = reinterpret_cast<PFN_vkCreateComputePipelines>(pipelinesRaw);
    destroyPipeline_ = reinterpret_cast<PFN_vkDestroyPipeline>(destroyPipelinesRaw);
    getDescriptor_ = reinterpret_cast<PFN_vkGetDescriptorEXT>(descriptorRaw);

        // ---------- the driver's own descriptor-element costs ----------
    // Physical-level property reads follow the same route capabilities.cpp already
    // takes (the caller's loader). Reading is not enabling: a device that has not
    // enabled VK_EXT_descriptor_buffer reports zero sizes and the build refuses.
    DescriptorSizes sizes;
    if (backend.physicalDevice) {
        VkPhysicalDeviceDescriptorBufferPropertiesEXT db{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_BUFFER_PROPERTIES_EXT};
        VkPhysicalDeviceProperties2 props{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        props.pNext = &db;
        vkGetPhysicalDeviceProperties2(backend.physicalDevice, &props);
        sizes.storageBuffer = db.storageBufferDescriptorSize;
        sizes.uniformBuffer = db.uniformBufferDescriptorSize;
        sizes.combinedImageSampler = db.combinedImageSamplerDescriptorSize;
        sizes.storageImage = db.storageImageDescriptorSize;
        sizes.sampler = db.samplerDescriptorSize;
        sizes.available = true;
        // Kept for a later real dispatch, which must serialize at THESE sizes.
        sizesReady_ = true;
        elemStorageBuffer_ = sizes.storageBuffer;
        elemUniformBuffer_ = sizes.uniformBuffer;
        elemCombinedImageSampler_ = sizes.combinedImageSampler;
        elemStorageImage_ = sizes.storageImage;
        elemSampler_ = sizes.sampler;
    }
    if (!allSizesUsable(sizes)) {
        reason = "driver reports no descriptor-buffer element sizes "
                 "(VK_EXT_descriptor_buffer not enabled on the caller's device)";
        facts.firstFailure = reason;
        return false;
    }

    // ---------- a memory type read off the device, never assumed ----------
    VkPhysicalDeviceMemoryProperties memoryProperties{};
    vkGetPhysicalDeviceMemoryProperties(backend.physicalDevice, &memoryProperties);
    memoryProperties_ = memoryProperties;
    haveMemoryProperties_ = true;
    const bool wantAddress = capabilities.bufferDeviceAddress;
    auto allocate = [&](VkDeviceSize bytes, VkDeviceMemory* out) -> bool {
        *out = VK_NULL_HANDLE;
        if (bytes == 0) {
            return false;
        }
        VkMemoryAllocateFlagsInfo flags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
        flags.flags = wantAddress ? VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT : 0u;
        VkMemoryAllocateInfo info{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        info.pNext = wantAddress ? &flags : nullptr;
        info.allocationSize = bytes;
        info.memoryTypeIndex = 0xFFFFFFFFu;
        for (std::uint32_t index = 0; index < memoryProperties.memoryTypeCount; ++index) {
            if (memoryProperties.memoryTypes[index].propertyFlags &
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) {
                info.memoryTypeIndex = index;
                break;
            }
        }
        if (info.memoryTypeIndex == 0xFFFFFFFFu) {
            return false;
        }
        return allocateMemory_(device_, &info, nullptr, out) == VK_SUCCESS;
    };

    // ---------- the plan's intermediates, at the host's resolutions ----------
    const auto sized = ffgplan::sizeTargets(plan.facts, renderWidth, renderHeight,
                                           displayWidth, displayHeight);
    bool targetsOk = !sized.entries.empty();
    for (const auto& entry : sized.entries) {
        Target target;
        target.role = entry.role;
        target.width = entry.width;
        target.height = entry.height;
        target.bytes = entry.bytes;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        if (allocate(static_cast<VkDeviceSize>(entry.bytes), &memory)) {
            target.memory = memory;
            allocations_.push_back(memory);
            ++facts.targetsAllocated;
            targets_.push_back(std::move(target));
        } else {
            targetsOk = false;
            if (facts.firstFailure.empty()) {
                facts.firstFailure = "vkAllocateMemory refused the " + entry.role +
                                     " intermediate (" + std::to_string(entry.bytes) +
                                     " bytes)";
                facts.firstFailureModule = entry.role;
            }
        }
    }
    for (const auto& target : targets_) {
        facts.targetBytes += target.bytes;
    }
    facts.intermediatesComplete =
        targetsOk && targets_.size() == sized.entries.size();

    // ---------- one descriptor pool, sized from the driver's element costs ----------
    std::uint64_t poolBytes = 0;
    for (const auto& module : plan.modules) {
        if (!selected(module.stage, includeNeuralSubset)) {
            continue;
        }
        for (const auto& group : module.groups) {
            poolBytes += static_cast<std::uint64_t>(sizeFor(sizes, group.type));
        }
    }
    facts.descriptorPoolBytes = poolBytes;
    if (poolBytes > 0) {
        VkDeviceMemory memory = VK_NULL_HANDLE;
        if (allocate(static_cast<VkDeviceSize>(poolBytes), &memory)) {
            allocations_.push_back(memory);
            pool_ = memory;
            facts.poolAllocated = true;
        } else if (facts.firstFailure.empty()) {
            facts.firstFailure = "vkAllocateMemory refused the " +
                                 std::to_string(poolBytes) +
                                 " byte descriptor pool";
            facts.firstFailureModule = "descriptor-pool";
        }
    }

        // ---------- census backing objects: real resources, deliberately tiny ----------
    // vkGetDescriptorEXT dereferences whatever it is asked to describe, so the census
    // cannot probe with null handles (measured: the driver crashes inside its descriptor
    // serialization). It therefore describes 256-byte buffers, one 4x4 sampled image,
    // one 4x4 storage image, their views and one sampler - objects NO pipeline is ever
    // bound with and NO work is ever submitted against. The entry points for these are
    // resolved separately and OPTIONALLY: a host that lacks them loses the census
    // (counted and printed as skipped) instead of losing the allocations the reported
    // usage numbers are made of.
    {
        std::vector<std::string> scratchMissing;
        auto wantMember = [&](const char* name, auto& slot) {
            PFN_vkVoidFunction raw =
                backend.getDeviceProcAddr
                    ? backend.getDeviceProcAddr(backend.device, name)
                    : nullptr;
            // The cast target is the POINTED-FUNCTION type of the member, not a reference
            // to it: assigning through the reference is what the member declaration does.
            slot = reinterpret_cast<std::remove_reference_t<decltype(slot)>>(raw);
            if (!slot) {
                scratchMissing.push_back(name);
            }
        };
        wantMember("vkCreateBuffer", createBuffer_);
        wantMember("vkDestroyBuffer", destroyBuffer_);
        wantMember("vkGetBufferMemoryRequirements", bufferMemoryRequirements_);
        wantMember("vkBindBufferMemory", bindBufferMemory_);
        wantMember("vkGetBufferDeviceAddress", getBufferDeviceAddress_);
        wantMember("vkCreateImage", createImage_);
        wantMember("vkDestroyImage", destroyImage_);
        wantMember("vkGetImageMemoryRequirements", imageMemoryRequirements_);
        wantMember("vkBindImageMemory", bindImageMemory_);
        wantMember("vkCreateImageView", createImageView_);
        wantMember("vkDestroyImageView", destroyImageView_);
        wantMember("vkCreateSampler", createSampler_);
        wantMember("vkDestroySampler", destroySampler_);
        auto createScratchBacking = [&](SubstrateFacts& census) -> bool {
            auto makeBuffer = [&](VkBufferUsageFlagBits usage, VkBuffer& handle,
                                   VkDeviceAddress& address) -> bool {
                VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
                info.size = 256u;
                info.usage = usage;
                info.flags = 0u;
                info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
                if (createBuffer_(device_, &info, nullptr, &handle) != VK_SUCCESS) {
                    return false;
                }
                ++census.scratchBackings;
                VkMemoryRequirements requirements{};
                bufferMemoryRequirements_(device_, handle, &requirements);
                VkDeviceMemory memory = VK_NULL_HANDLE;
                if (!allocate(requirements.size, &memory)) {
                    return false;
                }
                scratchMemory_.push_back(memory);
                if (bindBufferMemory_(device_, handle, memory, 0u) != VK_SUCCESS) {
                    return false;
                }
                VkBufferDeviceAddressInfo addressInfo{
                    VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, nullptr, handle};
                address = getBufferDeviceAddress_(device_, &addressInfo);
                return address != 0u;
            };
            if (!makeBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                            scratchStorageBuffer_, scratchStorageAddress_)) {
                return false;
            }
            if (!makeBuffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                            scratchUniformBuffer_, scratchUniformAddress_)) {
                return false;
            }
            auto makeImage = [&](VkImageUsageFlagBits usage, VkImage& imageHandle,
                                  VkImageView& viewHandle) -> bool {
                VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
                info.imageType = VK_IMAGE_TYPE_2D;
                info.extent = {4u, 4u, 1u};
                info.mipLevels = 1u;
                info.arrayLayers = 1u;
                info.samples = VK_SAMPLE_COUNT_1_BIT;
                info.format = VK_FORMAT_R8G8B8A8_UNORM;
                info.tiling = VK_IMAGE_TILING_OPTIMAL;
                info.usage = usage;
                info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
                info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                if (createImage_(device_, &info, nullptr, &imageHandle) != VK_SUCCESS) {
                    return false;
                }
                ++census.scratchBackings;
                VkMemoryRequirements requirements{};
                imageMemoryRequirements_(device_, imageHandle, &requirements);
                VkDeviceMemory memory = VK_NULL_HANDLE;
                if (!allocate(requirements.size, &memory)) {
                    return false;
                }
                scratchMemory_.push_back(memory);
                if (bindImageMemory_(device_, imageHandle, memory, 0u) != VK_SUCCESS) {
                    return false;
                }
                VkImageViewCreateInfo view{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
                view.image = imageHandle;
                view.viewType = VK_IMAGE_VIEW_TYPE_2D;
                view.format = VK_FORMAT_R8G8B8A8_UNORM;
                view.components = {VK_COMPONENT_SWIZZLE_IDENTITY,
                                   VK_COMPONENT_SWIZZLE_IDENTITY,
                                   VK_COMPONENT_SWIZZLE_IDENTITY,
                                   VK_COMPONENT_SWIZZLE_IDENTITY};
                view.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0u, 1u, 0u, 1u};
                if (createImageView_(device_, &view, nullptr, &viewHandle) != VK_SUCCESS) {
                    return false;
                }
                ++census.scratchBackings;
                return true;
            };
            if (!makeImage(VK_IMAGE_USAGE_SAMPLED_BIT, scratchSampled_,
                           scratchSampledView_)) {
                return false;
            }
            if (!makeImage(VK_IMAGE_USAGE_STORAGE_BIT, scratchStorage_,
                           scratchStorageView_)) {
                return false;
            }
            VkSamplerCreateInfo sampler{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
            sampler.magFilter = VK_FILTER_NEAREST;
            sampler.minFilter = VK_FILTER_NEAREST;
            sampler.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
            sampler.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            sampler.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            sampler.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            sampler.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
            sampler.compareOp = VK_COMPARE_OP_NEVER;
            sampler.anisotropyEnable = VK_FALSE;
            sampler.maxAnisotropy = 1.0f;
            if (createSampler_(device_, &sampler, nullptr, &scratchSampler_) != VK_SUCCESS) {
                return false;
            }
            ++census.scratchBackings;
            scratchReady_ = true;
            return true;
        };
        if (!scratchMissing.empty()) {
            facts.descriptorCensusSkipped = true;
            if (facts.firstFailure.empty()) {
                std::string joined;
                for (const auto& name : scratchMissing) {
                    if (!joined.empty()) {
                        joined += ", ";
                    }
                    joined += name;
                }
                facts.firstFailure =
                    "census skipped, host exports none of: " + joined;
            }
        } else if (!createScratchBacking(facts)) {
            facts.descriptorCensusSkipped = true;
        }
    }

    // ---------- one layout + one real compute pipeline per module of the subset ----------
    // Bindings come from the plan's OWN de-duplicated census. A module the reader could
    // not resolve is skipped and counted, never bound with something invented.
    for (const auto& module : plan.modules) {
        if (!selected(module.stage, includeNeuralSubset)) {
            continue;
        }
        ++facts.modulesAttempted;
        if (!module.resolved) {
            ++facts.modulesSkippedUnresolved;
            continue;
        }
        std::uint32_t maxSet = 0;
        for (const auto& group : module.groups) {
            maxSet = group.set > maxSet ? group.set : maxSet;
        }
        Layout layout;
        layout.sets.assign(static_cast<std::size_t>(maxSet) + 1u, VK_NULL_HANDLE);
        bool layoutOk = true;
        for (std::uint32_t set = 0; set <= maxSet && layoutOk; ++set) {
            std::vector<VkDescriptorSetLayoutBinding> bindings;
            std::vector<VkDescriptorBindingFlags> flags;
            for (const auto& group : module.groups) {
                if (group.set != set) {
                    continue;
                }
                VkDescriptorSetLayoutBinding binding{};
                binding.binding = group.binding;
                binding.descriptorType = group.type;
                binding.descriptorCount = group.arrayed ? 0xFFFFFFFFu : 1u;
                binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
                bindings.push_back(binding);
                flags.push_back(group.arrayed ? VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT
                                              : 0u);
            }
            // A gap in the declared set indices is NOT a hole in the layout array: pSetLayouts
            // positions ARE set indices and EVERY entry must be a valid layout handle, so a
            // set with no bindings of its own gets an EMPTY layout - the same thing the SPIR-V's
            // own EMPTY-set declaration means - never VK_NULL_HANDLE.
            VkDescriptorSetLayoutCreateInfo info{
                VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
            // The transport these layouts exist for is the plan's: a layout that was not
            // created as a DESCRIPTOR_BUFFER layout has no defined size at all, and the
            // measurement that proved it is the one recorded in README - vkGetDescriptorSetLayoutSizeEXT
            // on a plain layout answered ~17 GB (a plausible-looking pile of uninitialized
            // bytes), which then turned into 67M vkCmdUpdateBuffer calls. Sizing questions
            // only have answers once the layout declares which transport it is for.
            info.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT;
            info.bindingCount = static_cast<std::uint32_t>(bindings.size());
            info.pBindings = bindings.empty() ? nullptr : bindings.data();
            VkDescriptorSetLayoutBindingFlagsCreateInfo extra{
                VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO};
            bool anyFlag = false;
            for (const auto flag : flags) {
                anyFlag = anyFlag || flag != 0u;
            }
            if (anyFlag) {
                extra.bindingCount = static_cast<std::uint32_t>(bindings.size());
                extra.pBindingFlags = flags.data();
                info.pNext = &extra;
            }
            if (createSetLayout_(device_, &info, nullptr, &layout.sets[set]) != VK_SUCCESS) {
                layoutOk = false;
                if (facts.firstFailure.empty()) {
                    facts.firstFailure = "vkCreateDescriptorSetLayout refused set " +
                                         std::to_string(set) + " of " + module.name;
                    facts.firstFailureModule = module.name;
                }
            } else {
                ++facts.setLayouts;
            }
        }
        layouts_.push_back(layout);
        if (!layoutOk) {
            ++facts.pipelineFailures;
            continue;
        }

        const auto spv = ffgassets::read(ffgassets::packRoot() / ("modules/" + module.name));
        if (spv.empty()) {
            ++facts.pipelineFailures;
            if (facts.firstFailure.empty()) {
                facts.firstFailure = "pack has no SPIR-V payload for " + module.name;
                facts.firstFailureModule = module.name;
            }
            continue;
        }
        VkShaderModule shader = VK_NULL_HANDLE;
        VkShaderModuleCreateInfo shaderInfo{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                            nullptr,
                                            0,
                                            static_cast<std::uint32_t>(spv.size()),
                                            reinterpret_cast<const std::uint32_t*>(spv.data())};
        if (createShaderModule_(device_, &shaderInfo, nullptr, &shader) != VK_SUCCESS) {
            ++facts.pipelineFailures;
            if (facts.firstFailure.empty()) {
                facts.firstFailure = "vkCreateShaderModule refused " + module.name;
                facts.firstFailureModule = module.name;
            }
            continue;
        }
        VkPipelineLayoutCreateInfo pipelineInfo{
            VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pipelineInfo.setLayoutCount = static_cast<std::uint32_t>(layout.sets.size());
        pipelineInfo.pSetLayouts = layout.sets.data();
        VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0u, module.pushBytes};
        pipelineInfo.pushConstantRangeCount = module.pushBytes != 0u ? 1u : 0u;
        pipelineInfo.pPushConstantRanges = module.pushBytes != 0u ? &push : nullptr;
        VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
        if (createPipelineLayout_(device_, &pipelineInfo, nullptr, &pipelineLayout) !=
            VK_SUCCESS) {
            destroyShaderModule_(device_, shader, nullptr);
            ++facts.shaderModulesDestroyed;
            ++facts.pipelineFailures;
            if (facts.firstFailure.empty()) {
                facts.firstFailure = "vkCreatePipelineLayout refused " + module.name;
                facts.firstFailureModule = module.name;
            }
            continue;
        }
        ++facts.pipelineLayouts;
        VkComputePipelineCreateInfo stage{};
        // Measured element sizes imply the driver has the descriptor-buffer transport
        // active, and then these pipelines must be created for it, exactly as the upscaler
        // runtime does: an unflagged pipeline against a descriptor-buffer layout is a
        // mismatch the driver is free to reject.
        stage.flags = sizes.available ? VK_PIPELINE_CREATE_DESCRIPTOR_BUFFER_BIT_EXT : 0u;
        stage.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        stage.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.stage.module = shader;
        stage.stage.pName = "main";
        stage.layout = pipelineLayout;
        VkPipeline pipeline = VK_NULL_HANDLE;
        const VkResult created =
            createComputePipelines_(device_, pipelineCache, 1u, &stage, nullptr, &pipeline);
        // The shader module is released as soon as its pipeline exists: the pipeline
        // keeps the driver's compiled state (and the context's namespaced "fg4" cache
        // lets a warm start reuse it), so holding every module alive would only cost
        // the caller VRAM it never asked for.
        destroyShaderModule_(device_, shader, nullptr);
        ++facts.shaderModulesDestroyed;
        if (created != VK_SUCCESS || pipeline == VK_NULL_HANDLE) {
            ++facts.pipelineFailures;
            if (facts.firstFailure.empty()) {
                facts.firstFailure =
                    std::string("vkCreateComputePipelines VkResult=") +
                    vkResultName(created) + " for " + module.name;
                facts.firstFailureModule = module.name;
            }
            continue;
        }
        ++facts.pipelinesCreated;
        built_.push_back({module.name, module.stage, module.entry, module.order, module.workgroup,
                          module.pushBytes, module.groups, layout.sets, pipelineLayout, pipeline});

    }

    // ---------- host-population census: one serialized descriptor per binding ----------
    // vkGetDescriptorEXT is asked to serialize each binding KIND at the driver's own
    // element size through the descriptor-buffer transport - the transport the plan says
    // a Frame Generation runtime must use. It is fed the scratch backing objects created
    // above (256 B buffers, 4x4 images, one sampler), because a driver that dereferences
    // its target crashes on a null handle: describing a REAL tiny resource is the honest
    // probe, and it still says nothing about any pipeline's eventual target - no write,
    // no bind, no queue, no dispatch happens here.
    std::vector<std::uint8_t> censusBuffer;
    if (!scratchReady_) {
        facts.descriptorCensusSkipped = true;
    }
    for (const auto& module : plan.modules) {
        if (!scratchReady_) {
            break;  // the census is skipped as a whole, and counted as skipped
        }
        if (!selected(module.stage, includeNeuralSubset) || !module.resolved) {
            continue;
        }
        bool builtModule = false;
        for (const auto& entry : built_) {
            builtModule = builtModule || entry.name == module.name;
        }
        if (!builtModule) {
            continue;
        }
        for (const auto& group : module.groups) {
            const auto element = sizeFor(sizes, group.type);
            if (element == 0) {
                continue;
            }
            ++facts.descriptorBindings;
            censusBuffer.assign(static_cast<std::size_t>(element), 0u);
            VkDescriptorGetInfoEXT get{VK_STRUCTURE_TYPE_DESCRIPTOR_GET_INFO_EXT, nullptr,
                                       group.type};
            // Both source objects are hoisted to a scope that outlives the call: the
            // driver reads them during serialization, so per-branch temporaries would
            // leave a dangling pointer (the crash shape the harness documents).
            VkDescriptorAddressInfoEXT address{
                VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT, nullptr, 0u, 256u,
                VK_FORMAT_UNDEFINED};
            VkDescriptorImageInfo image{};
            if (group.type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) {
                address.address = scratchStorageAddress_;
                get.data.pStorageBuffer = &address;
            } else if (group.type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER) {
                address.address = scratchUniformAddress_;
                get.data.pUniformBuffer = &address;
            } else if (group.type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) {
                image.sampler = scratchSampler_;
                image.imageView = scratchSampledView_;
                image.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                get.data.pCombinedImageSampler = &image;
            } else if (group.type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE) {
                image.sampler = VK_NULL_HANDLE;
                image.imageView = scratchStorageView_;
                image.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                get.data.pStorageImage = &image;
            } else if (group.type == VK_DESCRIPTOR_TYPE_SAMPLER) {
                get.data.pSampler = &scratchSampler_;
            } else {
                continue;  // a KIND the plan never maps to a size is counted, not guessed
            }
            // This call returns void: what is measured is whether the driver produced
            // real bytes for that KIND at its own element size. A zero-filled result is
            // counted separately instead of being reported as a successful serialization.
            getDescriptor_(device_, &get, static_cast<VkDeviceSize>(element),
                           censusBuffer.data());
            bool wroteBytes = false;
            for (const auto byte : censusBuffer) {
                if (byte != 0u) {
                    wroteBytes = true;
                    break;
                }
            }
            if (wroteBytes) {
                ++facts.descriptorBindingsOk;
                facts.descriptorBytesSerialized += static_cast<std::uint64_t>(element);
            } else if (facts.firstFailure.empty()) {
                facts.firstFailure =
                    "vkGetDescriptorEXT produced no bytes for a " +
                    std::string(kindName(group.type)) + " binding";
                facts.firstFailureModule = module.name;
            }
        }
    }

    // ---------- what the GPU-memory-usage query may therefore say ----------
    // Only real allocations the driver handed back are reported, and only once every
    // intermediate the plan asked for exists: a partial build never produces a partial
    // number dressed up as the effect's footprint.
    if (facts.intermediatesComplete && facts.poolAllocated) {
        facts.usageReportable = true;
        facts.totalUsageInBytes = facts.targetBytes + facts.descriptorPoolBytes;
        // The pool is the part a later runtime reuses across modules/frames, which is
        // exactly what aliasableUsageInBytes names in the SDK contract.
        facts.aliasableUsageInBytes = facts.descriptorPoolBytes;
    }
    if (!facts.targetsAllocated && reason.empty()) {
        reason = "no intermediate of the plan could be allocated";
    }
    return true;
}

// ---------------------------------------------------------------------------
// dispatchOne: the first REAL work this provider submits.
//
// Everything it binds is a resource the plan named: the pool the substrate already sized
// (per-set sizes asked of THIS driver for the plan's OWN set layouts), images at the plan's
// own intermediate extent, buffers for the SB/UB slots, a sampler. The commands are the ones
// this host's driver was measured to export - VK_EXT_descriptor_buffer at spec version 1 is
// the 0.14.2 draft generation, so binding a pool is vkCmdBindDescriptorBuffersEXT plus
// vkCmdSetDescriptorBufferOffsetsEXT, and the final-revision vkCmdPushPipelineDataEXT /
// vkCmdBindDescriptorRangesEXT resolve to NULL here; each name is resolved through the
// CALLER's resolver, because a device-scope lookup returns NULL for these commands even
// when the instance-scope one succeeds.
//
// What it does NOT claim: that the bound resources are the ones the pass means SEMANTICALLY.
// Slot kinds and arrayness come from the plan; WHICH intermediate each slot names is the
// naming phase. So an output that changes here proves the chain RUNS, not that a generated
// frame is correct - and the printed line says so.
bool PlanRuntime::dispatchOne(std::uint32_t planOrder, std::uint32_t canvasWidth,
                              std::uint32_t canvasHeight, std::uint32_t dispatchRepeat,
                              DispatchFacts& facts, std::string& reason) {
    facts = DispatchFacts{};
    if (!device_ || !sizesReady_) {
        reason = "no substrate built here: dispatch needs the plan's pipelines and the driver's "
                 "own descriptor element sizes";
        return false;
    }
    const Built* found = nullptr;
    for (const auto& entry : built_) {
        if (entry.order == planOrder) {
            found = &entry;
            break;
        }
    }
    if (found == nullptr) {
        reason = "no built module carries plan order " + std::to_string(planOrder) + " (this substrate built " +
                 std::to_string(built_.size()) + " of them)";
        return false;
    }
    facts.attempted = true;
    facts.module = found->name;
    facts.stage = found->stage;
    facts.entry = found->entry;
    facts.order = planOrder;
    facts.workgroup = found->workgroup;
    facts.sets = static_cast<std::uint32_t>(found->sets.size());
    facts.bindings = static_cast<std::uint32_t>(found->groups.size());

    // Resolve the consumption + submission surface by NAME, and refuse by NAME if this
    // driver's revision does not carry it. This is the measurement that separates the
    // draft generation (present here) from the final revision (absent here).
    std::vector<std::string> lacked;
    auto want = [&](const char* name, PFN_vkVoidFunction& slot) {
        if (slot != nullptr) {
            return;
        }
        slot = resolve_ ? resolve_(device_, name) : nullptr;
        if (slot == nullptr) {
            lacked.push_back(name);
        }
    };
    want("vkGetDescriptorSetLayoutSizeEXT", reinterpret_cast<PFN_vkVoidFunction&>(setLayoutSize_));
    want("vkGetDescriptorSetLayoutBindingOffsetEXT",
         reinterpret_cast<PFN_vkVoidFunction&>(layoutBindingOffset_));
    want("vkGetDeviceQueue", reinterpret_cast<PFN_vkVoidFunction&>(getQueue_));
    want("vkCreateCommandPool", reinterpret_cast<PFN_vkVoidFunction&>(createCommandPool_));
    want("vkAllocateCommandBuffers", reinterpret_cast<PFN_vkVoidFunction&>(allocateCommandBuffers_));
    want("vkFreeCommandBuffers", reinterpret_cast<PFN_vkVoidFunction&>(freeCommandBuffers_));
    want("vkDestroyCommandPool", reinterpret_cast<PFN_vkVoidFunction&>(destroyCommandPool_));
    want("vkBeginCommandBuffer", reinterpret_cast<PFN_vkVoidFunction&>(beginCommandBuffer_));
    want("vkEndCommandBuffer", reinterpret_cast<PFN_vkVoidFunction&>(endCommandBuffer_));
    want("vkCmdBindDescriptorBuffersEXT", reinterpret_cast<PFN_vkVoidFunction&>(bindDescriptorBuffers_));
    want("vkCmdSetDescriptorBufferOffsetsEXT", reinterpret_cast<PFN_vkVoidFunction&>(setDescriptorBufferOffsets_));
    want("vkCmdBindPipeline", reinterpret_cast<PFN_vkVoidFunction&>(bindPipeline_));
    want("vkCmdDispatch", reinterpret_cast<PFN_vkVoidFunction&>(dispatch_));
    want("vkCmdPipelineBarrier", reinterpret_cast<PFN_vkVoidFunction&>(pipelineBarrier_));
    want("vkCmdClearColorImage", reinterpret_cast<PFN_vkVoidFunction&>(clearImage_));
    want("vkCmdUpdateBuffer", reinterpret_cast<PFN_vkVoidFunction&>(updateBuffer_));
    want("vkCmdCopyImageToBuffer", reinterpret_cast<PFN_vkVoidFunction&>(copyImageToBuffer_));
    want("vkCmdCopyBuffer", reinterpret_cast<PFN_vkVoidFunction&>(copyBuffer_));
    want("vkQueueSubmit", reinterpret_cast<PFN_vkVoidFunction&>(queueSubmit_));
    want("vkQueueWaitIdle", reinterpret_cast<PFN_vkVoidFunction&>(queueWaitIdle_));
    want("vkMapMemory", reinterpret_cast<PFN_vkVoidFunction&>(mapMemory_));
    want("vkUnmapMemory", reinterpret_cast<PFN_vkVoidFunction&>(unmapMemory_));
    want("vkCmdPushConstants", reinterpret_cast<PFN_vkVoidFunction&>(pushConstants_));
    if (!lacked.empty()) {
        std::string joined;
        for (const auto& name : lacked) {
            if (!joined.empty()) {
                joined += ", ";
            }
            joined += name;
        }
        reason = "driver exports none of: " + joined + " - VK_EXT_descriptor_buffer at spec version 1 is "
                 "the 0.14.2 draft generation, and this pinned SDK targets it; the final-revision names "
                 "are not this driver's API";
        return false;
    }
    // ---- the plan's own largest allocated intermediate is RECORDED; the pass runs over an
    // explicit canvas the CALLER named (default 128x128), never an invented extent ----
    // Two reasons this is a parameter and not simply the plan's 3840x2160: the whole point of
    // the measurement is a CHEAP, defined read-back (a full-extent FP32 image is 132 MB, its
    // two read-back buffers 264 MB), and a pass that faults only when the images are big is a
    // different finding from one that faults at any extent. Both extents are reported so the
    // reader can tell which claim is being made.
    std::uint32_t planMaxWidth = 0;
    std::uint32_t planMaxHeight = 0;
    for (const auto& target : targets_) {
        if (target.width > planMaxWidth) {
            planMaxWidth = target.width;
        }
        if (target.height > planMaxHeight) {
            planMaxHeight = target.height;
        }
    }
    if (planMaxWidth == 0 || planMaxHeight == 0) {
        reason = "no allocated intermediate extent to report alongside a pass";
        return false;
    }
    facts.planMaxWidth = planMaxWidth;
    facts.planMaxHeight = planMaxHeight;
    const std::uint32_t width = canvasWidth != 0u ? canvasWidth : 128u;
    const std::uint32_t height = canvasHeight != 0u ? canvasHeight : 128u;
    // Below one workgroup of the smallest class the plan declares (4x4) a dispatch would cover
    // no thread at all and "ran" would be vacuous; above 8192 the read-back stops being cheap
    // enough to repeat per module. Neither bound is a guess about the shader.
    if (width < 16u || height < 16u || width > 8192u || height > 8192u) {
        reason = "canvas " + std::to_string(width) + "x" + std::to_string(height) +
                 " is outside the 16..8192 band this measurement is defined for";
        return false;
    }
    facts.canvasWidth = width;
    facts.canvasHeight = height;

    // ---- only memory types this device reports; a host mapping is PROVEN by doing it ----
    auto pickType = [&](const VkMemoryRequirements& req, VkMemoryPropertyFlags flags) -> std::uint32_t {
        for (std::uint32_t index = 0; index < memoryProperties_.memoryTypeCount; ++index) {
            if ((req.memoryTypeBits & (1u << index)) &&
                (memoryProperties_.memoryTypes[index].propertyFlags & flags) == flags) {
                return index;
            }
        }
        return 0xFFFFFFFFu;
    };

    std::vector<VkDeviceMemory> frees;
    std::vector<VkBuffer> buffers;
    std::vector<VkImage> images;
    std::vector<VkImageView> views;
    VkSampler sampler = VK_NULL_HANDLE;
    VkCommandPool commandPool = VK_NULL_HANDLE;
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    auto fail = [&](const std::string& text) {
        facts.failure = text;
        return false;
    };

    // ---- the pool, sized by THIS driver for the plan's OWN set layouts ----
    std::vector<VkDeviceSize> setSizes(found->sets.size(), 0);
    std::vector<VkDeviceSize> setOffsets(found->sets.size(), 0);
    std::uint64_t poolBytes = 0;
    // The driver answers the size of each of the plan's own set layouts - but only for
    // a layout that declared which transport it is for, and only with a number that
    // makes sense. A layout with a VK_WHOLE_ARRAY binding has no bounded size at all
    // (that is what a runtime-sized array MEANS), so what sizes such a set is the
    // offset+element the plan's own bindings need, which is asked of the driver too.
    // Both routes are recorded, because which one produced a number changes what the
    // measurement is allowed to claim.
    const VkDeviceSize plausibleCeiling = 1u << 20;  // 1 MiB per set: far above any set the plan declares
    for (std::size_t set = 0; set < found->sets.size(); ++set) {
        if (found->sets[set] == VK_NULL_HANDLE) {
            continue;
        }
        VkDeviceSize answered = 0;
        setLayoutSize_(device_, found->sets[set], &answered);
        VkDeviceSize computed = 0;
        std::uint32_t declared = 0;
        for (const auto& group : found->groups) {
            if (group.set != set) {
                continue;
            }
            ++declared;
            VkDeviceSize element = 0;
            if (group.type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) {
                element = elemStorageBuffer_;
            } else if (group.type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER) {
                element = elemUniformBuffer_;
            } else if (group.type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) {
                element = elemCombinedImageSampler_;
            } else if (group.type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE) {
                element = elemStorageImage_;
            } else if (group.type == VK_DESCRIPTOR_TYPE_SAMPLER) {
                element = elemSampler_;
            }
            if (element == 0) {
                continue;
            }
            VkDeviceSize offset = 0;
            layoutBindingOffset_(device_, found->sets[set], group.binding, &offset);
            if (offset + element > computed) {
                computed = offset + element;
            }
        }
        if (declared == 0) {
            // The plan declares this SET INDEX but no binding of this module lives in it, which
            // is exactly the EMPTY-layout case the substrate had to create so the pSetLayouts
            // positions line up with set indices (gaps in the set sequence are not holes). A
            // set with nothing in it has size 0, and the driver agrees by answering 0. Treating
            // that answer as a failure - the shape of my first attempt here - turned a legal
            // empty set into "no bounded size for set 1" and no pass ever ran.
            facts.note += "set" + std::to_string(set) + "=empty ";
            continue;
        }
        // What the driver answers for a layout whose binding is a RUNTIME-SIZED ARRAY is not
        // 0 but a WRAPPED number: measured 4294967232 = 2^32 - 64 for a 64-byte element, i.e.
        // (uint32)-1 multiplied by the element size. That IS the "no bounded size" answer, and
        // the ceiling check is what keeps it from becoming a 16 GiB pool; the per-binding
        // offset + element route (also the driver's own numbers) is the bounded one. Where
        // both are bounded the LARGER is used, and which route produced the number is always
        // recorded, because it changes what the measurement is allowed to claim.
        if (computed == 0 || computed > plausibleCeiling) {
            return fail("no bounded size for set " + std::to_string(set) + " of " + facts.module +
                        " (" + std::to_string(declared) + " binding(s) declared; driver answered " +
                        std::to_string(answered) + "B, computed " + std::to_string(computed) + "B)");
        }
        const bool answeredBounded = answered != 0 && answered <= plausibleCeiling;
        if (answeredBounded && answered > computed) {
            setSizes[set] = answered;
            facts.note += "set" + std::to_string(set) + "=max(driver=" + std::to_string(answered) +
                          "B,computed=" + std::to_string(computed) + "B) ";
        } else if (answeredBounded) {
            setSizes[set] = computed;
            facts.note += "set" + std::to_string(set) + "=driver(" + std::to_string(answered) + "B) ";
        } else {
            setSizes[set] = computed;
            facts.note += "set" + std::to_string(set) + "=computed(" + std::to_string(computed) +
                          "B;driver_answered=" + std::to_string(answered) + "B=wrapped_runtime_array) ";
        }
        poolBytes += setSizes[set];
    }
    if (poolBytes == 0) {
        return fail("vkGetDescriptorSetLayoutSizeEXT reported no size for this module's sets");
    }
    if (poolBytes > plausibleCeiling) {
        return fail("this module's sets need " + std::to_string(poolBytes) + "B of pool, past the " +
                    std::to_string(plausibleCeiling) + "B band any set the plan declares can be");
    }
    // vkCmdUpdateBuffer (the inline upload this revision can do without mapping device
    // memory) requires a byte offset and size that are multiples of 4. Every element size the
    // driver reported for the plan's binding KINDS is 16/32/64 B, so a set size that is NOT
    // a multiple of 4 means the sizing came from somewhere other than those element sizes,
    // and the upload would be an invalid command. Refusing is the measurement; the other
    // route (vkCmdCopyBuffer from a host-staged buffer) is recorded in README as the way to
    // go if a future plan ever produces such a set.
    for (std::size_t set = 0; set < found->sets.size(); ++set) {
        if (setSizes[set] != 0 && (setSizes[set] % 4u) != 0) {
            return fail("set " + std::to_string(set) + " sized " + std::to_string(setSizes[set]) +
                        "B is not a multiple of 4, which no element size this driver reported can produce");
        }
    }
    facts.poolBytes = poolBytes;
    {
        VkDeviceSize running = 0;
        for (std::size_t set = 0; set < found->sets.size(); ++set) {
            setOffsets[set] = running;
            running += setSizes[set];
        }
        if (running != poolBytes) {
            return fail("per-set offsets sum to " + std::to_string(running) +
                        "B but the pool was sized " + std::to_string(poolBytes) + "B");
        }
    }
    // Those offsets index INTO the buffer created further down, so the buffer is sized FROM
    // them (rounded up to a page) and the relationship is CHECKED, not assumed: a buffer
    // smaller than the offsets recorded against it is precisely the shape that turns a
    // legal bind into a descriptor read outside every mapping (measured: GPUVM fault +
    // "the CS has been cancelled because the context is lost").
    //
    // The slack is for the RUNTIME-SIZED ARRAY sets the plan records: the ARRAYNESS of a
    // slot is known, its LENGTH is not, so a pass can index further into the pool than the
    // one descriptor serialized here occupies. Padding the allocation keeps that read INSIDE
    // the mapping (it lands on zero bytes) instead of turning a plumbing measurement into a
    // GPUVM fault that says nothing about either side. The slack is REPORTED in the same
    // line as the sizes, because it is a property of the measurement, not of the effect.
    const VkDeviceSize poolSlackBytes = 8192;
    const VkDeviceSize poolAllocBytes =
        (poolBytes + poolSlackBytes + 255u) & ~static_cast<VkDeviceSize>(255u);
    facts.note += "poolSlack=" + std::to_string(poolAllocBytes - poolBytes) + "B=zero(unbound_array_slots) ";
    // ---- the pool (at poolAllocBytes) and the 256 KiB buffer backs a SB/UB slot needs ----
    struct Resource {
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkDeviceAddress address = 0;
    };
    const VkDeviceSize backingBytes = 262144;  // 256 KiB: a pass may index a slot well past one page
    auto makeBuffer = [&](VkBufferUsageFlags usage,
                          VkDeviceSize size) -> Resource {
        Resource made;
        VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        info.size = size;
        info.usage = usage;
        info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if (createBuffer_(device_, &info, nullptr, &made.buffer) != VK_SUCCESS) {
            made.buffer = VK_NULL_HANDLE;
            return made;
        }
        buffers.push_back(made.buffer);
        VkMemoryRequirements req{};
        bufferMemoryRequirements_(device_, made.buffer, &req);
        VkMemoryAllocateFlagsInfo flags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
        flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
        VkMemoryAllocateInfo allocate{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocate.pNext = &flags;
        allocate.allocationSize = req.size;
        allocate.memoryTypeIndex = pickType(req, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (allocate.memoryTypeIndex == 0xFFFFFFFFu ||
            allocateMemory_(device_, &allocate, nullptr, &made.memory) != VK_SUCCESS) {
            made.buffer = VK_NULL_HANDLE;
            return made;
        }
        frees.push_back(made.memory);
        if (bindBufferMemory_(device_, made.buffer, made.memory, 0u) != VK_SUCCESS) {
            made.buffer = VK_NULL_HANDLE;
            return made;
        }
        VkBufferDeviceAddressInfo addressInfo{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, nullptr, made.buffer};
        made.address = getBufferDeviceAddress_(device_, &addressInfo);
        return made;
    };
    struct Surface {
        VkImage image = VK_NULL_HANDLE;
        VkImageView view = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
    };
    auto makeImage = [&](VkFormat format, VkImageUsageFlags usage) -> Surface {
        Surface made;
        VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        info.imageType = VK_IMAGE_TYPE_2D;
        info.format = format;
        info.extent = {width, height, 1u};
        info.mipLevels = 1u;
        info.arrayLayers = 1u;
        info.samples = VK_SAMPLE_COUNT_1_BIT;
        info.tiling = VK_IMAGE_TILING_OPTIMAL;
        info.usage = usage;
        info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        if (createImage_(device_, &info, nullptr, &made.image) != VK_SUCCESS) {
            return made;
        }
        images.push_back(made.image);
        VkMemoryRequirements req{};
        imageMemoryRequirements_(device_, made.image, &req);
        VkMemoryAllocateInfo allocate{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocate.allocationSize = req.size;
        allocate.memoryTypeIndex = pickType(req, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (allocate.memoryTypeIndex == 0xFFFFFFFFu ||
            allocateMemory_(device_, &allocate, nullptr, &made.memory) != VK_SUCCESS) {
            made.image = VK_NULL_HANDLE;
            return made;
        }
        frees.push_back(made.memory);
        if (bindImageMemory_(device_, made.image, made.memory, 0u) != VK_SUCCESS) {
            made.image = VK_NULL_HANDLE;
            return made;
        }
        VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        viewInfo.image = made.image;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = format;
        viewInfo.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0u, 1u, 0u, 1u};
        if (createImageView_(device_, &viewInfo, nullptr, &made.view) != VK_SUCCESS) {
            made.image = VK_NULL_HANDLE;
            return made;
        }
        views.push_back(made.view);
        return made;
    };
    // ---- the objects the plan's slots will actually be bound to ----
    // The pool buffer gets the size the offsets above were built from; the two buffer backs
    // keep the 256 KiB backing a SB/UB slot may index into. Same call, different sizes -
    // which is the point: a pool is not a backing store, and sizing one by the other's
    // constant is what made a legal bind read descriptors outside every mapping.
    Resource poolBuffer = makeBuffer(VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                                     VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                     VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                     poolAllocBytes);
    // TRANSFER_SRC on the BACKS is not decoration: the census copies FROM them, and a
    // buffer created without that usage cannot legally be a copy source (the driver is
    // free to accept it and produce nothing, which would read as "the pass wrote nothing").
    Resource storageBacking =
        makeBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                       VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                   backingBytes);
    Resource uniformBacking =
        makeBuffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                       VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                   backingBytes);
    if (!poolBuffer.buffer || !storageBacking.buffer || !uniformBacking.buffer) {
        return fail("could not create the pool or the buffer backs a SB/UB slot needs");
    }
    // The canvas format is named once and its texel cost derived from IT, because the read
    // -back buffer below must be sized by the same number: allocating w*h*4 for an RGBA
    // FP32 image (16 B/texel) makes vkCmdCopyImageToBuffer write 4x past the allocation,
    // which is a GPUVM fault with nothing wrong in the compute pass itself - the shape of
    // the first failure this measurement produced. FP32 is chosen because it is the
    // widest of the plan's own texel costs (4/8/12/16 B), so a pass that writes <= 4
    // components of <= 32-bit floats cannot land outside the image by format alone.
    const VkFormat canvasFormat = VK_FORMAT_R32G32B32A32_SFLOAT;
    const std::uint32_t bytesPerTexel = 16u;
    facts.bytesPerTexel = bytesPerTexel;
    Surface sampled = makeImage(canvasFormat,
                                VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                                    VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    Surface storage = makeImage(canvasFormat,
                                VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                                    VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    if (!sampled.image || !storage.image) {
        return fail("could not create the sampled/storage images at the plan's extent");
    }
    {
        VkSamplerCreateInfo info{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        info.magFilter = VK_FILTER_NEAREST;
        info.minFilter = VK_FILTER_NEAREST;
        info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        info.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
        info.compareOp = VK_COMPARE_OP_NEVER;
        info.anisotropyEnable = VK_FALSE;
        info.maxAnisotropy = 1.0f;
        if (createSampler_(device_, &info, nullptr, &sampler) != VK_SUCCESS || sampler == VK_NULL_HANDLE) {
            return fail("could not create the sampler an image-sampler slot needs");
        }
    }

    // ---- read-back census: EVERY writable resource the pass was handed, not just one ----
    // Order is fixed and documented, because the code below addresses these by index:
    //   [0] storage image BEFORE   [1] storage image AFTER
    //   [2] sampled image AFTER    [3] STORAGE_BUFFER backing AFTER
    //   [4] UNIFORM_BUFFER backing AFTER
    // The sampled image and both backs were CLEARED/zeroed by the same command buffer
    // immediately before the dispatch, so no "before" copy is needed for them: a non-zero
    // byte afterwards is proof the pass wrote there.
    struct HostReadback {
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        void* map = nullptr;
        VkDeviceSize bytes = 0;
    };
    std::vector<HostReadback> readbacks;
    bool readbackOk = true;
    const VkDeviceSize imageBytes = static_cast<VkDeviceSize>(width) * height * bytesPerTexel;
    auto makeReadback = [&](VkDeviceSize size) {
        HostReadback made;
        VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        info.size = size;
        info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if (createBuffer_(device_, &info, nullptr, &made.buffer) != VK_SUCCESS || !made.buffer) {
            return false;
        }
        buffers.push_back(made.buffer);
        VkMemoryRequirements req{};
        bufferMemoryRequirements_(device_, made.buffer, &req);
        made.bytes = req.size;
        VkMemoryAllocateInfo allocate{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocate.allocationSize = req.size;
        allocate.memoryTypeIndex =
            pickType(req, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (allocate.memoryTypeIndex == 0xFFFFFFFFu ||
            allocateMemory_(device_, &allocate, nullptr, &made.memory) != VK_SUCCESS) {
            made.memory = VK_NULL_HANDLE;
            return false;
        }
        frees.push_back(made.memory);
        if (bindBufferMemory_(device_, made.buffer, made.memory, 0u) != VK_SUCCESS) {
            made.memory = VK_NULL_HANDLE;
            return false;
        }
        void* mapped = nullptr;
        if (mapMemory_(device_, made.memory, 0u, req.size, 0u, &mapped) != VK_SUCCESS || mapped == nullptr) {
            // A mapping is PROVEN by DOING it; a host-visible memory type that refuses to map
            // is counted as "census unavailable", never as "the pass wrote nothing".
            made.memory = VK_NULL_HANDLE;
            return false;
        }
        made.map = mapped;
        readbacks.push_back(made);
        return true;
    };
    readbackOk = makeReadback(imageBytes) && makeReadback(imageBytes) && makeReadback(imageBytes) &&
                  makeReadback(backingBytes) && makeReadback(backingBytes);
    if (readbackOk) {
        facts.readBackBytes = imageBytes * 3u + backingBytes * 2u;
        facts.readbackAvailable = true;
    } else {
        readbacks.clear();
        facts.note += "readback=unavailable(host_visible_or_map_refused) ";
    }
    // ---- host-serialize every binding of every set, at THIS driver's own stub offsets ----
    std::vector<std::vector<std::uint8_t>> hostSets(found->sets.size());
    for (std::size_t set = 0; set < found->sets.size(); ++set) {
        if (setSizes[set] == 0) {
            continue;
        }
        hostSets[set].assign(static_cast<std::size_t>(setSizes[set]), 0u);
        for (const auto& group : found->groups) {
            if (group.set != set) {
                continue;
            }
            VkDeviceSize element = 0;
            if (group.type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) {
                element = elemStorageBuffer_;
            } else if (group.type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER) {
                element = elemUniformBuffer_;
            } else if (group.type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) {
                element = elemCombinedImageSampler_;
            } else if (group.type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE) {
                element = elemStorageImage_;
            } else if (group.type == VK_DESCRIPTOR_TYPE_SAMPLER) {
                element = elemSampler_;
            }
            if (element == 0) {
                continue;
            }
            VkDeviceSize offset = 0;
            layoutBindingOffset_(device_, found->sets[set], group.binding, &offset);
            if (offset + element > setSizes[set]) {
                return fail("driver placed a binding offset past its own set size for " + facts.module);
            }
            VkDescriptorGetInfoEXT get{VK_STRUCTURE_TYPE_DESCRIPTOR_GET_INFO_EXT, nullptr, group.type};
            // Hoisted out of the branch on purpose: the driver reads these during serialization,
            // and a per-branch temporary would leave a dangling pointer - the crash shape this
            // transport is documented to have.
            VkDescriptorAddressInfoEXT address{VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT, nullptr, 0u,
                                               backingBytes, VK_FORMAT_UNDEFINED};
            VkDescriptorImageInfo image{};
            if (group.type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) {
                address.address = storageBacking.address;
                get.data.pStorageBuffer = &address;
            } else if (group.type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER) {
                address.address = uniformBacking.address;
                get.data.pUniformBuffer = &address;
            } else if (group.type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) {
                image.sampler = sampler;
                image.imageView = sampled.view;
                image.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                get.data.pCombinedImageSampler = &image;
            } else if (group.type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE) {
                image.sampler = VK_NULL_HANDLE;
                image.imageView = storage.view;
                image.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                get.data.pStorageImage = &image;
            } else if (group.type == VK_DESCRIPTOR_TYPE_SAMPLER) {
                get.data.pSampler = &sampler;
            } else {
                continue;  // a KIND the plan never maps to a size is counted, never guessed
            }
            getDescriptor_(device_, &get, element, hostSets[set].data() + offset);
        }
    }
    facts.bound = true;
    // ---- record: upload the pool, bind it, set offsets, bind, dispatch, read back ----
    VkCommandPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    if (createCommandPool_(device_, &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
        return fail("vkCreateCommandPool refused");
    }
    VkCommandBufferAllocateInfo allocInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, nullptr, commandPool,
                                         VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1u};
    if (allocateCommandBuffers_(device_, &allocInfo, &commandBuffer) != VK_SUCCESS || !commandBuffer) {
        return fail("vkAllocateCommandBuffers refused");
    }
    getQueue_(device_, 0u, 0u, &queue);
    if (!queue) {
        return fail("vkGetDeviceQueue gave no queue");
    }
    const auto recordStart = std::chrono::steady_clock::now();
    VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr, 0u, nullptr};
    if (beginCommandBuffer_(commandBuffer, &beginInfo) != VK_SUCCESS) {
        return fail("vkBeginCommandBuffer refused");
    }
    {
        // Park both images in the layout their bindings claim and clear them: a defined
        // input is what makes the before/after comparison mean something.
        VkImageMemoryBarrier toGeneral[2] = {};
        toGeneral[0].image = sampled.image;
        toGeneral[1].image = storage.image;
        for (auto& barrier : toGeneral) {
            barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            barrier.srcAccessMask = 0u;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0u, 1u, 0u, 1u};
        }
        pipelineBarrier_(commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0u,
                         0u, nullptr, 0u, nullptr, 2u, toGeneral);
        VkClearColorValue zero{};
        VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0u, 1u, 0u, 1u};
        clearImage_(commandBuffer, sampled.image, VK_IMAGE_LAYOUT_GENERAL, &zero, 1u, &range);
        clearImage_(commandBuffer, storage.image, VK_IMAGE_LAYOUT_GENERAL, &zero, 1u, &range);
        // Zero the SB/UB backs too, so a read outside what the pass wrote is defined memory.
        const std::vector<std::uint8_t> chunk(256, 0u);
        for (const Resource* backing : {&storageBacking, &uniformBacking}) {
            for (VkDeviceSize at = 0; at < backingBytes; at += 256u) {
                updateBuffer_(commandBuffer, backing->buffer, at, 256u, chunk.data());
            }
        }
        // Upload the serialized bytes of each set into the pool. Inline data in the command,
        // so no host mapping of device memory is needed - which is what this revision can do.
        for (std::size_t set = 0; set < found->sets.size(); ++set) {
            if (setSizes[set] == 0) {
                continue;
            }
            const std::uint8_t* src = hostSets[set].data();
            for (VkDeviceSize at = 0; at < setSizes[set]; at += 256u) {
                const VkDeviceSize piece = (setSizes[set] - at < 256u) ? (setSizes[set] - at) : 256u;
                updateBuffer_(commandBuffer, poolBuffer.buffer, setOffsets[set] + at, piece, src + at);
                facts.descriptorBytesWritten += static_cast<std::uint64_t>(piece);
            }
        }
        if (readbackOk) {
            VkBufferImageCopy before{};
            before.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0u, 0u, 1u};
            before.imageExtent = {width, height, 1u};
            copyImageToBuffer_(commandBuffer, storage.image, VK_IMAGE_LAYOUT_GENERAL,
                               readbacks[0].buffer, 1u, &before);
        }
    }
    {
        // The bind this driver was measured to export for spec version 1: one pool, then
        // per-set offsets into it. No vkCmdPushPipelineDataEXT here - that name does not
        // resolve on this host, in either scope.
        VkDescriptorBufferBindingInfoEXT binding{VK_STRUCTURE_TYPE_DESCRIPTOR_BUFFER_BINDING_INFO_EXT, nullptr,
                                                 poolBuffer.address,
                                                 VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                                                     VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                                                     VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
                                                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT};
        bindDescriptorBuffers_(commandBuffer, 1u, &binding);
        std::vector<std::uint32_t> bufferIndices(found->sets.size(), 0u);
        setDescriptorBufferOffsets_(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, found->pipelineLayout, 0u,
                                    static_cast<std::uint32_t>(found->sets.size()), bufferIndices.data(),
                                    setOffsets.data());
        // Every module in the pack declares a 32-byte push range, and NOTHING measured so
        // far says what a pass expects inside it. So the range is filled with ZEROS
        // BY DESIGN OF THE MEASUREMENT and the note says so out loud: a pass that branches
        // on those values may take its empty path, and a reader of "out_changed=no" has
        // to be able to tell that apart from "the pass provably did nothing". The per-set
        // sizing basis is already recorded next to each set, so no extra claim is made here.
        if (found->pushBytes != 0u) {
            const std::vector<std::uint8_t> pushZero(found->pushBytes, 0u);
            // Two traps in this one call, both measured rather than remembered. (1) The
            // parameter order is (cmd, LAYOUT, STAGEFLAGS, ...) - the LAYOUT comes FIRST here,
            // where vkCmdSetDescriptorBufferOffsetsEXT above takes (cmd, BINDPOINT, LAYOUT,
            // ...); writing them the same way is a compile error ("cannot convert VkPipelineBind
            // Point to VkPipelineLayout"). (2) The third parameter is VkShaderStageFlags, NOT
            // a bind point, and a VkPipelineBindPoint converts to VkFlags SILENTLY (both are
            // integers here) - so the wrong-but-compiling version would have named stage 2
            // (VERTEX) while the layout's only push range was declared VK_SHADER_STAGE_COMPUTE
            // BIT, which is a validation error the driver may or may not report. What the spec
            // requires is that stageFlags/offset/size fall inside a range the LAYOUT declares,
            // and setupPipeline() declared exactly (COMPUTE, 0, pushBytes).
            pushConstants_(commandBuffer, found->pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0u,
                           found->pushBytes, pushZero.data());
            facts.note += "push=" + std::to_string(found->pushBytes) + "B=zero(uninitialized) ";
        }
        bindPipeline_(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, found->pipeline);
        const auto divide = [](std::uint32_t count, std::uint32_t group) {
            return group == 0u ? 1u : (count + group - 1u) / group;
        };
        facts.groups = {divide(width, found->workgroup[0]), divide(height, found->workgroup[1]), 1u};
        const std::uint32_t repeats = dispatchRepeat != 0u ? dispatchRepeat : 1u;
        facts.repeats = repeats;
        // The barrier between two dispatches of the SAME pool is where VK_EXT_descriptor_buffer
        // asks for VK_DEPENDENCY_BY_TRANSITION (writes into the pool must be visible to the
        // next dispatch). That flag is NOT in these pinned headers - grepping the name across
        // third_party/Vulkan-Headers returns nothing - so the only barrier available here is a
        // PLAIN one, and that is RECORDED rather than papered over: a real pass chain whose
        // correctness depends on by-transition semantics cannot be declared working against
        // this header revision.
        facts.barrierFlags =
            repeats > 1u ? "plain(VK_DEPENDENCY_BY_TRANSITION_absent_from_pinned_headers)"
                         : "none(single_dispatch)";
        for (std::uint32_t repeat = 0; repeat < repeats; ++repeat) {
            if (repeat != 0u) {
                // vkCmdPipelineBarrier takes the 1.0-era stage flag names here: the COMPUTE
                // one is VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT (0x800), and the shorter
                // VK_PIPELINE_STAGE_COMPUTE_BIT belongs to the VkPipelineStageFlagBits2 family
                // that this pinned header revision does not carry (measured: "was not declared
                // in this scope; did you mean VK_PIPELINE_STAGE_HOST_BIT?").
                pipelineBarrier_(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0u, 0u, nullptr, 0u, nullptr, 0u,
                                 nullptr);
            }
            dispatch_(commandBuffer, facts.groups[0], facts.groups[1], facts.groups[2]);
        }
        if (repeats > 1u) {
            // The write census below cannot be trusted for THIS call: a repeat run on this
            // host was measured to produce a driver hard-recovery ("the CS has been cancelled")
            // while vkQueueWaitIdle still answered VK_SUCCESS, so "wrote nothing" may simply
            // mean "the cancelled CS never ran". Saying it out loud is the difference between
            // a measurement and an excuse.
            facts.note += "census_may_be_confounded_by_cancelled_cs(repeats>1) ";
        }
        if (readbackOk) {
            // Three image copies and two buffer copies, all AFTER the dispatch and BEFORE
            // any wait: they are part of the SAME submit, which is what makes the after-
            // state the state the pass left behind rather than a later race. The sampled
            // image is copied even though it was handed as INPUT - a pass that writes into
            // its own input is a finding, not a detail.
            VkBufferImageCopy after{};
            after.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0u, 0u, 1u};
            after.imageExtent = {width, height, 1u};
            copyImageToBuffer_(commandBuffer, storage.image, VK_IMAGE_LAYOUT_GENERAL,
                               readbacks[1].buffer, 1u, &after);
            copyImageToBuffer_(commandBuffer, sampled.image, VK_IMAGE_LAYOUT_GENERAL,
                               readbacks[2].buffer, 1u, &after);
            // The backs of the buffer slots. Their whole 256 KiB is copied, not just the
            // first page: a pass is free to index deep into a STORAGE_BUFFER, and only the
            // bytes it actually touched matter for the census. vkCmdCopyBuffer takes REGIONS
            // (VkBufferCopy{sOffset,dOffset,size}); the (src,dst,srcOff,dstOff,size) shape
            // belongs to vkCmdCopyBuffer2EXT, which is not what this name resolves to.
            VkBufferCopy region{0u, 0u, backingBytes};
            copyBuffer_(commandBuffer, storageBacking.buffer, readbacks[3].buffer, 1u, &region);
            copyBuffer_(commandBuffer, uniformBacking.buffer, readbacks[4].buffer, 1u, &region);
        }
    }
    if (endCommandBuffer_(commandBuffer) != VK_SUCCESS) {
        return fail("vkEndCommandBuffer refused: nothing was submitted");
    }
    // The span INCLUDES vkEndCommandBuffer: that is where a driver that needs it finalizes
    // the command list, so excluding it would let cost fall into no bucket at all. Which
    // bucket the measured ~2 s of these passes lands in is what this split is for.
    facts.msRecord = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - recordStart).count();
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0u, nullptr, nullptr, 1u, &commandBuffer, 0u,
                        nullptr};
    const auto submitStart = std::chrono::steady_clock::now();
    const VkResult submitted = queueSubmit_(queue, 1u, &submit, VK_NULL_HANDLE);
    facts.msSubmit = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - submitStart).count();
    if (submitted != VK_SUCCESS) {
        // A submit that answers VK_ERROR_OTHER is the same class of driver state as a
        // faulted CS: the device is no longer usable in this process, whatever the pass
        // itself did. Marking it here is what stops a sweep from reporting the 91 passes
        // AFTER the first such failure as "the driver cannot run them".
        facts.contextLost = true;
        return fail(std::string("vkQueueSubmit VkResult=") + vkResultName(submitted));
    }
    facts.submitted = true;
    const auto waitStart = std::chrono::steady_clock::now();
    const VkResult waited = queueWaitIdle_(queue);
    const auto waitedAt = std::chrono::steady_clock::now();
    facts.msWait = std::chrono::duration<double, std::milli>(waitedAt - waitStart).count();
    facts.ms = facts.msSubmit + facts.msWait;
    facts.ok = (waited == VK_SUCCESS);
    if (waited != VK_SUCCESS) {
        facts.failure = std::string("vkQueueWaitIdle VkResult=") + vkResultName(waited);
        // A faulted CS does not merely fail THIS submit: RADV puts the device into hard
        // recovery ("the CS has been cancelled because the context is lost"), after which
        // every later vkQueueSubmit on the same device returns VK_ERROR_OTHER. Without this
        // flag a 92-module sweep inside one process would read as "the driver can run none
        // of them" once the first one faults, which is the wrong finding: the caller must
        // re-run in a fresh process (the sweep harness does exactly that).
        facts.contextLost = true;
    }
    if (readbackOk && readbacks.size() == 5u) {
        auto checksumOf = [&](const HostReadback& made) {
            const auto* bytes = reinterpret_cast<const std::uint8_t*>(made.map);
            std::uint64_t hash = 1469598103934665603ull;
            for (std::uint64_t i = 0; i < made.bytes; ++i) {
                hash = (hash * 1099511627689ull + bytes[i]) ^ (hash >> 47);
            }
            return hash;
        };
        auto nonZeroBytes = [&](const HostReadback& made) {
            const auto* bytes = reinterpret_cast<const std::uint8_t*>(made.map);
            std::uint64_t count = 0;
            for (std::uint64_t i = 0; i < made.bytes; ++i) {
                if (bytes[i] != 0u) {
                    ++count;
                }
            }
            return count;
        };
        facts.checksumBefore = checksumOf(readbacks[0]);
        facts.checksumAfter = checksumOf(readbacks[1]);
        facts.outputChanged = (facts.checksumBefore != facts.checksumAfter);
        // Three separate answers instead of one: WHICH writable resource the pass touched
        // is the information the next phase needs in order to guess less and bind better.
        facts.wroteSampledImage = nonZeroBytes(readbacks[2]) != 0ull;
        facts.nonZeroStorageBufferBytes = nonZeroBytes(readbacks[3]);
        facts.nonZeroUniformBufferBytes = nonZeroBytes(readbacks[4]);
        facts.anyWritableOutputChanged = facts.outputChanged || facts.wroteSampledImage ||
                                         facts.nonZeroStorageBufferBytes != 0ull ||
                                         facts.nonZeroUniformBufferBytes != 0ull;
    }

    // ---- teardown only what THIS call created; the substrate's objects stay alive ----
    for (const auto& made : readbacks) {
        if (made.memory) {
            unmapMemory_(device_, made.memory);
        }
    }
    for (const auto& view : views) {
        destroyImageView_(device_, view, nullptr);
    }
    for (const auto& image : images) {
        destroyImage_(device_, image, nullptr);
    }
    if (sampler) {
        destroySampler_(device_, sampler, nullptr);
    }
    for (const auto& buffer : buffers) {
        destroyBuffer_(device_, buffer, nullptr);
    }
    for (const auto& memory : frees) {
        freeMemory_(device_, memory, nullptr);
    }
    if (commandBuffer) {
        freeCommandBuffers_(device_, commandPool, 1u, &commandBuffer);
    }
    if (commandPool) {
        destroyCommandPool_(device_, commandPool, nullptr);
    }
    if (facts.ok && facts.failure.empty()) {
        reason = "ran";
    }
    return facts.ok;
}

}  // namespace ffgvk
