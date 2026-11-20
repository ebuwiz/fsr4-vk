// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 fsr4-vk contributors.
// See LICENSE in the project root.

#include "fg-runtime.hpp"

#include <algorithm>
#include <cstring>
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
        built_.push_back({module.name, pipelineLayout, pipeline});

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

}  // namespace ffgvk
