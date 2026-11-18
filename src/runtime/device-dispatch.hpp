// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 fsr4-vk contributors.
// See LICENSE in the project root.

#pragma once

#include <array>
#include <cstddef>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#include <vulkan/vulkan.h>

namespace fsr4vk {

// One row per device function group: the promoted spelling and the extension
// spelling, with a null khr when there is no alternative. The runtime resolves
// this table in DeviceDispatch::load and the provider's contract check walks
// the same rows, so the functions the runtime is allowed to call are the
// functions the provider promised. They used to be three separately worded
// lists - the capability table, DeviceDispatch's literals and five hand-written
// lookups in createDescriptorLayouts - and nothing stopped them from drifting.
struct EntryPointGroup {
    const char* core = nullptr;
    const char* khr = nullptr;
};

constexpr EntryPointGroup kRequiredEntryPointGroups[] = {
    {"vkGetDescriptorEXT", nullptr},
    {"vkCmdBindDescriptorBuffersEXT", nullptr},
    {"vkCmdSetDescriptorBufferOffsetsEXT", nullptr},
    {"vkGetDescriptorSetLayoutSizeEXT", nullptr},
    {"vkGetDescriptorSetLayoutBindingOffsetEXT", nullptr},
    {"vkCmdPipelineBarrier2", "vkCmdPipelineBarrier2KHR"},
    {"vkCmdWriteTimestamp2", "vkCmdWriteTimestamp2KHR"},
    {"vkGetBufferDeviceAddress", "vkGetBufferDeviceAddressKHR"},
};

constexpr const size_t kRequiredEntryPointGroupCount =
    sizeof(kRequiredEntryPointGroups) / sizeof(kRequiredEntryPointGroups[0]);

// Row index of each table entry, which is also the DeviceDispatch member it
// lands in. load() asserts the count, so a row added here without a member bound
// below fails the build rather than handing the runtime a null function pointer.
constexpr const size_t kRowGetDescriptor = 0;
constexpr const size_t kRowBindDescriptorBuffers = 1;
constexpr const size_t kRowSetDescriptorBufferOffsets = 2;
constexpr const size_t kRowGetDescriptorSetLayoutSize = 3;
constexpr const size_t kRowGetDescriptorSetLayoutBindingOffset = 4;
constexpr const size_t kRowPipelineBarrier2 = 5;
constexpr const size_t kRowWriteTimestamp2 = 6;
constexpr const size_t kRowGetBufferDeviceAddress = 7;

// How a row reads in a message: every spelling that would have satisfied it.
inline std::string entryPointLabel(const EntryPointGroup& group) {
    std::string label(group.core);
    if (group.khr != nullptr) {
        label += " or ";
        label += group.khr;
    }
    return label;
}

// Which spelling this device answers for a row. A loader may implement either,
// and Vulkan 1.1 and 1.2 ask for the KHR name first because some of them answer
// the promoted name with a trampoline that cannot be called; 1.3 and newer
// prefer the promoted spelling. answered_name reports the choice, which is what
// the [entry-points] line logs, so a loader that only answers extension names
// is visible in a log instead of being guessed at.
inline PFN_vkVoidFunction resolveEntryPoint(VkDevice device,
                                            PFN_vkGetDeviceProcAddr get_device_proc_addr,
                                            uint32_t api_version,
                                            const EntryPointGroup& group,
                                            const char** answered_name) {
    if (!device || !get_device_proc_addr) {
        return nullptr;
    }

    const bool prefer_core =
        VK_API_VERSION_MAJOR(api_version) > 1 ||
        (VK_API_VERSION_MAJOR(api_version) == 1 && VK_API_VERSION_MINOR(api_version) >= 3);

    const char* const first = prefer_core ? group.core : group.khr;
    const char* const second = prefer_core ? group.khr : group.core;

    for (const char* name : {first, second}) {
        if (name == nullptr) {
            continue;
        }
        if (const auto function = get_device_proc_addr(device, name)) {
            if (answered_name != nullptr) {
                *answered_name = name;
            }
            return function;
        }
    }

    return nullptr;
}

// Rows the device cannot satisfy at all. Whether the API version would have
// preferred one spelling or the other is irrelevant to a refusal: the row is
// satisfied if any spelling answers, so the version used here only fixes the
// order names are tried.
inline std::vector<std::string> missingEntryPointGroups(
    VkDevice device, PFN_vkGetDeviceProcAddr get_device_proc_addr) {
    std::vector<std::string> missing;
    for (const auto& group : kRequiredEntryPointGroups) {
        if (!resolveEntryPoint(device,
                               get_device_proc_addr,
                               VK_API_VERSION_1_3,
                               group,
                               nullptr)) {
            missing.emplace_back(entryPointLabel(group));
        }
    }
    return missing;
}

// Every row of the table lands in a typed member below, so a device that cannot
// satisfy a row fails at load() naming the row, rather than later inside a
// command buffer or a descriptor write. The debug object-name entry point is
// deliberately not a row: the runtime skips it instead of refusing the device.
struct DeviceDispatch {
    PFN_vkGetDescriptorEXT get_descriptor = nullptr;
    PFN_vkCmdBindDescriptorBuffersEXT cmd_bind_descriptor_buffers = nullptr;
    PFN_vkCmdSetDescriptorBufferOffsetsEXT cmd_set_descriptor_buffer_offsets = nullptr;
    PFN_vkGetDescriptorSetLayoutSizeEXT get_descriptor_set_layout_size = nullptr;
    PFN_vkGetDescriptorSetLayoutBindingOffsetEXT get_descriptor_set_layout_binding_offset = nullptr;
    PFN_vkCmdPipelineBarrier2 cmd_pipeline_barrier2 = nullptr;
    PFN_vkCmdWriteTimestamp2 cmd_write_timestamp2 = nullptr;
    PFN_vkGetBufferDeviceAddress get_buffer_device_address = nullptr;

    // Which spelling answered each row, indexed by the row constants above.
    std::array<const char*, kRequiredEntryPointGroupCount> answered{};

    PFN_vkSetDebugUtilsObjectNameEXT set_debug_utils_object_name = nullptr;

    DeviceDispatch() = default;

    DeviceDispatch(VkDevice device,
                   PFN_vkGetDeviceProcAddr get_device_proc_addr,
                   uint32_t api_version = VK_API_VERSION_1_3) {
        load(device, get_device_proc_addr, api_version);
    }

    void load(VkDevice device,
              PFN_vkGetDeviceProcAddr get_device_proc_addr,
              uint32_t api_version = VK_API_VERSION_1_3) {
        static_assert(kRequiredEntryPointGroupCount == 8,
                      "bind every entry point row to a DeviceDispatch member");

        std::array<PFN_vkVoidFunction, kRequiredEntryPointGroupCount> functions{};
        std::vector<std::string> missing;

        for (size_t row = 0; row < kRequiredEntryPointGroupCount; ++row) {
            const auto& group = kRequiredEntryPointGroups[row];
            functions[row] = resolveEntryPoint(
                device, get_device_proc_addr, api_version, group, &answered[row]);
            if (functions[row] == nullptr) {
                missing.emplace_back(entryPointLabel(group));
            }
        }

        if (!missing.empty()) {
            std::string message = "missing Vulkan device entry point: ";
            for (size_t index = 0; index < missing.size(); ++index) {
                if (index != 0) {
                    message += ", ";
                }
                message += missing[index];
            }
            throw std::runtime_error(message);
        }

        get_descriptor =
            reinterpret_cast<PFN_vkGetDescriptorEXT>(functions[kRowGetDescriptor]);
        cmd_bind_descriptor_buffers =
            reinterpret_cast<PFN_vkCmdBindDescriptorBuffersEXT>(
                functions[kRowBindDescriptorBuffers]);
        cmd_set_descriptor_buffer_offsets =
            reinterpret_cast<PFN_vkCmdSetDescriptorBufferOffsetsEXT>(
                functions[kRowSetDescriptorBufferOffsets]);
        get_descriptor_set_layout_size =
            reinterpret_cast<PFN_vkGetDescriptorSetLayoutSizeEXT>(
                functions[kRowGetDescriptorSetLayoutSize]);
        get_descriptor_set_layout_binding_offset =
            reinterpret_cast<PFN_vkGetDescriptorSetLayoutBindingOffsetEXT>(
                functions[kRowGetDescriptorSetLayoutBindingOffset]);
        cmd_pipeline_barrier2 =
            reinterpret_cast<PFN_vkCmdPipelineBarrier2>(functions[kRowPipelineBarrier2]);
        cmd_write_timestamp2 =
            reinterpret_cast<PFN_vkCmdWriteTimestamp2>(functions[kRowWriteTimestamp2]);
        get_buffer_device_address =
            reinterpret_cast<PFN_vkGetBufferDeviceAddress>(
                functions[kRowGetBufferDeviceAddress]);

        if (get_device_proc_addr) {
            set_debug_utils_object_name = reinterpret_cast<PFN_vkSetDebugUtilsObjectNameEXT>(
                get_device_proc_addr(device, "vkSetDebugUtilsObjectNameEXT"));
        }
    }

    // The [entry-points] line: how many rows this device satisfied, and for the
    // groups that have two spellings which one the loader actually answered.
    // A driver that implements only the KHR names is worth seeing in a log.
    std::string describe() const {
        size_t resolved = 0;
        size_t promoted = 0;
        size_t promoted_core = 0;
        for (size_t row = 0; row < answered.size(); ++row) {
            if (answered[row] != nullptr) {
                ++resolved;
            }
            if (kRequiredEntryPointGroups[row].khr != nullptr) {
                ++promoted;
                const std::string_view name =
                    answered[row] == nullptr ? std::string_view()
                                             : std::string_view(answered[row]);
                if (name == std::string_view(kRequiredEntryPointGroups[row].core)) {
                    ++promoted_core;
                }
            }
        }

        return "groups=" + std::to_string(kRequiredEntryPointGroupCount) +
               " resolved=" + std::to_string(resolved) + " promoted=" +
               std::to_string(promoted) + " promoted_core=" +
               std::to_string(promoted_core);
    }
};

inline void setDebugName(const DeviceDispatch& dispatch,
                         VkDevice device,
                         VkObjectType type,
                         uint64_t handle,
                         const char* name) {
    if (dispatch.set_debug_utils_object_name && device && handle && name) {
        VkDebugUtilsObjectNameInfoEXT info{VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT};
        info.objectType = type;
        info.objectHandle = handle;
        info.pObjectName = name;
        dispatch.set_debug_utils_object_name(device, &info);
    }
}

} // namespace fsr4vk
