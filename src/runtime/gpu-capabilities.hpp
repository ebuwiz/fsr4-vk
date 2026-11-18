// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 fsr4-vk contributors.
// See LICENSE in the project root.

// One physical device probed once: identity, extensions, features and the
// cooperative matrix shapes the shipped model shaders are built around. Every
// query here is physical-device level, so the probe never needs an instance
// handle and never creates one of its own. A query the loader refuses leaves
// that field unprobed rather than looking unsupported; only a probe that
// answered can rule a model out on shape grounds.
//
// The provider consults this snapshot before it is allowed to select a model,
// so an unsupported FP8 request fails with a reason instead of a shader compile
// error deep inside the runtime.

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <vulkan/vulkan.h>

#include "resources.hpp"

namespace fsr4vk {

// Cooperative matrix tile shared by every FP8 model shader variant.
constexpr uint32_t kModelMatrixExtent = 16;

// Device functions the runtime resolves through the loader's device. Group
// entries are alternatives, so "vkCmdPipelineBarrier2|vkCmdPipelineBarrier2KHR"
// accepts either spelling.
constexpr const char* kRequiredEntryPointGroups[] = {
    "vkGetDescriptorEXT",
    "vkCmdBindDescriptorBuffersEXT",
    "vkCmdSetDescriptorBufferOffsetsEXT",
    "vkGetDescriptorSetLayoutSizeEXT",
    "vkGetDescriptorSetLayoutBindingOffsetEXT",
    "vkCmdPipelineBarrier2|vkCmdPipelineBarrier2KHR",
    "vkCmdWriteTimestamp2|vkCmdWriteTimestamp2KHR",
    "vkGetBufferDeviceAddress|vkGetBufferDeviceAddressKHR",
};

// Model request, before any device is consulted.
enum class ModelChoice : std::uint8_t { Auto, ForceInt8, ForceFp8 };

// FSR4_VK_MODEL values. Auto keeps the historical INT8 default and must never
// probe for FP8 support; only an explicit request is allowed to gate on it.
std::optional<ModelChoice> parseModelChoice(const char* value);

struct GpuCapabilities {
    // Identity ------------------------------------------------------------
    VkPhysicalDeviceProperties properties{};
    VkPhysicalDeviceIDProperties identity{};
    VkPhysicalDeviceDriverProperties driver{};

    // Properties ----------------------------------------------------------
    VkPhysicalDeviceSubgroupProperties subgroups{};
    VkPhysicalDeviceSubgroupSizeControlProperties subgroup_sizes{};
    VkPhysicalDeviceFloatControlsProperties float_controls{};
    VkPhysicalDeviceCooperativeMatrixPropertiesKHR cooperative{};

    std::vector<VkExtensionProperties> extensions;

    // Features -----------------------------------------------------------
    VkPhysicalDeviceFeatures core{};
    VkPhysicalDeviceShaderFloat16Int8Features float16_int8{};
    VkPhysicalDevice8BitStorageFeatures storage8{};
    VkPhysicalDeviceVulkanMemoryModelFeatures memory_model{};
    VkPhysicalDeviceSubgroupSizeControlFeatures subgroup_size_control{};
    VkPhysicalDeviceBufferDeviceAddressFeatures buffer_device_address{};
    VkPhysicalDeviceDescriptorIndexingFeatures descriptor_indexing{};
    VkPhysicalDeviceDescriptorBufferFeaturesEXT descriptor_buffer{};
    VkPhysicalDeviceShaderFloat8FeaturesEXT shader_float8{};
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR cooperative_matrix{};

    // Cooperative matrix shapes offered for compute workloads. Empty is
    // ambiguous on its own, so probed records whether the query answered: a
    // host that refused it leaves both fields untouched. A caller that owns an
    // instance may fill them itself, as tools/cap-probe.cpp does.
    std::vector<VkCooperativeMatrixPropertiesKHR> matrix_shapes;
    bool matrix_shapes_probed = false;

    bool hasExtension(const char* wanted) const;
    // Feature chain and subgroup contract the runtime always needs.
    bool supportsRuntimeContract() const;

    // Cooperative matrix tiles the FP8 model shaders are built around.
    bool supportsModelMatrixShapes() const;

    // Full INT8 and FP8 model contracts.
    bool supportsInt8Model() const;
    bool supportsFp8Model() const;

    // Human readable reasons for rejecting either model.
    std::vector<std::string> int8Blockers() const;
    std::vector<std::string> fp8Blockers() const;

    // Vendor profiles, so shader packaging decisions can stop guessing.
    bool amdProprietaryDriver() const;
    bool rdna2Device() const;

    std::string describe() const;

    bool hasMatrixShape(VkComponentTypeKHR input_type) const;

private:
    const char* matrixInputName(VkComponentTypeKHR type) const;
    std::string missingMatrixShapeName(VkComponentTypeKHR type) const;
};

std::string joinBlockers(const std::vector<std::string>& blockers);

// Queries one physical device. Every command used here takes the physical device
// itself, so the provider can build the snapshot from the handle the loader hands
// it. A command the host does not export leaves its field unprobed rather than
// looking unsupported.
GpuCapabilities detect(VkPhysicalDevice physical_device);

// Entry points missing from the device the loader handed over.
std::vector<std::string> missingEntryPoints(VkDevice device,
                                            PFN_vkGetDeviceProcAddr get_device_proc_addr);

const char* modelName(ModelChoice choice);

} // namespace fsr4vk
