// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 fsr4-vk contributors.
// See LICENSE in the project root.

#pragma once

#include "assets/assets.hpp"
#include "runtime/color-space.hpp"
#include "runtime/gpu-capabilities.hpp"
#include "runtime/resources.hpp"

#include <cstdint>
#include <functional>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>

namespace fsr4_411 {

struct ContextCreateInfo {
    VkPhysicalDevice physical_device = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;

    std::filesystem::path asset_root;
    std::filesystem::path diagnostic_root;

    PFN_vkGetDeviceProcAddr get_device_proc_addr = nullptr;

    bool auto_exposure = false;
    bool depth_inverted = false;
    bool high_resolution_motion_vectors = false;
    bool motion_vector_jitter_cancellation = false;
    bool extensive_logging = false;

    uint32_t max_render_width = 0;
    uint32_t max_render_height = 0;
    uint32_t max_output_width = 0;
    uint32_t max_output_height = 0;

    uint32_t api_version = VK_API_VERSION_1_1;
};

struct OverlayOptions {
    std::string_view label;
    bool watermark = false;
    bool debug_view = false;
    bool hdr = false;
};

class QualityContext {
public:
    virtual ~QualityContext() = default;

    QualityContext(const QualityContext&) = delete;
    QualityContext& operator=(const QualityContext&) = delete;

    virtual void recordFrame(VkCommandBuffer command_buffer,
                             Image color,
                             Image depth,
                             Image motion,
                             Image exposure,
                             Image optional_mask,
                             Image output,
                             OptimizedConstants constants,
                             bool use_staging_output,
                             ColorSpace color_space = ColorSpace::Linear,
                             OverlayOptions overlay = {}) = 0;

protected:
    QualityContext() = default;
};

using QualityContextPtr = std::unique_ptr<QualityContext>;

} // namespace fsr4_411

namespace fsr4runtime {
enum class PassDependency;
}

namespace fsr4fp8 {
namespace fs = std::filesystem;

// The id, and what asking for it means, live in runtime/gpu-capabilities.hpp next to
// ModelChoice::ForceFp8 so one table drives both the provider's registry and the model
// ladder. Note the number is this project's own: 0xF5A5CA1E_01001002 is the reference
// DLL's single published id and stays model-agnostic, which is why a strict FP8 pin
// could not keep using it.
inline constexpr uint64_t kVersionId = fsr4vk::kFp8VersionId;
inline constexpr const char* kVersionName = "4.1.2 Vulkan FP8 (experimental)";

bool hasAssets(const fs::path& root);
fsr4_411::QualityContextPtr createContext(const fsr4_411::ContextCreateInfo& info);

std::array<uint32_t, 2> dispatchSize(uint32_t pass_index,
                                     uint32_t output_width,
                                     uint32_t output_height,
                                     uint32_t model_bucket_width,
                                     uint32_t model_bucket_height);
fsr4runtime::PassDependency
passDependency(uint32_t pass_index, uint32_t next_pass_index, bool use_wmma);
} // namespace fsr4fp8

namespace fsr4int8 {
namespace fs = std::filesystem;

inline constexpr uint64_t kVersionId = fsr4vk::kInt8VersionId;
inline constexpr const char* kVersionName = "4.1.2 Vulkan INT8 (experimental)";

const char* resolutionBucket(uint32_t output_width, uint32_t output_height);

bool hasAssets(const fs::path& root);
fs::path modelRoot(const fs::path& asset_root);
fsr4_411::QualityContextPtr createContext(const fsr4_411::ContextCreateInfo& info);

std::array<uint32_t, 2> dispatchSize(uint32_t pass_index,
                                     uint32_t output_width,
                                     uint32_t output_height,
                                     uint32_t model_bucket_width,
                                     uint32_t model_bucket_height);
fsr4runtime::PassDependency passDependency(uint32_t pass_index, uint32_t next_pass_index);
} // namespace fsr4int8

namespace fsr4int8 {
struct ShaderVariants {
    std::filesystem::path root;
    std::string profile;

    ShaderVariants(const std::filesystem::path& int8_root, bool enabled);

    std::filesystem::path candidate(const std::filesystem::path& original) const;
};
} // namespace fsr4int8
