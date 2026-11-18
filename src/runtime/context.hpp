// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 fsr4-vk contributors.
// See LICENSE in the project root.

#pragma once

#include "diagnostics/diagnostics.hpp"
#include "models/models.hpp"
#include "runtime/frame.hpp"
#include "runtime/gpu-capabilities.hpp"
#include "runtime/resources.hpp"
#include "runtime/rcas.hpp"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <sstream>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace fsr4runtime {
// Models choose shader variants for the shared runtime.
struct ShaderRecipe {
    std::string profile;

    std::function<std::filesystem::path(const std::filesystem::path&)> select;

    std::filesystem::path candidate(const std::filesystem::path& path) const {
        return select ? select(path) : path;
    }
};
} // namespace fsr4runtime

namespace fsr4runtime {
struct ShaderSelectionOptions {
    bool use_wmma;
    bool use_int8_dispatch;
    bool numbered_shader_files;
    bool windows_fp8_optimized;
    bool windows_int8_optimized;
    bool extensive_logging;
};

// Share fallback logging state across presets.
class ModelShaderSelector {
    ShaderSelectionOptions options;
    const ShaderRecipe& recipe;
    bool exact_fallback_logged = false;

    std::filesystem::path stockShaderPath(const std::filesystem::path& directory,
                                          uint32_t pass_index,
                                          const std::filesystem::path& prepass_override,
                                          const std::filesystem::path& postpass_override) const;
    std::filesystem::path selectFp8Shader(const std::filesystem::path& directory,
                                          uint32_t pass_index,
                                          const std::filesystem::path& prepass_override,
                                          const std::filesystem::path& postpass_override);
    std::filesystem::path selectInt8Shader(const std::filesystem::path& directory,
                                           uint32_t pass_index,
                                           const std::filesystem::path& prepass_override,
                                           const std::filesystem::path& postpass_override);
    std::filesystem::path windowsInt8ShaderPath(const std::filesystem::path& directory,
                                                uint32_t pass_index,
                                                const std::filesystem::path& stock_path) const;
    std::filesystem::path selectInt8Variant(const std::filesystem::path& original);

public:
    ModelShaderSelector(ShaderSelectionOptions options, const ShaderRecipe& recipe)
        : options(options), recipe(recipe) {}

    std::filesystem::path select(const std::filesystem::path& directory,
                                 uint32_t pass_index,
                                 const std::filesystem::path& prepass_override,
                                 const std::filesystem::path& postpass_override);
};
} // namespace fsr4runtime

namespace fsr4runtime {
namespace fs = std::filesystem;

using fsr4_411::Buffer;
using fsr4_411::check;
using fsr4_411::computeBarrier;
using fsr4_411::createBuffer;
using fsr4_411::createImage;
using fsr4_411::destroyBuffer;
using fsr4_411::destroyImage;
using fsr4_411::deviceAddress;
using fsr4_411::Image;
using fsr4_411::imageBarrier;
using fsr4_411::kConstantBufSize;
using fsr4_411::kDescriptorCount;
using fsr4_411::kNumDescriptorSets;
using fsr4_411::OptimizedConstants;

// The transport enum and its names live in fsr4vk beside the entry-point table
// they scope; the runtime refers to them unqualified from here on.
using fsr4vk::DescriptorBackend;
using fsr4_411::readBytes;
using fsr4_411::readModelSpirv;
using fsr4_411::readSpirv;

struct ModelRuntimeOptions {
    fs::path rcas_shader_root;
    fs::path color_shader_root;
    bool use_wmma = false;
    bool use_int8_dispatch = false;
    bool numbered_shader_files = false;
    bool execute_padding_passes = false;
    bool direct_display_motion_vectors = false;

    fs::path ultraperformance_shader_dir;
    fs::path ultraperformance_model_data_path;

    ColorEndpointOptions color_endpoints;

    ModelPassPolicy pass_policy;
    ShaderRecipe shader_recipe;
};

class QualityContext final : public fsr4_411::QualityContext {
    VkPhysicalDevice physical_device;
    VkDevice device;
    fsr4vk::DeviceDispatch dispatch;

    // Which transport this context ended up using, chosen from the device snapshot
    // below and overruled only by an exact FSR4_DESCRIPTOR_BACKEND spelling. Every
    // descriptor path in the runtime reads this one value.
    fsr4vk::DescriptorBackend descriptor_backend =
        fsr4vk::DescriptorBackend::DescriptorBuffers;

    VkDeviceSize descriptor_capacity = 0;
    std::array<VkDeviceSize, kNumDescriptorSets> set_strides{};
    std::array<VkDeviceSize, kNumDescriptorSets> set_binding_offsets{};

    // Dynamic offsets for the sets transport. The current layouts use ordinary
    // buffer descriptors, so this remains empty rather than imposing dynamic
    // offset binding semantics on a conventional descriptor set.
    // uint32_t rather than VkDynamicOffset: the pinned headers type pDynamicOffsets
    // as uint32_t, and the VkDynamicOffset alias only arrives with the 1.3 write-
    // descriptor-set-2 surface this host's headers do not declare at all.
    std::vector<uint32_t> descriptor_dynamic_offsets;

    VkSampler sampler{};
    VkPipelineLayout pipeline_layout{};
    VkPipelineCache pipeline_cache{};
    fs::path pipeline_cache_path;
    bool pipeline_cache_initial_data = false;
    double pipeline_creation_ms = 0.0;
    uint32_t pipeline_creation_count = 0;
    std::array<VkDescriptorSetLayout, kNumDescriptorSets> set_layouts{};

    // Only the selected shaders own pipelines; inactive variants are retired.
    std::vector<VkPipeline> pipelines;
    std::vector<fs::path> active_shader_paths;
    std::array<std::vector<fs::path>, 2> model_shader_paths;
    unsigned model_pipeline_variant = 2;
    fs::path auto_exposure_shader_path;
    VkPipeline auto_exposure_pipeline{};
    VkPipeline overlay_pipeline{};
    bool overlay_pipeline_blend = false;

    struct PendingFrameUse {
        VkEvent completion{};
        std::vector<VkPipeline> pipelines;
        std::vector<VkImage> internal_images;
        VkImage staging_output{};
    };

    std::vector<PendingFrameUse> pending_frame_uses;
    std::vector<VkEvent> available_frame_events;
    std::vector<VkPipeline> retired_pipelines;
    std::vector<Image> retired_staging_outputs;
    std::vector<Image> retired_internal_images;
    VkFormat overlay_read_format = VK_FORMAT_UNDEFINED;
    bool overlay_read_supported = false;
    // One probe of the physical device answers identity, drivers, subgroups and
    // the cooperative matrix tiles; the shader policy below reads this snapshot
    // instead of asking the same driver questions a second time.
    fsr4vk::GpuCapabilities device_capabilities{};
    bool device_is_rdna2 = false;
    uint32_t default_subgroup_size = 0;
    // The wave32 pin is diagnosed once per context, not once per pipeline.
    bool wave32_diagnostic_logged = false;
    std::string automatic_wave_label;
    std::string model_wave_label;
    Buffer overlay_font{};
    VkDeviceAddress overlay_font_address{};

    struct RcasPipeline {
        VkPipeline handle{};
        VkExtent2D output_tile{4, 64};
    };

    RcasPipeline rcas_pipeline{};
    unsigned rcas_pipeline_variant = 10;
    ColorSpace color_space = ColorSpace::Linear;
    fs::path color_shader_root;
    ColorEndpointOptions color_endpoints;
    uint32_t color_api_version = VK_API_VERSION_1_3;
    std::array<bool, 2> color_wave32{};

    Image rcas_input{};
    bool rcas_input_initialized = false;
    uint32_t max_output_width = 0;
    uint32_t max_output_height = 0;
    fs::path rcas_shader_root;

    bool use_wmma = false;
    bool use_int8_dispatch = false;
    ModelPassPolicy model_pass_policy;
    bool numbered_shader_files = false;
    bool execute_padding_passes = false;
    bool direct_display_motion_vectors = false;
    bool auto_exposure_enabled = false;
    bool extensive_logging = false;
    bool depth_inverted_enabled = false;

    uint32_t max_render_width = 0;
    uint32_t max_render_height = 0;
    uint32_t model_bucket_width = 1920;
    uint32_t model_bucket_height = 1080;

    struct DescriptorSize {
        VkDeviceSize sampled_image;
        VkDeviceSize storage_image;
        VkDeviceSize sampler;
        VkDeviceSize storage_buffer;
        VkDeviceSize uniform_buffer;
    } descriptor_size{};

    Buffer scratch_buffer{};
    Buffer weight_buffer{};
    Buffer ultraperformance_weight_buffer{};

    std::string current_preset;

    std::vector<Image> images;
    Image staging_output{};

    // Cache exact-size color copies by active render size and format.
    struct AutoExposureColorEntry {
        Image image{};
        bool initialized = false;
        uint64_t last_used_frame = 0;
    };

    std::vector<AutoExposureColorEntry> auto_exposure_color_cache;

    // Recently used entries may exceed this cache target.
    static constexpr size_t kAutoExposureColorCacheSoftLimit = 8;

    Image fallback_depth{};
    bool fallback_depth_initialized = false;

    Image render_motion{};
    bool render_motion_initialized = false;

    bool initialized = false;
    float previous_exposure = 1.0f;

    ModelConstants cached_model_constants{};
    uint32_t cached_model_width = 0;
    uint32_t cached_model_height = 0;

    uint64_t frame_count = 0;
    double total_record_ms = 0;
    double min_record_ms = 1e9;
    double max_record_ms = 0;

    bool live_capture_enabled = false;
    bool live_capture_recording = false;
    bool live_capture_pending = false;
    VkEvent live_capture_completion{};
    bool live_capture_done = false;
    bool live_capture_run_auto_exposure = false;
    bool live_capture_use_staging_output = false;

    uint64_t live_capture_target_frame = 0;

    fs::path live_capture_trigger;
    fs::path live_capture_root;
    fs::path live_capture_frame_dir;

    static constexpr uint32_t kProfileFrameStart = 0;
    static constexpr uint32_t kProfileAfterExternalInputs = 1;
    static constexpr uint32_t kProfileAfterExposure = 2;
    static constexpr uint32_t kProfileAfterRecordFrame = 3;
    static constexpr uint32_t kProfileAfterStagingCopy = 4;
    static constexpr uint32_t kProfileFrameEnd = 5;

    static constexpr uint32_t kProfilePassBase = 6;
    static constexpr uint32_t kProfilePassStride = 3;
    static constexpr uint32_t kProfileQueryCount =
        kProfilePassBase + kProfilePassStride * kPasses.size();

    bool gpu_profile_enabled = false;
    bool gpu_profile_summary_only = false;
    bool gpu_benchmark_enabled = false;
    bool gpu_benchmark_passes = false;
    bool gpu_benchmark_reported = false;
    uint32_t gpu_profile_interval = 60;
    uint32_t gpu_benchmark_warmup = 30;
    uint32_t gpu_benchmark_frames = 300;
    double gpu_timestamp_period_ns = 0.0;
    std::vector<double> gpu_benchmark_samples;

    struct Frame {
        // What a conventional descriptor-set write points at. The write and the
        // payload live in the same frame slot, and both vectors are reserved to
        // this ceiling before any write is queued, because a reallocation would
        // leave the pointer inside VkWriteDescriptorSet aimed at freed storage.
        struct DescriptorWritePayload {
            VkDescriptorImageInfo image{};
            VkDescriptorBufferInfo buffer{};
        };
        // Sized to the heaviest single flush the fallback transport can produce.
        // Both vectors are cleared after vkUpdateDescriptorSets, while retaining
        // capacity so a reused frame slot cannot accumulate stale writes.
        static constexpr size_t kDescriptorWriteCeiling = 512;

        VkPipeline rcas_pipeline{};
        VkExtent2D rcas_dispatch{};
        Buffer constants{};
        Buffer model_constants{};
        Buffer exposure_upload{};

        struct CachedImageView {
            VkImage image = VK_NULL_HANDLE;
            VkFormat format = VK_FORMAT_UNDEFINED;
            VkImageAspectFlags aspect = 0;
            VkImageView view = VK_NULL_HANDLE;
            bool used = false;
        };
        std::vector<CachedImageView> image_view_cache;
        // Buffers transport: one mapped byte buffer per frame slot, six sets wide.
        Buffer descriptor_buffer{};
        // Sets transport: one pool per frame slot with six sets allocated from
        // it, plus the writes staged between binds. Destroying the pool releases
        // its sets, which is why no per-set free call exists anywhere.
        VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
        std::array<VkDescriptorSet, kNumDescriptorSets> descriptor_sets{};
        std::vector<VkWriteDescriptorSet> descriptor_writes;
        std::vector<DescriptorWritePayload> descriptor_payloads;

        VkBuffer model_weights{};

        VkDeviceAddress constants_address = 0;
        VkDeviceAddress descriptor_buffer_address = 0;

        VkQueryPool profile_queries = VK_NULL_HANDLE;
        uint64_t profile_frame_id = 0;
        bool profile_pending = false;
    };

    struct LiveCaptureItem {
        std::string relative_path;
        Buffer staging{};
        VkDeviceSize bytes = 0;
    };

    std::vector<LiveCaptureItem> live_capture_items;

    std::array<Frame, 64> frames{};
    size_t next_frame = 0;

    static void log(const std::string& text, fsr4log::Level level = fsr4log::Level::Info) noexcept;

    double profileMs(uint64_t begin, uint64_t end) const;
    void reportGpuBenchmark() noexcept;
    bool collectGpuProfile(Frame& frame);
    bool beginGpuProfile(VkCommandBuffer command_buffer, Frame& frame);
    void
    profileTimestamp(VkCommandBuffer command_buffer, Frame& frame, bool active, uint32_t query);

    static uint32_t bytesPerPixel(VkFormat format);
    static const char* liveCapturePointName(CapturePoint point);
    const char* liveCaptureModelName();
    void liveCaptureWrite(const fs::path& path, const void* data, size_t bytes) noexcept;
    void liveCaptureImageMeta(const fs::path& path,
                              const char* role,
                              const Image& image,
                              VkImageLayout layout,
                              uint32_t bpp,
                              VkDeviceSize bytes) noexcept;
    void liveCaptureManifest(const Image& color,
                             const Image& depth,
                             const Image& motion,
                             const Image& exposure,
                             const Image& output,
                             const OptimizedConstants& constants,
                             bool run_auto_exposure,
                             bool use_staging_output) noexcept;
    void liveCaptureImage(VkCommandBuffer command_buffer,
                          CapturePoint point,
                          const char* role,
                          const Image& image,
                          VkImageLayout layout);
    void liveCaptureBuffer(VkCommandBuffer command_buffer,
                           CapturePoint point,
                           const char* role,
                           const Buffer& buffer);
    void liveCaptureFlush() noexcept;
    void captureAtStage(CapturePoint point, VkCommandBuffer command_buffer);
    static void liveCaptureHook(void* opaque, CapturePoint point, VkCommandBuffer command_buffer);
    void release();

public:
    QualityContext(VkPhysicalDevice physical_device_,
                   VkDevice device_,
                   const fs::path& shader_dir,
                   const fs::path& weights_path,
                   const fs::path& auto_exposure_shader_path,
                   PFN_vkGetDeviceProcAddr get_device_proc_addr_,
                   ModelRuntimeOptions runtime_options,
                   bool auto_exposure_enabled_,
                   bool depth_inverted_enabled_,
                   bool extensive_logging_,
                   uint32_t max_input_width,
                   uint32_t max_input_height,
                   uint32_t max_output_width,
                   uint32_t max_output_height,
                   uint32_t api_version = VK_API_VERSION_1_3);
    ~QualityContext() override;

    void recordFrame(VkCommandBuffer command_buffer,
                     Image color,
                     Image depth,
                     Image motion,
                     Image exposure_image,
                     Image optional_mask,
                     Image output,
                     OptimizedConstants main_constants,
                     bool use_staging_output = false,
                     ColorSpace color_space = ColorSpace::Linear,
                     fsr4_411::OverlayOptions overlay = {}) override;

private:
    std::span<const VkPipeline> prepareColorSpacePipelines(ColorSpace requested,
                                                           bool debug_view = false);
    Image resampleDisplayMotionVectors(VkCommandBuffer command_buffer,
                                       const Image& source,
                                       uint32_t target_width,
                                       uint32_t target_height);
    Image resolveDepthInput(VkCommandBuffer command_buffer,
                            Image depth,
                            uint32_t active_width,
                            uint32_t active_height);
    Image prepareExactAutoExposureColor(VkCommandBuffer command_buffer,
                                        const Image& source,
                                        uint32_t active_width,
                                        uint32_t active_height);

    struct ShaderDriverPolicy {
        bool windows_amd_shaders_allowed;
        bool rdna2_device;
    };

    struct ModelPipelinePlan {
        bool windows_fp8_optimized = false;
        bool windows_int8_optimized = false;
        std::array<bool, kPasses.size()> require_wave32{};
    };

    ModelPipelinePlan model_pipeline_plan{};

    ModelPipelinePlan prepareFp8Pipelines(const fs::path& shader_dir,
                                          const ShaderDriverPolicy& policy);
    ModelPipelinePlan prepareInt8Pipelines(const ModelRuntimeOptions& options,
                                           const ShaderDriverPolicy& policy);
    void prepareModelWeights(Frame& frame);
    void writeExposureDescriptors(Frame& frame, const Image& exposure);
    void writeStaticExposureDescriptor(Frame& frame);

    struct DescriptorGeometry {
        VkDeviceSize alignment;
        VkDeviceSize max_layout_size;
    };

    ShaderDriverPolicy configureDeviceShaderPolicy();
    void configureLiveCapture();
    void configureGpuProfile();
    // The descriptor functions this needs are already in the dispatch table the
    // runtime loaded for its own barriers and addresses, so the call takes no
    // resolver: one owner resolves every device function.
    DescriptorGeometry createDescriptorLayouts();
    void configureModelPipelines(const fs::path& shader_dir,
                                 const fs::path& weights_path,
                                 const ModelRuntimeOptions& runtime_options,
                                 uint32_t api_version,
                                 const ShaderDriverPolicy& shader_policy);
    void createPipelineCache();
    void savePipelineCache() noexcept;
    void prepareModelPipelines();
    VkPipeline createComputePipeline(const std::vector<uint32_t>& code, bool wave32);
    void retirePipeline(VkPipeline& pipeline);
    void collectFrameUses();
    void beginFrameUse();
    void trackPipelineUse(VkPipeline pipeline);
    void trackInternalImageUse(VkImage image);
    void retireInternalImage(Image& image);
    void prepareRcas(VkCommandBuffer command_buffer,
                     Frame& frame,
                     const Image& output,
                     const Image& exposure,
                     bool use_staging_output);
    void recordRcas(VkCommandBuffer command_buffer,
                    Frame& frame,
                    const Image& output,
                    const OptimizedConstants& constants);
    void recordOverlay(VkCommandBuffer command_buffer,
                       Frame& frame,
                       const Image& output,
                       const Image& motion,
                       const OptimizedConstants& constants,
                       const fsr4_411::OverlayOptions& options);
    void createAutoExposurePipeline(const fs::path& auto_exposure_shader_path);
    void allocateModelBuffers(const fs::path& weights_path,
                              const ModelRuntimeOptions& runtime_options,
                              uint32_t max_input_width,
                              uint32_t max_output_width,
                              uint32_t max_output_height);
    void allocateFrameResources(const DescriptorGeometry& geometry);
    void allocateInternalImages(uint32_t max_input_width,
                                uint32_t max_input_height,
                                uint32_t max_output_width,
                                uint32_t max_output_height);

    void selectModelPreset(uint32_t render_width, OptimizedConstants& main_constants);
    void
    writeFrameConstants(Frame& frame, OptimizedConstants& main_constants, bool run_auto_exposure);
    void initializeFrameHistory(VkCommandBuffer command_buffer,
                                Frame& frame,
                                const OptimizedConstants& main_constants);
    // Takes the frame slot by reference, not const: publishing descriptors to a
    // stage under the sets transport applies that frame's queued writes first.
    void recordAutoExposure(VkCommandBuffer command_buffer,
                            Frame& frame,
                            const OptimizedConstants& main_constants);
    void writeAutoExposureConstants(Frame& frame, const OptimizedConstants& main_constants);
    void prepareStagingOutput(VkCommandBuffer command_buffer,
                              const Image& output,
                              bool use_staging_output);
    void
    copyStagingOutput(VkCommandBuffer command_buffer, const Image& output, bool use_staging_output);

    void prepareExternalViews(Frame& frame,
                              Image& color,
                              Image& depth,
                              Image& motion,
                              Image& exposure,
                              Image& optional_mask,
                              Image& output);
    bool updateDynamicDescriptors(VkCommandBuffer command_buffer,
                                  Frame& frame,
                                  Image& color,
                                  Image& depth,
                                  Image& motion,
                                  Image& exposure_image,
                                  Image& optional_mask,
                                  Image& output,
                                  Image& reprojected,
                                  bool use_staging_output);
    void transitionExternalInputs(VkCommandBuffer command_buffer,
                                  const Image& color,
                                  const Image& depth,
                                  const Image& motion,
                                  const Image& exposure,
                                  const Image& optional_mask,
                                  const Image& output);

    void writeImageDescriptorBuffer(
        Frame& frame,
        uint32_t set_index,
        uint32_t binding_index,
        VkDescriptorType type,
        VkImageView view,
        VkImageLayout image_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    uint8_t* descriptorAddress(Frame& frame, uint32_t set_index, uint32_t binding_index);
    void writeSampledImageDescriptor(
        Frame& frame,
        uint32_t set_index,
        uint32_t binding_index,
        VkImageView view,
        VkImageLayout image_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    void writeStorageImageDescriptor(Frame& frame,
                                     uint32_t set_index,
                                     uint32_t binding_index,
                                     VkImageView view);
    void writeSamplerDescriptor(Frame& frame,
                                uint32_t set_index,
                                uint32_t binding_index,
                                VkSampler sampler_handle);
    void writeStorageBufferDescriptor(Frame& frame,
                                      uint32_t set_index,
                                      uint32_t binding_index,
                                      VkBuffer buffer,
                                      VkDeviceSize size);
    void fillStorageBufferDescriptors(Frame& frame,
                                      uint32_t set_index,
                                      VkBuffer buffer,
                                      VkDeviceSize size);
    void writeUniformBufferDescriptor(Frame& frame,
                                      uint32_t set_index,
                                      uint32_t binding_index,
                                      VkBuffer buffer,
                                      VkDeviceSize size);
    void writeWeightDescriptors(Frame& frame, const Buffer& model_data);

    // Conventional descriptor sets, for the fallback transport. Writes are queued
    // per frame and applied at the next bind, because vkCmdBindDescriptorSets reads
    // the set contents when the command is recorded rather than when the frame
    // runs, so a stage that writes descriptors after another stage bound them has
    // to bind again before it dispatches.
    void allocateDescriptorSets(Frame& frame);
    void queueDescriptorWrite(Frame& frame,
                              uint32_t set_index,
                              uint32_t binding_index,
                              VkDescriptorType type,
                              const VkDescriptorImageInfo* image_info,
                              const VkDescriptorBufferInfo* buffer_info);
    void flushDescriptorWrites(Frame& frame);
    void bindDescriptors(VkCommandBuffer command_buffer, Frame& frame);

    void writeStaticDescriptors(Frame& frame);
    void writeAllStaticDescriptors();
};

} // namespace fsr4runtime
