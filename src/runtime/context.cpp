// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 fsr4-vk contributors.
// See LICENSE in the project root.

#include "runtime/context.hpp"
#include "assets/assets.hpp"
#include "diagnostics/diagnostics.hpp"

#include <cstdio>
#include <mutex>

namespace fsr4runtime {

void QualityContext::release() {
    if (!device) {
        return;
    }

    vkDeviceWaitIdle(device);
    for (auto& frame : frames) {
        if (!collectGpuProfile(frame)) {
            log("[gpu-profile] pending query was unavailable during release",
                fsr4log::Level::Warning);
        }
    }
    reportGpuBenchmark();

    if (live_capture_pending) {
        liveCaptureFlush();
    }

    for (auto& item : live_capture_items) {
        destroyBuffer(device, item.staging);
    }

    live_capture_items.clear();
    if (live_capture_completion) {
        vkDestroyEvent(device, live_capture_completion, nullptr);
        live_capture_completion = VK_NULL_HANDLE;
    }

    for (auto& frame : frames) {
        for (const auto& cached_view : frame.image_view_cache) {
            vkDestroyImageView(device, cached_view.view, nullptr);
        }

        destroyBuffer(device, frame.constants);
        destroyBuffer(device, frame.model_constants);
        destroyBuffer(device, frame.exposure_upload);
        if (frame.descriptor_pool) {
            // Destroying the pool releases the six sets allocated from it, which
            // is why the sets transport has no per-set destroy call anywhere; the
            // buffers transport instead releases a byte buffer per frame slot.
            vkDestroyDescriptorPool(device, frame.descriptor_pool, nullptr);
        } else {
            destroyBuffer(device, frame.descriptor_buffer);
        }

        if (frame.profile_queries) {
            vkDestroyQueryPool(device, frame.profile_queries, nullptr);
        }
    }

    destroyImage(device, rcas_input);
    if (rcas_pipeline.handle) {
        vkDestroyPipeline(device, rcas_pipeline.handle, nullptr);
    }

    if (staging_output.image) {
        destroyImage(device, staging_output);
    }

    for (auto& image : retired_staging_outputs) {
        destroyImage(device, image);
    }
    for (auto& image : retired_internal_images) {
        destroyImage(device, image);
    }

    for (auto& entry : auto_exposure_color_cache) {
        if (entry.image.image) {
            destroyImage(device, entry.image);
        }
    }

    auto_exposure_color_cache.clear();

    if (fallback_depth.image) {
        destroyImage(device, fallback_depth);
    }

    if (render_motion.image) {
        destroyImage(device, render_motion);
    }

    for (auto& image : images) {
        destroyImage(device, image);
    }

    destroyBuffer(device, scratch_buffer);
    destroyBuffer(device, weight_buffer);
    destroyBuffer(device, ultraperformance_weight_buffer);

    for (auto pipeline : pipelines) {
        if (pipeline) {
            vkDestroyPipeline(device, pipeline, nullptr);
        }
    }

    for (auto pipeline : retired_pipelines) {
        vkDestroyPipeline(device, pipeline, nullptr);
    }
    for (const auto& use : pending_frame_uses) {
        vkDestroyEvent(device, use.completion, nullptr);
    }
    for (auto event : available_frame_events) {
        vkDestroyEvent(device, event, nullptr);
    }
    if (overlay_pipeline) {
        vkDestroyPipeline(device, overlay_pipeline, nullptr);
    }
    destroyBuffer(device, overlay_font);

    if (auto_exposure_pipeline) {
        vkDestroyPipeline(device, auto_exposure_pipeline, nullptr);
    }

    if (pipeline_layout) {
        vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
    }
    savePipelineCache();
    if (pipeline_cache) {
        vkDestroyPipelineCache(device, pipeline_cache, nullptr);
    }

    for (auto& layout : set_layouts) {
        if (layout) {
            vkDestroyDescriptorSetLayout(device, layout, nullptr);
        }
    }

    if (sampler) {
        vkDestroySampler(device, sampler, nullptr);
    }
}

QualityContext::QualityContext(VkPhysicalDevice physical_device_,
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
                               uint32_t api_version)
    : physical_device(physical_device_), device(device_), use_wmma(runtime_options.use_wmma),
      use_int8_dispatch(runtime_options.use_int8_dispatch),
      model_pass_policy(runtime_options.pass_policy),
      numbered_shader_files(runtime_options.numbered_shader_files),
      execute_padding_passes(runtime_options.execute_padding_passes),
      direct_display_motion_vectors(runtime_options.direct_display_motion_vectors),
      auto_exposure_enabled(auto_exposure_enabled_), extensive_logging(extensive_logging_),
      depth_inverted_enabled(depth_inverted_enabled_), max_render_width(max_input_width),
      max_render_height(max_input_height),
      model_bucket_width(fsr4::modelBucket(max_output_width, max_output_height)[0]),
      model_bucket_height(fsr4::modelBucket(max_output_width, max_output_height)[1]) {
    try {
        // One snapshot answers every device question this context has: the
        // transport that carries descriptors, the shader packaging the driver
        // wants, and the model the assets may run. It is taken before the entry
        // point table is resolved, because that table is now transport-scoped:
        // the five VK_EXT_descriptor_buffer rows are only in the contract when
        // the buffers transport won its one distinguishing question.
        if (auto snapshot = fsr4vk::detectCached(physical_device)) {
            device_capabilities = *snapshot;
        }
        if (auto forced = fsr4vk::descriptorBackendOverride()) {
            descriptor_backend = *forced;
        } else {
            descriptor_backend = fsr4vk::selectDescriptorBackend(
                device_capabilities, device, get_device_proc_addr_);
        }
        const auto backend_blockers =
            fsr4vk::descriptorBackendBlockers(device_capabilities, descriptor_backend);
        if (!backend_blockers.empty()) {
            throw std::runtime_error(
                "descriptor backend '" +
                std::string(fsr4vk::descriptorBackendName(descriptor_backend)) +
                "' is unavailable: " + fsr4vk::joinBlockers(backend_blockers));
        }
        log("[descriptors] backend=" +
            std::string(fsr4vk::descriptorBackendName(descriptor_backend)) + " " +
            std::string(device_capabilities.supportsDescriptorBuffers() ? "buffers=yes"
                                                                       : "buffers=no") +
            " sets=" + std::string(device_capabilities.supportsDescriptorSets() ? "yes"
                                                                                : "no"));
        dispatch.load(device, get_device_proc_addr_, api_version, descriptor_backend);
        log("[entry-points] " + dispatch.describe());
        this->auto_exposure_shader_path = auto_exposure_shader_path;
        color_shader_root = runtime_options.color_shader_root;
        color_endpoints = runtime_options.color_endpoints;
        color_api_version = api_version;
        this->max_output_width = max_output_width;
        this->max_output_height = max_output_height;
        rcas_shader_root = runtime_options.rcas_shader_root;

        if (direct_display_motion_vectors) {
            log("[motion] direct display-resolution prepass sampling enabled");
        }

        if (model_pass_policy.passes.empty() || model_pass_policy.passes.size() > kPasses.size() ||
            !model_pass_policy.make_constants) {
            throw std::invalid_argument("unsupported model pass/constant layout");
        }
        const auto shader_policy = configureDeviceShaderPolicy();
        configureLiveCapture();
        configureGpuProfile();
        createPipelineCache();

        const auto descriptor_geometry = createDescriptorLayouts();
        configureModelPipelines(
            shader_dir, weights_path, runtime_options, api_version, shader_policy);

        allocateModelBuffers(
            weights_path, runtime_options, max_input_width, max_output_width, max_output_height);
        allocateFrameResources(descriptor_geometry);
        allocateInternalImages(
            max_input_width, max_input_height, max_output_width, max_output_height);

        writeAllStaticDescriptors();

    } catch (...) {
        release();
        throw;
    }
}

QualityContext::~QualityContext() {
    release();
}

void QualityContext::allocateModelBuffers(const fs::path& weights_path,
                                          const ModelRuntimeOptions& runtime_options,
                                          uint32_t max_input_width,
                                          uint32_t max_output_width,
                                          uint32_t max_output_height) {
    scratch_buffer = createBuffer(
        device_capabilities.memory,
        device,
        fsr4::scratchCapacity(max_output_width, max_output_height),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        true);

    const auto weight_bytes = readBytes(weights_path);
    weight_buffer = createBuffer(
        device_capabilities.memory,
        device,
        std::max<VkDeviceSize>(weight_bytes.size(), 256),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        true,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    std::memcpy(weight_buffer.mapped, weight_bytes.data(), weight_bytes.size());

    if (!runtime_options.ultraperformance_model_data_path.empty()) {
        const auto ultraperformance_weight_bytes =
            readBytes(runtime_options.ultraperformance_model_data_path);
        ultraperformance_weight_buffer = createBuffer(
            device_capabilities.memory,
            device,
            std::max<VkDeviceSize>(ultraperformance_weight_bytes.size(), 256),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            true,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        std::memcpy(ultraperformance_weight_buffer.mapped,
                    ultraperformance_weight_bytes.data(),
                    ultraperformance_weight_bytes.size());
    }

    const auto log_memory = [&](const char* role, const Buffer& buffer) {
        std::ostringstream message;
        message << "[memory] role=" << role << " size_bytes=" << buffer.size
                << " memory_type=" << buffer.memory_type_index
                << " heap=" << buffer.memory_heap_index
                << " properties=" << fsr4log::memoryProperties(buffer.memory_property_flags);
        log(message.str());
    };
    log_memory("scratch", scratch_buffer);
    log_memory("weights", weight_buffer);

    if (ultraperformance_weight_buffer.buffer) {
        log_memory("ultraperf-weights", ultraperformance_weight_buffer);
    }

    current_preset = fsr4::resolutionPreset(max_input_width, max_output_width);
    log("[model] initial_preset=" + current_preset + " weights_asset='" + weights_path.string() +
        "'");
}

void QualityContext::allocateFrameResources(const DescriptorGeometry& geometry) {
    const auto descriptor_offset_alignment = geometry.alignment;
    const auto max_layout_size = geometry.max_layout_size;
    const bool use_descriptor_sets =
        descriptor_backend == DescriptorBackend::DescriptorSets;

    // Alignment and capacity are byte arithmetic, so they belong to the buffers
    // transport alone; a sets backend reports both geometry fields as zero and
    // needs no byte buffer per frame slot at all.
    if (!use_descriptor_sets && !descriptor_offset_alignment) {
        throw std::runtime_error("invalid descriptor-buffer alignment");
    }

    if (!use_descriptor_sets) {
        descriptor_capacity =
            ((max_layout_size + descriptor_offset_alignment - 1) / descriptor_offset_alignment) *
            descriptor_offset_alignment;
        log("[descriptor] max_layout_bytes=" + std::to_string(max_layout_size) +
            " alignment_bytes=" + std::to_string(descriptor_offset_alignment) +
            " capacity_bytes_per_set=" + std::to_string(descriptor_capacity));
    }

    for (auto& frame : frames) {
        frame.constants = createBuffer(
            device_capabilities.memory,
            device,
            kConstantBufSize,
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            true);
        frame.model_constants = createBuffer(
            device_capabilities.memory,
            device,
            kConstantBufSize,
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            true);
        frame.exposure_upload = createBuffer(
            device_capabilities.memory,
            device,
            8,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        if (use_descriptor_sets) {
            allocateDescriptorSets(frame);
        } else {
            frame.descriptor_buffer =
                createBuffer(device_capabilities.memory,
                             device,
                             descriptor_capacity * kNumDescriptorSets,
                             VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT |
                                 VK_BUFFER_USAGE_SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT |
                                 VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                             true);
            std::memset(frame.descriptor_buffer.mapped,
                        0,
                        static_cast<std::size_t>(frame.descriptor_buffer.size));
        }

        frame.constants_address =
            deviceAddress(dispatch.get_buffer_device_address, device, frame.constants.buffer);
        if (!use_descriptor_sets) {
            frame.descriptor_buffer_address = deviceAddress(
                dispatch.get_buffer_device_address, device, frame.descriptor_buffer.buffer);
        }

        if (gpu_profile_enabled) {
            VkQueryPoolCreateInfo query_info{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
            query_info.queryType = VK_QUERY_TYPE_TIMESTAMP;
            query_info.queryCount = kProfileQueryCount;
            check(vkCreateQueryPool(device, &query_info, nullptr, &frame.profile_queries),
                  "vkCreateQueryPool(FSR4 GPU profile)");
        }
    }

    // The descriptor half of this line differs by transport on purpose: the buffers
    // path spends bytes per frame slot, the sets path spends sets and pool entries,
    // and a log that claimed one number for both would hide which is which.
    log("[frame] resources allocated slots=" + std::to_string(frames.size()) +
        " constants_bytes_per_slot=" + std::to_string(frames.front().constants.size) +
        (use_descriptor_sets ? std::string(" descriptor_sets_per_slot=") +
                                  std::to_string(frames.front().descriptor_sets.size()) +
                                  " dynamic_offsets_per_bind=" +
                                  std::to_string(descriptor_dynamic_offsets.size())
                              : std::string(" descriptor_bytes_per_slot=") +
                                    std::to_string(frames.front().descriptor_buffer.size)) +
        " profile_queries_per_slot=" +
        std::to_string(gpu_profile_enabled ? kProfileQueryCount : 0));

    if (extensive_logging) {
        const auto log_device_address_range = [&](const char* name, const Buffer& buffer) {
            if (!buffer.buffer) {
                return;
            }

            const VkDeviceAddress address =
                deviceAddress(dispatch.get_buffer_device_address, device, buffer.buffer);
            std::ostringstream message;
            message << "[address] " << name << "=[0x" << std::hex << address << ",0x"
                    << (address + buffer.size) << ")" << std::dec << " bytes=" << buffer.size;
            log(message.str(), fsr4log::Level::Debug);
        };
        log_device_address_range("scratch", scratch_buffer);
        log_device_address_range("weights", weight_buffer);
        log_device_address_range("ultraperf-weights", ultraperformance_weight_buffer);

        for (size_t frame_index = 0; frame_index < frames.size(); ++frame_index) {
            const auto& frame = frames[frame_index];
            std::ostringstream message;
            message << "[address] frame_slot=" << frame_index << " constants=[0x" << std::hex
                    << frame.constants_address << ",0x"
                    << (frame.constants_address + frame.constants.size) << ") descriptor=[0x"
                    << frame.descriptor_buffer_address << ",0x"
                    << (frame.descriptor_buffer_address + frame.descriptor_buffer.size) << ")"
                    << std::dec;
            log(message.str(), fsr4log::Level::Debug);
        }
    }
}

void QualityContext::allocateInternalImages(uint32_t max_input_width,
                                            uint32_t max_input_height,
                                            uint32_t max_output_width,
                                            uint32_t max_output_height) {
    const auto padded_output_width = fsr4::roundUp(max_output_width, 8);
    const auto padded_output_height = fsr4::roundUp(max_output_height, 8);
    const auto image_usage_flags = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT |
                                   VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                   VK_IMAGE_USAGE_TRANSFER_DST_BIT;

    images.resize(auto_exposure_enabled ? image_slot::exposure_mips + 1
                                        : image_slot::fallback_mask + 1);
    images[image_slot::history] = createImage(device_capabilities.memory,
                                              device,
                                              padded_output_width,
                                              padded_output_height,
                                              VK_FORMAT_R16G16B16A16_SFLOAT,
                                              image_usage_flags);
    images[image_slot::reprojected] = createImage(device_capabilities.memory,
                                                  device,
                                                  padded_output_width,
                                                  padded_output_height,
                                                  VK_FORMAT_R16G16B16A16_SFLOAT,
                                                  image_usage_flags);
    images[image_slot::recurrent] = createImage(device_capabilities.memory,
                                                device,
                                                padded_output_width,
                                                padded_output_height,
                                                VK_FORMAT_R8G8B8A8_UNORM,
                                                image_usage_flags);
    images[image_slot::wmma_scratch0] = createImage(device_capabilities.memory,
                                                    device,
                                                    padded_output_width,
                                                    padded_output_height,
                                                    VK_FORMAT_R16G16B16A16_SFLOAT,
                                                    image_usage_flags);
    images[image_slot::wmma_scratch1] = createImage(device_capabilities.memory,
                                                    device,
                                                    padded_output_width,
                                                    padded_output_height,
                                                    VK_FORMAT_R16G16B16A16_SFLOAT,
                                                    image_usage_flags);
    // Prepass reads current and previous exposure from this 2x1 image.
    images[image_slot::exposure] =
        createImage(device_capabilities.memory, device, 2, 1, VK_FORMAT_R32_SFLOAT,
                    image_usage_flags);
    images[image_slot::fallback_mask] =
        createImage(device_capabilities.memory, device, 1, 1, VK_FORMAT_R8_UNORM,
                    image_usage_flags);

    if (auto_exposure_enabled) {
        images[image_slot::exposure_counter] =
            createImage(device_capabilities.memory, device, 1, 1, VK_FORMAT_R32_UINT,
                        image_usage_flags);
        images[image_slot::exposure_mips] = createImage(device_capabilities.memory,
                                                        device,
                                                        fsr4::divideUp(max_input_width, 16),
                                                        fsr4::divideUp(max_input_height, 16),
                                                        VK_FORMAT_R32_SFLOAT,
                                                        image_usage_flags);
    }

    fsr4vk::setDebugName(dispatch,
                         device,
                         VK_OBJECT_TYPE_BUFFER,
                         (uint64_t)scratch_buffer.buffer,
                         "FSR4_ScratchBuffer");
    fsr4vk::setDebugName(dispatch,
                         device,
                         VK_OBJECT_TYPE_BUFFER,
                         (uint64_t)weight_buffer.buffer,
                         "FSR4_WeightBuffer");
    fsr4vk::setDebugName(dispatch,
                         device,
                         VK_OBJECT_TYPE_IMAGE,
                         (uint64_t)images[image_slot::history].image,
                         "FSR4_HistoryImage");
    fsr4vk::setDebugName(dispatch,
                         device,
                         VK_OBJECT_TYPE_IMAGE,
                         (uint64_t)images[image_slot::reprojected].image,
                         "FSR4_ReprojectedImage");
    fsr4vk::setDebugName(dispatch,
                         device,
                         VK_OBJECT_TYPE_IMAGE,
                         (uint64_t)images[image_slot::recurrent].image,
                         "FSR4_RecurrentImage");

    if (auto_exposure_enabled) {
        fsr4vk::setDebugName(dispatch,
                             device,
                             VK_OBJECT_TYPE_IMAGE,
                             (uint64_t)images[image_slot::exposure].image,
                             "FSR4_AutoExposure");
        fsr4vk::setDebugName(dispatch,
                             device,
                             VK_OBJECT_TYPE_IMAGE,
                             (uint64_t)images[image_slot::exposure_counter].image,
                             "FSR4_AutoExposureAtomic");
        fsr4vk::setDebugName(dispatch,
                             device,
                             VK_OBJECT_TYPE_IMAGE,
                             (uint64_t)images[image_slot::exposure_mips].image,
                             "FSR4_AutoExposureMips");
    }
}

void QualityContext::log(const std::string& text, fsr4log::Level level) noexcept {
    fsr4runtime::log(text, level);
}

} // namespace fsr4runtime

namespace fsr4runtime {
namespace fs = std::filesystem;

fs::path ModelShaderSelector::stockShaderPath(const fs::path& directory,
                                              uint32_t pass_index,
                                              const fs::path& prepass_override,
                                              const fs::path& postpass_override) const {
    const auto pass_name = kPasses[pass_index].name;
    char numbered_name[32]{};
    std::snprintf(numbered_name, sizeof(numbered_name), "pass-%02u.spv", pass_index);

    std::string filename =
        options.numbered_shader_files ? numbered_name : std::string(pass_name) + ".spv";

    fs::path shader_path = directory / filename;

    if (pass_index == kPrepassIndex && !prepass_override.empty()) {
        shader_path = prepass_override;
    }

    if (options.numbered_shader_files && pass_index == kPostpassIndex &&
        !postpass_override.empty()) {
        shader_path = postpass_override;
    }

    return shader_path;
}

fs::path ModelShaderSelector::select(const fs::path& directory,
                                     uint32_t pass_index,
                                     const fs::path& prepass_override,
                                     const fs::path& postpass_override) {
    return options.use_int8_dispatch
               ? selectInt8Shader(directory, pass_index, prepass_override, postpass_override)
               : selectFp8Shader(directory, pass_index, prepass_override, postpass_override);
}
} // namespace fsr4runtime

namespace fsr4runtime {

QualityContext::ShaderDriverPolicy QualityContext::configureDeviceShaderPolicy() {
    // The provider already probed this device to choose a model; the runtime
    // reads that cached snapshot so the vendor, subgroup and matrix-shape
    // questions are answered once per device rather than once per concern.
    if (auto snapshot = fsr4vk::detectCached(physical_device)) {
        device_capabilities = *snapshot;
    }
    const auto& properties = device_capabilities.properties;
    const auto& driver = device_capabilities.driver;

    bool force_windows_amd_shaders = false;

    if (const char* value = std::getenv("FSR4_VK_FORCE_WINDOWS_AMD_SHADERS")) {
        force_windows_amd_shaders = std::strcmp(value, "0") != 0;
    }
#if defined(FSR4_NATIVE_ASSETS_ONLY)
    constexpr bool windows_shader_assets_available = false;
#else
    constexpr bool windows_shader_assets_available = true;
#endif
    const bool windows_amd_shaders_allowed =
        fsr4vk::allowWindowsAmdShaders(properties.vendorID,
                                       driver.driverID,
                                       force_windows_amd_shaders,
                                       windows_shader_assets_available);
    // The RDNA2 packaging verdict decides the wave32 pin and the AMD shader-pack
    // allowance. FSR4_VK_DEVICE_CLASS can pin or lift it for a device this release's
    // ID table does not name, which is the honest way to cover an AMD APU family:
    // the device IDs that would make it a rule have only ever been measured by their
    // owners, so the line below prints what the device does report and the operator
    // supplies the fact instead of the runtime inventing one.
    const auto device_class_value = std::getenv("FSR4_VK_DEVICE_CLASS");
    const auto pinned_rdna2 = fsr4vk::parseDeviceClass(device_class_value);
    const bool integrated_apu = device_capabilities.amdIntegratedApu();
    const bool rdna2_device = pinned_rdna2.value_or(device_capabilities.rdna2Device());
    device_is_rdna2 = rdna2_device;
    default_subgroup_size = device_capabilities.subgroups.subgroupSize;
    uint32_t minimum = default_subgroup_size, maximum = default_subgroup_size;
    // Subgroup size control is core in 1.3 and the probe reads the structure
    // whenever the loader understands it; a driver that left it untouched keeps
    // the default subgroup size as the only label worth trusting.
    if (properties.apiVersion >= VK_API_VERSION_1_3 &&
        device_capabilities.subgroup_sizes.minSubgroupSize != 0) {
        minimum = device_capabilities.subgroup_sizes.minSubgroupSize;
        maximum = device_capabilities.subgroup_sizes.maxSubgroupSize;
    }
    automatic_wave_label = "Wave" + std::to_string(minimum);
    if (minimum != maximum) {
        automatic_wave_label += (maximum == minimum * 2 ? "/" : "-") + std::to_string(maximum);
    }
    automatic_wave_label += " (auto)";

    const bool amd_proprietary = device_capabilities.amdProprietaryDriver();
    const std::string shape_summary = device_capabilities.matrixShapeSummary();

    // What decided the RDNA2 verdict, spelled out for the log: a pin the operator
    // asked for, a value the parser refused (naming it, so a typo is visible
    // rather than silently ignored), or the table's own answer.
    std::string device_class_label = "auto";
    if (pinned_rdna2.has_value()) {
        const std::string spelled(device_class_value);
        device_class_label = *pinned_rdna2 ? "pinned-rdna2(" + spelled + ")"
                                           : "pinned-not-rdna2(" + spelled + ")";
    } else if (device_class_value && *device_class_value) {
        device_class_label = std::string("refused(") + device_class_value + ")";
    }

    std::ostringstream policy_message;
    policy_message << "[device-policy] device="
                   << std::quoted(properties.deviceName)
                   << " vendor_id=" << fsr4log::hexValue(properties.vendorID)
                   << " device_id=" << fsr4log::hexValue(properties.deviceID)
                   << " driver_id=" << driver.driverID
                   << " driver_name=" << std::quoted(driver.driverName)
                   << " driver_info=" << std::quoted(driver.driverInfo)
                   << " amd_proprietary=" << (amd_proprietary ? "yes" : "no")
                   << " rdna2=" << (rdna2_device ? "yes" : "no")
                   << " device_type=" << properties.deviceType
                   << " integrated_amd=" << (integrated_apu ? "yes" : "no")
                   << " device_class=" << device_class_label
                   << " windows_shaders_available="
                   << (windows_shader_assets_available ? "yes" : "no")
                   << " windows_shaders_allowed=" << (windows_amd_shaders_allowed ? "yes" : "no")
                   << " force_windows_shaders=" << (force_windows_amd_shaders ? "yes" : "no")
                   << " matrix_shapes=" << shape_summary
                   << " matrix_probe=" << device_capabilities.matrixProbeRouteName()
                   << " wmma=" << (use_wmma ? "yes" : "no")
                   << " descriptor_sizes=" << device_capabilities.descriptorSizeSummary();
    log(policy_message.str());

    // An AMD device that reports itself integrated and whose device ID the RDNA2
    // table does not name: very likely an APU, very likely wanting RDNA2-shaped
    // waves, and the runtime has no measured fact to act on. Saying so, with the
    // exact ID the table would need, is what turns this into a rule later instead
    // of a guess now.
    if (integrated_apu && !rdna2_device) {
        log("[device-policy] this AMD device reports itself integrated (device_id=" +
                fsr4log::hexValue(properties.deviceID) +
                ") and no RDNA2 device ID matched, so RDNA2 packaging was not applied: "
                "if this is a Ryzen-class APU, FSR4_VK_DEVICE_CLASS=rdna2 pins it, and "
                "that device_id is the value isRdna2DeviceId needs before this becomes "
                "a rule rather than an override.",
            fsr4log::Level::Warning);
    }

    return {windows_amd_shaders_allowed, rdna2_device};
}

void QualityContext::configureModelPipelines(const fs::path& shader_dir,
                                             const fs::path& weights_path,
                                             const ModelRuntimeOptions& runtime_options,
                                             uint32_t api_version,
                                             const ShaderDriverPolicy& shader_policy) {
    const auto passes = model_pass_policy.passes;
    ModelPipelinePlan plan;
    if (use_int8_dispatch) {
        plan = prepareInt8Pipelines(runtime_options, shader_policy);
    } else {
        plan = prepareFp8Pipelines(shader_dir, shader_policy);
    }

    model_pipeline_plan = plan;
    color_wave32 = {plan.require_wave32.front(), plan.require_wave32[passes.size() - 1]};
    log(std::string("[shader-source] source=") + "original version=4.1.2 model=" +
        (model_pass_policy.model_name ? model_pass_policy.model_name
         : use_int8_dispatch          ? "int8"
                                      : "fp8"));

    ModelShaderSelector shader_selector(
        ShaderSelectionOptions{
            .use_wmma = use_wmma,
            .use_int8_dispatch = use_int8_dispatch,
            .numbered_shader_files = numbered_shader_files,
            .windows_fp8_optimized = plan.windows_fp8_optimized,
            .windows_int8_optimized = plan.windows_int8_optimized,
            .extensive_logging = extensive_logging,
        },
        runtime_options.shader_recipe);

    pipelines.resize(passes.size());
    active_shader_paths.resize(passes.size());
    model_shader_paths[0].resize(passes.size());
    if (!runtime_options.ultraperformance_shader_dir.empty()) {
        model_shader_paths[1].resize(passes.size());
    }
    for (uint32_t index = 0; index < passes.size(); ++index) {
        const auto type = passes[index].type;
        if (type == PassType::Prepass || type == PassType::Postpass ||
            (type == PassType::TensorPadding &&
             (!execute_padding_passes ||
              model_binding::paddingLayer(index) == model_binding::unused_padding_layer))) {
            continue;
        }
        const auto standard = shader_selector.select(shader_dir, index, {}, {});
        model_shader_paths[0][index] = standard;
        if (!model_shader_paths[1].empty()) {
            auto ultra =
                shader_selector.select(runtime_options.ultraperformance_shader_dir, index, {}, {});
            // Identical programs share their active handle across preset changes.
            if (ultra != standard &&
                readModelSpirv(ultra, api_version) == readModelSpirv(standard, api_version)) {
                ultra = standard;
            }
            model_shader_paths[1][index] = std::move(ultra);
        }
    }
}

} // namespace fsr4runtime

namespace fsr4runtime {
VkPipeline QualityContext::createComputePipeline(const std::vector<uint32_t>& code, bool wave32) {
    VkShaderModuleCreateInfo module_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    module_info.codeSize = code.size() * sizeof(uint32_t);
    module_info.pCode = code.data();
    VkShaderModule module{};
    check(vkCreateShaderModule(device, &module_info, nullptr, &module), "compute shader module");
    // The shipped shaders ask for full 32-lane subgroups and the stage below pins
    // that size, which the driver only accepts when it reports subgroup size
    // control and a range containing 32. Naming both inputs beats a pipeline
    // creation failure with no reason attached; the provider already refuses such
    // devices, so this line only fires when the runtime is used without that gate.
    if (wave32 && !device_capabilities.supportsRequiredSubgroupSize(32) &&
        !wave32_diagnostic_logged) {
        wave32_diagnostic_logged = true;
        std::ostringstream pinned;
        pinned << "[wave32] pinned 32-lane subgroups need reported range and size control ("
                << device_capabilities.subgroup_sizes.minSubgroupSize << '-'
                << device_capabilities.subgroup_sizes.maxSubgroupSize
                << " with size control "
                << (device_capabilities.subgroup_size_control.subgroupSizeControl ? "yes" : "no")
                << "); pipeline creation is expected to fail";
        log(pinned.str());
    }
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo subgroup{
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO};
    subgroup.requiredSubgroupSize = 32;
    VkComputePipelineCreateInfo info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    if (descriptor_backend == DescriptorBackend::DescriptorBuffers) {
        info.flags = VK_PIPELINE_CREATE_DESCRIPTOR_BUFFER_BIT_EXT;
    }
    info.layout = pipeline_layout;
    info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    info.stage.module = module;
    info.stage.pName = "main";
    if (wave32) {
        info.stage.flags = VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT;
        info.stage.pNext = &subgroup;
    }
    VkPipeline pipeline{};
    const auto result =
        vkCreateComputePipelines(device, pipeline_cache, 1, &info, nullptr, &pipeline);
    vkDestroyShaderModule(device, module, nullptr);
    if (result != VK_SUCCESS && pipeline) {
        vkDestroyPipeline(device, pipeline, nullptr);
    }
    check(result, "compute pipeline");
    return pipeline;
}

namespace {
std::mutex pipeline_cache_file_mutex;

fs::path pipelineCacheRoot() {
    if (const char* root = std::getenv("FSR4_VK_PIPELINE_CACHE_PATH")) {
        if (*root) {
            return root;
        }
    }
#if defined(_WIN32)
    if (const char* local = std::getenv("LOCALAPPDATA")) {
        return fs::path(local) / "fsr4-vk";
    }
#else
    if (const char* xdg = std::getenv("XDG_CACHE_HOME")) {
        if (*xdg) {
            return fs::path(xdg) / "fsr4-vk";
        }
    }
    if (const char* home = std::getenv("HOME")) {
        return fs::path(home) / ".cache" / "fsr4-vk";
    }
#endif
    return {};
}

std::uint64_t pipelineDeviceFingerprint(const fsr4vk::GpuCapabilities& capabilities) {
    std::uint64_t hash = 1469598103934665603ull;
    const auto consume = [&hash](const void* data, std::size_t size) {
        const auto* bytes = static_cast<const std::uint8_t*>(data);
        for (std::size_t index = 0; index < size; ++index) {
            hash ^= bytes[index];
            hash *= 1099511628211ull;
        }
    };
    consume(capabilities.identity.deviceUUID, sizeof(capabilities.identity.deviceUUID));
    consume(&capabilities.properties.vendorID, sizeof(capabilities.properties.vendorID));
    consume(&capabilities.properties.deviceID, sizeof(capabilities.properties.deviceID));
    consume(&capabilities.properties.driverVersion, sizeof(capabilities.properties.driverVersion));
    consume(&capabilities.driver.driverID, sizeof(capabilities.driver.driverID));
    return hash;
}
} // namespace

void QualityContext::createPipelineCache() {
    VkPipelineCacheCreateInfo info{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
    std::vector<uint8_t> initial_data;
    const char* disabled = std::getenv("FSR4_VK_PIPELINE_CACHE_DISABLED");
    const bool persistent = !disabled || std::strcmp(disabled, "0") == 0;
    if (persistent) {
        auto root = pipelineCacheRoot();
        if (!root.empty()) {
            const auto fingerprint = pipelineDeviceFingerprint(device_capabilities);
            std::ostringstream filename;
            filename << "v2-" << std::hex << fingerprint << ".bin";
            if (const char* exact = std::getenv("FSR4_VK_PIPELINE_CACHE_PATH");
                exact && *exact) {
                pipeline_cache_path = exact;
            } else {
                pipeline_cache_path = root / filename.str();
            }
        }
    }

    std::lock_guard<std::mutex> lock(pipeline_cache_file_mutex);
    if (!pipeline_cache_path.empty()) {
        std::error_code error;
        if (fs::exists(pipeline_cache_path, error) && !error) {
            const auto cache_size = fs::file_size(pipeline_cache_path, error);
            constexpr std::uintmax_t kMaximumCacheBytes = 256u * 1024u * 1024u;
            if (!error && cache_size && cache_size <= kMaximumCacheBytes) {
                std::ifstream stream(pipeline_cache_path, std::ios::binary);
                initial_data.resize(static_cast<std::size_t>(cache_size));
                stream.read(reinterpret_cast<char*>(initial_data.data()),
                            static_cast<std::streamsize>(initial_data.size()));
                if (!stream) {
                    log("[pipeline-cache] read failed path='" + pipeline_cache_path.string() +
                            "'; rebuilding",
                        fsr4log::Level::Warning);
                    initial_data.clear();
                }
            } else if (error) {
                log("[pipeline-cache] stat failed path='" + pipeline_cache_path.string() +
                        "' error=" + error.message(),
                    fsr4log::Level::Warning);
            } else if (cache_size > kMaximumCacheBytes) {
                log("[pipeline-cache] oversized cache ignored path='" +
                        pipeline_cache_path.string() + "'",
                    fsr4log::Level::Warning);
            }
        } else if (error) {
            log("[pipeline-cache] probe failed path='" + pipeline_cache_path.string() +
                    "' error=" + error.message(),
                fsr4log::Level::Warning);
        }
    }

    info.initialDataSize = initial_data.size();
    info.pInitialData = initial_data.empty() ? nullptr : initial_data.data();
    VkResult result = vkCreatePipelineCache(device, &info, nullptr, &pipeline_cache);
    if (result != VK_SUCCESS && info.initialDataSize != 0) {
        initial_data.clear();
        info.initialDataSize = 0;
        info.pInitialData = nullptr;
        result = vkCreatePipelineCache(device, &info, nullptr, &pipeline_cache);
        log("[pipeline-cache] rejected cache rebuilt path='" +
                pipeline_cache_path.string() + "'",
            fsr4log::Level::Warning);
    }
    check(result, "vkCreatePipelineCache");
    log("[pipeline-cache] status=" +
        std::string(initial_data.empty() ? "cold" : "loaded") + " persistent=" +
        (pipeline_cache_path.empty() ? "disabled" : "enabled") + " path='" +
        pipeline_cache_path.string() + "'");
}

void QualityContext::savePipelineCache() noexcept {
    if (!pipeline_cache || pipeline_cache_path.empty()) {
        return;
    }
    std::lock_guard<std::mutex> lock(pipeline_cache_file_mutex);
    try {
        std::size_t cache_size = 0;
        if (vkGetPipelineCacheData(device, pipeline_cache, &cache_size, nullptr) != VK_SUCCESS ||
            cache_size == 0) {
            log("[pipeline-cache] no data to save", fsr4log::Level::Warning);
            return;
        }
        std::vector<uint8_t> data(cache_size);
        check(vkGetPipelineCacheData(device, pipeline_cache, &cache_size, data.data()),
              "vkGetPipelineCacheData");
        if (!pipeline_cache_path.parent_path().empty()) {
            fs::create_directories(pipeline_cache_path.parent_path());
        }
        const fs::path temporary = pipeline_cache_path.string() + ".tmp";
        std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
        stream.write(reinterpret_cast<const char*>(data.data()),
                     static_cast<std::streamsize>(data.size()));
        stream.close();
        if (!stream) {
            throw std::runtime_error("failed to write pipeline cache temporary file");
        }
        std::error_code error;
        fs::rename(temporary, pipeline_cache_path, error);
        if (error) {
            fs::remove(pipeline_cache_path, error);
            error.clear();
            fs::rename(temporary, pipeline_cache_path, error);
        }
        if (error) {
            fs::remove(temporary, error);
            throw std::runtime_error("failed to replace pipeline cache: " + error.message());
        }
        log("[pipeline-cache] saved bytes=" + std::to_string(data.size()) + " path='" +
            pipeline_cache_path.string() + "'");
    } catch (const std::exception& error) {
        log(std::string("[pipeline-cache] save failed: ") + error.what(),
            fsr4log::Level::Warning);
    }
}

void QualityContext::prepareModelPipelines() {
    const unsigned variant = use_int8_dispatch && current_preset == "ultraperf" ? 1u : 0u;
    if (model_pipeline_variant == variant) {
        return;
    }
    const auto& paths = model_shader_paths[variant];
    if (paths.size() != pipelines.size()) {
        throw std::runtime_error("requested model pipeline set is unavailable");
    }
    const auto& plan = model_pipeline_plan;
    unsigned created = 0;
    for (uint32_t index = 0; index < paths.size(); ++index) {
        const auto type = model_pass_policy.passes[index].type;
        if (type == PassType::Prepass || type == PassType::Postpass ||
            (type == PassType::TensorPadding &&
             (!execute_padding_passes ||
              model_binding::paddingLayer(index) == model_binding::unused_padding_layer))) {
            continue;
        }
        if (pipelines[index] && active_shader_paths[index] == paths[index]) {
            continue;
        }
        const auto code = readModelSpirv(paths[index], color_api_version);
        const auto pipeline = createComputePipeline(code, plan.require_wave32[index]);
        retirePipeline(pipelines[index]);
        pipelines[index] = pipeline;
        active_shader_paths[index] = paths[index];
        if (type == PassType::ModelPass) {
            const auto wave = plan.require_wave32[index] ? std::string("Wave32")
                              : code[1] >= 0x00010600u
                                  ? automatic_wave_label
                                  : "Wave" + std::to_string(default_subgroup_size);
            if (model_wave_label.empty()) {
                model_wave_label = wave;
            } else if (model_wave_label != wave) {
                model_wave_label = "Wave mixed";
            }
        }
        ++created;
    }
    model_pipeline_variant = variant;
    log("[pipelines] model=" + std::string(variant ? "ultraperf" : "standard") +
        " created=" + std::to_string(created));
}

void QualityContext::retirePipeline(VkPipeline& pipeline) {
    if (pipeline) {
        retired_pipelines.push_back(pipeline);
        pipeline = VK_NULL_HANDLE;
    }
}

void QualityContext::retireInternalImage(Image& image) {
    if (!image.image) {
        return;
    }
    retired_internal_images.push_back(std::move(image));
    image = {};
}

void QualityContext::trackInternalImageUse(VkImage image) {
    if (!image || pending_frame_uses.empty()) {
        return;
    }
    auto& images = pending_frame_uses.back().internal_images;
    if (std::find(images.begin(), images.end(), image) == images.end()) {
        images.push_back(image);
    }
}

void QualityContext::collectFrameUses() {
    std::erase_if(pending_frame_uses, [&](const PendingFrameUse& use) {
        const auto status = vkGetEventStatus(device, use.completion);
        if (status == VK_EVENT_RESET) {
            return false;
        }
        if (status != VK_EVENT_SET) {
            check(status, "frame completion event");
        }
        available_frame_events.push_back(use.completion);
        return true;
    });
    std::erase_if(retired_pipelines, [&](VkPipeline pipeline) {
        for (const auto& use : pending_frame_uses) {
            if (std::find(use.pipelines.begin(), use.pipelines.end(), pipeline) !=
                use.pipelines.end()) {
                return false;
            }
        }
        vkDestroyPipeline(device, pipeline, nullptr);
        return true;
    });
    std::erase_if(retired_staging_outputs, [&](Image& image) {
        for (const auto& use : pending_frame_uses) {
            if (use.staging_output == image.image) {
                return false;
            }
        }
        destroyImage(device, image);
        return true;
    });
    std::erase_if(retired_internal_images, [&](Image& image) {
        for (const auto& use : pending_frame_uses) {
            if (std::find(use.internal_images.begin(),
                          use.internal_images.end(),
                          image.image) != use.internal_images.end()) {
                return false;
            }
        }
        destroyImage(device, image);
        return true;
    });
}

void QualityContext::beginFrameUse() {
    VkEvent event{};
    if (available_frame_events.empty()) {
        VkEventCreateInfo info{VK_STRUCTURE_TYPE_EVENT_CREATE_INFO};
        check(vkCreateEvent(device, &info, nullptr, &event), "frame completion event");
    } else {
        event = available_frame_events.back();
        check(vkResetEvent(device, event), "reset frame completion event");
        available_frame_events.pop_back();
    }
    pending_frame_uses.push_back({event, {}});
    pending_frame_uses.back().pipelines.reserve(pipelines.size() + 3);
}

void QualityContext::trackPipelineUse(VkPipeline pipeline) {
    if (pipeline) {
        pending_frame_uses.back().pipelines.push_back(pipeline);
    }
}
} // namespace fsr4runtime
