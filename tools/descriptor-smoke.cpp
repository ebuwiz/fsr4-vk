// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 fsr4-vk contributors.
// See LICENSE in the project root.

#include "assets/assets.hpp"
#include "models/models.hpp"
#include "runtime/gpu-capabilities.hpp"

#include <cstdlib>
#include <cstring>
#include <chrono>
#include <iostream>
#include <string>
#include <vector>

namespace {
using fsr4_411::check;
using fsr4_411::Image;
using fsr4_411::OptimizedConstants;

uint32_t findComputeQueue(VkPhysicalDevice physical_device) {
    uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &count, nullptr);
    std::vector<VkQueueFamilyProperties> families(count);
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &count, families.data());
    for (uint32_t index = 0; index < count; ++index) {
        if (families[index].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            return index;
        }
    }
    throw std::runtime_error("no compute queue family");
}

Image externalImage(const fsr4vk::GpuCapabilities& capabilities,
                    VkDevice device,
                    uint32_t width,
                    uint32_t height,
                    VkFormat format,
                    VkImageUsageFlags usage) {
    return fsr4_411::createImage(capabilities.memory, device, width, height, format, usage);
}

std::pair<uint32_t, uint32_t> smokeExtent(const char* environment, uint32_t fallback) {
    const char* value = std::getenv(environment);
    if (!value || !*value) {
        return {fallback, fallback};
    }
    const std::string text(value);
    const auto separator = text.find('x');
    if (separator == std::string::npos || separator == 0 || separator + 1 == text.size()) {
        throw std::runtime_error(std::string(environment) + " must use WIDTHxHEIGHT");
    }
    char* width_end = nullptr;
    char* height_end = nullptr;
    const unsigned long width = std::strtoul(text.c_str(), &width_end, 10);
    const unsigned long height = std::strtoul(text.c_str() + separator + 1, &height_end, 10);
    if (width_end != text.c_str() + separator || *height_end != '\0' ||
        width < 64 || height < 64 || width > 7680 || height > 4320) {
        throw std::runtime_error(std::string(environment) +
                                 " is outside supported WIDTHxHEIGHT bounds");
    }
    return {static_cast<uint32_t>(width), static_cast<uint32_t>(height)};
}

void transitionExternal(PFN_vkCmdPipelineBarrier2 barrier2,
                        VkCommandBuffer command_buffer,
                        const Image& image,
                        VkImageLayout source_layout,
                        VkImageLayout destination_layout) {
    VkImageMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    barrier.srcStageMask = source_layout == VK_IMAGE_LAYOUT_UNDEFINED
                               ? VK_PIPELINE_STAGE_2_NONE
                               : VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    barrier.srcAccessMask =
        source_layout == VK_IMAGE_LAYOUT_UNDEFINED ? VK_ACCESS_2_NONE
                                                   : VK_ACCESS_2_TRANSFER_WRITE_BIT;
    barrier.dstStageMask = destination_layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
                               ? VK_PIPELINE_STAGE_2_TRANSFER_BIT
                               : VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    barrier.dstAccessMask = destination_layout == VK_IMAGE_LAYOUT_GENERAL
                                ? VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                                      VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT
                            : destination_layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
                                ? VK_ACCESS_2_TRANSFER_WRITE_BIT
                                : VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
    barrier.oldLayout = source_layout;
    barrier.newLayout = destination_layout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image.image;
    barrier.subresourceRange = {image.aspect, 0, 1, 0, 1};
    VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dependency.imageMemoryBarrierCount = 1;
    dependency.pImageMemoryBarriers = &barrier;
    barrier2(command_buffer, &dependency);
}

std::vector<uint8_t> runBackend(VkPhysicalDevice physical_device,
                                const fsr4vk::GpuCapabilities& supported,
                                fsr4vk::DescriptorBackend backend,
                                bool fp8) {
    const auto context_start = std::chrono::steady_clock::now();
    const auto [render_width, render_height] =
        smokeExtent("FSR4_VK_SMOKE_RENDER_SIZE", 128);
    const auto [output_width, output_height] =
        smokeExtent("FSR4_VK_SMOKE_OUTPUT_SIZE", 128);
    const uint32_t queue_family = findComputeQueue(physical_device);
    const float priority = 1.0f;
    VkDeviceQueueCreateInfo queue_info{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queue_info.queueFamilyIndex = queue_family;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &priority;

    VkPhysicalDeviceFeatures core{};
    core.shaderInt16 = supported.core.shaderInt16;
    core.shaderStorageImageReadWithoutFormat =
        supported.core.shaderStorageImageReadWithoutFormat;
    core.shaderStorageImageWriteWithoutFormat =
        supported.core.shaderStorageImageWriteWithoutFormat;

    VkPhysicalDeviceShaderFloat16Int8Features float16_int8{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES};
    float16_int8.shaderFloat16 = supported.float16_int8.shaderFloat16;
    float16_int8.shaderInt8 = supported.float16_int8.shaderInt8;
    VkPhysicalDevice8BitStorageFeatures storage8{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES};
    storage8.storageBuffer8BitAccess = supported.storage8.storageBuffer8BitAccess;
    VkPhysicalDeviceBufferDeviceAddressFeatures buffer_address{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES};
    buffer_address.bufferDeviceAddress = supported.buffer_device_address.bufferDeviceAddress;
    VkPhysicalDeviceDescriptorIndexingFeatures descriptor_indexing{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES};
    descriptor_indexing.runtimeDescriptorArray =
        supported.descriptor_indexing.runtimeDescriptorArray;
    VkPhysicalDeviceVulkanMemoryModelFeatures memory_model{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_MEMORY_MODEL_FEATURES};
    memory_model.vulkanMemoryModel = supported.memory_model.vulkanMemoryModel;
    VkPhysicalDeviceVulkan13Features vulkan13{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    vulkan13.shaderIntegerDotProduct = supported.vulkan13.shaderIntegerDotProduct;
    VkPhysicalDeviceComputeShaderDerivativesFeaturesKHR derivatives{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COMPUTE_SHADER_DERIVATIVES_FEATURES_KHR};
    derivatives.computeDerivativeGroupLinear =
        supported.compute_derivatives.computeDerivativeGroupLinear;
    VkPhysicalDeviceSubgroupSizeControlFeatures subgroup_control{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES};
    subgroup_control.subgroupSizeControl =
        supported.subgroup_size_control.subgroupSizeControl;
    subgroup_control.computeFullSubgroups =
        supported.subgroup_size_control.computeFullSubgroups;
    VkPhysicalDeviceDescriptorBufferFeaturesEXT descriptor_buffer{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_BUFFER_FEATURES_EXT};
    descriptor_buffer.descriptorBuffer =
        backend == fsr4vk::DescriptorBackend::DescriptorBuffers &&
        supported.descriptor_buffer.descriptorBuffer;
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR cooperative{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
    cooperative.cooperativeMatrix =
        fp8 && supported.cooperative_matrix.cooperativeMatrix;
    VkPhysicalDeviceShaderFloat8FeaturesEXT shader_float8{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT};
    shader_float8.shaderFloat8 = fp8 && supported.shader_float8.shaderFloat8;
    shader_float8.shaderFloat8CooperativeMatrix =
        fp8 && supported.shader_float8.shaderFloat8CooperativeMatrix;

    float16_int8.pNext = &storage8;
    storage8.pNext = &buffer_address;
    buffer_address.pNext = &descriptor_indexing;
    descriptor_indexing.pNext = &memory_model;
    memory_model.pNext = &vulkan13;
    vulkan13.pNext = &derivatives;
    derivatives.pNext = &subgroup_control;
    subgroup_control.pNext = &descriptor_buffer;
    descriptor_buffer.pNext = &cooperative;
    cooperative.pNext = &shader_float8;

    std::vector<const char*> extensions;
    if (backend == fsr4vk::DescriptorBackend::DescriptorBuffers &&
        supported.hasExtension(VK_EXT_DESCRIPTOR_BUFFER_EXTENSION_NAME)) {
        extensions.push_back(VK_EXT_DESCRIPTOR_BUFFER_EXTENSION_NAME);
    }
    for (const char* extension : {VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
                                  VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME,
                                  VK_KHR_COMPUTE_SHADER_DERIVATIVES_EXTENSION_NAME,
                                  VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME,
                                  VK_EXT_SHADER_FLOAT8_EXTENSION_NAME}) {
        if (supported.hasExtension(extension)) {
            if (fp8 || (std::strcmp(extension,
                                    VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME) != 0 &&
                        std::strcmp(extension, VK_EXT_SHADER_FLOAT8_EXTENSION_NAME) != 0)) {
                extensions.push_back(extension);
            }
        }
    }

    VkDeviceCreateInfo device_info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;
    device_info.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    device_info.ppEnabledExtensionNames = extensions.data();
    device_info.pEnabledFeatures = &core;
    device_info.pNext = &float16_int8;

    VkDevice device = VK_NULL_HANDLE;
    check(vkCreateDevice(physical_device, &device_info, nullptr, &device), "vkCreateDevice");
    VkQueue queue = VK_NULL_HANDLE;
    vkGetDeviceQueue(device, queue_family, 0, &queue);

    VkCommandPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pool_info.queueFamilyIndex = queue_family;
    VkCommandPool pool = VK_NULL_HANDLE;
    check(vkCreateCommandPool(device, &pool_info, nullptr, &pool), "vkCreateCommandPool");

    const char* backend_name = fsr4vk::descriptorBackendName(backend);
    setenv("FSR4_VK_DESCRIPTOR_BACKEND", backend_name, 1);
    if (!std::getenv("FSR4_VK_PIPELINE_CACHE_DISABLED") &&
        !std::getenv("FSR4_VK_PIPELINE_CACHE_PATH")) {
        setenv("FSR4_VK_PIPELINE_CACHE_DISABLED", "1", 1);
    }
    auto get_device_proc_addr = vkGetDeviceProcAddr;
    const fsr4_411::ContextCreateInfo context_info{
        .physical_device = physical_device,
        .device = device,
        .asset_root = fsr4assets::root,
        .diagnostic_root = std::filesystem::current_path(),
        .get_device_proc_addr = get_device_proc_addr,
        .max_render_width = render_width,
        .max_render_height = render_height,
        .max_output_width = output_width,
        .max_output_height = output_height,
        .api_version = supported.properties.apiVersion,
    };
    auto context = fp8 ? fsr4fp8::createContext(context_info)
                       : fsr4int8::createContext(context_info);
    const auto context_ready = std::chrono::steady_clock::now();

    auto color = externalImage(
        supported, device, render_width, render_height, VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    auto depth = externalImage(
        supported, device, render_width, render_height, VK_FORMAT_R32_SFLOAT,
        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    auto motion = externalImage(
        supported, device, render_width, render_height, VK_FORMAT_R16G16_SFLOAT,
        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    auto output = externalImage(
        supported, device, output_width, output_height, VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT |
            VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);

    const char* requested_frames = std::getenv("FSR4_VK_SMOKE_FRAMES");
    const uint32_t frame_count =
        requested_frames ? static_cast<uint32_t>(std::strtoul(requested_frames, nullptr, 10))
                         : 2u;
    if (frame_count == 0) {
        throw std::runtime_error("FSR4_VK_SMOKE_FRAMES must be greater than zero");
    }
    double total_record_ms = 0.0;
    const auto frames_start = std::chrono::steady_clock::now();
    for (uint32_t frame = 0; frame < frame_count; ++frame) {
        VkCommandBufferAllocateInfo allocate{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        allocate.commandPool = pool;
        allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocate.commandBufferCount = 1;
        VkCommandBuffer command_buffer = VK_NULL_HANDLE;
        check(vkAllocateCommandBuffers(device, &allocate, &command_buffer),
              "vkAllocateCommandBuffers");
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        check(vkBeginCommandBuffer(command_buffer, &begin), "vkBeginCommandBuffer");
        if (frame == 0) {
            fsr4vk::DeviceDispatch dispatch(device, get_device_proc_addr,
                                            supported.properties.apiVersion, backend);
            transitionExternal(dispatch.cmd_pipeline_barrier2,
                               command_buffer,
                               color,
                               VK_IMAGE_LAYOUT_UNDEFINED,
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            transitionExternal(dispatch.cmd_pipeline_barrier2,
                               command_buffer,
                               depth,
                               VK_IMAGE_LAYOUT_UNDEFINED,
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            transitionExternal(dispatch.cmd_pipeline_barrier2,
                               command_buffer,
                               motion,
                               VK_IMAGE_LAYOUT_UNDEFINED,
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            transitionExternal(dispatch.cmd_pipeline_barrier2,
                               command_buffer,
                               output,
                               VK_IMAGE_LAYOUT_UNDEFINED,
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            VkClearColorValue clear{};
            const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            for (const Image* image : {&color, &depth, &motion, &output}) {
                vkCmdClearColorImage(command_buffer,
                                     image->image,
                                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                     &clear,
                                     1,
                                     &range);
            }
            transitionExternal(dispatch.cmd_pipeline_barrier2,
                               command_buffer,
                               color,
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                               VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            transitionExternal(dispatch.cmd_pipeline_barrier2,
                               command_buffer,
                               depth,
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                               VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            transitionExternal(dispatch.cmd_pipeline_barrier2,
                               command_buffer,
                               motion,
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                               VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            transitionExternal(dispatch.cmd_pipeline_barrier2,
                               command_buffer,
                               output,
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                               VK_IMAGE_LAYOUT_GENERAL);
        }

        OptimizedConstants constants{};
        const uint32_t padded_width = fsr4::roundUp(output_width, 8);
        const uint32_t padded_height = fsr4::roundUp(output_height, 8);
        constants.inv_size[0] = 1.0f / padded_width;
        constants.inv_size[1] = 1.0f / padded_height;
        constants.scale[0] = static_cast<float>(padded_width) / render_width;
        constants.scale[1] = static_cast<float>(padded_height) / render_height;
        constants.inv_scale[0] = 1.0f / constants.scale[0];
        constants.inv_scale[1] = 1.0f / constants.scale[1];
        constants.tex_size[0] = padded_width;
        constants.tex_size[1] = padded_height;
        constants.max_render_size[0] = render_width;
        constants.max_render_size[1] = render_height;
        constants.width = padded_width;
        constants.height = padded_height;
        constants.width_lr = render_width;
        constants.height_lr = render_height;
        constants.pre_exposure = 1.0f;
        constants.previous_pre_exposure = 1.0f;
        const auto record_start = std::chrono::steady_clock::now();
        context->recordFrame(command_buffer,
                             color,
                             depth,
                             motion,
                             {},
                             {},
                             output,
                             constants,
                             false,
                             fsr4_411::ColorSpace::Linear,
                             {});
        total_record_ms +=
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - record_start)
                .count();
        check(vkEndCommandBuffer(command_buffer), "vkEndCommandBuffer");
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &command_buffer;
        check(vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE), "vkQueueSubmit");
        check(vkQueueWaitIdle(queue), "vkQueueWaitIdle");
        vkFreeCommandBuffers(device, pool, 1, &command_buffer);
    }
    const double frame_wall_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
                                                  frames_start)
            .count();

    const VkDeviceSize output_bytes =
        static_cast<VkDeviceSize>(output_width) * output_height * 8;
    auto readback = fsr4_411::createBuffer(supported.memory,
                                          device,
                                          output_bytes,
                                          VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VkCommandBufferAllocateInfo readback_allocate{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    readback_allocate.commandPool = pool;
    readback_allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    readback_allocate.commandBufferCount = 1;
    VkCommandBuffer readback_command = VK_NULL_HANDLE;
    check(vkAllocateCommandBuffers(device, &readback_allocate, &readback_command),
          "vkAllocateCommandBuffers(readback)");
    VkCommandBufferBeginInfo readback_begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    check(vkBeginCommandBuffer(readback_command, &readback_begin), "vkBeginCommandBuffer(readback)");
    fsr4vk::DeviceDispatch dispatch(
        device, vkGetDeviceProcAddr, supported.properties.apiVersion, backend);
    VkImageMemoryBarrier2 output_barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    output_barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    output_barrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    output_barrier.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    output_barrier.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
    output_barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    output_barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    output_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    output_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    output_barrier.image = output.image;
    output_barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VkDependencyInfo readback_dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    readback_dependency.imageMemoryBarrierCount = 1;
    readback_dependency.pImageMemoryBarriers = &output_barrier;
    dispatch.cmd_pipeline_barrier2(readback_command, &readback_dependency);
    VkBufferImageCopy copy{};
    copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    copy.imageExtent = {output_width, output_height, 1};
    vkCmdCopyImageToBuffer(readback_command,
                           output.image,
                           VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           readback.buffer,
                           1,
                           &copy);
    check(vkEndCommandBuffer(readback_command), "vkEndCommandBuffer(readback)");
    VkSubmitInfo readback_submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    readback_submit.commandBufferCount = 1;
    readback_submit.pCommandBuffers = &readback_command;
    check(vkQueueSubmit(queue, 1, &readback_submit, VK_NULL_HANDLE), "vkQueueSubmit(readback)");
    check(vkQueueWaitIdle(queue), "vkQueueWaitIdle(readback)");
    std::vector<uint8_t> pixels(output_bytes);
    std::memcpy(pixels.data(), readback.mapped, pixels.size());
    vkFreeCommandBuffers(device, pool, 1, &readback_command);
    fsr4_411::destroyBuffer(device, readback);

    context.reset();
    fsr4_411::destroyImage(device, color);
    fsr4_411::destroyImage(device, depth);
    fsr4_411::destroyImage(device, motion);
    fsr4_411::destroyImage(device, output);
    vkDestroyCommandPool(device, pool, nullptr);
    vkDestroyDevice(device, nullptr);
    std::cout << "descriptor backend " << backend_name
              << " model=" << (fp8 ? "fp8" : "int8")
              << " render=" << render_width << 'x' << render_height
              << " output=" << output_width << 'x' << output_height
              << " frames=" << frame_count << " create_ms="
              << std::chrono::duration<double, std::milli>(context_ready - context_start).count()
              << " average_record_ms=" << total_record_ms / frame_count
              << " average_wall_ms=" << frame_wall_ms / frame_count
              << ": submitted and output read back successfully\n";
    return pixels;
}
} // namespace

int main() {
    setenv("FSR4_VK_MODEL", "int8", 1);
    VkApplicationInfo application{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    application.pApplicationName = "fsr4-vk descriptor smoke";
    application.apiVersion = VK_API_VERSION_1_4;
    VkInstanceCreateInfo instance_info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    instance_info.pApplicationInfo = &application;
    VkInstance instance = VK_NULL_HANDLE;
    check(vkCreateInstance(&instance_info, nullptr, &instance), "vkCreateInstance");

    uint32_t count = 0;
    check(vkEnumeratePhysicalDevices(instance, &count, nullptr), "vkEnumeratePhysicalDevices");
    std::vector<VkPhysicalDevice> devices(count);
    check(vkEnumeratePhysicalDevices(instance, &count, devices.data()),
          "vkEnumeratePhysicalDevices");

    VkPhysicalDevice selected = VK_NULL_HANDLE;
    fsr4vk::GpuCapabilities capabilities;
    for (const auto device : devices) {
        const auto candidate = fsr4vk::detect(device);
        if (candidate.properties.vendorID == fsr4vk::kAmdVendorId &&
            candidate.supportsInt8Model() &&
            candidate.hasExtension(VK_EXT_DESCRIPTOR_BUFFER_EXTENSION_NAME)) {
            selected = device;
            capabilities = candidate;
            break;
        }
    }
    if (!selected) {
        std::cerr << "no AMD device supporting the INT8 model and descriptor buffers\n";
        return 77;
    }

    try {
        for (const bool fp8 : {false, true}) {
            if (fp8 && !capabilities.supportsFp8Model()) {
                std::cout << "fp8 backend comparison skipped: physical capability blockers="
                          << fsr4vk::joinBlockers(capabilities.fp8Blockers()) << '\n';
                continue;
            }
            const auto buffer_pixels = runBackend(
                selected, capabilities, fsr4vk::DescriptorBackend::DescriptorBuffers, fp8);
            const auto set_pixels = runBackend(
                selected, capabilities, fsr4vk::DescriptorBackend::DescriptorSets, fp8);
            if (buffer_pixels != set_pixels) {
                throw std::runtime_error(std::string("descriptor backend outputs differ for ") +
                                         (fp8 ? "fp8" : "int8"));
            }
            std::cout << "descriptor backends model=" << (fp8 ? "fp8" : "int8")
                      << ": output bytes identical\n";
        }
    } catch (const std::exception& error) {
        std::cerr << "descriptor smoke failed: " << error.what() << '\n';
        vkDestroyInstance(instance, nullptr);
        return 1;
    }
    vkDestroyInstance(instance, nullptr);
    fsr4vk::clearCapabilityCache();
    return 0;
}
