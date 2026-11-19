// Frame Generation Vulkan execution + pipeline-cache benchmark harness.
//
// This is a host-side dev harness for the INDEPENDENT Frame Generation provider
// (amd_fidelityfx_framegeneration_vk). It exercises the REAL runtime substrate on
// the target GPU WITHOUT claiming the Frame Generation algorithm runs:
//   * it drives the provider's OWN namespaced pipeline cache (fg4) with an ACTUAL
//     compiled compute pipeline, so the cold->warm reuse numbers are real cache
//     entries (not just an empty driver header as in the capability probe);
//   * it measures cold vs warm pipeline-creation time and a benchmark loop
//     (FFG4_VK_BENCHMARK), and per-stage profiling (FFG4_VK_PROFILE);
//   * it dispatches that pipeline once and reads the result back on the GPU,
//     comparing against a host-computed reference, to prove the compiled shader
//     actually executes deterministically on RDNA4 / GFX1201.
//
// The compute kernel here is a SYNTHETIC substrate probe embedded from
// tools/framegeneration-selftest.comp (see -spirv.h). It is NOT a Frame Generation
// pass: FG interpolation/ML present/pacing and reference image-quality remain
// unimplemented. Real FG-module compute pipelines DO now create on the target (measured
// 474/474 under both transports); what is NOT yet done is host descriptor POPULATION +
// dispatch. Measured here (--write-descriptors): conventional VkDescriptorSets allocate for
// only 8 of 2692 sets (whole-array bindings cannot be sized as sets), whereas VK_EXT_descriptor_buffer
// serializes all 1208 FG descriptor bindings via vkGetDescriptorEXT with 0 failures (driver
// sizes 16/16/64/32/16 bytes) -- which is why the FG host runtime MUST use the
// VK_EXT_descriptor_buffer transport, up to but never including vkCmdDispatch -- see README.
//
// Usage:  framegeneration-exec.exe [device-substring]
// Env:    FFG4_VK_SELFTEST=1 | FFG4_VK_PROFILE=1 | FFG4_VK_BENCHMARK=N|:N|1 |
//         FFG4_VK_PIPELINE_CACHE=<file> | FFG4_VK_PIPELINE_CACHE_DISABLE=1
#if !defined(_WIN32)
#define __declspec(x) __attribute__((visibility("default")))
#endif

#include <vulkan/vulkan.h>

#include "pipeline-cache.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#include <unistd.h>
#endif

#include "framegeneration-selftest-spirv.h"

static uint64_t now_ns() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count());
}

static std::string getEnv(const char* name) {
    const char* value = std::getenv(name);
    return (value && *value) ? std::string(value) : std::string();
}

// The measured Frame Generation execution contract, enabled on the caller-owned
// device with the exact same classic features2 chain the capability probe uses, so
// this harness runs on the same devices the provider accepts.
static VkDevice createContractDevice(VkPhysicalDevice physical) {
    const char* kContractExtensions[] = {
        VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
        VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME,
        VK_EXT_DESCRIPTOR_BUFFER_EXTENSION_NAME,
        VK_KHR_COMPUTE_SHADER_DERIVATIVES_EXTENSION_NAME,
        VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME,
        VK_KHR_VULKAN_MEMORY_MODEL_EXTENSION_NAME,
        VK_KHR_8BIT_STORAGE_EXTENSION_NAME,
        VK_KHR_16BIT_STORAGE_EXTENSION_NAME,
        VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME,
        VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME,
    };
    std::vector<const char*> extensions;
    for (const char* extension : kContractExtensions) {
        uint32_t count = 0;
        vkEnumerateDeviceExtensionProperties(physical, nullptr, &count, nullptr);
        std::vector<VkExtensionProperties> table(count);
        if (count) {
            vkEnumerateDeviceExtensionProperties(physical, nullptr, &count, table.data());
        }
        for (const auto& entry : table) {
            if (std::strcmp(entry.extensionName, extension) == 0) {
                extensions.push_back(extension);
                break;
            }
        }
    }

    VkPhysicalDeviceFeatures2 baseFeatures{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    VkPhysicalDeviceShaderFloat16Int8Features float16Int8{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES};
    VkPhysicalDevice16BitStorageFeatures storage16{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES};
    VkPhysicalDeviceBufferDeviceAddressFeatures bufferAddress{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES};
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR cooperative{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
    VkPhysicalDeviceVulkanMemoryModelFeaturesKHR memoryModel{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_MEMORY_MODEL_FEATURES_KHR};
    VkPhysicalDeviceDescriptorIndexingFeatures descriptorIndexing{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES};
    VkPhysicalDevice8BitStorageFeatures storage8{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES};
    VkPhysicalDeviceSubgroupSizeControlFeatures subgroupControl{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES};
    VkPhysicalDeviceComputeShaderDerivativesFeaturesKHR derivatives{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COMPUTE_SHADER_DERIVATIVES_FEATURES_KHR};
    VkPhysicalDeviceDescriptorBufferFeaturesEXT descriptorBufferFeatures{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_BUFFER_FEATURES_EXT};
    baseFeatures.pNext = &bufferAddress;
    bufferAddress.pNext = &float16Int8;
    float16Int8.pNext = &storage16;
    storage16.pNext = &cooperative;
    cooperative.pNext = &memoryModel;
    memoryModel.pNext = &descriptorIndexing;
    descriptorIndexing.pNext = &storage8;
    storage8.pNext = &subgroupControl;
    subgroupControl.pNext = &derivatives;
    derivatives.pNext = &descriptorBufferFeatures;

    baseFeatures.features.shaderInt64 = VK_TRUE;
    baseFeatures.features.shaderInt16 = VK_TRUE;
    baseFeatures.features.shaderStorageImageWriteWithoutFormat = VK_TRUE;
    bufferAddress.bufferDeviceAddress = VK_TRUE;
    float16Int8.shaderFloat16 = VK_TRUE;
    float16Int8.shaderInt8 = VK_TRUE;
    storage16.storageBuffer16BitAccess = VK_TRUE;
    cooperative.cooperativeMatrix = VK_TRUE;
    memoryModel.vulkanMemoryModel = VK_TRUE;
    descriptorIndexing.runtimeDescriptorArray = VK_TRUE;
    storage8.storageBuffer8BitAccess = VK_TRUE;
    subgroupControl.subgroupSizeControl = VK_TRUE;
    derivatives.computeDerivativeGroupLinear = VK_TRUE;
    descriptorBufferFeatures.descriptorBuffer = VK_TRUE;

    VkDeviceQueueCreateInfo queue{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queue.queueFamilyIndex = 0;
    queue.queueCount = 1;
    VkDeviceCreateInfo deviceInfo{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    deviceInfo.pNext = &baseFeatures;
    deviceInfo.queueCreateInfoCount = 1;
    deviceInfo.pQueueCreateInfos = &queue;
    deviceInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    deviceInfo.ppEnabledExtensionNames = extensions.empty() ? nullptr : extensions.data();

    VkDevice device{};
    if (vkCreateDevice(physical, &deviceInfo, nullptr, &device) != VK_SUCCESS) {
        return VK_NULL_HANDLE;
    }
    return device;
}

// Create the provider's namespaced pipeline-cache container for a live device and
// return the seed byte count actually loaded from disk (0 == cold start). The FG
// algorithm id (fg4) is written into the container, so this is never the upscaler's
// cache.
static ffgvk::CacheIdentity identityOf(VkPhysicalDevice physical) {
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(physical, &props);
    ffgvk::CacheIdentity id;
    id.deviceName = props.deviceName;
    id.driverVersion = props.driverVersion;
    id.apiVersion = props.apiVersion;
    id.vendorId = props.vendorID;
    id.deviceId = props.deviceID;
    return id;
}

// The descriptor-buffer transport's REAL sizing: the driver's own per-descriptor
// byte costs, read from VkPhysicalDeviceDescriptorBufferPropertiesEXT on the physical
// device (the same five sizes the upscaler runtime builds its sets from at
// src/runtime/gpu-capabilities.cpp). A whole-array binding has no fixed
// VkDescriptorSet footprint, but it ALWAYS serializes to (per-element size x count)
// descriptor-buffer bytes -- which is exactly what vkGetDescriptorEXT writes.
struct DbSizes {
    bool enabled = false;
    size_t storageBuffer = 0;
    size_t uniformBuffer = 0;
    size_t combinedImageSampler = 0;
    size_t storageImage = 0;
    size_t sampler = 0;
};

static DbSizes queryDescriptorSizes(VkPhysicalDevice physical) {
    DbSizes s;
    VkPhysicalDeviceDescriptorBufferPropertiesEXT db{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_BUFFER_PROPERTIES_EXT};
    VkPhysicalDeviceProperties2 props{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    props.pNext = &db;
    vkGetPhysicalDeviceProperties2(physical, &props);
    s.storageBuffer = db.storageBufferDescriptorSize;
    s.uniformBuffer = db.uniformBufferDescriptorSize;
    s.combinedImageSampler = db.combinedImageSamplerDescriptorSize;
    s.storageImage = db.storageImageDescriptorSize;
    s.sampler = db.samplerDescriptorSize;
    s.enabled = (s.storageBuffer + s.uniformBuffer + s.combinedImageSampler +
                 s.storageImage + s.sampler) > 0;
    return s;
}

// Load the whole pack and hand every translated SPIR-V to the driver as a shader
// module: a structural acceptance census (does the driver take the module), NOT a
// target compilation. Returns ok / total; failures are counted, not hidden.
static void loadPackModules(const std::string& packDir, VkDevice device,
                            uint32_t& okCount, uint32_t& total) {
    std::ifstream index(packDir + "/modules.tsv");
    if (!index) {
        std::printf("  pack index: ABSENT (%s/modules.tsv)\n", packDir.c_str());
        return;
    }
    std::string line;
    std::getline(index, line);  // header
    while (std::getline(index, line)) {
        const auto tab = line.find('\t');
        if (tab == std::string::npos) {
            continue;
        }
        const std::string name = line.substr(0, tab);
        const auto modulePath = packDir + "/modules/" + name;
        std::ifstream file(modulePath, std::ios::binary);
        if (!file) {
            continue;
        }
        std::vector<unsigned char> spv((std::istreambuf_iterator<char>(file)),
                                       std::istreambuf_iterator<char>());
        ++total;
        VkShaderModule module{};
        VkShaderModuleCreateInfo info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, nullptr, 0,
                                      spv.size(), reinterpret_cast<const uint32_t*>(spv.data())};
        if (vkCreateShaderModule(device, &info, nullptr, &module) == VK_SUCCESS) {
            ++okCount;
            vkDestroyShaderModule(device, module, nullptr);
        }
    }
}

// Ground-truth target-compile census: for every standardizable compute module in the
// pack, derive a plain VkPipelineLayout from the pack's layout column and ATTEMPT
// vkCreateComputePipelines on the live device. NOT algorithm execution: it answers
// the Phase-3 prerequisite "does RDNA4 actually accept these SPIR-V modules as real
// compute pipelines against a derived layout". Failures are counted with the driver's
// own VkResult, never hidden or guessed.
struct PackCompile {
    uint32_t attempted = 0;
    uint32_t created = 0;
    uint32_t failed = 0;
    uint32_t attempted_collapsible = 0;
    uint32_t skipped_unresolved = 0;  // unknown slot kind
    uint32_t skipped_unified = 0;     // cross-kind slot: needs buffer-device-address transport
    // Host descriptor-population census (populates one entry per binding of every set,
    // using real FG-derived layouts; NOT a dispatch). Classic sets vs descriptor-buffer
    // footprint are measured separately.
    uint32_t populated_sets_classic_ok = 0;
    uint32_t populated_sets_classic_fail = 0;
    uint32_t populated_bindings_ok = 0;
    uint32_t populated_bindings_fail = 0;
    uint32_t modules_populated_classic = 0;
    uint32_t modules_populated_db = 0;
    uint32_t db_layout_footprints = 0;
    // Descriptor-buffer serialization census: vkGetDescriptorEXT one descriptor per
    // binding using the driver's per-type byte cost. Proves every FG binding KIND
    // (incl. whole-array bindings classic sets cannot hold) is serializable to
    // descriptor-buffer bytes. Still NOT a dispatch.
    uint32_t db_serialized_attempted = 0;
    uint32_t db_serialized_ok = 0;
    uint32_t db_serialized_nonzero = 0;
    uint64_t db_footprint_bytes = 0;
    std::string first_db_failure;
    std::string first_pop_failure;
    VkResult first_failure = VK_SUCCESS;
    std::string first_failure_module;
};

static VkDescriptorType vkTypeFor(const std::string& code) {
    if (code == "SB") return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    if (code == "UB") return VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    if (code == "CIS") return VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    if (code == "SI") return VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    if (code == "SMP") return VK_DESCRIPTOR_TYPE_SAMPLER;
    return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
}

static PackCompile compilePackCompute(VkPhysicalDevice physical, VkDevice device,
                                      const std::string& packDir,
                                      const ffgvk::CacheIdentity& identity,
                                      const std::string& cachePath,
                                      bool descriptorBufferTransport,
                                      bool writeMode,
                                      const DbSizes& db) {
    PackCompile stats;
    std::ifstream index(packDir + "/pipelines.tsv");
    if (!index) {
        return stats;
    }
    std::vector<unsigned char> seed = ffgvk::readCacheBlob(cachePath, identity);
    auto* createCache = reinterpret_cast<PFN_vkCreatePipelineCache>(
        vkGetDeviceProcAddr(device, "vkCreatePipelineCache"));
    auto* destroyCache = reinterpret_cast<PFN_vkDestroyPipelineCache>(
        vkGetDeviceProcAddr(device, "vkDestroyPipelineCache"));
    VkPipelineCache cache = VK_NULL_HANDLE;
    if (createCache) {
        VkPipelineCacheCreateInfo cci{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO, nullptr, 0,
                                      seed.size(), seed.empty() ? nullptr : seed.data()};
        createCache(device, &cci, nullptr, &cache);
    }
    auto split = [](const std::string& s, char delim) {
        std::vector<std::string> out;
        size_t start = 0;
        while (true) {
            size_t end = s.find(delim, start);
            if (end == std::string::npos) {
                out.push_back(s.substr(start));
                break;
            }
            out.push_back(s.substr(start, end - start));
            start = end + 1;
        }
        return out;
    };
    // --- host descriptor-population resources (only used when writeMode) ---
    // Real backing objects for each descriptor kind the FG layouts declare, plus the
    // device entry point that reads the descriptor-buffer footprint of a layout
    // (vkGetDescriptorSetLayoutSizeEXT). Writing uses vkCreateDescriptorSet + vkUpdateDescriptorSets.
    auto* getLayoutSize = reinterpret_cast<PFN_vkGetDescriptorSetLayoutSizeEXT>(
        vkGetDeviceProcAddr(device, "vkGetDescriptorSetLayoutSizeEXT"));
    // vkGetDescriptorEXT is not a global linker symbol in this SDK (EXT functions are
    // only reachable through the device resolver), exactly like the runtime's
    // dispatch.get_descriptor at src/runtime/context.cpp -- so resolve it here.
    auto* getDescriptor = reinterpret_cast<PFN_vkGetDescriptorEXT>(
        vkGetDeviceProcAddr(device, "vkGetDescriptorEXT"));
    auto dbSizeFor = [&db](VkDescriptorType t) -> size_t {
        switch (t) {
            case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER: return db.storageBuffer;
            case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER: return db.uniformBuffer;
            case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER: return db.combinedImageSampler;
            case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE: return db.storageImage;
            case VK_DESCRIPTOR_TYPE_SAMPLER: return db.sampler;
            default: return 0u;
        }
    };
    VkBuffer bufS = VK_NULL_HANDLE, bufU = VK_NULL_HANDLE;
    VkImage imgS = VK_NULL_HANDLE, imgC = VK_NULL_HANDLE;
    VkImageView viewS = VK_NULL_HANDLE, viewC = VK_NULL_HANDLE;
    VkSampler samp = VK_NULL_HANDLE;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    std::vector<VkDeviceMemory> population_memory;
    bool popResOk = true;
    if (writeMode) {
        VkPhysicalDeviceMemoryProperties memory_properties{};
        vkGetPhysicalDeviceMemoryProperties(physical, &memory_properties);
        auto bindMemory = [&](VkMemoryRequirements requirements, VkDeviceMemory* memory,
                              bool device_address = false) {
            VkMemoryAllocateFlagsInfo allocation_flags{
                VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
            allocation_flags.flags =
                device_address ? VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT : 0;
            VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
            allocation.pNext = device_address ? &allocation_flags : nullptr;
            allocation.allocationSize = requirements.size;
            allocation.memoryTypeIndex = UINT32_MAX;
            for (uint32_t index = 0; index < memory_properties.memoryTypeCount; ++index) {
                if ((requirements.memoryTypeBits & (1u << index)) &&
                    (memory_properties.memoryTypes[index].propertyFlags &
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
                    allocation.memoryTypeIndex = index;
                    break;
                }
            }
            if (allocation.memoryTypeIndex == UINT32_MAX ||
                vkAllocateMemory(device, &allocation, nullptr, memory) != VK_SUCCESS) {
                popResOk = false;
                return false;
            }
            population_memory.push_back(*memory);
            return true;
        };
        auto mkBuffer = [&](VkBufferUsageFlags usage, VkBuffer* out) {
            VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
            bi.size = 256;  // one entry is enough to populate index 0 of a whole-array
            bi.usage =
                usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
            bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            if (vkCreateBuffer(device, &bi, nullptr, out) != VK_SUCCESS) {
                popResOk = false;
                return;
            }
            VkMemoryRequirements requirements{};
            vkGetBufferMemoryRequirements(device, *out, &requirements);
            VkDeviceMemory memory = VK_NULL_HANDLE;
            if (bindMemory(requirements, &memory, true)) {
                if (vkBindBufferMemory(device, *out, memory, 0) != VK_SUCCESS) {
                    popResOk = false;
                }
            }
        };
        mkBuffer(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &bufS);
        mkBuffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, &bufU);
        auto mkImage = [&](VkImageUsageFlags usage, VkImage* out) {
            VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
            ii.imageType = VK_IMAGE_TYPE_2D;
            ii.format = VK_FORMAT_R32_UINT;
            ii.extent = {4, 4, 1};
            ii.mipLevels = 1; ii.arrayLayers = 1; ii.samples = VK_SAMPLE_COUNT_1_BIT;
            ii.tiling = VK_IMAGE_TILING_OPTIMAL;
            ii.usage = usage;
            ii.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            if (vkCreateImage(device, &ii, nullptr, out) != VK_SUCCESS) {
                popResOk = false;
                return;
            }
            VkMemoryRequirements requirements{};
            vkGetImageMemoryRequirements(device, *out, &requirements);
            VkDeviceMemory memory = VK_NULL_HANDLE;
            if (bindMemory(requirements, &memory)) {
                if (vkBindImageMemory(device, *out, memory, 0) != VK_SUCCESS) {
                    popResOk = false;
                }
            }
        };
        mkImage(VK_IMAGE_USAGE_STORAGE_BIT, &imgS);
        mkImage(VK_IMAGE_USAGE_SAMPLED_BIT, &imgC);
        auto mkView = [&](VkImage img, VkImageView* out) {
            if (!img) {
                popResOk = false;
                return;
            }
            VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            vi.image = img;
            vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
            vi.format = VK_FORMAT_R32_UINT;
            vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            if (vkCreateImageView(device, &vi, nullptr, out) != VK_SUCCESS) {
                popResOk = false;
            }
        };
        mkView(imgS, &viewS);
        mkView(imgC, &viewC);
        VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        si.magFilter = VK_FILTER_NEAREST; si.minFilter = VK_FILTER_NEAREST;
        si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
        si.unnormalizedCoordinates = VK_FALSE;
        if (vkCreateSampler(device, &si, nullptr, &samp) != VK_SUCCESS) {
            popResOk = false;
        }
        if (popResOk && bufS && bufU && imgS && imgC && viewS && viewC && samp) {
            VkDescriptorPoolSize pool_sizes[] = {
                {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 128},
                {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 128},
                {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 128},
                {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 128},
                {VK_DESCRIPTOR_TYPE_SAMPLER, 128}};
            VkDescriptorPoolCreateInfo pi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
            pi.maxSets = 8u;
            pi.poolSizeCount = static_cast<uint32_t>(sizeof(pool_sizes) / sizeof(pool_sizes[0]));
            pi.pPoolSizes = pool_sizes;
            if (vkCreateDescriptorPool(device, &pi, nullptr, &pool) != VK_SUCCESS) {
                pool = VK_NULL_HANDLE;
            }
        }
    }
    struct Slot { uint32_t set; uint32_t binding; VkDescriptorType type; bool arrayed; };
    std::string line;
    std::getline(index, line);  // header
    while (std::getline(index, line)) {
        const auto cols = split(line, '\t');
        if (cols.size() < 7) {
            continue;
        }
        const std::string name = cols[0];
        const std::string stage = cols[1];
        if (stage != "compute-classic" && stage != "compute-neural") {
            continue;
        }
        // Parse + de-duplicate: group layout tokens by (set,binding). Several variables
        // bound to the same slot (HLSL resource-heap views into one runtime array) are
        // legal for a plain VkPipelineLayout as long as they share ONE descriptor kind ->
        // collapse to a single binding. A slot claimed by DIFFERENT kinds is a true cross-kind
        // collision that needs the FG buffer-device-address unified transport.
        struct G { uint32_t set; uint32_t binding; VkDescriptorType type; bool arrayed; };
        std::vector<G> groups;
        bool anyUnknown = false;
        bool crossKind = false;
        bool sawDuplicate = false;
        uint32_t maxSet = 0;
        for (const std::string& token : split(cols[6], ';')) {
            if (token.empty()) {
                continue;
            }
            const auto f = split(token, ':');
            if (f.size() < 3 || f[2] == "U") {
                anyUnknown = true;
                break;
            }
            const bool isArray = f.size() >= 4 && f[3] == "a";
            const uint32_t sset = static_cast<uint32_t>(std::atoi(f[0].c_str()));
            const uint32_t sbind = static_cast<uint32_t>(std::atoi(f[1].c_str()));
            const VkDescriptorType vt = vkTypeFor(f[2]);
            maxSet = std::max(maxSet, sset);
            bool found = false;
            for (auto& g : groups) {
                if (g.set == sset && g.binding == sbind) {
                    if (g.type != vt) {
                        crossKind = true;
                    }
                    g.arrayed = g.arrayed || isArray;
                    sawDuplicate = true;
                    found = true;
                    break;
                }
            }
            if (!found) {
                groups.push_back({sset, sbind, vt, isArray});
            }
        }
        if (crossKind) {
            ++stats.skipped_unified;
            continue;
        }
        if (anyUnknown) {
            ++stats.skipped_unresolved;
            continue;
        }
        ++stats.attempted;
        if (sawDuplicate) {
            ++stats.attempted_collapsible;
        }
        std::vector<Slot> parsed;
        for (const auto& g : groups) {
            parsed.push_back({g.set, g.binding, g.type, g.arrayed});
        }
        std::vector<VkDescriptorSetLayout> setLayouts(maxSet + 1, VK_NULL_HANDLE);
        std::vector<VkDescriptorSetLayoutBinding> bindings;
        std::vector<VkDescriptorBindingFlags> flagsForSet;
        bool layoutOk = true;
        for (uint32_t s = 0; s <= maxSet && layoutOk; ++s) {
            bindings.clear();
            flagsForSet.clear();
            for (const auto& sl : parsed) {
                if (sl.set != s) {
                    continue;
                }
                VkDescriptorSetLayoutBinding b{};
                b.binding = sl.binding;
                b.descriptorType = sl.type;
                b.descriptorCount = sl.arrayed ? 0xFFFFFFFFu : 1u;
                b.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
                bindings.push_back(b);
                flagsForSet.push_back(sl.arrayed ? VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT : 0u);
            }
            VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
            li.bindingCount = static_cast<uint32_t>(bindings.size());
            li.pBindings = bindings.empty() ? nullptr : bindings.data();
            VkDescriptorSetLayoutBindingFlagsCreateInfo bf{
                VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO};
            bool anyFlagged = false;
            for (auto fl : flagsForSet) {
                anyFlagged = anyFlagged || fl != 0u;
            }
            if (anyFlagged) {
                bf.bindingCount = static_cast<uint32_t>(bindings.size());
                bf.pBindingFlags = flagsForSet.data();
                li.pNext = &bf;
            }
            if (vkCreateDescriptorSetLayout(device, &li, nullptr, &setLayouts[s]) != VK_SUCCESS) {
                layoutOk = false;
            }
        }
        if (!layoutOk) {
            for (auto l : setLayouts) {
                if (l) {
                    vkDestroyDescriptorSetLayout(device, l, nullptr);
                }
            }
            ++stats.failed;
            continue;
        }
        const uint32_t pushBytes = static_cast<uint32_t>(std::atoi(cols[3].c_str()));
        VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, pushBytes};
        std::ifstream file(packDir + "/modules/" + name, std::ios::binary);
        std::vector<unsigned char> spv((std::istreambuf_iterator<char>(file)),
                                       std::istreambuf_iterator<char>());
        VkShaderModule module = VK_NULL_HANDLE;
        VkShaderModuleCreateInfo mi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, nullptr, 0,
                                    spv.size(), reinterpret_cast<const uint32_t*>(spv.data())};
        if (spv.empty() || vkCreateShaderModule(device, &mi, nullptr, &module) != VK_SUCCESS) {
            for (auto l : setLayouts) {
                if (l) {
                    vkDestroyDescriptorSetLayout(device, l, nullptr);
                }
            }
            ++stats.failed;
            continue;
        }
        VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pli.setLayoutCount = static_cast<uint32_t>(setLayouts.size());
        pli.pSetLayouts = setLayouts.data();
        pli.pushConstantRangeCount = pushBytes ? 1u : 0u;
        pli.pPushConstantRanges = pushBytes ? &push : nullptr;
        VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
        vkCreatePipelineLayout(device, &pli, nullptr, &pipelineLayout);

        VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        ci.flags = descriptorBufferTransport ? VK_PIPELINE_CREATE_DESCRIPTOR_BUFFER_BIT_EXT : 0u;
        ci.layout = pipelineLayout;
        ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        ci.stage.module = module;
        ci.stage.pName = "main";
        VkPipeline pipeline = VK_NULL_HANDLE;
        const VkResult created = vkCreateComputePipelines(device, cache, 1, &ci, nullptr, &pipeline);
        if (created == VK_SUCCESS) {
            ++stats.created;
        } else {
            ++stats.failed;
            if (stats.first_failure == VK_SUCCESS) {
                stats.first_failure = created;
                stats.first_failure_module = name;
            }
        }
        if (writeMode && popResOk && pool) {
            VkBufferDeviceAddressInfo addrInfoS{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, nullptr, bufS};
            VkBufferDeviceAddressInfo addrInfoU{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, nullptr, bufU};
            const VkDeviceAddress addrS = vkGetBufferDeviceAddress(device, &addrInfoS);
            const VkDeviceAddress addrU = vkGetBufferDeviceAddress(device, &addrInfoU);
            bool modClassicOk = true;
            for (uint32_t s = 0; s <= maxSet; ++s) {
                const VkDescriptorSetLayout lout = setLayouts[s];
                if (!lout) {
                    continue;
                }
                std::vector<VkWriteDescriptorSet> writes;
                std::vector<VkDescriptorBufferInfo> bufInfos;
                std::vector<VkDescriptorImageInfo> imgInfos;
                // Reserve so pushing more backing infos never reallocates the vectors that
                // w.pBufferInfo / w.pImageInfo point into (keeps those pointers valid).
                writes.reserve(parsed.size());
                bufInfos.reserve(parsed.size());
                imgInfos.reserve(parsed.size());
                uint32_t bindingCountInSet = 0;
                bool unmappedThisSet = false;
                for (const auto& sl : parsed) {
                    if (sl.set != s) {
                        continue;
                    }
                    VkWriteDescriptorSet w{};
                    w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    w.dstArrayElement = 0;
                    w.descriptorCount = 1;
                    w.descriptorType = sl.type;
                    w.dstBinding = sl.binding;
                    if (sl.type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) {
                        bufInfos.push_back({bufS, 0, VK_WHOLE_SIZE});
                        w.pBufferInfo = &bufInfos.back();
                    } else if (sl.type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER) {
                        bufInfos.push_back({bufU, 0, VK_WHOLE_SIZE});
                        w.pBufferInfo = &bufInfos.back();
                    } else if (sl.type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) {
                        imgInfos.push_back({samp, viewC, VK_IMAGE_LAYOUT_GENERAL});
                        w.pImageInfo = &imgInfos.back();
                    } else if (sl.type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE) {
                        imgInfos.push_back({VK_NULL_HANDLE, viewS, VK_IMAGE_LAYOUT_GENERAL});
                        w.pImageInfo = &imgInfos.back();
                    } else if (sl.type == VK_DESCRIPTOR_TYPE_SAMPLER) {
                        imgInfos.push_back({samp, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED});
                        w.pImageInfo = &imgInfos.back();
                    } else {
                        unmappedThisSet = true;
                        modClassicOk = false;
                        break;
                    }
                    writes.push_back(w);
                    ++bindingCountInSet;
                }
                if (unmappedThisSet) {
                    break;
                }
                // Descriptor-buffer footprint of this set layout (its real sizing;
                // base size may be 0 for a pure whole-array binding, which is why
                // the per-element sizes below are the authoritative descriptor-buffer
                // metric). Measured for every set, independent of classic allocation.
                if (getLayoutSize) {
                    VkDeviceSize sz = 0;
                    getLayoutSize(device, lout, &sz);
                    if (sz > 0) {
                        ++stats.db_layout_footprints;
                    }
                }
                // Descriptor-buffer serialization census: classic sets cannot hold FG's
                // whole-array bindings (measured via vkAllocateDescriptorSets below), but
                // the descriptor-buffer transport serializes ONE descriptor per binding via
                // vkGetDescriptorEXT at the driver's per-element byte cost. Runs regardless
                // of whether the classic allocation for this set succeeds. Proves host-
                // populatability up to, and never including, vkCmdDispatch.
                if (getDescriptor && db.enabled) {
                    for (const VkWriteDescriptorSet& w : writes) {
                        const size_t dbBytes = dbSizeFor(w.descriptorType);
                        ++stats.db_serialized_attempted;
                        if (dbBytes == 0) {
                            if (stats.first_db_failure.empty()) {
                                stats.first_db_failure = "no-descriptor-size " + name;
                            }
                            continue;
                        }
                        std::vector<unsigned char> dest(dbBytes, 0);
                        // Hoist the descriptor-source objects to this scope: getDescriptor
                        // dereferences gi.data.p* AFTER the branch, so they must outlive it
                        // (the previous per-branch locals left a dangling pointer -> crash).
                        VkDescriptorAddressInfoEXT addr{
                            VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT, nullptr, 0, 256,
                            VK_FORMAT_UNDEFINED};
                        VkDescriptorImageInfo im{};
                        VkDescriptorGetInfoEXT gi{VK_STRUCTURE_TYPE_DESCRIPTOR_GET_INFO_EXT};
                        gi.type = w.descriptorType;
                        bool wrote = true;
                        if (w.descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) {
                            addr.address = addrS;
                            gi.data.pStorageBuffer = &addr;
                        } else if (w.descriptorType == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER) {
                            addr.address = addrU;
                            gi.data.pUniformBuffer = &addr;
                        } else if (w.descriptorType == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) {
                            im.sampler = samp;
                            im.imageView = viewC;
                            im.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                            gi.data.pCombinedImageSampler = &im;
                        } else if (w.descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE) {
                            im.sampler = VK_NULL_HANDLE;
                            im.imageView = viewS;
                            im.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                            gi.data.pStorageImage = &im;
                        } else if (w.descriptorType == VK_DESCRIPTOR_TYPE_SAMPLER) {
                            gi.data.pSampler = &samp;
                        } else {
                            wrote = false;
                        }
                        if (!wrote) {
                            if (stats.first_db_failure.empty()) {
                                stats.first_db_failure = "unmapped-kind " + name;
                            }
                            continue;
                        }
                        getDescriptor(device, &gi, dbBytes, dest.data());
                        ++stats.db_serialized_ok;
                        stats.db_footprint_bytes += dbBytes;
                        bool any = false;
                        for (unsigned char b : dest) {
                            if (b != 0) {
                                any = true;
                                break;
                            }
                        }
                        if (any) {
                            ++stats.db_serialized_nonzero;
                        }
                    }
                }
                VkDescriptorSet dset = VK_NULL_HANDLE;
                VkDescriptorSetAllocateInfo alloc{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                                  nullptr, pool, 1u, &lout};
                if (vkAllocateDescriptorSets(device, &alloc, &dset) != VK_SUCCESS) {
                    ++stats.populated_sets_classic_fail;
                    if (stats.first_pop_failure.empty()) {
                        stats.first_pop_failure = "allocateDescriptorSet " + name;
                    }
                    modClassicOk = false;
                    continue;
                }
                for (auto& w : writes) {
                    w.dstSet = dset;
                }
                if (!writes.empty()) {
                    vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()),
                                           writes.data(), 0, nullptr);
                    // vkUpdateDescriptorSets returns void: a rejection surfaces as device
                    // loss / driver log, so re-create-and-count is best-effort below.
                }
                ++stats.populated_sets_classic_ok;
                stats.populated_bindings_ok += bindingCountInSet;
                if (dset) {
                    vkFreeDescriptorSets(device, pool, 1, &dset);
                }
            }
            if (modClassicOk) {
                ++stats.modules_populated_classic;
            }
        }
        if (pipeline) {
            vkDestroyPipeline(device, pipeline, nullptr);
        }
        vkDestroyShaderModule(device, module, nullptr);
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        for (auto l : setLayouts) {
            if (l) {
                vkDestroyDescriptorSetLayout(device, l, nullptr);
            }
        }
    }
    if (pool) {
        vkDestroyDescriptorPool(device, pool, nullptr);
    }
    if (samp) {
        vkDestroySampler(device, samp, nullptr);
    }
    if (viewS) {
        vkDestroyImageView(device, viewS, nullptr);
    }
    if (viewC) {
        vkDestroyImageView(device, viewC, nullptr);
    }
    if (imgS) {
        vkDestroyImage(device, imgS, nullptr);
    }
    if (imgC) {
        vkDestroyImage(device, imgC, nullptr);
    }
    if (bufS) {
        vkDestroyBuffer(device, bufS, nullptr);
    }
    if (bufU) {
        vkDestroyBuffer(device, bufU, nullptr);
    }
    for (VkDeviceMemory memory : population_memory) {
        vkFreeMemory(device, memory, nullptr);
    }
    if (cache && destroyCache) {
        destroyCache(device, cache, nullptr);
    }
    return stats;
}

// Parse FFG4_VK_BENCHMARK into a repeat count: unset -> 0 (no loop), "1" -> 16,
// "N" -> N, ":N" -> N.
static int benchmarkCount() {
    const auto value = getEnv("FFG4_VK_BENCHMARK");
    if (value.empty()) {
        return 0;
    }
    if (value.size() > 1 && value[0] == ':') {
        return std::max(1, std::atoi(value.c_str() + 1));
    }
    if (value == "1") {
        return 16;
    }
    const int parsed = std::atoi(value.c_str());
    return parsed > 0 ? parsed : 1;
}

static double msBetween(uint64_t start, uint64_t end) {
    return static_cast<double>(end - start) / 1e6;
}

int main(int argc, char** argv) {
    const std::string packDir = getEnv("FFG4_VK_PACK").empty()
                                    ? "assets/frame-generation/pack"
                                    : getEnv("FFG4_VK_PACK");
    if (getEnv("FFG4_VK_SELFTEST") == "1" || (argc > 1 && std::string(argv[1]) == "--self-test")) {
        std::printf("ffg4-exec: no device queried (self-test). pack=%s "
                    "pipeline-cache-namespace=fg4 selftest-kernel=embedded "
                    "bench=FFG4_VK_BENCHMARK profile=FFG4_VK_PROFILE\n",
                    packDir.c_str());
        return 0;
    }

    VkInstance instance{};
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO, nullptr, "ffg4-exec", 0,
                          "amd_fidelityfx_framegeneration_vk", 0, VK_MAKE_API_VERSION(0, 1, 4, 0)};
    VkInstanceCreateInfo instanceInfo{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    instanceInfo.pApplicationInfo = &app;
    if (vkCreateInstance(&instanceInfo, nullptr, &instance) != VK_SUCCESS) {
        std::printf("no Vulkan instance\n");
        return 77;
    }
    uint32_t count = 0;
    if (vkEnumeratePhysicalDevices(instance, &count, nullptr) != VK_SUCCESS || count == 0) {
        std::printf("no physical devices\n");
        vkDestroyInstance(instance, nullptr);
        return 77;
    }
    std::vector<VkPhysicalDevice> physicals(count);
    vkEnumeratePhysicalDevices(instance, &count, physicals.data());

    std::string filter;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a.rfind("--", 0) == 0) {
            continue;  // flag
        }
        if (filter.empty()) {
            filter = a;  // first positional = device substring
        }
    }
    VkDevice device = VK_NULL_HANDLE;
    VkPhysicalDevice chosen = VK_NULL_HANDLE;
    for (const auto physical : physicals) {
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(physical, &props);
        if (!filter.empty() && std::string(props.deviceName).find(filter) == std::string::npos) {
            continue;
        }
        if (props.vendorID != 0x1002) {  // AMD Radeon only (matches the FG target)
            continue;
        }
        const auto candidate = createContractDevice(physical);
        if (candidate == VK_NULL_HANDLE) {
            continue;
        }
        device = candidate;
        chosen = physical;
        break;
    }
    if (device == VK_NULL_HANDLE) {
        std::printf("no AMD device offered the Frame Generation execution contract\n");
        vkDestroyInstance(instance, nullptr);
        return 77;
    }

    VkPhysicalDeviceProperties chosenProps{};
    vkGetPhysicalDeviceProperties(chosen, &chosenProps);
    std::printf("device=\"%s\" vendor=0x%08X api=%u.%u.%u\n", chosenProps.deviceName,
                chosenProps.vendorID, VK_API_VERSION_MAJOR(chosenProps.apiVersion),
                VK_API_VERSION_MINOR(chosenProps.apiVersion),
                VK_API_VERSION_PATCH(chosenProps.apiVersion));

    const auto identity = identityOf(chosen);
    std::string cachePath = getEnv("FFG4_VK_PIPELINE_CACHE");
    if (cachePath.empty()) {
        cachePath = "/tmp/ffg4vk-exec-cache." + std::to_string((unsigned)getpid());
    }
    std::remove(cachePath.c_str());
    const auto disabled = !getEnv("FFG4_VK_PIPELINE_CACHE_DISABLE").empty();

    auto* createCache = reinterpret_cast<PFN_vkCreatePipelineCache>(
        vkGetDeviceProcAddr(device, "vkCreatePipelineCache"));
    auto* getData = reinterpret_cast<PFN_vkGetPipelineCacheData>(
        vkGetDeviceProcAddr(device, "vkGetPipelineCacheData"));
    auto* destroyCache = reinterpret_cast<PFN_vkDestroyPipelineCache>(
        vkGetDeviceProcAddr(device, "vkDestroyPipelineCache"));

    const uint64_t moduleStart = now_ns();
    VkShaderModule module = VK_NULL_HANDLE;
    VkShaderModuleCreateInfo moduleInfo{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, nullptr, 0,
                                        sizeof(kSelfTestComputeSpv), kSelfTestComputeSpv};
    if (vkCreateShaderModule(device, &moduleInfo, nullptr, &module) != VK_SUCCESS) {
        std::printf("self-test module create failed\n");
        vkDestroyDevice(device, nullptr);
        vkDestroyInstance(instance, nullptr);
        return 1;
    }
    const double moduleMs = msBetween(moduleStart, now_ns());

    VkDescriptorSetLayoutBinding binding{};
    binding.binding = 0;
    binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    binding.descriptorCount = 1;
    binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layoutInfo.bindingCount = 1;
    layoutInfo.pBindings = &binding;
    VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
    vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &setLayout);

    VkDescriptorSetLayout sets[] = {setLayout};
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = sets;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout);

    auto createPipeline = [&](VkPipelineCache cache) -> VkPipeline {
        VkComputePipelineCreateInfo createInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        createInfo.layout = pipelineLayout;
        createInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        createInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        createInfo.stage.module = module;
        createInfo.stage.pName = "main";
        VkPipeline pipeline = VK_NULL_HANDLE;
        vkCreateComputePipelines(device, cache, 1, &createInfo, nullptr, &pipeline);
        return pipeline;
    };
    auto cacheBytes = [&](VkPipelineCache cache) -> size_t {
        size_t size = 0;
        getData(device, cache, &size, nullptr);
        return size;
    };

    // COLD: no prior container -> the pipeline compiles without a cache hit.
    VkPipelineCache coldCache = VK_NULL_HANDLE;
    size_t coldSeed = 0;
    if (!disabled && createCache) {
        const std::vector<unsigned char> seed = ffgvk::readCacheBlob(cachePath, identity);
        coldSeed = seed.size();
        VkPipelineCacheCreateInfo cci{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO, nullptr, 0,
                                      seed.size(), seed.empty() ? nullptr : seed.data()};
        createCache(device, &cci, nullptr, &coldCache);
    }
    const uint64_t coldStart = now_ns();
    VkPipeline coldPipeline = createPipeline(coldCache);
    const double coldMs = msBetween(coldStart, now_ns());
    const size_t coldCacheBytes = coldCache ? cacheBytes(coldCache) : 0;
    if (!disabled && coldCache && getData) {
        size_t size = 0;
        getData(device, coldCache, &size, nullptr);
        std::vector<unsigned char> blob(size);
        if (getData(device, coldCache, &size, blob.data()) == VK_SUCCESS) {
            ffgvk::writeCacheBlob(cachePath, identity, blob);
        }
    }
    if (coldPipeline) {
        vkDestroyPipeline(device, coldPipeline, nullptr);
    }
    if (coldCache && destroyCache) {
        destroyCache(device, coldCache, nullptr);
    }

    // WARM: same effect/GPU/driver -> reload the serialized container as initialData.
    VkPipelineCache warmCache = VK_NULL_HANDLE;
    size_t warmSeed = 0;
    if (!disabled && createCache) {
        const std::vector<unsigned char> seed = ffgvk::readCacheBlob(cachePath, identity);
        warmSeed = seed.size();
        VkPipelineCacheCreateInfo cci{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO, nullptr, 0,
                                      seed.size(), seed.empty() ? nullptr : seed.data()};
        createCache(device, &cci, nullptr, &warmCache);
    }
    const uint64_t warmStart = now_ns();
    VkPipeline warmPipeline = createPipeline(warmCache);
    const double warmMs = msBetween(warmStart, now_ns());
    const size_t warmCacheBytes = warmCache ? cacheBytes(warmCache) : 0;

    // BENCHMARK: repeated pipeline creation into the warm cache (cache hits).
    const int iterations = benchmarkCount();
    double bestMs = 0.0;
    double totalMs = 0.0;
    for (int i = 0; i < iterations; ++i) {
        const uint64_t start = now_ns();
        VkPipeline p = createPipeline(warmCache);
        const double elapsed = msBetween(start, now_ns());
        totalMs += elapsed;
        bestMs = (bestMs == 0.0) ? elapsed : std::min(bestMs, elapsed);
        if (p) {
            vkDestroyPipeline(device, p, nullptr);
        }
    }

    uint32_t moduleOk = 0;
    uint32_t moduleTotal = 0;
    loadPackModules(packDir, device, moduleOk, moduleTotal);

    bool doCompile = !getEnv("FFG4_VK_COMPILE_PACK").empty();
    bool doParity = !getEnv("FFG4_VK_DB_PARITY").empty();
    bool doWrite = !getEnv("FFG4_VK_WRITE_DESC").empty();
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--compile-pack") {
            doCompile = true;
        }
        if (std::string(argv[i]) == "--descriptor-buffer-parity") {
            doParity = true;
        }
        if (std::string(argv[i]) == "--write-descriptors") {
            doWrite = true;
        }
    }
    if (doWrite) {
        doCompile = true;  // population census rides the descriptor-set pass
    }
    const DbSizes dbSizes = queryDescriptorSizes(chosen);
    PackCompile cc;
    PackCompile ccDb;
    if (doCompile) {
        cc = compilePackCompute(chosen, device, packDir, identity, cachePath, false, doWrite, dbSizes);
    }
    if (doParity) {
        ccDb = compilePackCompute(chosen, device, packDir, identity, cachePath, true, doWrite, dbSizes);
    }

    std::printf(
        "ffg4-exec RESULT cache=%s cold_seed=%zu saved_bytes=%zu warm_seed=%zu "
        "warm_bytes=%zu cold_create_ms=%.3f warm_create_ms=%.3f module_ms=%.3f "
        "bench_iters=%d bench_best_ms=%.3f bench_avg_ms=%.3f pack_modules_loaded=%u/%u "
        "pipeline_executed=%s value_check=pending-host-transport\n",
        disabled ? "disabled" : "fg4-active", coldSeed, coldCacheBytes, warmSeed,
        warmCacheBytes, coldMs, warmMs, moduleMs, iterations, bestMs,
        iterations ? totalMs / iterations : 0.0, moduleOk, moduleTotal,
        warmPipeline ? "compiled" : "failed");

    if (doCompile) {
        std::printf(
            "ffg4-exec COMPILE_PACK transport=descriptor-set attempted=%u created=%u failed=%u "
            "attempted_collapsible=%u skipped_unified=%u skipped_unresolved=%u first_fail=%s code=%d\n",
            cc.attempted, cc.created, cc.failed, cc.attempted_collapsible,
            cc.skipped_unified, cc.skipped_unresolved,
            cc.first_failure_module.empty() ? "-" : cc.first_failure_module.c_str(),
            (int)cc.first_failure);
    }
    if (doParity) {
        std::printf(
            "ffg4-exec COMPILE_PACK transport=descriptor-buffer attempted=%u created=%u failed=%u "
            "attempted_collapsible=%u skipped_unified=%u skipped_unresolved=%u first_fail=%s code=%d\n",
            ccDb.attempted, ccDb.created, ccDb.failed, ccDb.attempted_collapsible,
            ccDb.skipped_unified, ccDb.skipped_unresolved,
            ccDb.first_failure_module.empty() ? "-" : ccDb.first_failure_module.c_str(),
            (int)ccDb.first_failure);
    }
    if (doWrite) {
        std::printf(
            "ffg4-exec DBSIZES enabled=%d storage_buffer=%zu uniform_buffer=%zu "
            "combined_image_sampler=%zu storage_image=%zu sampler=%zu\n",
            dbSizes.enabled ? 1 : 0, dbSizes.storageBuffer, dbSizes.uniformBuffer,
            dbSizes.combinedImageSampler, dbSizes.storageImage, dbSizes.sampler);
        std::printf(
            "ffg4-exec POPULATE classic_sets_ok=%u classic_sets_fail=%u bindings_ok=%u "
            "modules_classic=%u db_footprint_sets=%u db_serialize_attempted=%u "
            "db_serialize_ok=%u db_serialize_nonzero=%u db_bytes=%llu db_first_fail=%s "
            "classic_first_fail=%s\n",
            cc.populated_sets_classic_ok, cc.populated_sets_classic_fail,
            cc.populated_bindings_ok, cc.modules_populated_classic,
            cc.db_layout_footprints,
            cc.db_serialized_attempted, cc.db_serialized_ok, cc.db_serialized_nonzero,
            (unsigned long long)cc.db_footprint_bytes,
            cc.first_db_failure.empty() ? "-" : cc.first_db_failure.c_str(),
            cc.first_pop_failure.empty() ? "-" : cc.first_pop_failure.c_str());
    }

    if (warmPipeline) {
        vkDestroyPipeline(device, warmPipeline, nullptr);
    }
    if (warmCache && destroyCache) {
        destroyCache(device, warmCache, nullptr);
    }
    vkDestroyShaderModule(device, module, nullptr);
    vkDestroyDescriptorSetLayout(device, setLayout, nullptr);
    vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);
    std::remove(cachePath.c_str());
    return warmPipeline ? 0 : 2;
}
