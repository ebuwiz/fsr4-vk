// Measures which generation of VK_EXT_descriptor_buffer's CONSUMPTION API this driver
// actually exports, because that decides how the runnable FG dispatch must be written.
//
// Measured on this repo's PINNED Vulkan-Headers: they declare the 0.14.2-era draft
// (vkCmdBindDescriptorBuffersEXT / vkCmdSetDescriptorBufferOffsetsEXT /
// vkCmdBindDescriptorBufferEmbeddedSamplersEXT / vkGetPhysicalDeviceDescriptorSizeEXT /
// vkWriteResourceDescriptorsEXT) and NOT the final-revision names (vkCmdPushPipelineDataEXT /
// vkCmdBindDescriptorRangesEXT appear nowhere in vulkan_core.h). A newer RADV may speak the
// final names instead, so both generations are probed through the host's own resolver.
//
//   g++ -std=c++20 -I third_party/Vulkan-Headers/include tools/db-command-surface.cpp -o build/db-surface -lvulkan
//   ./build/db-surface Radeon
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <vulkan/vulkan.h>

static const char* kWanted[] = {
    // Serialization half (the Phase-1 substrate already uses these successfully).
    "vkGetDescriptorEXT", "vkGetDescriptorSetLayoutSizeEXT", "vkGetDescriptorSetLayoutBindingOffsetEXT",
    "vkGetPhysicalDeviceDescriptorSizeEXT",
    // Consumption half, 0.14.2-era draft - what the pinned headers DO declare.
    "vkCmdBindDescriptorBuffersEXT", "vkCmdSetDescriptorBufferOffsetsEXT",
    "vkCmdBindDescriptorBufferEmbeddedSamplersEXT", "vkCmdSetDescriptorBufferOffsets2EXT",
    "vkWriteResourceDescriptorsEXT", "vkWriteSamplerDescriptorsEXT",
    // Consumption half, final revision - what the pinned headers do NOT declare.
    "vkCmdPushPipelineDataEXT", "vkCmdBindDescriptorRangesEXT", "vkCmdBindDescriptorRanges2EXT",
    // Controls: functions that exist on any device, to prove the resolver works at all.
    "vkCmdBindPipeline", "vkCmdDispatch", "vkQueueSubmit", "vkQueueWaitIdle", "vkGetBufferDeviceAddress",
};

int main(int argc, char** argv) {
    const char* wantDevice = argc > 1 ? argv[1] : "Radeon";
    VkApplicationInfo app{};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "db-surface";
    VkInstanceCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &app;
    VkInstance instance = VK_NULL_HANDLE;
    if (vkCreateInstance(&ici, nullptr, &instance) != VK_SUCCESS) { std::printf("NO_INSTANCE\n"); return 1; }

    std::uint32_t n = 0;
    vkEnumeratePhysicalDevices(instance, &n, nullptr);
    std::vector<VkPhysicalDevice> devices(n ? n : 1);
    vkEnumeratePhysicalDevices(instance, &n, devices.data());
    VkPhysicalDevice pick = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties props{};
    for (std::uint32_t i = 0; i < n; ++i) {
        VkPhysicalDeviceProperties p{};
        vkGetPhysicalDeviceProperties(devices[i], &p);
        if (std::strstr(p.deviceName, wantDevice)) { pick = devices[i]; props = p; break; }
    }
    if (!pick) { std::printf("NO_DEVICE_MATCHING=%s\n", wantDevice); vkDestroyInstance(instance, nullptr); return 1; }

    std::uint32_t ecount = 0;
    vkEnumerateDeviceExtensionProperties(pick, nullptr, &ecount, nullptr);
    std::vector<VkExtensionProperties> exts(ecount);
    vkEnumerateDeviceExtensionProperties(pick, nullptr, &ecount, exts.data());
    std::printf("device=\"%s\" api=%u.%u.%u vendor=%u extensions=%u\n", props.deviceName,
                VK_API_VERSION_MAJOR(props.apiVersion), VK_API_VERSION_MINOR(props.apiVersion),
                VK_API_VERSION_PATCH(props.apiVersion), props.vendorID, ecount);
    for (const auto& e : exts) {
        if (std::strstr(e.extensionName, "criptor") || std::strstr(e.extensionName, "OperativeMatrix") ||
            std::strstr(e.extensionName, "Derivative")) {
            std::printf("  ext %-52s spec=%u\n", e.extensionName, e.specVersion);
        }
    }

    // Enable the feature through this fork's own shape: the extension struct chains into
    // VkDeviceCreateInfo::pNext, which is where the pinned headers read it.
    VkPhysicalDeviceDescriptorBufferFeaturesEXT db{};
    db.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_BUFFER_FEATURES_EXT;
    db.descriptorBuffer = VK_TRUE;
    db.descriptorBufferPushDescriptors = VK_TRUE;
    VkDeviceCreateInfo di{};
    di.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    di.pNext = &db;
    VkDevice dev = VK_NULL_HANDLE;
    if (vkCreateDevice(pick, &di, nullptr, &dev) != VK_SUCCESS) {
        std::printf("NO_DEVICE (descriptor-buffer enablement refused by this driver)\n");
        vkDestroyInstance(instance, nullptr);
        return 1;
    }

    int devPresent = 0, instPresent = 0;
    for (const char* name : kWanted) {
        const void* d = reinterpret_cast<const void*>(vkGetDeviceProcAddr(dev, name));
        const void* i = reinterpret_cast<const void*>(vkGetInstanceProcAddr(instance, name));
        std::printf("%-48s device=%s instance=%s\n", name, d ? "PRESENT" : "ABSENT", i ? "PRESENT" : "ABSENT");
        if (d) ++devPresent;
        if (i) ++instPresent;
    }
    std::printf("RESOLVED census: deviceScope=%d instanceScope=%d of %zu probed\n", devPresent, instPresent,
                sizeof(kWanted) / sizeof(kWanted[0]));
    vkDestroyDevice(dev, nullptr);
    vkDestroyInstance(instance, nullptr);
    return 0;
}
