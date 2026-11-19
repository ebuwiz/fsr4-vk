// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 fsr4-vk contributors.
// See LICENSE in the project root.

// Namespaced Vulkan pipeline cache container shared by the Frame Generation provider
// and its on-device probe. A VkPipelineCache is driver-authored and opaque, so an
// application cannot set its identity; reusing one cache file across algorithms /
// GPUs / drivers risks feeding an incompatible blob. We wrap the driver blob in our
// OWN container header that pins the algorithm namespace (frame-generation "fg4",
// distinct from the upscaler's "fs4"), the exact device name + driver/api/vendor/device
// ids, and a format version. The blob is only fed to vkCreatePipelineCache as initial
// data when that identity matches the live device; otherwise the cache cold-starts.

#pragma once

#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace ffgvk {

constexpr std::uint32_t fourcc(char a, char b, char c) {
    return static_cast<std::uint32_t>(static_cast<unsigned char>(a)) |
           (static_cast<std::uint32_t>(static_cast<unsigned char>(b)) << 8) |
           (static_cast<std::uint32_t>(static_cast<unsigned char>(c)) << 16);
}

constexpr std::uint32_t kCacheAlgorithmFrameGeneration = fourcc('f', 'g', '4');
constexpr char kCacheMagic[8] = {'F', 'F', 'G', '4', 'P', 'C', '1', '\0'};
constexpr std::uint32_t kCacheFormatVersion = 1u;

struct CacheIdentity {
    std::uint32_t algorithm = kCacheAlgorithmFrameGeneration;
    std::string deviceName;
    std::uint32_t driverVersion = 0;
    std::uint32_t apiVersion = 0;
    std::uint32_t vendorId = 0;
    std::uint32_t deviceId = 0;
    std::uint32_t namespaceSalt = 0; // bumped to invalidate on shader-set changes
};

static void put32(std::vector<unsigned char>& out, std::uint32_t v) {
    for (int shift = 0; shift <= 24; shift += 8) {
        out.push_back(static_cast<unsigned char>((v >> shift) & 0xFFu));
    }
}

static std::uint32_t get32(const unsigned char* p) {
    return static_cast<std::uint32_t>(p[0]) | (static_cast<std::uint32_t>(p[1]) << 8) |
           (static_cast<std::uint32_t>(p[2]) << 16) | (static_cast<std::uint32_t>(p[3]) << 24);
}

// Serialize our container: magic, format, algorithm id, identity scalars + name,
// then the opaque driver blob length + bytes.
void writeCacheBlob(const std::string& path, const CacheIdentity& id,
                    const std::vector<unsigned char>& blob) {
    std::vector<unsigned char> out;
    for (char c : kCacheMagic) out.push_back(static_cast<unsigned char>(c));
    put32(out, kCacheFormatVersion);
    put32(out, id.algorithm);
    put32(out, id.driverVersion);
    put32(out, id.apiVersion);
    put32(out, id.vendorId);
    put32(out, id.deviceId);
    put32(out, id.namespaceSalt);
    put32(out, static_cast<std::uint32_t>(id.deviceName.size()));
    for (unsigned char c : id.deviceName) out.push_back(c);
    put32(out, static_cast<std::uint32_t>(blob.size()));
    out.insert(out.end(), blob.begin(), blob.end());
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char*>(out.data()),
               static_cast<std::streamsize>(out.size()));
}
// Read the driver blob back ONLY when the container matches this live identity;
// otherwise return empty so the cache cold-starts (never reuses a foreign cache).
std::vector<unsigned char> readCacheBlob(const std::string& path,
                                         const CacheIdentity& id) {
    std::ifstream file(path, std::ios::binary);
    if (!file) return {};
    std::vector<unsigned char> data((std::istreambuf_iterator<char>(file)),
                                    std::istreambuf_iterator<char>());
    file.close();
    const unsigned char* p = data.data();
    const std::size_t need = sizeof(kCacheMagic) + 6u * 4u;
    if (data.size() < need) return {};
    if (std::memcmp(p, kCacheMagic, sizeof(kCacheMagic)) != 0) return {};
    std::size_t off = sizeof(kCacheMagic);
    if (get32(p + off) != kCacheFormatVersion) return {};
    off += 4;
    const std::uint32_t algorithm = get32(p + off); off += 4;
    const std::uint32_t driverVersion = get32(p + off); off += 4;
    const std::uint32_t apiVersion = get32(p + off); off += 4;
    const std::uint32_t vendorId = get32(p + off); off += 4;
    const std::uint32_t deviceId = get32(p + off); off += 4;
    off += 4; // namespaceSalt
    const std::uint32_t nameLen = get32(p + off); off += 4;
    if (data.size() < off + nameLen + 4u) return {};
    std::string storedName(reinterpret_cast<const char*>(p + off), nameLen);
    off += nameLen;
    const std::uint32_t blobLen = get32(p + off); off += 4;
    if (data.size() < off + blobLen) return {};
    if (algorithm != id.algorithm || storedName != id.deviceName ||
        driverVersion != id.driverVersion || apiVersion != id.apiVersion ||
        vendorId != id.vendorId || deviceId != id.deviceId) {
        return {}; // identity mismatch -> refuse reuse, cold start
    }
    return std::vector<unsigned char>(p + off, p + off + blobLen);
}

// A cache file name that itself carries the algorithm + GPU + driver identity, so
// several providers/GPUs coexist on one host without overwriting each other.
std::string defaultCachePath(const CacheIdentity& id) {
    return "amd_fidelityfx_framegeneration_vk.pcache." + id.deviceName + "." +
           std::to_string(id.driverVersion);
}

} // namespace ffgvk
