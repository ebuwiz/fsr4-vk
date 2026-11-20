// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 fsr4-vk contributors.
// See LICENSE in the project root.

#include "assets.hpp"

#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string_view>

#if defined(FFG4_EMBEDDED_ASSETS)
extern "C" {
extern const unsigned char fsr4fg_embedded_assets_begin[];
extern const unsigned char fsr4fg_embedded_assets_end[];
}
#endif

namespace ffgassets {
namespace {
constexpr std::string_view kPrefix = "__fg4_embedded__/";

#if defined(FFG4_EMBEDDED_ASSETS)
struct EmbeddedEntry {
    std::string name;
    std::span<const std::uint8_t> data;
};
std::vector<EmbeddedEntry> embedded;

std::span<const std::uint8_t> dataView() {
    return {reinterpret_cast<const std::uint8_t*>(fsr4fg_embedded_assets_begin),
            static_cast<std::size_t>(fsr4fg_embedded_assets_end -
                                     fsr4fg_embedded_assets_begin)};
}
#endif

std::string key(const std::filesystem::path& path) {
    const std::string text = path.generic_string();
    if (text.starts_with(kPrefix)) {
        return text.substr(kPrefix.size());
    }
    if (const char* root = std::getenv("FFG4_VK_ASSET_ROOT")) {
        const std::filesystem::path external(root);
        const auto relative = path.lexically_relative(external);
        if (!relative.empty() && !relative.is_absolute() &&
            *relative.begin() != "..") {
            return relative.generic_string();
        }
    }
    return text;
}
} // namespace

void initializeEmbedded() {
#if defined(FFG4_EMBEDDED_ASSETS)
    if (!embedded.empty()) {
        return;
    }
    const auto data = dataView();
    if (data.size() < 16 || std::memcmp(data.data(), "FFG4AS1\0", 8) != 0) {
        throw std::runtime_error("invalid embedded Frame Generation asset container");
    }
    // Every integer below is read by copying its bytes, never through a reinterpret_cast:
    // the index rows are packed (2 + 8 + 8, then a variable-length name), so a row's fields
    // sit at arbitrary offsets and a typed load there is a MISALIGNED load - UBSan flags it,
    // and it works on x86 only by the compiler's tolerance, not by the format's guarantee.
    auto readU32 = [&](std::size_t at) {
        std::uint32_t value = 0;
        std::memcpy(&value, data.data() + at, sizeof(value));
        return value;
    };
    auto readU16 = [&](std::size_t at) {
        std::uint16_t value = 0;
        std::memcpy(&value, data.data() + at, sizeof(value));
        return value;
    };
    auto readU64 = [&](std::size_t at) {
        std::uint64_t value = 0;
        std::memcpy(&value, data.data() + at, sizeof(value));
        return value;
    };
    const std::uint32_t count = readU32(8);
    const std::uint32_t payload_size = readU32(12);
    std::size_t offset = 16;
    struct PendingEntry {
        std::string name;
        std::uint64_t offset;
        std::uint64_t size;
    };
    std::vector<PendingEntry> pending;
    pending.reserve(count);
    for (std::uint32_t index = 0; index < count; ++index) {
        if (offset + 18 > data.size()) {
            throw std::runtime_error("truncated Frame Generation asset index");
        }
        const std::uint16_t name_size = readU16(offset);
        const std::uint64_t payload_offset = readU64(offset + 2);
        const std::uint64_t size = readU64(offset + 10);
        offset += 18;
        if (offset + name_size > data.size()) {
            throw std::runtime_error("truncated Frame Generation asset name");
        }
        std::string name(reinterpret_cast<const char*>(data.data() + offset), name_size);
        offset += name_size;
        pending.push_back({std::move(name), payload_offset, size});
    }
    if (offset + payload_size != data.size()) {
        throw std::runtime_error("invalid Frame Generation payload size");
    }
    const std::size_t payload_base = offset;
    for (const auto& entry : pending) {
        if (entry.offset > payload_size || entry.size > payload_size - entry.offset) {
            throw std::runtime_error("Frame Generation asset range is outside the payload");
        }
        embedded.push_back({
            entry.name,
            {data.data() + payload_base + entry.offset,
             static_cast<std::size_t>(entry.size)},
        });
    }
#endif
}

std::filesystem::path packRoot() {
    if (const char* root = std::getenv("FFG4_VK_ASSET_ROOT")) {
        return root;
    }
#if defined(FFG4_EMBEDDED_ASSETS)
    return "__fg4_embedded__";
#endif
    throw std::runtime_error("Frame Generation assets not found; set FFG4_VK_ASSET_ROOT");
}

bool exists(const std::filesystem::path& path) {
#if defined(FFG4_EMBEDDED_ASSETS)
    initializeEmbedded();
    const std::string wanted = key(path);
    for (const auto& entry : embedded) {
        if (entry.name == wanted) {
            return true;
        }
    }
#endif
    return std::filesystem::is_regular_file(path);
}

std::span<const std::uint8_t> readView(const std::filesystem::path& path) {
#if defined(FFG4_EMBEDDED_ASSETS)
    initializeEmbedded();
    const std::string wanted = key(path);
    for (const auto& entry : embedded) {
        if (entry.name == wanted) {
            return entry.data;
        }
    }
#endif
    throw std::runtime_error("external Frame Generation asset reads require a copied buffer: " +
                             path.string());
}

std::vector<std::uint8_t> read(const std::filesystem::path& path) {
#if defined(FFG4_EMBEDDED_ASSETS)
    try {
        const auto view = readView(path);
        return {view.begin(), view.end()};
    } catch (const std::exception&) {
        if (path.generic_string().starts_with(kPrefix)) {
            throw;
        }
    }
#endif
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("cannot read Frame Generation asset: " + path.string());
    }
    return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
}

} // namespace ffgassets
