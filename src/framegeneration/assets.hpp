// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 fsr4-vk contributors.
// See LICENSE in the project root.

#pragma once

#include <filesystem>
#include <span>
#include <string>
#include <vector>
#include <cstdint>

namespace ffgassets {

void initializeEmbedded();
bool exists(const std::filesystem::path& path);
std::vector<std::uint8_t> read(const std::filesystem::path& path);
std::span<const std::uint8_t> readView(const std::filesystem::path& path);
std::filesystem::path packRoot();

} // namespace ffgassets
