// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 fsr4-vk contributors.
// See LICENSE in the project root.

// Deterministic FSR upscaler helper math, factored out of the provider so it is
// one source of truth AND unit-testable with no Vulkan device / assets / shader
// (same shape as upscaler-checks.hpp and framegeneration-checks.hpp). These are the
// pure functions a host calls through ffxQuery before dispatch: quality -> ratio,
// display -> render resolution, and the jitter sequence (phase count + R2 offset).
// Ratios are fixed by AMD's ffx_upscale.h doc comments (NativeAA 1.0x, Quality 1.5x,
// Balanced 1.7x, Performance 2.0x, Ultra-Performance 3.0x).

#pragma once

#include <cmath>
#include <cstdint>

namespace fsr4query {

// AMD's documented per-dimension upscaling ratio for a quality mode. Out-of-range
// mode returns 0.0f, which callers treat as "invalid mode" (parameter error).
constexpr std::uint32_t kQualityModeCount = 5u;
float upscaleRatio(std::uint32_t qualityMode) {
    constexpr float ratios[kQualityModeCount] = {1.0f, 1.5f, 1.7f, 2.0f, 3.0f};
    return qualityMode < kQualityModeCount ? ratios[qualityMode] : 0.0f;
}

// The render dimension for a display dimension at a given upscale ratio: the
// provider rounds UP (ceil) so the render target never undershoots the ratio.
// A non-positive ratio yields 0 (caller rejects).
std::uint32_t renderDimension(std::uint32_t displayDimension, float ratio) {
    if (ratio <= 0.0f) {
        return 0u;
    }
    return static_cast<std::uint32_t>(
        std::ceil(static_cast<float>(displayDimension) / ratio));
}

// Jitter sequence length for a render->display scaling factor. Matches AMD's
// modern ffx_upscale GetJitterPhaseCount: int(8.0 * (displayWidth/renderWidth)^2).
// A zero renderWidth is invalid (caller rejects) -> 0.
std::int32_t jitterPhaseCount(std::uint32_t renderWidth, std::uint32_t displayWidth) {
    if (renderWidth == 0u) {
        return 0;
    }
    const float ratio = static_cast<float>(displayWidth) / static_cast<float>(renderWidth);
    return static_cast<std::int32_t>(8.0f * ratio * ratio);
}

// Van der Corput / Halton low-discrepancy inverse: the radical inverse of `index`
// in `base`, used to spread sub-pixel jitter uniformly over time.
float radicalInverse(std::uint32_t index, std::uint32_t base) {
    if (base < 2u) {
        return 0.0f;
    }
    float value = 0.0f;
    const float inverseBase = 1.0f / static_cast<float>(base);
    float fraction = inverseBase;
    while (index != 0u) {
        value += static_cast<float>(index % base) * fraction;
        index /= base;
        fraction *= inverseBase;
    }
    return value;
}

// Sub-pixel jitter offset for (index, phaseCount): the sample position is
// index % phaseCount + 1, and the two axes use different radical bases (2 and 3)
// so the pair is a 2D low-discrepancy sequence. Both axes stay in [-0.5, 0.5).
struct Offset2 {
    float x = 0.0f;
    float y = 0.0f;
};
Offset2 jitterOffset(std::int32_t index, std::int32_t phaseCount) {
    if (phaseCount <= 0 || index < 0) {
        return Offset2{};
    }
    const auto sample = static_cast<std::uint32_t>(index % phaseCount) + 1u;
    return Offset2{radicalInverse(sample, 2u) - 0.5f, radicalInverse(sample, 3u) - 0.5f};
}

// The upscaler provider's OWN GPU memory usage (ffxQueryDescUpscaleGetGPUmemoryUsage{,V2}).
// This provider is a command-buffer recorder: every color/depth/motion/exposure/reactive/
// transparency/history surface is supplied by the host at dispatch, and the FSR4 intermediate
// targets it drives are allocated through the CALLER's allocator -- the context itself retains
// no persistent GPU resources. So the effect's owned usage is honestly {0,0}: the provider's
// real amount, NOT a stand-in for AMD's DLL internal allocations (this provider performs none).
struct GPUMemoryUsage {
    std::uint64_t totalUsageInBytes = 0u;
    std::uint64_t aliasableUsageInBytes = 0u;
};
GPUMemoryUsage gpuMemoryUsage() {
    return GPUMemoryUsage{0u, 0u};
}

} // namespace fsr4query
