// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 fsr4-vk contributors.
// See LICENSE in the project root.

// GPU-free self-test for the deterministic FSR upscaler helper math in
// src/provider/upscaler-queries.hpp — the exact functions the provider answers through
// ffxQuery (GetUpscaleRatioFromQualityMode / GetRenderResolutionFromQualityMode /
// GetJitterPhaseCount / GetJitterOffset). Ratios are pinned to AMD's ffx_upscale.h
// documented modes (1.0/1.5/1.7/2.0/3.0). No device, no shader: it locks the math so
// a refactor cannot silently change what a host gets back. Exits 0/1.

#include "provider/upscaler-queries.hpp"

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <string>

static int failures = 0;
static void check(bool cond, const std::string& label) {
    if (!cond) {
        std::printf("FAIL %s\n", label.c_str());
        ++failures;
    }
}
static bool approx(float a, float b, float tol = 1e-5f) {
    return std::fabs(a - b) <= tol;
}

int main() {
    // --- quality -> ratio (AMD's documented modes) ----------------------------
    {
        const float want[5] = {1.0f, 1.5f, 1.7f, 2.0f, 3.0f};
        for (std::uint32_t m = 0; m < 5u; ++m) {
            check(approx(fsr4query::upscaleRatio(m), want[m]),
                  "ratio mode " + std::to_string(m));
        }
        check(fsr4query::upscaleRatio(5u) == 0.0f, "ratio: out-of-range mode -> 0");
    }

    // --- display -> render dimension (ceil) --------------------------------
    {
        check(fsr4query::renderDimension(1920u, 3.0f) == 640u, "render 1920 @3.0 = 640");
        check(fsr4query::renderDimension(1920u, 1.7f) == 1130u, "render 1920 @1.7 = 1130");
        check(fsr4query::renderDimension(1080u, 1.7f) == 636u, "render 1080 @1.7 = 636");
        check(fsr4query::renderDimension(100u, 0.0f) == 0u, "render: ratio 0 -> 0");
    }

    // --- jitter phase count = int(8 * (display/render)^2) ------------------
    {
        check(fsr4query::jitterPhaseCount(640u, 1920u) == 72, "phaseCount 640->1920 = 72");
        check(fsr4query::jitterPhaseCount(960u, 1920u) == 32, "phaseCount 960->1920 = 32");
        check(fsr4query::jitterPhaseCount(1920u, 1920u) == 8, "phaseCount 1:1 = 8");
        check(fsr4query::jitterPhaseCount(0u, 1920u) == 0, "phaseCount: renderWidth 0 -> 0");
    }

    // --- radical inverse (van der corput) --------------------------------
    {
        check(approx(fsr4query::radicalInverse(0u, 2u), 0.0f), "radInv(0,2)=0");
        check(approx(fsr4query::radicalInverse(1u, 2u), 0.5f), "radInv(1,2)=0.5");
        check(approx(fsr4query::radicalInverse(2u, 2u), 0.25f), "radInv(2,2)=0.25");
        check(approx(fsr4query::radicalInverse(1u, 3u), 1.0f / 3.0f), "radInv(1,3)=1/3");
    }

    // --- jitter offset: bounds, determinism, sample wrap --------------------
    {
        const auto o = fsr4query::jitterOffset(0, 8);
        check(approx(o.x, 0.0f, 1e-6f), "offset(0,8).x = radInv(1,2)-0.5 = 0");
        check(approx(o.y, 1.0f / 3.0f - 0.5f, 1e-6f), "offset(0,8).y = radInv(1,3)-0.5");

        // Same sample position modulo phaseCount -> identical offset.
        const auto a = fsr4query::jitterOffset(3, 8);
        const auto b = fsr4query::jitterOffset(11, 8);  // 11 % 8 == 3
        check(a.x == b.x && a.y == b.y, "offset: index modulo phaseCount is stable");

        bool inRange = true;
        for (std::int32_t i = 0; i < 72; ++i) {
            const auto off = fsr4query::jitterOffset(i, 72);
            if (!(off.x >= -0.5f && off.x < 0.5f) ||
                !(off.y >= -0.5f && off.y < 0.5f)) {
                inRange = false;
            }
        }
        check(inRange, "offset: both axes stay in [-0.5, 0.5)");

        // Invalid input rejected (caller-side guard mirrors this).
        const auto bad = fsr4query::jitterOffset(0, 0);
        check(bad.x == 0.0f && bad.y == 0.0f, "offset: phaseCount<=0 -> {0,0}");
    }

    // --- provider-owned GPU memory usage (answered, not refused) --------------------
    {
        const auto mem = fsr4query::gpuMemoryUsage();
        check(mem.totalUsageInBytes == 0ull, "gpu-memory: provider owns 0 bytes");
        check(mem.aliasableUsageInBytes == 0ull, "gpu-memory: 0 aliasable bytes");
    }

    if (failures == 0) {
        std::printf("upscaler-query self-test OK (ratio/render/jitter)\n");
        return 0;
    }
    std::printf("upscaler-query self-test FAILED: %d\n", failures);
    return 1;
}
