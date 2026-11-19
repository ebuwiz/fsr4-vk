// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 fsr4-vk contributors.
// See LICENSE in the project root.

// GPU-free self-test for the documented AMD FSR Upscaling 4.1.2 debug checker
// (src/provider/upscaler-checks.hpp). The provider routes these warnings through
// providerMessage on the first dispatch; no existing on-device test drives the
// provider ABI with DEBUG_CHECKING + the bad inputs, so this pins the decision
// logic directly: the checker is inert unless requested, each documented warning
// fires on its own condition, both fire together, and a well-formed frame is
// silent. Non-zero exit means regression.

#include "provider/upscaler-checks.hpp"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace fs = fsr4provider;

static std::uint32_t checked(const std::initializer_list<std::uint32_t>& bits) {
    std::uint32_t flags = 0u;
    for (std::uint32_t b : bits) {
        flags |= b;
    }
    return flags;
}

int main() {
    const std::uint32_t dc = FFX_UPSCALE_ENABLE_DEBUG_CHECKING;
    const std::uint32_t inv = FFX_UPSCALE_ENABLE_DEPTH_INVERTED;
    const std::uint32_t inf = FFX_UPSCALE_ENABLE_DEPTH_INFINITE;

    // 1) Checker NOT requested -> silent even with bad values.
    if (!fs::upscaleDebugCheckerWarnings(0u, 0.5f, 0.5f).empty()) {
        std::printf("FAIL: debug checker must be silent when not requested\n");
        return 1;
    }

    // 2) Only the depth warning (inverted + infinite + low cameraFar, valid frame time).
    {
        const auto w = fs::upscaleDebugCheckerWarnings(checked({dc, inv, inf}), 0.5f, 16.6f);
        if (w.size() != 1 || w[0].find("depth separation") == std::string::npos) {
            std::printf("FAIL: expected exactly the depth-separation warning\n");
            return 1;
        }
    }

    // 3) Only the frame-time warning (checked, sane camera, frameTimeDelta < 1.0f).
    {
        const auto w = fs::upscaleDebugCheckerWarnings(dc, 1000.0f, 0.5f);
        if (w.size() != 1 || w[0].find("frameTimeDelta") == std::string::npos) {
            std::printf("FAIL: expected exactly the frameTimeDelta warning\n");
            return 1;
        }
    }

    // 4) Both conditions at once -> two warnings.
    if (fs::upscaleDebugCheckerWarnings(checked({dc, inv, inf}), 0.5f, 0.5f).size() != 2) {
        std::printf("FAIL: expected two warnings when both conditions hold\n");
        return 1;
    }

    // 5) Well-formed frame with the checker ON -> silent.
    if (!fs::upscaleDebugCheckerWarnings(checked({dc, inv, inf}), 1000.0f, 16.6f).empty()) {
        std::printf("FAIL: expected no warnings for a well-formed frame\n");
        return 1;
    }

    std::printf("upscaler-debugcheck self-test OK (silent/1/1/2/silent)\n");
    return 0;
}
