// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 fsr4-vk contributors.
// See LICENSE in the project root.

// Pure, GPU-free predicates for the documented AMD FSR Upscaling 4.1.2 debug
// checker (FFX_UPSCALE_ENABLE_DEBUG_CHECKING). The provider routes every returned
// string through providerMessage(..., FFX_API_MESSAGE_TYPE_WARNING, ...), so the
// host's fpMessage callback and the runtime log both see it; keeping the decision
// logic here (side-effect free) lets it be unit-tested without driving a dispatch.

#pragma once

// The FFX API headers use __declspec(dllexport/dllimport) for the shared entry points;
// on non-Windows hosts that needs the same visibility shim provider.hpp/framegeneration.hpp
// apply before pulling in the API. Kept here so this header is self-contained.
#if !defined(_WIN32)
#define __declspec(x) __attribute__((visibility("default")))
#endif
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#include "ffx_upscale.h"

#include <cstdint>
#include <string>
#include <vector>

namespace fsr4provider {

// createFlags carries the FFX_UPSCALE_ENABLE_* create-context bits; cameraFar and
// frameTimeDelta come from ffxDispatchDescUpscale. Returns exactly the warnings
// that apply for this frame, empty when the checker was not requested.
//   * DEPTH_INVERTED + DEPTH_INFINITE with a very low cameraFar can cause depth-
//     separation artifacting (cameraFar < 1.0f is a stated approximation of the
//     SDK's "very low" under an infinite far plane).
//   * frameTimeDelta is milliseconds; < 1.0f means the host is not reporting a
//     realistic frame time (~16.6f for 60fps).
std::vector<std::string> upscaleDebugCheckerWarnings(std::uint32_t createFlags,
                                                     float cameraFar,
                                                     float frameTimeDelta) {
    std::vector<std::string> warnings;
    if ((createFlags & FFX_UPSCALE_ENABLE_DEBUG_CHECKING) == 0) {
        return warnings;
    }
    const bool depth_inverted =
        (createFlags & FFX_UPSCALE_ENABLE_DEPTH_INVERTED) != 0;
    const bool depth_infinite =
        (createFlags & FFX_UPSCALE_ENABLE_DEPTH_INFINITE) != 0;
    if (depth_inverted && depth_infinite && cameraFar < 1.0f) {
        warnings.push_back(
            "FSR_API_DEBUG_WARNING: FFX_UPSCALE_ENABLE_DEPTH_INFINITE and "
            "FFX_UPSCALE_ENABLE_DEPTH_INVERTED present, cameraFar value is very "
            "low which may result in depth separation artifacting");
    }
    if (frameTimeDelta < 1.0f) {
        warnings.push_back(
            "FSR_API_DEBUG_WARNING: frameTimeDelta is less than 1.0f - this "
            "value should be milliseconds (~16.6f for 60fps)");
    }
    return warnings;
}

} // namespace fsr4provider
