// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 fsr4-vk contributors.
// See LICENSE in the project root.

// Frame Generation API argument/order validation, extracted from the provider as
// pure predicates over AMD's own descriptor structs (vendored ffx_frame_generation.h)
// so they are unit-testable with no Vulkan device, no assets and no game-style
// dispatch — the same shape as upscaler-checks.hpp. The provider routes the verdicts
// through these on the create/configure/dispatch paths and still refuses to run the
// algorithm (the interpolation / ML / present runtime is not implemented), so this
// file enforces the DOCUMENTED mandatory call order and camera contract, NOT frame
// generation itself.

#pragma once

// ffx_frame_generation.h needs ffx_api.h's __declspec export macro; on non-Windows
// hosts that macro is absent, so apply the same visibility shim the provider header
// uses before including it.
#if !defined(_WIN32)
#if !defined(__declspec)
#define __declspec(x) __attribute__((visibility("default")))
#endif
#endif
#include "ffx_frame_generation.h"

#include <cmath>
#include <cstdint>
#include <string>

namespace ffgcheck {

struct Verdict {
    bool ok = true;
    std::string reason;
};

static void note(Verdict& v, const std::string& text) {
    v.ok = false;
    if (!v.reason.empty()) {
        v.reason += "; ";
    }
    v.reason += text;
}

static bool finite3(const float* a) {
    return a != nullptr && std::isfinite(a[0]) && std::isfinite(a[1]) && std::isfinite(a[2]);
}
static float dot3(const float* a, const float* b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
static float norm3(const float* a) {
    return std::sqrt(dot3(a, a));
}

// A camera-info block (linked after a Prepare descriptor) must carry finite,
// mutually-orthonormal basis vectors: unit length and pairwise |dot| near zero.
// This is the contract the SDK documents for FSR3.1.4+ ("required input ... best
// quality"), independent of handedness so it cannot reject a valid left- or right-
// handed app convention.
bool cameraBasisOk(const ffxDispatchDescFrameGenerationPrepareCameraInfo* cam,
                   std::string& reason) {
    if (cam == nullptr) {
        reason = "camera-info block (type 0x0002000a) not linked after Prepare";
        return false;
    }
    if (!finite3(cam->cameraUp) || !finite3(cam->cameraRight) || !finite3(cam->cameraForward)) {
        reason = "camera basis vector is not finite";
        return false;
    }
    const float tol = 1e-3f;
    for (const float* v : {cam->cameraUp, cam->cameraRight, cam->cameraForward}) {
        if (std::fabs(norm3(v) - 1.0f) > tol) {
            reason = "camera basis vector is not unit length";
            return false;
        }
    }
    if (std::fabs(dot3(cam->cameraUp, cam->cameraRight)) > tol ||
        std::fabs(dot3(cam->cameraUp, cam->cameraForward)) > tol ||
        std::fabs(dot3(cam->cameraRight, cam->cameraForward)) > tol) {
        reason = "camera basis vectors are not orthogonal";
        return false;
    }
    return true;
}

// Create-context sanity: a real display/render resolution and a known surface
// format. backBufferFormat 0 is FFX_SURFACE_FORMAT_UNKNOWN.
Verdict createDescOk(const ffxCreateContextDescFrameGeneration* d) {
    Verdict v;
    if (d == nullptr) {
        note(v, "null create descriptor");
        return v;
    }
    if (d->displaySize.width == 0u || d->displaySize.height == 0u) {
        note(v, "displaySize must be non-zero");
    }
    if (d->maxRenderSize.width == 0u || d->maxRenderSize.height == 0u) {
        note(v, "maxRenderSize must be non-zero");
    }
    if (d->backBufferFormat == 0u) {
        note(v, "backBufferFormat is FFX_SURFACE_FORMAT_UNKNOWN");
    }
    if (d->maxRenderSize.width < d->displaySize.width ||
        d->maxRenderSize.height < d->displaySize.height) {
        note(v, "maxRenderSize smaller than displaySize");
    }
    return v;
}

// Prepare validation. depthInfinite is the create-time
// FFX_FRAMEGENERATION_ENABLE_DEPTH_INFINITE bit; frameTimeDelta is milliseconds.
// When debugChecking is set, the frameTimeDelta-in-milliseconds check and the
// required camera-info block are enforced (mirrors AMD's own debug checker).
Verdict prepareDescOk(const ffxDispatchDescFrameGenerationPrepare* p,
                      bool depthInfinite,
                      bool debugChecking) {
    Verdict v;
    if (p == nullptr) {
        note(v, "null Prepare descriptor");
        return v;
    }
    if (p->renderSize.width == 0u || p->renderSize.height == 0u) {
        note(v, "renderSize must be non-zero");
    }
    if (!std::isfinite(p->frameTimeDelta) || p->frameTimeDelta <= 0.0f) {
        note(v, "frameTimeDelta must be a positive time in milliseconds");
    }
    if (debugChecking && p->frameTimeDelta < 1.0f) {
        note(v, "frameTimeDelta looks sub-millisecond (expected ~16.6 for 60fps)");
    }
    if (!std::isfinite(p->cameraNear) || !std::isfinite(p->cameraFar)) {
        note(v, "cameraNear/cameraFar must be finite");
    } else if (!depthInfinite && p->cameraNear > p->cameraFar) {
        note(v, "cameraNear must be <= cameraFar for finite depth");
    }
    if (!std::isfinite(p->jitterOffset.x) || !std::isfinite(p->jitterOffset.y) ||
        !std::isfinite(p->motionVectorScale.x) || !std::isfinite(p->motionVectorScale.y)) {
        note(v, "jitterOffset/motionVectorScale must be finite");
    }
    if (!std::isfinite(p->cameraFovAngleVertical) || p->cameraFovAngleVertical <= 0.0f) {
        note(v, "cameraFovAngleVertical must be a positive finite radians value");
    }
    return v;
}

// Generate validation: one output per generated frame (outputs[4]), 1..3 frames
// (the pinned effect produces at most 3), and a sane luminance range when HDR.
Verdict generateDescOk(const ffxDispatchDescFrameGeneration* g) {
    Verdict v;
    if (g == nullptr) {
        note(v, "null generate descriptor");
        return v;
    }
    if (g->numGeneratedFrames < 1u || g->numGeneratedFrames > 3u) {
        note(v, "numGeneratedFrames must be 1..3");
    }
    if (!std::isfinite(g->minMaxLuminance[0]) || !std::isfinite(g->minMaxLuminance[1])) {
        note(v, "minMaxLuminance must be finite");
    } else if (g->minMaxLuminance[0] > g->minMaxLuminance[1]) {
        note(v, "minMaxLuminance[0] must be <= minMaxLuminance[1]");
    }
    return v;
}

// The SDK's Frame Generation host-integration contract (ffxConfigureDescFrameGeneration,
// 0x00020002). When the host has frame generation ENABLED it MUST hand over the swap chain the
// runtime presents through AND both callbacks it drives back (presentCallback + frameGeneration
// callback) with their user contexts; HUDLessColor and generationRect are optional. generationRect
// width/height must be non-negative (0 == the "whole frame" sentinel). This is the contract the
// (future) present phase drives, so it is validated + remembered here even while the compute
// algorithm is still refused.
Verdict configureDescOk(const ffxConfigureDescFrameGeneration* c) {
    Verdict v;
    if (c == nullptr) {
        note(v, "null configure descriptor");
        return v;
    }
    if (c->frameGenerationEnabled) {
        if (c->swapChain == nullptr) {
            note(v, "frameGenerationEnabled requires a host swapChain");
        }
        if (c->presentCallback == nullptr) {
            note(v, "frameGenerationEnabled requires presentCallback");
        }
        if (c->frameGenerationCallback == nullptr) {
            note(v, "frameGenerationEnabled requires frameGenerationCallback");
        }
    }
    if (c->generationRect.width < 0 || c->generationRect.height < 0) {
        note(v, "generationRect width/height must be non-negative (0 = whole frame)");
    }
    return v;
}

// The documented mandatory call order: Configure -> Dispatch(Prepare) -> Dispatch(Frame
// Generation) -> Present. These mirror AMD's debug-checker order enforcement so a host that
// skips a step is refused with a precise reason rather than reaching an out-of-order dispatch.
Verdict prepareOrderOk(bool configured) {
    Verdict v;
    if (!configured) {
        note(v, "Dispatch(Prepare) requires a preceding ffxConfigure(ffxConfigureDescFrameGeneration)");
    }
    return v;
}
Verdict generateOrderOk(bool configured, bool prepared) {
    Verdict v;
    if (!configured) {
        note(v, "Dispatch(Frame Generation) requires a preceding ffxConfigure");
    }
    if (!prepared) {
        note(v, "Dispatch(Frame Generation) requires a preceding Dispatch(Prepare) for the frame");
    }
    return v;
}

// A registered distortion field (ffxConfigureDescFrameGenerationRegisterDistortionFieldResource,
// 0x00020008) must point at a real resource; a null resource is not a registerable lens-distortion
// field. This is host-boundary state (lens distortion is applied at present time), validated +
// remembered here even while the compute/present runtime is refused.
Verdict registerDistortionOk(
    const ffxConfigureDescFrameGenerationRegisterDistortionFieldResource* d) {
    Verdict v;
    if (d == nullptr) {
        note(v, "null distortion-field descriptor");
        return v;
    }
    if (d->distortionField.resource == nullptr) {
        note(v, "distortionField.resource must be non-null");
    }
    return v;
}

// The HUDLess create block (ffxCreateContextDescFrameGenerationHudless, 0x00020009) asks the
// provider for a HUD-less back buffer; its format must be a real surface (0 == UNKNOWN).
Verdict hudlessCreateDescOk(const ffxCreateContextDescFrameGenerationHudless* d) {
    Verdict v;
    if (d == nullptr) {
        note(v, "null HUDLess create descriptor");
        return v;
    }
    if (d->hudlessBackBufferFormat == 0u) {
        note(v, "hudlessBackBufferFormat is FFX_SURFACE_FORMAT_UNKNOWN");
    }
    return v;
}

// Walk a dispatch descriptor's pNext chain for the linked camera-info block
// (type FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION_PREPARE_CAMERAINFO).
const ffxDispatchDescFrameGenerationPrepareCameraInfo* findCameraInfo(
    const ffxDispatchDescHeader* header) {
    for (const auto* cur = (header ? header->pNext : nullptr); cur != nullptr; cur = cur->pNext) {
        if (cur->type == FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION_PREPARE_CAMERAINFO) {
            return reinterpret_cast<const ffxDispatchDescFrameGenerationPrepareCameraInfo*>(cur);
        }
    }
    return nullptr;
}

} // namespace ffgcheck
