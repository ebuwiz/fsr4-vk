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

// Shared orthonormal-basis test: finite, unit-length, pairwise |dot| near zero, handedness-independent.
// Used by BOTH the deprecated linked CameraInfo (0x0a) and the 2.4.0 PrepareV2 (0x0c) EMBEDDED camera
// vectors, so the two entry points enforce ONE contract.
bool basisOk(const float* up, const float* right, const float* fwd, std::string& reason) {
    if (!finite3(up) || !finite3(right) || !finite3(fwd)) {
        reason = "camera basis vector is not finite";
        return false;
    }
    const float tol = 1e-3f;
    for (const float* v : {up, right, fwd}) {
        if (std::fabs(norm3(v) - 1.0f) > tol) {
            reason = "camera basis vector is not unit length";
            return false;
        }
    }
    if (std::fabs(dot3(up, right)) > tol || std::fabs(dot3(up, fwd)) > tol ||
        std::fabs(dot3(right, fwd)) > tol) {
        reason = "camera basis vectors are not orthogonal";
        return false;
    }
    return true;
}

// A camera-info block (linked after a Prepare descriptor) must carry finite, mutually-orthonormal basis
// vectors: unit length and pairwise |dot| near zero. This is the contract the SDK documents for FSR3.1.4+
// ("required input ... best quality"), independent of handedness so it cannot reject a valid left- or
// right-handed app convention.
bool cameraBasisOk(const ffxDispatchDescFrameGenerationPrepareCameraInfo* cam,
                   std::string& reason) {
    if (cam == nullptr) {
        reason = "camera-info block (type 0x0002000a) not linked after Prepare";
        return false;
    }
    return basisOk(cam->cameraUp, cam->cameraRight, cam->cameraForward, reason);
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

// ffxDispatchDescFrameGenerationPrepareV2 (0x0c) is absent from the vendored ffx_frame_generation.h this
// provider includes. Mirror it EXACTLY from the pinned 2.4.0 SDK
// (Kits/FidelityFX/framegeneration/include/ffx_framegeneration.h L225..L249): its leading fields are ABI-identical
// to ffxDispatchDescFrameGenerationPrepare (0x04) - the only change is `unused_reset` renamed to `reset` - and
// the camera basis is carried EMBEDDED as four [3]-float vectors, SUPERSEDING the deprecated linked CameraInfo
// (0x0a). Reading a host PrepareV2 through this mirror is what lets the provider validate the embedded camera
// WITHOUT repointing the whole provider at the newer header. check-test asserts sizeof(mirror) ==
// sizeof(base Prepare) + 4*3 floats, so the ABI assumption is machine-checked, not assumed.
struct PrepareV2Desc {
    ffxDispatchDescHeader      header;
    uint64_t                   frameID;
    uint32_t                   flags;
    void*                      commandList;
    struct FfxApiDimensions2D  renderSize;
    struct FfxApiFloatCoords2D jitterOffset;
    struct FfxApiFloatCoords2D motionVectorScale;
    float                        frameTimeDelta;
    bool                         reset;
    float                        cameraNear;
    float                        cameraFar;
    float                        cameraFovAngleVertical;
    float                        viewSpaceToMetersFactor;
    struct FfxApiResource        depth;
    struct FfxApiResource        motionVectors;
    float                        cameraPosition[3];
    float                        cameraUp[3];
    float                        cameraRight[3];
    float                        cameraForward[3];
};

// PrepareV2 (0x0c) argument + embedded-camera validation: the SAME numeric argument contract as prepareDescOk
// over the layout-identical prefix, PLUS the EMBEDDED basis via basisOk (which the old Prepare path only got
// when a separate CameraInfo block was linked under debug-checking). This is the completion of the provider's
// PrepareV2 TODO ("add the embedded-camera validation the injector relies on"); it does NOT run the algorithm.
Verdict prepareV2DescOk(const PrepareV2Desc* v2, bool depthInfinite, bool debugChecking) {
    Verdict v;
    if (v2 == nullptr) {
        note(v, "null PrepareV2 descriptor");
        return v;
    }
    const auto base = prepareDescOk(
        reinterpret_cast<const ffxDispatchDescFrameGenerationPrepare*>(v2), depthInfinite, debugChecking);
    if (!base.ok) {
        note(v, base.reason);
    }
    std::string br;
    if (!basisOk(v2->cameraUp, v2->cameraRight, v2->cameraForward, br)) {
        note(v, "PrepareV2 embedded camera: " + br);
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
