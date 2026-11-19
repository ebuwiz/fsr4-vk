// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 fsr4-vk contributors.
// See LICENSE in the project root.

// GPU-free self-test for the Frame Generation argument/order/camera predicates in
// src/framegeneration/framegeneration-checks.hpp. No Vulkan device, no assets, no
// algorithm: it pins the mandatory-order + camera contract the provider enforces
// under FFX_FRAMEGENERATION_ENABLE_DEBUG_CHECKING, so a drift in the checker fails
// without a game-style dispatch. Exits 0 on success, 1 with a printed reason.

#include "framegeneration-checks.hpp"

#include <cstdio>
#include <string>

static int failures = 0;
static void check(bool cond, const std::string& label) {
    if (!cond) {
        std::printf("FAIL %s\n", label.c_str());
        ++failures;
    }
}

static ffxDispatchDescFrameGenerationPrepareCameraInfo makeCam() {
    // A right-handed orthonormal basis: forward -Z, up +Y, right +X.
    ffxDispatchDescFrameGenerationPrepareCameraInfo c{};
    c.header.type = FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION_PREPARE_CAMERAINFO;
    c.header.pNext = nullptr;
    const float p[3] = {0.0f, 0.0f, 0.0f};
    const float up[3] = {0.0f, 1.0f, 0.0f};
    const float right[3] = {1.0f, 0.0f, 0.0f};
    const float fwd[3] = {0.0f, 0.0f, -1.0f};
    for (int i = 0; i < 3; ++i) {
        c.cameraPosition[i] = p[i];
        c.cameraUp[i] = up[i];
        c.cameraRight[i] = right[i];
        c.cameraForward[i] = fwd[i];
    }
    return c;
}

static ffxReturnCode_t cbPresent(ffxCallbackDescFrameGenerationPresent*, void*) {
    return FFX_API_RETURN_OK;
}
static ffxReturnCode_t cbFg(ffxDispatchDescFrameGeneration*, void*) {
    return FFX_API_RETURN_OK;
}

int main() {
    // --- create descriptor -----------------------------------------------------
    {
        ffxCreateContextDescFrameGeneration d{};
        d.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATION;
        d.flags = 0u;
        d.displaySize = {1920u, 1080u};
        d.maxRenderSize = {1920u, 1080u};
        d.backBufferFormat = 1u;
        check(ffgcheck::createDescOk(&d).ok, "create: valid desc accepted");
        d.backBufferFormat = 0u;  // UNKNOWN
        check(!ffgcheck::createDescOk(&d).ok, "create: unknown surface format rejected");
        d.backBufferFormat = 1u;
        d.displaySize = {0u, 0u};
        check(!ffgcheck::createDescOk(&d).ok, "create: zero displaySize rejected");
    }

    // --- camera basis ----------------------------------------------------------
    {
        const auto good = makeCam();
        std::string reason;
        check(ffgcheck::cameraBasisOk(&good, reason), "camera: orthonormal basis accepted");
        check(!ffgcheck::cameraBasisOk(nullptr, reason), "camera: missing block rejected");
        auto bad = makeCam();
        bad.cameraUp[1] = 2.0f;  // not unit
        check(!ffgcheck::cameraBasisOk(&bad, reason), "camera: non-unit vector rejected");
        bad = makeCam();
        bad.cameraRight[1] = 1.0f;  // up.dot(right) != 0
        check(!ffgcheck::cameraBasisOk(&bad, reason), "camera: non-orthogonal rejected");
    }

    // --- prepare descriptor + depth-infinite / debug-checking ------------------
    {
        ffxDispatchDescFrameGenerationPrepare p{};
        p.header.type = FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION_PREPARE;
        p.renderSize = {1920u, 1080u};
        p.jitterOffset = {0.0f, 0.0f};
        p.motionVectorScale = {1.0f, 1.0f};
        p.frameTimeDelta = 16.6f;
        p.cameraNear = 0.1f;
        p.cameraFar = 1000.0f;
        p.cameraFovAngleVertical = 1.0f;
        p.flags = 0u;
        check(ffgcheck::prepareDescOk(&p, false, false).ok, "prepare: valid desc accepted");
        p.cameraNear = 2000.0f;  // near > far, finite depth
        check(!ffgcheck::prepareDescOk(&p, false, false).ok, "prepare: near>far finite rejected");
        check(ffgcheck::prepareDescOk(&p, true, false).ok,
              "prepare: near>far allowed when depth-infinite");
        p.cameraNear = 0.1f;
        p.frameTimeDelta = 0.0f;
        check(!ffgcheck::prepareDescOk(&p, false, false).ok, "prepare: non-positive time rejected");
        p.frameTimeDelta = 0.5f;  // sub-millisecond, only flagged under debug
        check(ffgcheck::prepareDescOk(&p, false, false).ok, "prepare: sub-ms ok when no debug");
        check(!ffgcheck::prepareDescOk(&p, false, true).ok, "prepare: sub-ms rejected under debug");
    }

    // --- generate descriptor --------------------------------------------------
    {
        ffxDispatchDescFrameGeneration g{};
        g.header.type = FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION;
        g.numGeneratedFrames = 1u;
        g.minMaxLuminance[0] = 0.0f;
        g.minMaxLuminance[1] = 1000.0f;
        check(ffgcheck::generateDescOk(&g).ok, "generate: 1 frame accepted");
        g.numGeneratedFrames = 3u;
        check(ffgcheck::generateDescOk(&g).ok, "generate: 3 frames accepted");
        g.numGeneratedFrames = 0u;
        check(!ffgcheck::generateDescOk(&g).ok, "generate: 0 frames rejected");
        g.numGeneratedFrames = 4u;
        check(!ffgcheck::generateDescOk(&g).ok, "generate: 4 frames rejected");
        g.numGeneratedFrames = 2u;
        g.minMaxLuminance[0] = 5.0f;
        g.minMaxLuminance[1] = 1.0f;  // inverted range
        check(!ffgcheck::generateDescOk(&g).ok, "generate: inverted luminance rejected");
    }

    // --- linked camera-info lookup on a Prepare pNext chain -------------------
    {
        ffxDispatchDescFrameGenerationPrepare p{};
        p.header.type = FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION_PREPARE;
        p.header.pNext = nullptr;
        check(ffgcheck::findCameraInfo(&p.header) == nullptr, "findCam: absent returns null");
        auto cam = makeCam();
        p.header.pNext = &cam.header;
        check(ffgcheck::findCameraInfo(&p.header) == &cam, "findCam: linked block found");
    }

    // --- configure host-integration contract + mandatory order --------------------
    {
        ffxConfigureDescFrameGeneration c{};
        c.header.type = FFX_API_CONFIGURE_DESC_TYPE_FRAMEGENERATION;
        c.frameGenerationEnabled = true;
        c.swapChain = (void*)&c;
        c.presentCallback = cbPresent;
        c.presentCallbackUserContext = nullptr;
        c.frameGenerationCallback = cbFg;
        c.frameGenerationCallbackUserContext = nullptr;
        c.allowAsyncWorkloads = false;
        c.onlyPresentGenerated = false;
        c.generationRect = {0, 0, 0, 0};
        c.flags = 0u;
        c.frameID = 1u;
        check(ffgcheck::configureDescOk(&c).ok, "configure: enabled + full contract accepted");
        c.swapChain = nullptr;
        check(!ffgcheck::configureDescOk(&c).ok, "configure: enabled without swapChain rejected");
        c.swapChain = (void*)&c;
        c.presentCallback = nullptr;
        check(!ffgcheck::configureDescOk(&c).ok, "configure: enabled without presentCallback rejected");
        c.presentCallback = cbPresent;
        c.frameGenerationCallback = nullptr;
        check(!ffgcheck::configureDescOk(&c).ok, "configure: enabled without frameGenerationCallback rejected");
        c.frameGenerationCallback = cbFg;
        c.generationRect = {0, 0, -1, 0};
        check(!ffgcheck::configureDescOk(&c).ok, "configure: negative generationRect.width rejected");
        c.generationRect = {0, 0, 0, 0};
        c.frameGenerationEnabled = false;  // disabled => callbacks are optional
        c.swapChain = nullptr;
        c.presentCallback = nullptr;
        c.frameGenerationCallback = nullptr;
        check(ffgcheck::configureDescOk(&c).ok, "configure: disabled contract with no callbacks accepted");

        check(!ffgcheck::prepareOrderOk(false).ok, "order: Prepare without Configure rejected");
        check(ffgcheck::prepareOrderOk(true).ok, "order: Prepare after Configure ok");
        check(!ffgcheck::generateOrderOk(false, false).ok,
              "order: Generate without Configure rejected");
        check(!ffgcheck::generateOrderOk(true, false).ok,
              "order: Generate without Prepare rejected");
        check(ffgcheck::generateOrderOk(true, true).ok,
              "order: Generate after Configure + Prepare ok");
    }

    // --- optional FG create/configure blocks (distortion field + HUDLess) ---------
    {
        ffxConfigureDescFrameGenerationRegisterDistortionFieldResource d{};
        d.header.type = FFX_API_CONFIGURE_DESC_TYPE_FRAMEGENERATION_REGISTERDISTORTIONRESOURCE;
        check(!ffgcheck::registerDistortionOk(&d).ok,
              "distortion: null resource rejected");
        int resToken = 0;
        d.distortionField.resource = &resToken;
        check(ffgcheck::registerDistortionOk(&d).ok, "distortion: real resource accepted");

        ffxCreateContextDescFrameGenerationHudless h{};
        h.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATION_HUDLESS;
        h.hudlessBackBufferFormat = 0u;  // FFX_SURFACE_FORMAT_UNKNOWN
        check(!ffgcheck::hudlessCreateDescOk(&h).ok, "hudless: UNKNOWN format rejected");
        h.hudlessBackBufferFormat = 1u;
        check(ffgcheck::hudlessCreateDescOk(&h).ok, "hudless: known format accepted");
    }

    // --- PrepareV2 (0x0c): the provider order-checks it; here pin arg + EMBEDDED camera basis -----
    {
        ffgcheck::PrepareV2Desc v2{};
        v2.renderSize = {1920u, 1080u};
        v2.jitterOffset = {0.0f, 0.0f};
        v2.motionVectorScale = {1.0f, 1.0f};
        v2.frameTimeDelta = 16.6f;
        v2.reset = false;
        v2.cameraNear = 0.1f;
        v2.cameraFar = 1000.0f;
        v2.cameraFovAngleVertical = 1.0f;
        v2.viewSpaceToMetersFactor = 0.0f;
        const float pos[3] = {0.0f, 0.0f, 0.0f}, up[3] = {0.0f, 1.0f, 0.0f},
                      right[3] = {1.0f, 0.0f, 0.0f}, fwd[3] = {0.0f, 0.0f, -1.0f};
        for (int i = 0; i < 3; ++i) {
            v2.cameraPosition[i] = pos[i]; v2.cameraUp[i] = up[i];
            v2.cameraRight[i] = right[i]; v2.cameraForward[i] = fwd[i];
        }
        check(ffgcheck::prepareV2DescOk(&v2, false, true).ok, "prepareV2: valid args + embedded basis accepted");
        auto bad = v2; bad.cameraUp[1] = 2.0f;   // not unit
        check(!ffgcheck::prepareV2DescOk(&bad, false, true).ok, "prepareV2: non-unit embedded basis rejected");
        bad = v2; bad.cameraRight[1] = 1.0f;     // up.dot(right) != 0
        check(!ffgcheck::prepareV2DescOk(&bad, false, true).ok, "prepareV2: non-orthogonal embedded basis rejected");
        bad = v2; bad.renderSize = {0u, 0u};
        check(!ffgcheck::prepareV2DescOk(&bad, false, true).ok, "prepareV2: zero renderSize rejected");
        check(!ffgcheck::prepareV2DescOk(nullptr, false, true).ok, "prepareV2: null descriptor rejected");
        // ABI mirror check: the 4 embedded camera vectors begin exactly at sizeof(base Prepare).
        check(sizeof(ffgcheck::PrepareV2Desc) ==
                  sizeof(ffxDispatchDescFrameGenerationPrepare) + 4u * 3u * sizeof(float),
              "prepareV2: mirror size == base Prepare + 4x camera[3]");
    }

    if (failures == 0) {
        std::printf("framegeneration-check self-test OK (order + camera + create/generate/prepareV2)\n");
        return 0;
    }
    std::printf("framegeneration-check self-test FAILED: %d\n", failures);
    return 1;
}
