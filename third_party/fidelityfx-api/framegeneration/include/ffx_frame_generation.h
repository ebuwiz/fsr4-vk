// This file is part of the FidelityFX SDK.
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files(the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and /or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions :
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

// Vendored from AMD's FidelityFX SDK Frame Generation public API header
// (ffx_frame_generation.h), 2024 copyright, kept byte-faithful so the provider's
// descriptor type ids and field layouts are the SDK's own, not guesses: the earlier
// ffgvk constants used the Frame Generation EFFECT id (0x00020000) as the create-desc
// type, which does not match what a real host passes (0x00020001). This file is the
// source of truth for those ids and for the mandatory call order
// (Configure -> Dispatch(Prepare) -> Dispatch(Frame Generation) -> Present).

#pragma once

#include "ffx_api.h"
#include "ffx_api_types.h"

#define FFX_API_EFFECT_ID_FRAMEGENERATION 0x00020000u

#if defined(__cplusplus)
extern "C" {
#endif
enum FfxApiCreateContextFramegenerationFlags
{
    FFX_FRAMEGENERATION_ENABLE_ASYNC_WORKLOAD_SUPPORT              = (1<<0),
    FFX_FRAMEGENERATION_ENABLE_DISPLAY_RESOLUTION_MOTION_VECTORS   = (1<<1),
    FFX_FRAMEGENERATION_ENABLE_MOTION_VECTORS_JITTER_CANCELLATION  = (1<<2),
    FFX_FRAMEGENERATION_ENABLE_DEPTH_INVERTED                      = (1<<3),
    FFX_FRAMEGENERATION_ENABLE_DEPTH_INFINITE                      = (1<<4),
    FFX_FRAMEGENERATION_ENABLE_HIGH_DYNAMIC_RANGE                  = (1<<5),
    FFX_FRAMEGENERATION_ENABLE_DEBUG_CHECKING                      = (1<<6),
};

enum FfxApiDispatchFramegenerationFlags
{
    FFX_FRAMEGENERATION_FLAG_DRAW_DEBUG_TEAR_LINES       = (1 << 0),
    FFX_FRAMEGENERATION_FLAG_DRAW_DEBUG_RESET_INDICATORS = (1 << 1),
    FFX_FRAMEGENERATION_FLAG_DRAW_DEBUG_VIEW = (1 << 2),
    FFX_FRAMEGENERATION_FLAG_NO_SWAPCHAIN_CONTEXT_NOTIFY = (1 << 3),
    FFX_FRAMEGENERATION_FLAG_DRAW_DEBUG_PACING_LINES = (1 << 4),
    FFX_FRAMEGENERATION_FLAG_RESERVED_1 = (1 << 5),
    FFX_FRAMEGENERATION_FLAG_RESERVED_2 = (1 << 6),
};

enum FfxApiUiCompositionFlags
{
    FFX_FRAMEGENERATION_UI_COMPOSITION_FLAG_USE_PREMUL_ALPHA                    = (1 << 0),
    FFX_FRAMEGENERATION_UI_COMPOSITION_FLAG_ENABLE_INTERNAL_UI_DOUBLE_BUFFERING = (1 << 1),
};

#define FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATION 0x00020001u
struct ffxCreateContextDescFrameGeneration
{
    ffxCreateContextDescHeader header;
    uint32_t flags;
    struct FfxApiDimensions2D displaySize;
    struct FfxApiDimensions2D maxRenderSize;
    uint32_t backBufferFormat;
};

#define FFX_API_CALLBACK_DESC_TYPE_FRAMEGENERATION_PRESENT 0x00020005u
typedef struct ffxCallbackDescFrameGenerationPresent
{
    ffxDispatchDescHeader header;
    void* device;
    void* commandList;
    struct FfxApiResource currentBackBuffer;
    struct FfxApiResource currentUI;
    struct FfxApiResource outputSwapChainBuffer;
    bool isGeneratedFrame;
    uint64_t              frameID;
} ffxCallbackDescFrameGenerationPresent;

#define FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION 0x00020003u
typedef struct ffxDispatchDescFrameGeneration
{
    ffxDispatchDescHeader header;
    void*                 commandList;
    struct FfxApiResource presentColor;
    struct FfxApiResource outputs[4];
    uint32_t              numGeneratedFrames;
    bool                  reset;
    uint32_t              backbufferTransferFunction;
    float                 minMaxLuminance[2];
    struct FfxApiRect2D   generationRect;
    uint64_t              frameID;
} ffxDispatchDescFrameGeneration;

typedef ffxReturnCode_t(*FfxApiPresentCallbackFunc)(ffxCallbackDescFrameGenerationPresent* params, void* pUserCtx);
typedef ffxReturnCode_t(*FfxApiFrameGenerationDispatchFunc)(ffxDispatchDescFrameGeneration* params, void* pUserCtx);

#define FFX_API_CONFIGURE_DESC_TYPE_FRAMEGENERATION 0x00020002u
struct ffxConfigureDescFrameGeneration
{
    ffxConfigureDescHeader header;
    void* swapChain;
    FfxApiPresentCallbackFunc presentCallback;
    void* presentCallbackUserContext;
    FfxApiFrameGenerationDispatchFunc frameGenerationCallback;
    void* frameGenerationCallbackUserContext;
    bool frameGenerationEnabled;
    bool allowAsyncWorkloads;
    struct FfxApiResource HUDLessColor;
    uint32_t flags;
    bool onlyPresentGenerated;
    struct FfxApiRect2D generationRect;
    uint64_t frameID;
};

#define FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION_PREPARE 0x00020004u
struct ffxDispatchDescFrameGenerationPrepare
{
    ffxDispatchDescHeader      header;
    uint64_t                   frameID;
    uint32_t                   flags;
    void*                      commandList;
    struct FfxApiDimensions2D  renderSize;
    struct FfxApiFloatCoords2D jitterOffset;
    struct FfxApiFloatCoords2D motionVectorScale;

    float                 frameTimeDelta;
    bool                  unused_reset;
    float                 cameraNear;
    float                 cameraFar;
    float                 cameraFovAngleVertical;
    float                 viewSpaceToMetersFactor;
    struct FfxApiResource depth;
    struct FfxApiResource motionVectors;
};

#define FFX_API_CONFIGURE_DESC_TYPE_FRAMEGENERATION_KEYVALUE 0x00020006u
struct ffxConfigureDescFrameGenerationKeyValue
{
    ffxConfigureDescHeader  header;
    uint64_t                key;
    uint64_t                u64;
    void*                   ptr;
};

enum FfxApiConfigureFrameGenerationKey
{
    // No values.
};

#define FFX_API_QUERY_DESC_TYPE_FRAMEGENERATION_GPU_MEMORY_USAGE 0x00020007u
struct ffxQueryDescFrameGenerationGetGPUMemoryUsage
{
    ffxQueryDescHeader header;
    struct FfxApiEffectMemoryUsage* gpuMemoryUsageFrameGeneration;
};

#define FFX_API_CONFIGURE_DESC_TYPE_FRAMEGENERATION_REGISTERDISTORTIONRESOURCE 0x00020008u
struct ffxConfigureDescFrameGenerationRegisterDistortionFieldResource
{
    ffxConfigureDescHeader header;
    struct FfxApiResource distortionField;
};

#define FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATION_HUDLESS 0x00020009u
struct ffxCreateContextDescFrameGenerationHudless
{
    ffxCreateContextDescHeader header;
    uint32_t hudlessBackBufferFormat;
};

#define FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION_PREPARE_CAMERAINFO 0x0002000au
// Linked after ffxDispatchDescFrameGenerationPrepare. Required input for FSR3.1.4 and
// onwards for best quality: the camera basis vectors used to reconstruct motion.
struct ffxDispatchDescFrameGenerationPrepareCameraInfo
{
    ffxConfigureDescHeader header;
    float                  cameraPosition[3];
    float                  cameraUp[3];
    float                  cameraRight[3];
    float                  cameraForward[3];
};

#if defined(__cplusplus)
} // extern "C"
#endif
