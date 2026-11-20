#!/usr/bin/env bash
# tools/FGTest/build.sh - compile the animated-triangle FSR4 host sample + its SPIR-V, then run it offscreen (headless).
# For the visible 30-second app (the deliverable of this phase) use tools/FGTest/run.sh instead; this script is the
# headless gate that every change must keep green. FGTEST_BUILD_ONLY=1 compiles without running (run.sh uses it).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
BUILD="$HERE/build"; mkdir -p "$BUILD"

# 1) shaders -> SPIR-V
glslc -o "$BUILD/triangle.vert.spv" "$HERE/triangle.vert"
glslc -o "$BUILD/triangle.frag.spv" "$HERE/triangle.frag"
glslc -o "$BUILD/present.vert.spv" "$HERE/present.vert"
glslc -o "$BUILD/present.frag.spv" "$HERE/present.frag"

# 2) the app (SDL2 drives the optional --window present mode)
g++ -std=c++20 -Wall -o "$BUILD/FGTest" "$HERE/main.cpp" \
    $(sdl2-config --cflags) \
    -I"$REPO/third_party/fidelityfx-api/api/include" \
    -I"$REPO/third_party/fidelityfx-api/framegeneration/include" \
    -I"$REPO/third_party/fidelityfx-api/upscalers/include" \
    -I"$REPO/third_party/Vulkan-Headers/include" \
    -lvulkan -ldl $(sdl2-config --libs)
echo "built $BUILD/FGTest"

# 3) run headless against the built providers (pass their real paths); frames land in $BUILD
# The modular upscaler provider resolves its shader/model sidecars from FSR4_VK_ASSET_ROOT
# (otherwise create() throws "cannot locate sidecars" -> RUNTIME_ERROR). Point it at the unified pack.
# FGTEST_BUILD_ONLY=1 skips this step - tools/FGTest/run.sh uses it to compile the app and then open
# the window itself, so the two entry points never double-run the proof.
if [ "${FGTEST_BUILD_ONLY:-0}" != "1" ]; then
export FSR4_FGTEST_OUTDIR="$BUILD"
export FSR4_VK_ASSET_ROOT="$REPO/assets/pack-unified"
"$BUILD/FGTest" \
    "$REPO/build/linux/amd_fidelityfx_upscaler_vk.so" \
    "$REPO/build/linux/amd_fidelityfx_framegeneration_vk.so" \
    "$BUILD/triangle.vert.spv" "$BUILD/triangle.frag.spv"
fi

# To SEE it in a real window (Wayland/X11) instead of offscreen PPMs, run with --window [SECONDS]:
#   "$BUILD/FGTest" <upscaler.so> <framegeneration.so> <vert.spv> <frag.spv> --window=15
# It opens a 1440x1080 SDL window showing the LIVE moving triangle **upscaled by the REAL AMD FSR4 compute**:
# each frame is rendered offscreen at 480x270 by the app's own VkInstance, ffxDispatch(FFX_API_DISPATCH_DESC_TYPE_UPSCALE)
# records the AMD compute into a host command buffer (0.6 ms/frame on RX 9070 XT), the 960x540 FP16 result is copied
# back and decoded to RGBA8, then blitted by SDL2's SOFTWARE renderer (the WSI swapchain is never touched, so this
# stays crash-free on drivers with no working vkCreateSwapchainKHR). A top-right FPS readout (5x7 digit glyphs + bar)
# is baked into the image, and each second prints "t=N frames=.. fps=.." plus the average FSR dispatch cost (~25 ms
# per frame end to end, dominated by the CPU read-back, so ~21 fps). FG is driven per frame too with REAL images
# (depth / motion vectors / presentColor + backbuffer outputs); `generated` stays 0 because AMD's provider refuses
# ffxDispatch(generate) with RUNTIME_ERROR - "refusing to fake frame generation" - until the interpolation/ML/present
# runtime exists, and the documented cancel path (numGeneratedFrames == 0) returns OK. Nothing is faked.
# Add --present to attempt the real Vulkan-WSI FSR blit (1280x720 -> 1440x1080); it prints NO_PRESENT_MODES /
# NO_SWAPCHAIN and exits cleanly where the driver has no working swapchain.
if [ "${FGTEST_WINDOW:-0}" = "1" ]; then
    "$BUILD/FGTest" \
        "$REPO/build/linux/amd_fidelityfx_upscaler_vk.so" \
        "$REPO/build/linux/amd_fidelityfx_framegeneration_vk.so" \
        "$BUILD/triangle.vert.spv" "$BUILD/triangle.frag.spv" --window="${FGTEST_SECONDS:-15}"
fi
