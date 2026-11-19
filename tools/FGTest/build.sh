#!/usr/bin/env bash
# tools/FGTest/build.sh - compile the animated-triangle FSR4 host sample + its SPIR-V, offscreen (headless).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
BUILD="$HERE/build"; mkdir -p "$BUILD"

# 1) shaders -> SPIR-V
glslc -o "$BUILD/triangle.vert.spv" "$HERE/triangle.vert"
glslc -o "$BUILD/triangle.frag.spv" "$HERE/triangle.frag"

# 2) the app
g++ -std=c++20 -Wall -o "$BUILD/FGTest" "$HERE/main.cpp" \
    -I"$REPO/third_party/fidelityfx-api/api/include" \
    -I"$REPO/third_party/fidelityfx-api/framegeneration/include" \
    -I"$REPO/third_party/fidelityfx-api/upscalers/include" \
    -I"$REPO/third_party/Vulkan-Headers/include" \
    -lvulkan -ldl
echo "built $BUILD/FGTest"

# 3) run headless against the built providers (pass their real paths); frames land in $BUILD
# The modular upscaler provider resolves its shader/model sidecars from FSR4_VK_ASSET_ROOT
# (otherwise create() throws "cannot locate sidecars" -> RUNTIME_ERROR). Point it at the unified pack.
export FSR4_FGTEST_OUTDIR="$BUILD"
export FSR4_VK_ASSET_ROOT="$REPO/assets/pack-unified"
"$BUILD/FGTest" \
    "$REPO/build/linux/amd_fidelityfx_upscaler_vk.so" \
    "$REPO/build/linux/amd_fidelityfx_framegeneration_vk.so" \
    "$BUILD/triangle.vert.spv" "$BUILD/triangle.frag.spv"
