#!/usr/bin/env bash
# tools/FGTest/run.sh - THE RUNNABLE MINI GRAPHIC APP OF THIS PHASE.
#
# Opens a 1440x1080 SDL window for 30 seconds showing a LIVE animated triangle that goes through the
# REAL pipeline: the app renders it offscreen at 480x270, amd_fidelityfx_upscaler_vk.so renders the
# upscale to 960x540 through ffxDispatch(FFX_API_DISPATCH_DESC_TYPE_UPSCALE), the FP16 result is read back
# and blitted to the window, and the Frame Generation provider is driven with the real images it owes
# (depth / motion vectors / presentColor + backbuffers). The HUD readout - resolutions top-left, clocks
# top-right, the frame-generation multiplier as GOT n X REQ n X - is painted into every frame by a 5x7
# bitmap font, so what you see is what the providers actually did.
#
#   bash tools/FGTest/run.sh                 # window, 30 s (Esc / q / closing the window ends it early)
#   bash tools/FGTest/run.sh --seconds=12    # shorter run
#   bash tools/FGTest/run.sh --headless      # no window at all: proof + HUD corners printed as ASCII
#   bash tools/FGTest/run.sh --fg=2x         # REQUEST 2 generated frames per base frame (provider still refuses)
#   bash tools/FGTest/run.sh --present       # attempt the real Vulkan-WSI swapchain path
#
# Exit codes: 0 ran and displayed; 77 no display / no suitable device (it then runs the SAME proof
# offscreen and prints the HUD as ASCII, but keeps 77 so a GPU-less or CI host is honestly reported);
# 2 the repo's provider .so files are missing (it prints the command that builds them).
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
BUILD="$HERE/build"; mkdir -p "$BUILD"
DURATION=30; MODE=window; EXTRA=()
for a in "$@"; do
    case "$a" in
        --seconds=*)  DURATION="${a#--seconds=}" ;;
        --headless)   MODE=headless ;;
        --build)      MODE=build-only ;;
        --hud-dump|--fg=*|--present) EXTRA+=("$a") ;;
        *) echo "run.sh: ignoring unknown argument '$a'" >&2 ;;
    esac
done
UP="$REPO/build/linux/amd_fidelityfx_upscaler_vk.so"
FG="$REPO/build/linux/amd_fidelityfx_framegeneration_vk.so"
if [ ! -f "$UP" ] || [ ! -f "$FG" ]; then
    echo "  missing provider builds: $UP and/or $FG"
    echo "  build them first:  bash tools/build/linux.sh"
    exit 2
fi
if [ ! -x "$BUILD/FGTest" ] || [ "$HERE/main.cpp" -nt "$BUILD/FGTest" ]; then
    echo "  compiling the app (build.sh, run step skipped)"
    FGTEST_BUILD_ONLY=1 bash "$HERE/build.sh" || exit 2
fi
export FSR4_FGTEST_OUTDIR="$BUILD"
export FSR4_VK_ASSET_ROOT="$REPO/assets/pack-unified"
SPV=("$BUILD/triangle.vert.spv" "$BUILD/triangle.frag.spv")
if [ "$MODE" = build-only ]; then
    echo "  built only: run it with  bash tools/FGTest/run.sh"
    exit 0
fi
if [ "$MODE" = window ]; then
    echo "  window mode: ${DURATION}s at 1440x1080 (Esc / q / close to end early)"
    "$BUILD/FGTest" "$UP" "$FG" "${SPV[@]}" "--window=$DURATION" "${EXTRA[@]}"
    rc=$?
    if [ "$rc" = 77 ]; then
        echo "  no window/device here: re-running the SAME proof offscreen with the HUD printed as ASCII"
        "$BUILD/FGTest" "$UP" "$FG" "${SPV[@]}" --hud-dump "${EXTRA[@]}"
        rc2=$?
        echo "  headless proof rc=$rc2 (PPM frames in $BUILD); exit code 77 kept: nothing was displayed"
        exit 77
    fi
    exit "$rc"
fi
"$BUILD/FGTest" "$UP" "$FG" "${SPV[@]}" --hud-dump "${EXTRA[@]}"
rc=$?
echo "  headless: PPM frames + ASCII HUD crops land under $FSR4_FGTEST_OUTDIR"
exit "$rc"
