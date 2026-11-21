#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
# tools/capture/frame-generation/run-capture.sh - run AMD's OWN Frame Generation provider under Wine, recorded
# by this repo's capture shim, which is the sanctioned source of the classic chain's vkCmdDispatch ORDER (and
# of each dispatched module's identity, by SPIR-V digest). Steps: stage the three pinned DLLs -> build the shim
# (LINUX .so, LD_PRELOADED into the wine process) -> build the WINDOWS host with winegcc -> run it under wine
# -> print what the capture holds and join it against the pack with capture-diff.py.
# Everything self-skips (exit 77) where wine / winegcc are absent, so this never fakes a capture.
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../../.." && pwd)"
run="$here/build"
dlls="$repo/dlls"
capdir="$run/capture"

command -v wine >/dev/null 2>&1 || { echo "SKIP run-capture: no wine"; exit 77; }
command -v winegcc >/dev/null 2>&1 || { echo "SKIP run-capture: no winegcc"; exit 77; }

echo '=== 1/4 stage the pinned AMD DLLs (hash-checked) ==='
python3 "$here/stage-vk-dlls.py" --dest "$dlls" --apply || { echo 'stage failed'; exit 77; }
ls -la "$dlls" | sed -n '1,6p'

echo '=== 2/4 build the LINUX capture shim (LD_PRELOAD target) ==='
mkdir -p "$run" "$capdir"
g++ -std=c++20 -O2 -shared -fPIC -I "$repo/third_party/Vulkan-Headers/include" \
    -I "$here" "$here/capture-shim.cpp" -o "$run/libffxvk-capture.so" \
    || { echo 'shim build failed'; exit 77; }
echo "built $run/libffxvk-capture.so"

echo '=== 3/4 build the WINDOWS host with winegcc ==='
bash "$here/build-host.sh" "$run" || { echo 'host build failed (see errors above)'; exit 77; }

echo '=== 4/4 run AMD's provider under wine, shim preloaded ==='
# The DLL must be reachable by the WINDOWS image: pass the path Wine sees, which for a staged file in the repo
# is the same relative location (Wine maps Z: to the filesystem root).
fg_dll_windows="$(winepath -w "$dlls/amd_fidelityfx_framegeneration_vk.dll" 2>/dev/null || echo '')"
echo "  wine path for the provider dll: ${fg_dll_windows:-<winepath failed>}"
FSR4_VK_CAPTURE_DIR="$capdir" LD_PRELOAD="$run/libffxvk-capture.so" \
    WINEDEBUG=-all wine "$run/fgcap-host.exe" ${fg_dll_windows:+"$fg_dll_windows"} > "$run/host-out.txt" 2>&1
    host_rc=$?
    sed -n '1,60p' "$run/host-out.txt"
    echo "  host rc=$host_rc"

echo '=== capture contents ==='
if [[ -f "$capdir/capture.tsv" ]]; then
    echo "  dispatch rows: $(($(grep -c '' "$capdir/capture.tsv")))  spv dumps: $(ls "$capdir"/mod-*.spv 2>/dev/null | grep -c '')"
    sed -n '1,12p' "$capdir/capture.tsv"
    echo '=== join against the pinned pack (identity by SPIR-V digest) ==='
    python3 "$here/capture-diff.py" --capture "$capdir/capture.tsv" 2>&1 | sed -n '1,24p'
else
    echo '  NO capture.tsv: AMD'\''s provider made no recorded Vulkan call before it stopped (see host output above)'
fi
