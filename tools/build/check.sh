#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 fsr4-vk contributors.
# See LICENSE in the project root.

# Reproducible post-build validation gate. Every check asserts exit 0 and `set -e`
# stops the gate on the first failure, so a green run is a single pass/fail verdict
# a host or CI can depend on.
#
# Two validators are deliberately NOT in this Linux gate because they need an artefact
# this host cannot produce: tools/validate/windows-binary.py wants the Windows PE
# build (needs a MinGW-w64 cross build or a Windows host) and tools/validate/ab-frames
# .py wants two live frame captures. Both are documented in README as gated on the
# Windows/capture path, and run by CI there.

set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
build_dir="$repo_root/build/linux"
pack="$repo_root/assets/pack-unified"
py="${PYTHON:-python3}"

echo "[check] SPIR-V shader validation (reconstructed modules must all validate)"
"$py" "$repo_root/tools/validate/shaders.py" "$pack"

# Binary-level proof, not manifest-level: locate the `fsr4_embedded_assets_begin`
# symbol inside the built provider .so, decompress its embedded bytes with the SAME
# per-payload Zstandard index the build used, and hash-check every payload against
# the pack manifest (asserts complete=true, windows_payloads=0, omitted_shader_tiers=[],
# read_only + private_data_symbol). The --storage-index argument is REQUIRED: without
# it the tool re-derives uncompressed sizes and reports a spurious "packed payload
# mismatch", which is why a bare invocation looks red even on a healthy build.
if [[ -f "$build_dir/generated/embedded-storage.json" && \
      -f "$build_dir/amd_fidelityfx_upscaler_vk.so" ]]; then
    echo "[check] embedded ELF asset verification (upscaler provider .so vs pack)"
    "$py" "$repo_root/tools/validate/elf-assets.py" \
        "$build_dir/amd_fidelityfx_upscaler_vk.so" "$pack" \
        "$build_dir/generated/elf-asset-report.json" \
        --storage-index "$build_dir/generated/embedded-storage.json"
else
    echo "[check] skipping embedded ELF asset verification (embedded assets not built)" >&2
fi

echo "[check] Frame Generation ABI self-tests (entry points + honest refusal)"
"$py" "$repo_root/tools/validate/frame-generation/tests.py"

# The FSR 4.1.2 debug-checker predicates (src/provider/upscaler-checks.hpp) have a
# GPU-free unit test that already builds as a target; running it here means the
# warning decision logic cannot silently drift without failing the Linux gate.
if [[ -x "$build_dir/upscaler-debugcheck-test" ]]; then
    echo "[check] FSR 4.1.2 debug-checker predicates (GPU-free)"
    "$build_dir/upscaler-debugcheck-test"
fi

# Frame Generation mandatory-order + camera + create/generate predicates
# (src/framegeneration/framegeneration-checks.hpp): GPU-free, no algorithm.
if [[ -x "$build_dir/framegeneration-check-test" ]]; then
    echo "[check] Frame Generation order/camera predicates (GPU-free)"
    "$build_dir/framegeneration-check-test"
fi

# Frame Generation pass-plan substrate (roadmap Phase 0). Regenerate it from the pack with
# tools/extract/frame-generation/plan.py, then assert the independent C++ re-parse agrees with
# plan.py (474 compute modules, 1208 de-duplicated binding groups, a valid order scaffold). GPU-free.
if [[ -f "$repo_root/assets/frame-generation/pack/pipelines.tsv" ]]; then
    echo "[check] FG pass-plan reconstruction (plan.py)"
    "$py" "$repo_root/tools/extract/frame-generation/plan.py" \
        --pack "$repo_root/assets/frame-generation/pack"
    if [[ -x "$build_dir/framegeneration-plan-test" ]]; then
        echo "[check] FG pass-plan consistency (GPU-free)"
        "$build_dir/framegeneration-plan-test"
    fi
    # Cooperative-dataflow BOUNDARY probe: asserts the measured reality that the 156 ML cooperative
    # passes declare weights (set 1 SSBO) + activation (set 4 SSBO) + CBV (set 5 Uniform) at binding 0,
    # but each OpCooperativeMatrixLoad/StoreKHR pointer is a BINDLESS/dynamic buffer address, so NO
    # pass->pass dataflow edge (hence the ML execution ORDER) is statically derivable from the shipped
    # SPIR-V. Pins edges=0 as a drift gate: if a future pack/toolchain made order derivable this fails
    # loudly, and ffxDispatch stays honestly refused until then. GPU-free (uses spirv-dis).
    if [[ -f "$repo_root/assets/frame-generation/pack/mlfi-graph.json" ]]; then
        echo "[check] FG cooperative-dataflow boundary probe (GPU-free)"
        "$py" "$repo_root/tools/extract/frame-generation/dataflow.py" \
            --pack "$repo_root/assets/frame-generation/pack"
    fi
fi

# Deterministic FSR query helper math (quality->ratio, render resolution, jitter
# phase-count + R2 offset) in src/provider/upscaler-queries.hpp: GPU-free.
if [[ -x "$build_dir/upscaler-query-test" ]]; then
    echo "[check] FSR upscaler query helper math (GPU-free)"
    "$build_dir/upscaler-query-test"
fi

# Also run the compiled CTest regression set (upscaler-debugcheck, cap-probe ladder
# gates, descriptor-smoke byte-identity, frame-generation-validators, and the
# framegeneration-compile-parity-474 census + the FG pass-plan gates (plan-consistency
# GPU-free, plan-drift on-device)). GPU-dependent entries auto-skip on exit code 77, and
# code 77, and the FG-parity entry is only registered when the generated FG pack is
# present, so ctest stays green without a suitable device or a converted pack. This
# makes linux.sh / `--target check` a single canonical command over both the Python
# validators above and the compiled tests.
if [[ -f "$build_dir/CTestTestfile.cmake" ]]; then
    echo "[check] ctest compiled regressions"
    ctest --test-dir "$build_dir" --output-on-failure
fi

echo "[check] all Linux gates passed"
