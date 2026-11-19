#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 fsr4-vk contributors.
# See LICENSE in the project root.

set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cmake -S "$repo_root" -B "$repo_root/build/linux" -DCMAKE_BUILD_TYPE=Release -DFSR4_EMBEDDED_ASSETS=ON \
    -DFSR4_ASSET_ROOT="$repo_root/assets/pack-unified" "$@"
build_dir="$repo_root/build/linux"
jobs="${FSR4_BUILD_JOBS:-4}"
# Build the asset-free Frame Generation provider first and by explicit name: the
# FG .so has no dependency on the upscaler's embedded-asset step, so pinning it
# as a target guarantees it is produced even if a later (asset-heavy) target fails
# or a generator's default "all" ordering leaves it unbuilt.
cmake --build "$build_dir" --parallel "$jobs" --target amd_fidelityfx_framegeneration_vk
# Then the remaining targets (upscaler provider, probes, tools).
cmake --build "$build_dir" --parallel "$jobs"
# Both independent provider deliverables must exist; fail loudly by name if not.
for provider_lib in amd_fidelityfx_framegeneration_vk amd_fidelityfx_upscaler_vk; do
    if [[ ! -f "$build_dir/$provider_lib.so" ]]; then
        echo "error: $provider_lib.so was not emitted (looked in $build_dir)" >&2
        exit 1
    fi
    printf 'emitted %s/%s.so\n' "$build_dir" "$provider_lib"
done
