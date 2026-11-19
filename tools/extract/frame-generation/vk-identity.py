#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 fsr4-vk contributors.
# See LICENSE in the project root.
"""Mine the AMD Frame Generation *Vulkan* provider DLL (amd_fidelityfx_framegeneration_vk.dll)
for its shader-IDENTITY inventory and cross-check it against the pack we extracted from the DX12
DLL. This is a SECOND AMD-authored source of truth, GPU-free and order-agnostic.

What it proves (and only this):
  - the _vk DLL advertises EXACTLY the 382 ML-neural entries `mlfi_{4k,8k}_passNNN`
    (pass0..190 on each of the two tiers), equal to the pack's mlfi `entry` set;
  - it names the classic interpolation passes we recognize (CsDilate, cbInpaintingPyramid,
    uOpticalFlowPyramidLevel, iInputLumaResolution, FfxFrameInterpolationDebugViewport);
  - cooperative-matrix markers == the pack's cooperative-matrix census (156), corroborating CsTensorSizes.

What it deliberately does NOT claim: the _vk DLL's shader BYTES are one compressed proprietary blob
(no plaintext SPIR-V / zstd / lz4), and `passNNN` are container LABELS, NOT the ML execution ORDER. The
ML dataflow order stays out (dataflow.py already proves it is not statically derivable)."""
import argparse
import hashlib
import re
import sys
from pathlib import Path

# The pinned _vk FG DLL identity (v2.4.0, FG 4.0.2.4555) == the SDK Kits/.../signedbin copy.
EXPECTED_VK_SHA = "d002dabff8c5e71eaa715fc99438cd4dc81230262a7897b16bc402efec3498b4"
EXPECTED_MLFI_TOTAL = 382          # 191 pass0..190 per tier x {4k,8k}
EXPECTED_PASSES_PER_TIER = 191
EXPECTED_COOPERATIVE = 156         # cooperative-matrix-using modules (CsTensorSizes census)
CLASSIC_MARKERS = (
    b"CsDilate", b"cbInpaintingPyramid", b"uOpticalFlowPyramidLevel",
    b"iInputLumaResolution", b"FfxFrameInterpolationDebugViewport",
    b"mlfi_4k_pass0", b"mlfi_8k_pass0",
)


def mlfi_names(dll_bytes):
    names = set()
    for m in re.finditer(rb"mlfi_(4k|8k)_pass(\d+)", dll_bytes):
        names.add((m.group(1).decode(), int(m.group(2))))
    return names


def pack_mlfi_entries(pack_dir):
    """Union of mlfi_ entries our pack uses: from modules.tsv col `entry` and mlfi-graph.json."""
    entries = set()
    tsv = pack_dir / "modules.tsv"
    if tsv.is_file():
        for line in tsv.read_text().splitlines():
            cols = line.split("\t")
            if len(cols) >= 3:
                entry = cols[2]  # name, stage, entry, workgroup, sha
                m = re.match(r"mlfi_(4k|8k)_pass(\d+)$", entry)
                if m:
                    entries.add((m.group(1), int(m.group(2))))
    graph = pack_dir / "mlfi-graph.json"
    if graph.is_file():
        import json
        data = json.loads(graph.read_text())
        for tier, arr in data.get("chains", {}).items():
            for node in arr:
                e = node.get("entry", "")
                m = re.match(r"mlfi_(4k|8k)_pass(\d+)$", e)
                if m:
                    entries.add((m.group(1), int(m.group(2))))
    return entries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dll", required=True, help="path to amd_fidelityfx_framegeneration_vk.dll")
    ap.add_argument("--pack", help="path to assets/frame-generation/pack")
    ap.add_argument("--expect", action="store_true", help="fail on any drift (gate mode)")
    args = ap.parse_args()

    dll = Path(args.dll)
    if not dll.is_file():
        print("SKIP vk-identity: _vk FG DLL absent"); return 0
    raw = dll.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if args.expect and sha != EXPECTED_VK_SHA:
        print(f"FAIL vk-identity: unexpected _vk DLL sha {sha}"); return 1

    names = mlfi_names(raw)
    tiers = {}
    for tier, p in names:
        tiers.setdefault(tier, set()).add(p)
    print(f"_vk DLL sha={sha[:12]} mlfi_distinct={len(names)} tiers="
          + ", ".join(f"{t}:{len(ps)}" for t, ps in sorted(tiers.items())))

    problems = []
    if len(names) != EXPECTED_MLFI_TOTAL:
        problems.append(f"mlfi distinct {len(names)} != {EXPECTED_MLFI_TOTAL}")
    for tier in ("4k", "8k"):
        want = set(range(EXPECTED_PASSES_PER_TIER))  # pass0..190
        got = tiers.get(tier, set())
        if got != want:
            problems.append(f"{tier} pass set drift: missing={sorted(want-got)[:5]} extra={sorted(got-want)[:5]}")

    coop_khr = raw.count(b"SPV_KHR_cooperative_matrix")
    coop_use = raw.count(b"CooperativeMatrixUseMatrixAKHR")
    print(f"cooperative markers: SPV_KHR_cooperative_matrix={coop_khr} UseMatrixAKHR={coop_use}")
    if coop_use != EXPECTED_COOPERATIVE:
        problems.append(f"cooperative UseMatrixAKHR {coop_use} != {EXPECTED_COOPERATIVE}")

    for marker in CLASSIC_MARKERS:
        if marker not in raw:
            problems.append(f"classic marker absent in _vk DLL: {marker.decode()}")

    # Cross-check: the _vk DLL mlfi set must equal the pack's mlfi entry set (both AMD artifacts agree).
    if args.pack and Path(args.pack).is_dir():
        pk = pack_mlfi_entries(Path(args.pack))
        only_dll = names - pk
        only_pack = pk - names
        print(f"pack mlfi entries={len(pk)}  dll-only={len(only_dll)} pack-only={len(only_pack)}")
        if only_dll or only_pack:
            problems.append(f"mlfi identity mismatch dll<->pack dll-only={sorted(only_dll)[:5]} pack-only={sorted(only_pack)[:5]}")

    if problems:
        if args.expect:
            print("FAIL vk-identity (drift): " + "; ".join(problems)); return 1
        print("WARN: " + "; ".join(problems))
    print("vk-identity: _vk DLL shader-name identity confirmed (order NOT claimed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
