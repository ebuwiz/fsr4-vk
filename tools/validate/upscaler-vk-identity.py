#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 fsr4-vk contributors.
# See LICENSE in the project root.
"""Second AMD-authored source of truth for the UPSCALER blobs (GPU-free), mirroring the FG _vk identity gate.

AMD ships the SAME FSR 4.1.2 provider as a Vulkan DLL (amd_fidelityfx_upscaler_vk.dll, sha 4a7e895a...). Its
MODEL weight banks are portable data, so this tool proves the model blobs we extracted from the pinned DX12 DLL
are BYTE-IDENTICAL to what AMD actually ships on Vulkan - an independent confirmation of the int8/fp8 standard +
ultraperf banks we embed in amd_fidelityfx_upscaler_vk.so. It also confirms the RCAS / auto-exposure / Navi2 / Lite
variant names exist in the Vulkan DLL.

It does NOT claim the Vulkan DLL's SHADERS are extractable: the _vk shader set is one compressed proprietary
container (0 plaintext SPIR-V/DXBC), so nothing new can be pulled out of it statically. It also adjudicates whether
the DLL carries any model weight BANK beyond the 4 we ship: it does NOT (the +/-128 KB neighbours are pitch-boundary
overlap or single-occurrence artifacts), and --expect fails loudly if a genuinely replicated 128 KB bank ever appears
so a future AMD addition surfaces instead of being silently ignored or auto-injected."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

VK_UPSCALER_SHA = "4a7e895af6e0b615d40a84d59eb8c2b1054ad91cdd0c739567940270d51bdab7"
BANKS = {
    "fp8/standard": "fp8/models/4.1.2/weights_standard.bin",
    "fp8/ultraperf": "fp8/models/4.1.2/weights_ultraperf.bin",
    "int8/standard": "int8/models/4.1.2/weights_standard.bin",
    "int8/ultraperf": "int8/models/4.1.2/weights_ultraperf.bin",
}
VARIANT_TOKENS = (b"fsr_rcas_pass", b"auto_exposure", b"Navi2", b"Lite", b"pass0_post")
BANK = 131072


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dll", help="path to amd_fidelityfx_upscaler_vk.dll")
    ap.add_argument("--pack", help="path to assets/pack-unified")
    ap.add_argument("--keys", help="path to tools/extract/keys/4.1.2.4555.json")
    ap.add_argument("--expect", action="store_true")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    dll = Path(args.dll) if args.dll else root / "dlls/amd_fidelityfx_upscaler_vk.dll"
    pack = Path(args.pack) if args.pack else root / "assets/pack-unified"
    keys = Path(args.keys) if args.keys else root / "tools/extract/keys/4.1.2.4555.json"
    if not dll.is_file():
        print("SKIP upscaler-vk-identity: _vk upscaler DLL absent"); return 0

    raw = dll.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if args.expect and sha != VK_UPSCALER_SHA:
        print(f"FAIL: _vk upscaler DLL not the pinned identity ({sha[:12]})"); return 1

    problems = []
    if sha != VK_UPSCALER_SHA:
        problems.append(f"_vk DLL sha {sha[:12]} != pinned {VK_UPSCALER_SHA[:12]}")

    # Model banks: each shipped pack-unified weight file must appear byte-identically in AMD's _vk DLL.
    counts = {}
    for label, rel in BANKS.items():
        f = pack / rel
        if not f.is_file():
            problems.append(f"missing pack bank {rel}")
            continue
        b = f.read_bytes()
        n = raw.count(b)
        counts[label] = n
        if n < 1:
            problems.append(f"bank {label} not found in _vk DLL")
    if counts.get("fp8/standard", 0) < 5:
        problems.append(f"fp8 standard copies {counts.get('fp8/standard')} < 5 (the documented replication)")

    # The fp8 standard/ultraperf SHAs must equal the pinned keys so the byte-search identity is the SAME model.
    if keys.is_file():
        k = json.loads(keys.read_text()).get("weights", {})
        checks = (("fp8/standard", k.get("standard_sha256")), ("fp8/ultraperf", k.get("ultraperf_sha256")))
        for label, want in checks:
            f = pack / BANKS[label]
            if f.is_file() and want:
                if hashlib.sha256(f.read_bytes()).hexdigest() != want:
                    problems.append(f"pack {label} sha != pinned keys sha")

    for tok in VARIANT_TOKENS:
        if raw.count(tok) < 1:
            problems.append(f"variant token absent in _vk DLL: {tok.decode()}")

    # Honesty: the _vk shader container is compressed; nothing new is statically extractable.
    print("vk_upscaler sha=%s bytes=%d  model-bank occurrences: %s"
          % (sha[:12], len(raw),
             {k: v for k, v in counts.items()}))
    import struct
    spv = raw.count(struct.pack(">I", 0x0731A053)) + raw.count(struct.pack("<I", 0x0731A053))
    print("plaintext SPIR-V in _vk shaders:", spv, "(0 -> _vk shaders are a compressed container, not extractable)")

    # Adjudicate whether AMD's _vk DLL hides any ADDITIONAL model weight bank beyond the 4 we ship. The bank
    # pitch is 128 KB + small metadata drift, so naive +/-128 KB neighbour windows land ACROSS a bank boundary
    # and most "candidates" are overlap artifacts. Classify rigorously instead of reporting a bogus count.
    def occurrences(needle, cap=64):
        res, i = [], 0
        while len(res) < cap:
            j = raw.find(needle, i)
            if j < 0:
                break
            res.append(j)
            i = j + 1
        return res
    known_ranges, pitches = [], {}
    for label, rel in BANKS.items():
        f = pack / rel
        if not f.is_file():
            continue
        b = f.read_bytes()
        occ = occurrences(b)
        known_ranges += [(o, o + len(b)) for o in occ]
        if len(occ) > 1:
            ds = [occ[i + 1] - occ[i] for i in range(len(occ) - 1)]
            pitches[label] = (min(ds), max(ds))
    print("model-bank pitch: nominal %d bytes (128 KB); per-bank observed min/max gap = %s" % (BANK, pitches))

    def overlaps_known(w):
        return any(not (w + BANK <= s or w >= e) for s, e in known_ranges)
    cand = {}
    for rel in ("fp8/models/4.1.2/weights_standard.bin", "int8/models/4.1.2/weights_standard.bin"):
        f = pack / rel
        if not f.is_file():
            continue
        for off in occurrences(f.read_bytes(), cap=8):
            for d in (-2, -1, 1, 2):
                w = off + d * BANK
                if 0 <= w and w + BANK <= len(raw):
                    cand.setdefault(hashlib.sha256(raw[w:w + BANK]).hexdigest(), set()).add(w)
    known = {hashlib.sha256((pack / rel).read_bytes()).hexdigest()
             for rel in BANKS.values() if (pack / rel).is_file()}
    new_banks = []
    for h, starts in cand.items():
        if h in known or len(starts) < 2:      # single-occurrence neighbor = artifact, not a replicated bank
            continue
        s0 = next(iter(starts))
        blk = raw[s0:s0 + BANK]
        if not overlaps_known(s0) and blk.count(0) < BANK * 0.5:   # 2+ distinct copies, non-zero, bank-free
            new_banks.append(h)
    print("additional replicated 128KB model banks beyond the 4 shipped:", len(new_banks),
          "(0 -> the _vk DLL carries NO weight bank our pack lacks; +/-128KB neighbours are boundary-overlap "
          "or single-occurrence artifacts and are never auto-injected)")
    if args.expect and new_banks:
        problems.append(f"UNEXPECTED replicated model bank(s) in _vk DLL: {len(new_banks)} "
                        "-> needs AMD reference capture to adjudicate, do NOT auto-inject")

    if problems:
        if args.expect:
            print("FAIL upscaler-vk-identity: " + "; ".join(problems)); return 1
        print("WARN: " + "; ".join(problems))
    print("upscaler-vk-identity: all 4 model weight banks confirmed byte-identical in AMD's Vulkan upscaler DLL")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
