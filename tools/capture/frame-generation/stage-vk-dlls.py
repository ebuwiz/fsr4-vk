#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 fsr4-vk contributors.
# See LICENSE in the project root.
"""Stage AMD's pinned *Vulkan* Frame Generation provider DLLs from the SDK into a runtime directory so a
reference capture can actually be RUN. The FG effect on Vulkan needs three AMD binaries, not one:
the Frame Generation provider, the Upscaler provider it composites through, and the loader that resolves the
effect chain. All three ship (MIT, signed) in the pinned v2.4.0 submodule under Kits/FidelityFX/signedbin.

This tool verifies each SOURCE DLL against its pinned SHA-256 before staging it, and refuses to stage anything
that is not the exact pinned identity, so a capture rig can never silently mix versions. It does NOT fetch amdvkffx64
.dll (the AMD Vulkan-FFX *driver* module, a separate OS/AMD component from the third-party mirror) - stage that by
hand and record its own hash."""
import argparse
import hashlib
import shutil
import sys
from pathlib import Path

SDK_SIGNEDBIN = "third_party/fidelityfx-sdk/Kits/FidelityFX/signedbin"
PINNED = {
    "amd_fidelityfx_framegeneration_vk.dll":
        "d002dabff8c5e71eaa715fc99438cd4dc81230262a7897b16bc402efec3498b4",
    "amd_fidelityfx_upscaler_vk.dll":
        "4a7e895af6e0b615d40a84d59eb8c2b1054ad91cdd0c739567940270d51bdab7",
    "amd_fidelityfx_loader_vk.dll":
        "38a56a01fb3565c0757d9f5c1e4e69de4b65e5f2e5769b323b200dc1f142bab9",
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-dir", default=SDK_SIGNEDBIN)
    ap.add_argument("--dest", default="dlls")
    ap.add_argument("--check", action="store_true", help="only verify source DLLs match the pinned hashes")
    ap.add_argument("--apply", action="store_true", help="copy missing/mismatched DLLs into --dest")
    args = ap.parse_args()

    src = Path(args.source_dir)
    if not src.is_dir():
        print("SKIP stage: SDK signedbin dir absent (run submodule init)"); return 0
    checked = staged = 0
    problems = []
    for name, want in PINNED.items():
        s = src / name
        if not s.is_file():
            problems.append(f"missing source {name}")
            continue
        got = sha256(s)
        if got != want:
            problems.append(f"SOURCE NOT PINNED {name}: {got} != {want}")
            continue
        checked += 1
        if args.apply:
            dst = Path(args.dest)
            dst.mkdir(parents=True, exist_ok=True)
            t = dst / name
            if not t.is_file() or sha256(t) != want:
                shutil.copy2(s, t)
            if sha256(t) != want:
                problems.append(f"staged copy mismatch {name}"); continue
            staged += 1
    print(f"checked={checked}/{len(PINNED)} staged={staged}")
    if problems:
        print("FAIL: " + "; ".join(problems)); return 1
    verb = "staged" if args.apply else "verified (source only; pass --apply to copy)"
    print(f"stage-vk-dlls: {verb} the 3 pinned Vulkan FG provider DLLs (loader+upscaler+framegeneration)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
