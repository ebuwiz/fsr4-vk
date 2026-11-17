#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 fsr4-vk contributors.
# See LICENSE in the project root.

"""Convert the 4.1.2 prepass and postpass color endpoint variants.

Each converted shader is validated, and the endpoint tree it consumes has to
come from the pinned 4.1.2 image (tools/extract/extract.py).
"""

import sys
from pathlib import Path
import argparse
import concurrent.futures
import hashlib
import json
import subprocess

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "extract"))
from sdk import ASSET_VERSION, SDK_HASH, SDK_VERSION


def endpoint_alias(family, tier, color, flags, preset, pass_index, debug=False,
                   split_postpass=False):
    """Map SDK permutations to prepass and postpass asset names.

    The postpass name carries a preset directory whenever the SDK dispatch binds
    different postpass containers for the Ultra Performance branch than for the
    other presets. The extracted FP8 matrix model uses shared postpasses.
    """
    family = family.removeprefix("sdk-")
    if family not in ("fp8", "int8") or pass_index not in (0, 26):
        raise ValueError("Unknown endpoint family/pass")
    root = Path(family) / f"shaders/{ASSET_VERSION}/color-endpoints" / str(tier)
    exposure = "auto" if flags & 4 else "external"
    if pass_index == 0:
        motion = (
            "display-mv"
            if not flags & 2
            else "render-mv-inverted-depth" if flags & 1 else "render-mv-normal-depth"
        )
        jitter = "jitter.spv" if flags & 8 else "no-jitter.spv"
        return (root / "prepass" / color / motion / exposure / jitter).as_posix()
    root = root / "postpass" / color
    if family == "int8" or split_postpass:
        root /= "ultraperf" if preset == 5 else "standard"
    return (root / (exposure + ("-debug.spv" if debug else ".spv"))).as_posix()


def split_postpass(extracted, files):
    """(family, tier) pairs whose postpass branches bind different containers."""
    branches = {}
    for path in files:
        family, tier, color, flags, preset, filename = path.relative_to(extracted).parts
        if not filename.startswith("pass-26"):
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        state = branches.setdefault((family, tier), {})
        state.setdefault((color, flags, filename), set()).add(digest)
    return {pair for pair, state in branches.items()
            if any(len(digests) > 1 for digests in state.values())}




def main():
    from toolchain import find_tool

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("sdk", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument(
        "translator",
        type=Path,
        nargs="?",
        help="dxil-spirv executable (defaults to the bundled submodule)",
    )
    parser.add_argument("--spirv-val", help="SPIR-V validator (auto-detected by default)")
    parser.add_argument("--family", choices=("fp8", "int8"), help="convert one model family")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--debug-postpass", action="store_const", const=True, default=None,
        help="convert only debug postpasses (default: normal and debug)",
    )
    selection.add_argument(
        "--no-debug-postpass", dest="debug_postpass", action="store_const",
        const=False, help="convert only normal endpoints",
    )
    args = parser.parse_args()
    if args.sdk.is_file() and hashlib.sha256(args.sdk.read_bytes()).hexdigest() != SDK_HASH:
        parser.error("Unsupported reference DLL: " + str(args.sdk))
    if args.output.exists():
        parser.error("Use a fresh output directory")
    try:
        args.translator = find_tool("dxil-spirv", args.translator, "DXIL_SPIRV")
        validator = find_tool("spirv-val", args.spirv_val)
    except ValueError as error:
        parser.error(str(error))
    args.output.mkdir(parents=True)
    extracted = args.sdk
    if not extracted.is_dir():
        parser.error(
            f"{SDK_VERSION} color endpoint variants have to be extracted first: "
            "pass the directory written by tools/extract/extract.py, not the DLL"
        )
    files = sorted((extracted / ("sdk-" + args.family)).rglob("*.dxbc")
                   if args.family else extracted.rglob("*.dxbc"))
    if not files:
        parser.error("No extracted endpoint shaders")
    split = split_postpass(extracted, files)
    aliases, unique = {}, {}
    for path in files:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        relative = path.relative_to(extracted)
        family, tier, color, flags, preset, filename = relative.parts
        key = endpoint_alias(
            family,
            tier,
            color,
            int(flags, 16),
            5 if preset == "ultraperf" else 1,
            int(Path(filename).stem.split("-")[1]),
            args.debug_postpass is True or filename.endswith("-debug.dxbc"),
            (family, tier) in split,
        )

        if key in aliases and aliases[key] != digest:
            raise ValueError("Endpoint settings cannot be collapsed: " + key)
        aliases[key] = digest
        unique.setdefault(digest, path)
    converted = args.output / "converted"
    converted.mkdir()

    def convert(item):
        digest, path = item
        output = converted / (digest + ".spv")
        flags = (
            ["--full-wmma", "1", "0"]
            if "sdk-fp8" in path.parts
            else ["--enable-shader-i8-dot", "--assume-ssbo-32bit-wrapping"]
        )
        subprocess.run(
            [
                str(args.translator),
                str(path),
                "--output",
                str(output),
                "--bindless",
                "--ssbo-uav",
                "--ssbo-srv",
                *flags,
            ],
            check=True,
        )
        validation = subprocess.run(
            [validator, "--target-env", "vulkan1.3", str(output)],
            capture_output=True,
            text=True,
        )
        if validation.returncode:
            raise ValueError(str(path) + ": " + validation.stderr)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(convert, unique.items()))

    # The converted files are named by their source DXBC digest, so the catalog
    # can only be validated by name when this sidecar records what each name is.
    (args.output / "manifest.json").write_text(json.dumps({
        "schema": 1,
        "sdk_version": SDK_VERSION,
        "asset_version": ASSET_VERSION,
        "converted": len(unique),
        "aliases": aliases,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
