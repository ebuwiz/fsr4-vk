# SPDX-License-Identifier: GPL-3.0-only
"""Extend a verified INT8 pack with the genuine 4.1.2 FP8 matrix model."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

from build_int8 import endpoint_alias, COLORS, PRESETS, TIERS

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "extract"))
from fp8 import PASS_NAMES
from sdk import ASSET_VERSION


def build(base, model, endpoints, output):
    if output.exists():
        raise ValueError("Use a fresh unified pack directory")
    manifest = json.loads((base / "manifest.json").read_text())
    converted = json.loads((endpoints / "manifest.json").read_text())
    if manifest.get("sdk_version") != "4.1.2.4555" or converted.get("sdk_version") != "4.1.2.4555":
        raise ValueError("FP8 or INT8 assets are not from 4.1.2.4555")
    expected = set()
    for tier in TIERS:
        for color in COLORS:
            for flags in range(16):
                expected.add(endpoint_alias("fp8", tier, color, flags, 0, 0))
                for debug in (False, True):
                    expected.add(endpoint_alias("fp8", tier, color, flags, 0, 26, debug))
    if set(converted["aliases"]) != expected:
        raise ValueError("FP8 endpoint conversion is missing or has unexpected variants")

    weight_hashes = {}
    for preset in PRESETS:
        path = model / "1080" / f"weights_{preset}.bin"
        weight_hashes[preset] = hashlib.sha256(path.read_bytes()).hexdigest()
    if any(weight_hashes[preset] != weight_hashes["native"]
           for preset in PRESETS if preset != "ultraperf"):
        raise ValueError("An FP8 preset requires a separate weight bank")

    shutil.copytree(base, output)
    rows = manifest["payloads"]
    aliases = manifest["aliases"]
    by_hash = {row["sha256"]: row["id"] for row in rows}

    def add(name, source, physical=False):
        blob = source.read_bytes()
        digest = hashlib.sha256(blob).hexdigest()
        if digest not in by_hash:
            index = len(rows)
            by_hash[digest] = index
            destination = output / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(blob)
            rows.append(dict(id=index, file=name, kind="full", base=None, size=len(blob),
                             sha256=digest, stored_size=len(blob), stored_sha256=digest))
        index = by_hash[digest]
        if name in aliases and aliases[name] != index:
            raise ValueError("Conflicting FP8 alias: " + name)
        aliases[name] = index
        if physical and not (output / name).exists():
            (output / name).parent.mkdir(parents=True, exist_ok=True)
            (output / name).write_bytes(blob)

    for tier in TIERS:
        for pass_name in PASS_NAMES[1:-1]:
            source = model / tier / (pass_name + ".spv")
            canonical = f"fp8/shaders/{ASSET_VERSION}/stock/{tier}/{pass_name}.spv"
            add(canonical, source, physical=True)
            add(f"fp8/{tier}/{pass_name}.spv", source)
    for preset, original in (("standard", "native"), ("ultraperf", "ultraperf")):
        add(f"fp8/models/{ASSET_VERSION}/weights_{preset}.bin",
            model / "1080" / f"weights_{original}.bin", physical=True)
    add(f"fp8/shaders/{ASSET_VERSION}/stock/auto-exposure.spv",
        model / "auto-exposure.spv", physical=True)
    add("fp8/auto-exposure.spv", model / "auto-exposure.spv")
    for name, digest in converted["aliases"].items():
        if not re_digest(digest):
            raise ValueError("Invalid FP8 endpoint digest: " + name)
        add(name, endpoints / "converted" / (digest + ".spv"))

    manifest["source_payload_bytes"] = sum(row["size"] for row in rows)
    manifest["stored_payload_bytes"] = sum(row["stored_size"] for row in rows)
    manifest["fp8_matrix_model"] = True
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Built unified pack: {len(rows)} payloads, {len(aliases)} aliases")


def re_digest(value):
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "model", "endpoints", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    build(args.base, args.model, args.endpoints, args.output)


if __name__ == "__main__":
    main()
