#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 fsr4-vk contributors.

"""Build a stock-only FSR 4.1.2 INT8 asset pack from validated conversions."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "extract"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "convert"))
from endpoints import endpoint_alias
from sdk import ASSET_VERSION, COLORS, PRESETS, TIERS

VERSION = ASSET_VERSION


def validate_weights(directory, weights):
    if set(weights) != set(PRESETS):
        raise ValueError("INT8 weights must include all six SDK presets")
    for preset in PRESETS:
        info = weights[preset]
        source = directory / info["file"]
        if hashlib.sha256(source.read_bytes()).hexdigest() != info["sha256"]:
            raise ValueError("INT8 weights changed: " + str(source))
        if preset != "ultraperf" and info["sha256"] != weights["native"]["sha256"]:
            raise ValueError("INT8 preset requires its own weights: " + preset)


def validate_endpoints(aliases):
    expected = set()
    for tier in TIERS:
        for color in COLORS:
            for flags in range(16):
                for preset in (0, 5):
                    expected.add(endpoint_alias("int8", tier, color, flags, preset, 0))
                    for debug in (False, True):
                        expected.add(endpoint_alias("int8", tier, color, flags, preset, 26, debug))
    missing = expected - aliases.keys()
    unexpected = aliases.keys() - expected
    if missing or unexpected:
        raise ValueError(
            f"INT8 endpoint aliases differ from the SDK matrix: "
            f"{len(missing)} missing, {len(unexpected)} unexpected"
            + (f"; first missing: {min(missing)}" if missing else "")
            + (f"; first unexpected: {min(unexpected)}" if unexpected else "")
        )


def build(int8, endpoints, rcas, output):
    if output.exists():
        raise ValueError("Use a fresh asset pack directory: " + str(output))
    conversion = json.loads((int8 / "manifest.json").read_text())
    endpoint_manifest = json.loads((endpoints / "manifest.json").read_text())
    if conversion["selection_count"] != len(conversion["shaders"]):
        raise ValueError("Incomplete INT8 conversion manifest")
    if endpoint_manifest.get("sdk_version") != "4.1.2.4555":
        raise ValueError("Unsupported endpoint conversion")
    validate_weights(int8, conversion["weights"])
    validate_endpoints(endpoint_manifest["aliases"])

    sources = {}
    for row in conversion["shaders"]:
        path = int8 / row["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["spirv_sha256"]:
            raise ValueError("Converted shader changed: " + str(path))
        sources[row["file"]] = path

    names = {}

    def add(name, source):
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("Invalid asset alias: " + name)
        source = Path(source)
        if not source.is_file():
            raise ValueError("Missing asset: " + str(source))
        if name in names and names[name].read_bytes() != source.read_bytes():
            raise ValueError("Conflicting asset alias: " + name)
        names[name] = source

    for tier in TIERS:
        for stage in range(27):
            standard = f"stock/{tier}/standard/pass-{stage:02}.spv"
            ultra = f"stock/{tier}/ultraperf/pass-{stage:02}.spv"
            if standard not in sources:
                raise ValueError("Missing INT8 pass: " + standard)
            for preset, source in (("native", sources[standard]),
                                   ("ultraperf", sources.get(ultra, sources[standard]))):
                add(f"int8/shaders/{VERSION}/stock/{tier}/{preset}/pass-{stage:02}.spv", source)
                add(f"int8/{tier}/{preset}/pass-{stage:02}.spv", source)

    exposure = int8 / "auto-exposure.spv"
    if hashlib.sha256(exposure.read_bytes()).hexdigest() != conversion["auto_exposure"]["spirv_sha256"]:
        raise ValueError("Converted auto-exposure shader changed")
    add(f"int8/shaders/{VERSION}/stock/auto-exposure.spv", exposure)
    add("int8/auto-exposure.spv", exposure)
    for preset, source_name in (("standard", "native"), ("ultraperf", "ultraperf")):
        info = conversion["weights"][source_name]
        source = int8 / info["file"]
        add(f"int8/models/{VERSION}/weights_{preset}.bin", source)
    for name, digest in endpoint_manifest["aliases"].items():
        if not name.startswith(f"int8/shaders/{VERSION}/color-endpoints/"):
            raise ValueError("Unexpected endpoint alias: " + name)
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("Invalid endpoint digest: " + name)
        add(name, endpoints / "converted" / (digest + ".spv"))
    for name in ("linear", "linear-auto", "nonlinear", "srgb", "pq"):
        add(f"common/shaders/{VERSION}/rcas/{name}.spv", rcas / (name + ".spv"))

    output.mkdir(parents=True)
    rows, aliases, by_hash = [], {}, {}
    # Stock names must be the canonical payload names when a shader is shared.
    ordered = sorted(names, key=lambda name: ("/stock/" not in name, name))
    for name in ordered:
        blob = names[name].read_bytes()
        digest = hashlib.sha256(blob).hexdigest()
        if digest not in by_hash:
            index = len(rows)
            by_hash[digest] = index
            destination = output / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(blob)
            rows.append(dict(id=index, file=name, kind="full", base=None, size=len(blob),
                             sha256=digest, stored_size=len(blob), stored_sha256=digest))
        aliases[name] = by_hash[digest]
        if "/stock/" in name or name.startswith(f"int8/models/{VERSION}/"):
            destination = output / name
            if not destination.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(names[name], destination)
    manifest = dict(schema=1, version=VERSION, sdk_version="4.1.2.4555",
                    encoding="uncompressed-copy-insert-v1", payloads=rows,
                    aliases=aliases, source_payload_bytes=sum(r["size"] for r in rows),
                    stored_payload_bytes=sum(r["stored_size"] for r in rows),
                    delta_payloads=0, max_delta_depth=1,
                    shader_storage_policy="stock-full-patched-direct-v1",
                    endpoint_selection_layout="stage-specific-v1",
                    omitted_shader_tiers=[], windows_assets_omitted=True)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Built INT8 asset pack: {len(rows)} payloads, {len(aliases)} aliases")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--int8", required=True, type=Path)
    parser.add_argument("--endpoints", required=True, type=Path)
    parser.add_argument("--rcas", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        build(args.int8, args.endpoints, args.rcas, args.output)
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
