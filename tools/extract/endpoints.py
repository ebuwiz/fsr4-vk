#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 fsr4-vk contributors.
# See LICENSE in the project root.

"""Extract the INT8 color endpoint containers selected by the model dispatcher.

Every model switch selects its prepass (pass 0) and postpass (pass 26) through a
branch table indexed by a key built from the upscaler pipeline state, which is
what `family.endpoint_key()` states; the other model passes are indexed by a
runtime field that never reaches the image.  The endpoint containers are
therefore enumerable from the switch itself:

* prepass - 64 keys: 4 color spaces x 8 motion-vector, inverted-depth and
  exposure flag combinations x jitter cancellation.  Keys that differ only in the
  inverted-depth bit while the render-motion-vector bit is clear (display motion
  vectors) select the same container, and the Ultra Performance branch selects
  exactly what the standard branch does - that is why the prepass asset name
  carries neither a preset nor a debug dimension.
* postpass - 16 keys: 4 color spaces x exposure x debug.  The index table is
  longer than the semantic key space, and every entry above key 15 repeats a
  container from below it. Both preset branches are written and named by
  `tools/convert/endpoints.py`.

The tree matches the layout that converter consumes::

    endpoints/sdk-<family>/<tier>/<color>/<flags>/<preset>/pass-<NN>[-debug].dxbc
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import dispatch
from family import decode_key, endpoint_key
from sdk import COLORS, SDK, TIERS

PHASES = (("prepass", 0, "_prepass"), ("postpass", 26, "_postpass"))
PRESETS = ("standard", "ultraperf")
POSTPASS_KEYS = 16


def phase_entry(spec, family, phase, suffix):
    """The one entry of a model switch that holds one endpoint phase."""
    found = sorted(entry for entry in spec["passes"] if entry.endswith(suffix))
    if len(found) != 1:
        raise ValueError(f"the {family} model switch reaches {len(found)} {phase} "
                         f"entries: {', '.join(found)}")
    return found[0]


def phase_keys(phase, family, tier, preset, records):
    """The semantic table indices of one branch, over-read entries verified away."""
    semantic = {key: row for key, row in records.items()
                if phase == "prepass" or key < POSTPASS_KEYS}
    wanted = 64 if phase == "prepass" else POSTPASS_KEYS
    missing = sorted(set(range(wanted)) - set(semantic))
    if missing:
        raise ValueError(f"{family} {phase} {tier}/{preset} reaches no table index "
                         f"{missing[0]}..{missing[-1]} ({len(missing)} of {wanted})")
    if phase != "prepass":
        known = {row["dxbc_sha256"] for row in semantic.values()}
        above = sorted(key for key, row in records.items()
                       if key >= POSTPASS_KEYS and row["dxbc_sha256"] not in known)
        if above:
            raise ValueError(f"{family} {phase} {tier}/{preset} selects containers for "
                             f"table indices {above} above the {POSTPASS_KEYS} indices "
                             "the endpoint key packing can express")
    return semantic


def branch_agreement(phase, family, tier, records):
    """Whether the two preset branches of one tier bind the same containers."""
    standard, ultraperf = records["standard"], records["ultraperf"]
    if phase == "prepass":
        for key, row in sorted(standard.items()):
            other = ultraperf.get(key)
            if other is None or other["dxbc_sha256"] != row["dxbc_sha256"]:
                raise ValueError(
                    f"{family} prepass {tier}: the Ultra Performance branch binds a "
                    f"different container for table index {key}, and the prepass asset "
                    "name carries no preset dimension to tell the two apart")
        for color in range(len(COLORS)):
            for flags in range(8):
                if flags & 2:
                    continue  # render motion vectors: inverted depth really is a variant
                for jitter in (0, 8):
                    left = standard[endpoint_key(0, flags | jitter, color)]
                    right = standard[endpoint_key(0, flags | jitter | 1, color)]
                    if left["dxbc_sha256"] != right["dxbc_sha256"]:
                        raise ValueError(
                            f"{family} prepass {tier}/{COLORS[color]}: display motion "
                            "vectors bind different containers for inverted depth, so "
                            "the endpoint key packing loses a bit")
    return ("shared" if all(ultraperf[key]["dxbc_sha256"] == standard[key]["dxbc_sha256"]
                            for key in standard) else "split")


def extract(dll, output):
    """Write every INT8 endpoint container under `output`."""
    sdk = SDK(dll)
    dispatchers = dispatch.load(sdk)
    output = Path(output)
    if output.exists():
        raise ValueError("Use a fresh extraction directory")
    output.mkdir(parents=True)
    files, branches = {}, {}
    for family in ("int8",):
        spec = dispatchers[family]
        for phase, pass_index, suffix in PHASES:
            entry = phase_entry(spec, family, phase, suffix)
            for tier in TIERS:
                tiers = spec["passes"][entry]["tiers"][tier]
                records = {preset: phase_keys(phase, family, tier, preset, tiers[preset])
                           for preset in PRESETS}
                branches[f"{family}/{phase}/{tier}"] = branch_agreement(
                    phase, family, tier, records)
                for preset in PRESETS:
                    for key, row in sorted(records[preset].items()):
                        color, flags, debug = decode_key(phase, pass_index, key)
                        name = f"pass-{pass_index:02d}" + ("-debug" if debug else "")
                        path = (output / f"sdk-{family}" / tier / COLORS[color]
                                / f"{flags:02x}" / preset / f"{name}.dxbc")
                        path.parent.mkdir(parents=True, exist_ok=True)

                        path.write_bytes(sdk.read(row["dxbc_rva"], row["size"]))
                        files[path.relative_to(output).as_posix()] = row["dxbc_sha256"]
    manifest = {
        "schema": "fsr4vk-endpoints-v1",
        "sdk_version": sdk.version,
        "extracted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": str(dll),
        "source_sha256": sdk.sha256,
        "endpoint_key_rules": {
            "prepass": "index = (flags & 7) | (color << 3) | ((flags & 8) << 2)",
            "postpass": "index = color | (flags & 4) | (debug << 3)",
            "flags": "bit0 inverted depth, bit1 render motion vectors, bit2 auto "
                     "exposure, bit3 motion vector jitter cancellation",
            "color": "index into sdk.COLORS: " + ", ".join(COLORS),
        },
        "branches": branches,
        "files": files,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dll", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        manifest = extract(args.dll, args.output)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(f"wrote {len(manifest['files'])} endpoint containers to {args.output}")
    for name, state in sorted(manifest["branches"].items()):
        print(f"  {name:<28} preset branches bind {state}")


if __name__ == "__main__":
    raise SystemExit(main())
