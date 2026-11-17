# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 fsr4-vk contributors.
# See LICENSE in the project root.

"""Extract the quantized int8 model of the pinned SDK build.

The image carries an indexed INT8 pass switch next to a reference switch. Both
select the same ``fsr4_model_*`` entry names, so the only honest way to tell
them apart is the content they select: ``dispatch.families`` decodes both
switches and labels the one whose containers do not carry the program version of
the pinned RCAS and auto-exposure shaders ``int8``, and refuses the pair unless
they reach one pass set and never share a container.  The container written here
for a (pass, tier, preset branch) is therefore the record this same image binds
for the quantized model, never an address taken from another build.

The INT8 weight banks come out of the same copy pattern the FP8 banks use, and
the digests that name them come from the reference catalog's int8 model rows:
this build carries those payloads byte for byte, which ``--write`` verifies
before pinning them in ``keys/<version>.json``::

    tools/extract/int8.py --dll dlls/<version>/<name>.dll --write
    tools/extract/int8.py --dll dlls/<version>/<name>.dll --output assets/extracted/int8

Only the stock variant (endpoint key 0) of every (pass, tier, preset branch) is
written, because 4.1.2 folds the quality presets into the six weight banks: one
container per pass serves the whole preset matrix, and an Ultra Performance
container is written only where that branch selects code of its own.  The
endpoint and permutation copies the prepass and postpass tables reach beyond the
stock record are left to the endpoint stage: a prepass branch reaches all 64
endpoint keys while a postpass branch stops part way, so naming the rest needs
the table length rule that ``dispatch`` does not pin yet.
"""

import argparse
import hashlib
import json
from pathlib import Path

import dispatch
import fp8
from sdk import KEYS_DIR, SDK, TIERS, WEIGHT_BANK

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "tools/assets/reference/manifest.json"
STOCK = dispatch.STOCK_KEY
BRANCHES = ("standard", "ultraperf")
MODEL = fp8.MODEL_PREFIX


def stage_number(entry):
    """The pass index of one dispatch entry, from the fp8 pass names.

    The quantized switch reuses the fp8 entry names, and the fp8 names are the
    order the fp8 model map is pinned in, so one name table numbers both
    families and the ``pass-NN`` files mean the same pass either way.
    """
    name = entry.removeprefix(MODEL)
    if name not in fp8.PASS_NAMES:
        raise ValueError(f"{entry} is not one of the {len(fp8.PASS_NAMES)} model passes")
    return fp8.PASS_NAMES.index(name)


def reference_weights(family="int8"):
    """The weight payload digests the reference catalog names for one family."""
    rows = json.loads(REFERENCE.read_text())["payloads"]
    found = {}
    for row in rows:
        parts = Path(row["file"]).parts
        if len(parts) == 4 and parts[0] == family and parts[1] == "models" \
                and parts[3].startswith("weights_") and parts[3].endswith(".bin"):
            found[parts[3].removeprefix("weights_").removesuffix(".bin")] = row["sha256"]
    if set(found) != {"standard", "ultraperf"}:
        raise ValueError(f"{REFERENCE} names {sorted(found)} {family} model weights, "
                         "expected standard and ultraperf")
    return {f"{kind}_sha256": digest for kind, digest in found.items()}


def pin_weights(sdk, path):
    """Pin the int8 model weight payloads this image carries.

    The payloads are named by the reference catalog's int8 model rows, which the
    4.1.1 build established; ``sdk.weight_banks`` only accepts them if this image
    really holds them, so a build that retrained the int8 model stops here
    instead of writing 4.1.1 bytes under a 4.1.2 version.
    """
    digests = reference_weights()
    banks = sdk.weight_banks("int8", overrides=digests)
    pins = dict(digests, rule=(
        "the payload of the reference catalog's int8 model rows, carried byte for "
        "byte by this image: five identical 128 KiB copies for the presets that "
        "share one model and one separate copy for Ultra Performance, found by the "
        "same bank-spacing copy pattern the fp8 weights use and never by an address"
    ))
    keys = json.loads(path.read_text())
    keys["int8_weights"] = pins
    path.write_text(json.dumps(keys, indent=2) + "\n")
    return pins, banks



def stock_records(sdk, spec):
    """[(pass, entry, tier, branch, record)] the quantized switch selects."""
    rows = []
    for entry, detail in spec["passes"].items():
        stage = stage_number(entry)
        for tier in TIERS:
            branches = detail["tiers"].get(tier)
            if branches is None:
                raise ValueError(f"the int8 switch has no {tier} tier for {entry}")
            for branch in BRANCHES:
                record = branches.get(branch, {}).get(STOCK)
                if record is None:
                    raise ValueError(
                        f"the int8 switch selects no endpoint key {STOCK} record "
                        f"for {tier}/{branch} of {entry}")
                rows.append((stage, entry, tier, branch, record))
    return sorted(rows, key=lambda row: (row[0], row[2], row[3]))


def write_weights(sdk, output, manifest):
    directory = output / "weights"
    directory.mkdir()
    for preset, offset in sorted(sdk.weight_banks("int8").items(), key=lambda item: item[1]):
        data = sdk.data[offset : offset + WEIGHT_BANK]
        path = directory / f"weights_{preset}.bin"
        path.write_bytes(data)
        manifest["weights"][preset] = {
            "file": path.relative_to(output).as_posix(), "rva": sdk.image().rva(offset),
            "size": len(data), "sha256": hashlib.sha256(data).hexdigest(),
        }


def extract(dll, output):
    sdk = SDK(dll)
    output = Path(output)
    if output.exists():
        raise ValueError("use a fresh extraction output directory")
    spec = dispatch.load(sdk)["int8"]
    output.mkdir(parents=True)
    manifest = {"sdk_version": sdk.version, "source_sha256": sdk.sha256, "model": "int8",
                "jump_table": spec["jump_table"], "cases": spec["cases"],
                "program_version": spec["program_version"], "weights": {}, "passes": {}}
    write_weights(sdk, output, manifest)
    written = {}
    for stage, entry, tier, branch, record in stock_records(sdk, spec):
        blob = sdk.pinned_record(record["record_rva"], record["dxbc_sha256"],
                                 record["size"], entry)
        if branch == "ultraperf" and written.get((stage, tier)) == blob["sha256"]:
            continue  # that branch binds the very same code as the standard one
        path = output / "stock" / tier / branch / f"pass-{stage:02d}.dxbc"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(blob["data"])
        written[(stage, tier)] = blob["sha256"]
        row = manifest["passes"].setdefault(
            f"pass-{stage:02d}", {"entry": entry, "case": spec["passes"][entry]["case"],
                                  "files": {}})
        row["files"][f"{tier}/{branch}"] = {
            "file": path.relative_to(output).as_posix(), "record_rva": blob["record_rva"],
            "rva": blob["rva"], "size": blob["size"], "sha256": blob["sha256"],
        }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"int8: {len(written)} stock passes, {len(manifest['weights'])} weight banks "
          f"-> {output}")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dll", type=Path, required=True)
    parser.add_argument("--output", type=Path,
                        help="write the quantized model tree under this directory")
    parser.add_argument("--write", action="store_true",
                        help="pin the int8 weight bank payloads in keys/<version>.json")
    args = parser.parse_args(argv)
    try:
        sdk = SDK(args.dll)
        if args.write:
            pins, banks = pin_weights(sdk, KEYS_DIR / f"{sdk.version}.json")
            for kind in ("standard", "ultraperf"):
                print(f"{kind}_sha256 = {pins[f'{kind}_sha256']}")
            for preset, offset in sorted(banks.items(), key=lambda item: item[1]):
                print(f"  {preset:<11} rva {sdk.image().rva(offset):#x}")
        if args.output:
            extract(args.dll, args.output)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
