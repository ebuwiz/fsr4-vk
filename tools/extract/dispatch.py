#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 fsr4-vk contributors.
# See LICENSE in the project root.

"""Decode and pin the indexed dispatch tables used to identify INT8.

The historical ``fp8`` key denotes the reference indexed dispatcher, not the
FP8 matrix model. The genuine FP8 dispatcher is decoded separately by fp8.py.
The reference and INT8 dispatchers share entry names but have disjoint records
and distinct DXIL program versions.

Key 0 of a branch is the shipping stock variant; the other keys are the endpoint
permutations the dispatch reaches through the same index table: the low nibble
is the motion vector, depth, jitter and exposure flag set, the nibble above it
indexes the ``sdk.COLORS`` color space.  Run with ``--write`` after pointing the
repository at a new SDK build::

    tools/extract/dispatch.py --dll dlls/<version>/<name>.dll --write
"""

import argparse
import hashlib
import json
from pathlib import Path

from sdk import (CASE_BRANCHES, COLORS, ENDPOINT_KEYS, MODEL_PREFIX, PRESET_GUARD,
                 PRESETS, SDK, TIERS)

KEYS_DIR = Path(__file__).resolve().parent / "keys"
STOCK_KEY = 0
FAMILIES = ("fp8", "int8")


def branch_case(sdk, switch, index):
    """The (entry, branches) one case of a pass switch reaches, or (None, [])."""
    cases = switch["cases"]
    pairs = sdk.dispatch_pairs()
    start = cases[index]
    stop = cases[index + 1] if index + 1 < len(cases) else None
    inside = [pair for pair in pairs if pair[0] >= start and
              (stop is None or pair[0] < stop)]
    if not inside:
        return None, []
    return sdk.case_selection(inside)


def branch_case_guarded(sdk, switch, index):
    """``branch_case`` plus the preset-guard count that bounds a model case."""
    entry, branches = branch_case(sdk, switch, index)
    if entry is None or not entry.startswith(MODEL_PREFIX):
        return entry, branches, 0
    rows = sdk.instructions()
    start = switch["cases"][index]
    guards = sum(1 for rva, text in rows
                 if start <= rva <= branches[-1][0] and PRESET_GUARD.match(text))
    return entry, branches, guards


def decode(sdk, switch):
    """One pass switch as {entry: {tier: {preset: {key: record}}}}, with its shape.

    Several cases of a switch can reach the same entry name, and their tables
    disagree about how far the key run goes: the case that carries the stock
    record wins, and a case that disagrees about a key it also reaches aborts
    the decode instead of overwriting what another case proved.
    """
    passes, tables = {}, set()
    for index in range(len(switch["cases"])):
        entry, branches, guards = branch_case_guarded(sdk, switch, index)
        if entry is None:
            continue
        if len(branches) != CASE_BRANCHES or guards != len(TIERS):
            raise ValueError(
                f"{entry}: dispatch case at {switch['cases'][index]:#x} has "
                f"{len(branches)} branches and {guards} preset guards, "
                f"expected {CASE_BRANCHES} and {len(TIERS)}")
        spec = passes.setdefault(entry, {"case": index, "tiers": {}})
        for position, (site, table, array, found) in enumerate(branches):
            tier = TIERS[position // 2]
            preset = "ultraperf" if position % 2 == 0 else "standard"
            rows = spec["tiers"].setdefault(tier, {}).setdefault(preset, {})
            for key, slot in found.items():
                row = {"record_rva": slot["rva"], "dxbc_rva": slot["dxbc_rva"],
                       "size": slot["size"], "dxbc_sha256": slot["sha256"]}
                known = rows.get(key)
                if known is not None and known["dxbc_sha256"] != row["dxbc_sha256"]:
                    raise ValueError(
                        f"{entry} {tier}/{preset} key {key}: the dispatch cases at "
                        f"{passes[entry]['case']:#x} and {switch['cases'][index]:#x} "
                        "select different containers")
                rows.setdefault(key, row)
            tables.add((table, array))
    return {"passes": passes, "selection_sites": sorted(tables)}


def program_version(sdk, decoded):
    """The DXIL program version every container of one switch carries."""
    versions = set()
    for spec in decoded["passes"].values():
        for tiers in spec["tiers"].values():
            for rows in tiers.values():
                slot = rows.get(STOCK_KEY)
                if slot:
                    versions.add(sdk.dxil_version(slot["dxbc_rva"], slot["size"]))
    if not versions:
        raise ValueError("a model switch selected no stock record")
    if len(versions) > 1:
        raise ValueError("one model switch mixes program versions "
                         + ", ".join(sorted(version.hex() for version in versions)))
    return versions.pop()


def families(sdk):
    """{"fp8": switch, "int8": switch}, labelled by the pinned program version."""
    wanted = sdk.shader_format()
    found = {}
    for switch in sdk.dispatch_switches():
        decoded = decode(sdk, switch)
        if not decoded["passes"]:
            continue
        version = program_version(sdk, decoded)
        family = "fp8" if version == wanted else "int8"
        if family in found:
            raise ValueError(f"two model switches carry program version {version.hex()}")
        found[family] = dict(jump_table=switch["table"], cases=len(switch["cases"]),
                             program_version=version.hex(), **decoded)
    if set(found) != set(FAMILIES):
        raise ValueError(f"decoded model switches {sorted(found)}, expected {list(FAMILIES)}")
    fp8, int8 = found["fp8"], found["int8"]
    pinned = sdk.keys().get("model") or {}
    if not pinned:
        raise ValueError(f'keys/{sdk.version}.json has "model": null: pin the shipping '
                         "variant map with tools/extract/modelmap.py --dll <dll> --write")
    for tier, rows in sorted(pinned.items()):
        for name, pin in sorted(rows.items()):
            selected = {}
            for family, spec in found.items():
                matches = [entry for entry in spec["passes"]
                           if entry == name or entry.endswith("_" + name)]
                if len(matches) != 1:
                    raise ValueError(f"the {family} switch reaches {len(matches)} "
                                     f"entries for the pinned pass {name}")
                tiers = spec["passes"][matches[0]]["tiers"]
                selected[family] = tiers.get(tier, {}).get("standard", {}).get(STOCK_KEY)
                if selected[family] is None:
                    raise ValueError(f"{family} switch selects no {tier}/{name} "
                                     "stock record")
            for field in ("record_rva", "dxbc_sha256"):
                if pin.get(field) != selected["fp8"][field]:
                    raise ValueError(
                        f"keys/{sdk.version}.json model {tier}/{name} {field} is "
                        f"{pin.get(field)}, the fp8 switch selects "
                        f"{selected['fp8'][field]}")
    if set(fp8["passes"]) != set(int8["passes"]):
        raise ValueError("the two model switches reach different pass sets: "
                         + ", ".join(sorted(set(fp8["passes"]) ^ set(int8["passes"]))))
    def containers(spec):
        return {row["dxbc_sha256"] for entry in spec["passes"].values()
                for tiers in entry["tiers"].values()
                for records in tiers.values() for row in records.values()}
    shared = containers(int8) & containers(fp8)
    if shared:
        raise ValueError(f"the two model switches share {len(shared)} containers")
    return found


def load(sdk):
    """The pinned dispatch of this build, re-derived and checked against the pins."""
    path = KEYS_DIR / f"{sdk.version}.json"
    if not path.is_file():
        raise ValueError(f"no semantic key map at {path}")
    keys = json.loads(path.read_text())
    if keys.get("source_sha256") != sdk.sha256:
        raise ValueError(f"{path.name} is pinned to a different DLL")
    pinned = keys.get("dispatchers")
    decoded = families(sdk)
    if not pinned:
        raise ValueError(
            f'keys/{sdk.version}.json has "dispatchers": null: pin both model '
            "switches with tools/extract/dispatch.py --dll <dll> --write")
    for family in FAMILIES:
        for field in ("jump_table", "program_version", "cases"):
            if pinned[family].get(field) != decoded[family][field]:
                raise ValueError(
                    f"the pinned {family} dispatcher {field} is "
                    f"{pinned[family].get(field)}, this image decodes "
                    f"{decoded[family][field]}: re-derive the dispatch and re-pin it")
    return decoded


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dll", type=Path, required=True)
    parser.add_argument("--write", action="store_true",
                        help="store the labelled switches in keys/<version>.json")
    args = parser.parse_args(argv)
    try:
        sdk = SDK(args.dll)
        decoded = families(sdk)
        for family in FAMILIES:
            spec = decoded[family]
            label = "reference" if family == "fp8" else family
            print(f"{label}: jump table {spec['jump_table']:#x}, {spec['cases']} cases, "
                  f"{len(spec['passes'])} model passes, program version "
                  f"{spec['program_version']}")
            for entry, rows in sorted(spec["passes"].items(), key=lambda item: item[1]["case"]):
                keys = sorted({key for tiers in rows["tiers"].values()
                               for records in tiers.values() for key in records})
                widths = sorted({len(records) for tiers in rows["tiers"].values()
                                 for records in tiers.values()})
                print(f"  case {rows['case']:>2} {entry[11:]:<26} endpoint keys "
                      f"{keys[0]}..{keys[-1]}, variants per branch {widths}")
        if args.write:
            path = KEYS_DIR / f"{sdk.version}.json"
            keys = json.loads(path.read_text())
            keys["dispatchers"] = {family: {
                "jump_table": spec["jump_table"], "cases": spec["cases"],
                "program_version": spec["program_version"]}
                for family, spec in decoded.items()}
            keys["dispatcher_rule"] = (
                "both model switches decode out of .text; the one whose containers "
                "carry the program version of the pinned RCAS and auto-exposure "
                "shaders is the reference indexed dispatch pinned as fp8 for "
                "historical reasons; the other is the int8 dispatch. The real "
                "fp8 matrix dispatch is extracted separately."
            )
            path.write_text(json.dumps(keys, indent=2) + "\n")
            print(f"wrote the dispatch pins to {path}")
    except (ValueError, OSError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
