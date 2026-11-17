#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Extract the optimized FP8 matrix dispatcher from the pinned 4.1.2 DLL."""

import argparse
import hashlib
import json
from pathlib import Path
import re
from family import decode_key
from sdk import COLORS, SDK, TIERS, extract_weights

MODEL_PREFIX = "fsr4_model_v07_fp8_no_scale_"
PASS_NAMES = (
    "prepass",
    "pass0_post", "pass1", "pass1_post",
    "pass2", "pass2_post", "pass3", "pass3_post",
    "pass4", "pass4_post", "pass5", "pass5_post",
    "pass6", "pass6_post", "pass7", "pass7_post",
    "pass8", "pass8_post", "pass9", "pass9_post",
    "pass10", "pass10_post", "pass11", "pass11_post",
    "pass12", "pass12_post",
    "postpass",
)


INDEXED_TABLE = re.compile(r"mov\s+eax, dword ptr \[r14 \+ 4\*rax \+ (0x[0-9a-f]+)\]")
INDEXED_RECORD = re.compile(
    r"mov\s+rax, qword ptr \[(?:r14 \+ rcx|rcx \+ r14) \+ (0x[0-9a-f]+)\]"
)
DIRECT_RECORD = re.compile(r"mov\s+eax, dword ptr \[rip [^\]]+\]\s+# (0x[0-9a-f]+)")


def dispatcher(sdk):
    instructions = sdk.instructions()
    matches = []
    for pos, (_, text) in enumerate(instructions):
        if not re.fullmatch(r"cmp\s+edx, 0x1a", text):
            continue
        for _, line in instructions[pos + 1:pos + 12]:
            match = INDEXED_TABLE.fullmatch(line)
            if not match:
                continue
            table = int(match.group(1), 16)
            cases = [sdk.uint(table + i * 4) for i in range(len(PASS_NAMES))]
            if cases != sorted(cases) or len(set(cases)) != len(PASS_NAMES):
                continue
            by_rva = dict(instructions)
            if by_rva.get(cases[2]) != "mov\tecx, dword ptr [r8 + 0x4]":
                continue
            matches.append((table, cases))
    if len(matches) != 1:
        raise ValueError(f"Expected one optimized FP8 dispatch, found {len(matches)}")
    return matches[0]


def extract(dll, output):
    sdk = SDK(dll)
    if output.exists():
        raise ValueError("Use a fresh FP8 extraction directory")
    table, cases = dispatcher(sdk)
    instructions = sdk.instructions()
    slots = sdk.record_slots()
    output.mkdir(parents=True)
    manifest = {"sdk_version": sdk.version, "source_sha256": sdk.sha256,
                "switch_rva": table, "files": {}}

    def write(path, slot):
        data = sdk.read(slot["dxbc_rva"], slot["size"])
        if hashlib.sha256(data).hexdigest() != slot["sha256"]:
            raise ValueError("FP8 shader record hash mismatch")
        destination = output / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        manifest["files"][path.as_posix()] = slot["sha256"]

    for stage, name in enumerate(PASS_NAMES):
        start = cases[stage]
        end = cases[stage + 1] if stage + 1 < len(cases) else table
        lines = [(rva, text) for rva, text in instructions if start <= rva < end]
        if len(lines) < 8 or lines[0][1] != "mov\tecx, dword ptr [r8 + 0x4]":
            raise ValueError(f"Unexpected tier selector for {name}")
        targets = []
        for _, text in lines[:7]:
            match = re.match(r"je\s+(0x[0-9a-f]+)", text)
            if match:
                targets.append(int(match.group(1), 16) - sdk.image_base)
        if len(targets) != 2 or not (start < targets[1] < targets[0] < end):
            raise ValueError(f"Unexpected tier branches for {name}")
        ranges = [(targets[0], end), (targets[1], targets[0]), (start, targets[1])]
        for tier_index, (tier, (begin, stop)) in enumerate(zip(TIERS, ranges)):
            branch = [text for rva, text in lines if begin <= rva < stop]
            if stage not in (0, 26):
                records = []
                for text in branch:
                    match = DIRECT_RECORD.fullmatch(text)
                    if match:
                        slot = slots.get(int(match.group(1), 16) - sdk.image_base)
                        if slot and slot["name"] == MODEL_PREFIX + name:
                            records.append(slot)
                if len(records) != 1:
                    raise ValueError(f"Expected one FP8 record for {tier}/{name}")
                write(Path("model/sm66") / tier / (name + ".dxbc"), records[0])
                continue

            indices = [int(m.group(1), 16) for text in branch
                       if (m := INDEXED_TABLE.fullmatch(text))]
            arrays = [int(m.group(1), 16) - 8 for text in branch
                      if (m := INDEXED_RECORD.fullmatch(text))]
            if len(indices) != 1 or not arrays:
                raise ValueError(f"Unexpected FP8 endpoint table for {tier}/{name}")
            records = sdk.table_records(indices[0], arrays[0])
            phase = "prepass" if stage == 0 else "postpass"
            for key in range(64 if stage == 0 else 16):
                table_key = key if stage == 0 else key + 16 * tier_index
                slot = records.get(table_key)
                if not slot or slot["name"] != MODEL_PREFIX + name:
                    raise ValueError(f"Missing FP8 {tier}/{name} key {table_key}")
                color, flags, debug = decode_key(phase, stage, key)
                filename = f"pass-{stage:02d}" + ("-debug" if debug else "") + ".dxbc"
                path = (Path("endpoints/sdk-fp8") / tier / COLORS[color] /
                        f"{flags:02x}" / "standard" / filename)
                write(path, slot)

    extract_weights(sdk, output / "model/weights")
    (output / "model/auto-exposure.dxbc").write_bytes(sdk.auto_exposure()["data"])
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Extracted FP8 matrix dispatcher: {len(manifest['files'])} selections")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dll", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    extract(args.dll, args.output)


if __name__ == "__main__":
    main()
