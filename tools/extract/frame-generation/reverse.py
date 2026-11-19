#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 fsr4-vk contributors.
# See LICENSE in the project root.

"""Reverse the pinned FG 4.0.2 PE's ML graph without committing model bytes.

The target has two complete 191-pass neural chains (4k and 8k). The shader
 inventory supplies every program identity; this tool correlates those identities
 with the PE's descriptor records, root-constant patterns from SPIR-V, and the
 aligned 9 MiB model-data candidate. It emits hashes and offsets only. Actual model
 bytes are extracted during the local asset workflow into the gitignored pack.
"""

import argparse
import hashlib
import json
import math
import re
import struct
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SOURCE_SHA256 = "e6843252f58b0341a0d80f6a20bc0d840643ee95bc48c22cac08745344577bb6"
MLFI = re.compile(r"^mlfi_(4k|8k)_pass(\d+)$")
MODEL_RVA = 0xF00000
MODEL_SIZE = 0x900000


def u16(data, offset):
    return struct.unpack_from("<H", data, offset)[0]


def u32(data, offset):
    return struct.unpack_from("<I", data, offset)[0]


def u64(data, offset):
    return struct.unpack_from("<Q", data, offset)[0]


class PeImage:
    def __init__(self, path):
        self.path = Path(path)
        self.data = self.path.read_bytes()
        self.sha256 = hashlib.sha256(self.data).hexdigest()
        if self.sha256 != SOURCE_SHA256:
            raise ValueError(
                "unsupported Frame Generation DLL: expected "
                f"{SOURCE_SHA256}, got {self.sha256}"
            )
        if self.data[:2] != b"MZ":
            raise ValueError("Frame Generation input is not a PE image")
        pe = u32(self.data, 0x3C)
        if self.data[pe : pe + 4] != b"PE\0\0":
            raise ValueError("Frame Generation input has an invalid PE header")
        self.image_base = u64(self.data, pe + 24 + 24)
        count = u16(self.data, pe + 6)
        optional_size = u16(self.data, pe + 20)
        table = pe + 24 + optional_size
        self.sections = []
        for index in range(count):
            entry = table + index * 40
            name = self.data[entry : entry + 8].rstrip(b"\0").decode("latin-1")
            virtual_size, rva, raw_size, raw = struct.unpack_from(
                "<IIII", self.data, entry + 8
            )
            self.sections.append(
                {
                    "name": name,
                    "rva": rva,
                    "size": max(virtual_size, raw_size),
                    "raw": raw,
                    "raw_size": raw_size,
                }
            )

    def offset(self, rva, size=1):
        for section in self.sections:
            if section["rva"] <= rva and rva + size <= section["rva"] + section["size"]:
                if rva + size > section["rva"] + section["raw_size"]:
                    raise ValueError(
                        f"RVA {rva:#x} is in virtual, non-file-backed section data"
                    )
                return section["raw"] + rva - section["rva"]
        raise ValueError(f"RVA {rva:#x} is outside the PE image")

    def rva(self, offset):
        for section in self.sections:
            if section["raw"] <= offset < section["raw"] + section["raw_size"]:
                return section["rva"] + offset - section["raw"]
        return None

    def blob(self, rva, size):
        offset = self.offset(rva, size)
        return self.data[offset : offset + size]


def entropy(data, sample=1 << 16):
    sample = data[:sample]
    if not sample:
        return 0.0
    counts = Counter(sample)
    return -sum(
        (count / len(sample)) * math.log2(count / len(sample))
        for count in counts.values()
    )


def inspect_spirv(disassembler, shader):
    result = subprocess.run(
        [str(disassembler), str(shader), "-o", "-"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    roots = sorted(
        set(
            int(value)
            for value in re.findall(
                r"OpAccessChain %_ptr_PushConstant_uint %registers %uint_(\d+)",
                result,
            )
        )
    )
    sets = sorted(
        {
            int(value)
            for value in re.findall(
                r"OpDecorate %\w+ DescriptorSet (\d+)", result
            )
        }
    )
    operations = {
        "cooperative_matrix": "OpTypeCooperativeMatrixKHR" in result,
        "float16": "OpTypeFloat 16" in result,
        "int8": "OpTypeInt 8" in result,
        "dot_product": bool(re.search(r"\bOp[US]?Dot\b", result)),
    }
    return {"push_constants": roots, "descriptor_sets": sets, **operations}


def shader_records(pe, inventory):
    containers = {}
    for row in inventory["shaders"]:
        for location in row["locations"]:
            rva = int(location["rva"], 16)
            containers[pe.image_base + rva] = {
                **row,
                "record_rva": rva,
            }
    records = []
    model_pointer = struct.pack("<Q", pe.image_base + MODEL_RVA)
    for offset in range(len(pe.data) - 7):
        value = u64(pe.data, offset)
        row = containers.get(value)
        if row is None:
            continue
        start = max(0, offset - 64)
        stop = min(len(pe.data), offset + 256)
        records.append(
            {
                "record_offset": offset,
                "record_rva": pe.rva(offset),
                "section": next(
                    (
                        section["name"]
                        for section in pe.sections
                        if section["raw"]
                        <= offset
                        < section["raw"] + section["raw_size"]
                    ),
                    "unknown",
                ),
                "entry": row.get("entry_point"),
                "stage": row["shader_stage"],
                "container_rva": row["record_rva"],
                "container_sha256": row["container_sha256"],
                "model_base_near_record": pe.data.find(
                    model_pointer, start, stop
                )
                != -1,
            }
        )
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dll", type=Path, required=True)
    parser.add_argument(
        "--inventory",
        type=Path,
        default=ROOT / "build/fg-capabilities/inventory.json",
    )
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--spirv-dis", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--extract-model", type=Path)
    args = parser.parse_args(argv)

    pe = PeImage(args.dll)
    inventory = json.loads(args.inventory.read_text())
    if inventory.get("source_sha256") != pe.sha256:
        raise ValueError("inventory SHA-256 does not match the pinned DLL")
    disassembler = args.spirv_dis or ROOT / "build/tools/SPIRV-Tools/bin/spirv-dis"
    if not disassembler.is_file():
        raise ValueError("spirv-dis not found: " + str(disassembler))

    chains = defaultdict(dict)
    pack_manifest = json.loads((args.pack / "manifest.json").read_text())
    by_entry = {
        row.get("entry_point"): (name, row)
        for name, row in pack_manifest["modules"].items()
    }
    for entry, (name, row) in by_entry.items():
        match = MLFI.match(entry or "")
        if not match:
            continue
        tier, pass_index = match.group(1), int(match.group(2))
        shader = args.pack / "modules" / name
        chains[tier][pass_index] = {
            "entry": entry,
            "module": name,
            "spirv_sha256": row["spirv_sha256"],
            **inspect_spirv(disassembler, shader),
        }
    for tier, passes in chains.items():
        expected = set(range(191))
        if set(passes) != expected:
            raise ValueError(
                f"{tier} ML chain does not contain passes 0..190; "
                f"missing={sorted(expected - set(passes))}"
            )

    model = pe.blob(MODEL_RVA, MODEL_SIZE)
    records = shader_records(pe, inventory)
    neural_records = [row for row in records if row["stage"] == "compute-neural"]
    model_record_refs = [row for row in neural_records if row["model_base_near_record"]]
    if len(model_record_refs) != 2:
        raise ValueError(
            "expected exactly two ML records to reference the model base "
            f"(one per 4k/8k chain), found {len(model_record_refs)}"
        )

    report = {
        "schema": 1,
        "component": "frame-generation",
        "sdk_version": "4.0.2.4555",
        "source_sha256": pe.sha256,
        "model": {
            "rva": f"0x{MODEL_RVA:x}",
            "size": len(model),
            "sha256": hashlib.sha256(model).hexdigest(),
            "entropy": entropy(model),
            "record_references": model_record_refs,
            "extracted": bool(args.extract_model),
        },
        "chains": {
            tier: [passes[index] for index in range(191)]
            for tier, passes in sorted(chains.items())
        },
        "shader_records": {
            "total": len(records),
            "neural": len(neural_records),
            "sections": dict(Counter(row["section"] for row in records)),
        },
        "order_basis": (
            "Complete 4k/8k chains verified as mlfi_<tier>_pass 0..190; "
            "within-tier execution order is fixed by pass index. "
            "Classic optical-flow/interpolation ordering still requires "
            "control-flow correlation."
        ),
    }
    if args.extract_model:
        args.extract_model.parent.mkdir(parents=True, exist_ok=True)
        args.extract_model.write_bytes(model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        f"Frame Generation RE: ML chains 4k=191 8k=191; model "
        f"{MODEL_SIZE} bytes sha256={report['model']['sha256']}; "
        f"records={len(records)} model_refs={len(model_record_refs)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
