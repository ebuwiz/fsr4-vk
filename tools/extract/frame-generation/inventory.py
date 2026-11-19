#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 fsr4-vk contributors.
# See LICENSE in the project root.

"""Inventory the pinned AMD FidelityFX Frame Generation 4.0.2.4555 DX12 DLL.

This is an independent Frame Generation audit. It does not reuse the FSR 4.1.2
upscaler inventory: the Frame Generation image stores its programs as Shader Model
6.6 *immediate shader graph* DXBC containers (they carry ISG1/OSG1 chunks and no
FCSR/RSIG root signature), and its entry-point families (mlfi_* neural inference,
the classic interpolation CS/main set, and the mainVS/mainPS swap-chain blit) are
Frame Generation specific. Every address is derived from the PE image itself.

The tool validates the PE magic, the exact SHA-256 and the pinned SDK version before
extracting anything, fails on any unsupported container layout rather than skipping
it, deduplicates containers by cryptographic hash, classifies each container's
shader stage, records the resource-graph chunk inventory, and preserves the source
DLL identity in the generated manifest so a generated asset cannot silently describe
a different build than the one being extracted.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import struct
import sys
from collections import Counter

ROOT = Path(__file__).resolve().parents[3]

# Verified Frame Generation identity. Only this image is supported.
DLL_SHA256 = "e6843252f58b0341a0d80f6a20bc0d840643ee95bc48c22cac08745344577bb6"
SDK_VERSION = "4.0.2.4555"
COMPONENT = "amd_fidelityfx_framegeneration"
TOOL_VERSION = "framegeneration-inventory-v1"

CORE_CHUNK_TAGS = ("SFI0", "PSV0", "DXIL", "HASH")
IMMEDIATE_TAGS = ("ISG1", "OSG1")
CLASSIC_SIG_TAGS = ("FCSR", "RSIG")


class Image:
    """Minimal PE/COFF reader: RVAs, file offsets and raw reads, never executed."""

    def __init__(self, path, expect_sha256=DLL_SHA256):
        self.path = Path(path)
        self.data = self.path.read_bytes()
        digest = hashlib.sha256(self.data).hexdigest()
        self.sha256 = digest
        if expect_sha256 is not None and digest != expect_sha256:
            raise ValueError(
                "unsupported Frame Generation DLL: expected SHA-256 "
                f"{expect_sha256}, got {digest}"
            )
        if self.data[:2] != b"MZ":
            raise ValueError("not a PE image: missing MZ magic")
        pe = struct.unpack_from("<I", self.data, 0x3C)[0]
        if self.data[pe:pe + 4] != b"PE\x00\x00":
            raise ValueError("not a PE image: missing PE magic at header offset")
        count = struct.unpack_from("<H", self.data, pe + 6)[0]
        optional = pe + 24
        optional_size = struct.unpack_from("<H", self.data, pe + 20)[0]
        self.image_base = struct.unpack_from("<Q", self.data, optional + 24)[0]
        self.sections = []
        for index in range(count):
            entry = optional + optional_size + index * 40
            name = self.data[entry:entry + 8].rstrip(b"\x00").decode("latin-1")
            virtual_size, rva, raw_size, raw_offset = struct.unpack_from(
                "<4I", self.data, entry + 8
            )
            self.sections.append({
                "name": name, "rva": rva, "virtual_size": virtual_size,
                "raw_offset": raw_offset, "raw_size": raw_size,
            })

    def section_for_offset(self, offset, size=1):
        for section in self.sections:
            if section["raw_offset"] <= offset and offset + size <= (
                section["raw_offset"] + section["raw_size"]
            ):
                return section
        return None



def u32(data, offset):
    return struct.unpack_from("<I", data, offset)[0]


def parse_container(data, offset):
    """Return (blob, parts) for one DXBC container, or None if malformed."""
    if offset + 32 > len(data):
        return None
    size, count = struct.unpack_from("<II", data, offset + 24)
    if size < 32 or offset + size > len(data) or count > (size - 32) // 4:
        return None
    parts = []
    for i in range(count):
        start = u32(data, offset + 32 + i * 4)
        if start < 32 + 4 * count or start + 8 > size:
            return None
        length = u32(data, offset + start + 4)
        if start + 8 + length > size:
            return None
        tag = data[offset + start:offset + start + 4]
        parts.append((tag, data[offset + start + 8:offset + start + 8 + length]))
    if sum(tag == b"DXIL" for tag, _ in parts) != 1:
        return None
    return data[offset:offset + size], parts


def read_entry_point(psv0):
    """Decode the PSV0 (runtime info) entry point, failing on unsupported layouts."""
    if len(psv0) < 60:
        raise ValueError("PSV0 truncated before the runtime-info header")
    if u32(psv0, 0) != 52:
        # 52 is the SM 6.6 runtime-info size the Frame Generation blobs carry.
        # Anything else is an unsupported layout, surfaced, never skipped.
        return None
    name_offset = u32(psv0, 52)
    count = u32(psv0, 56)
    pos = 60
    if count:
        if pos + 4 > len(psv0):
            raise ValueError("PSV0 truncated resource stride")
        stride = u32(psv0, pos)
        pos += 4 + count * stride
    if pos + 4 > len(psv0):
        raise ValueError("PSV0 truncated resource table")
    length = u32(psv0, pos)
    pos += 4
    if pos + length > len(psv0) or name_offset >= length:
        raise ValueError("invalid PSV0 string table")
    end = psv0.find(b"\x00", pos + name_offset, pos + length)
    if end == -1:
        raise ValueError("unterminated PSV0 entry point")
    return psv0[pos + name_offset:end].decode("ascii")


def classify_stage(entry_point):
    """Map a Frame Generation entry point to a shader stage/classification."""
    if entry_point is None:
        return "unsupported-psv"
    if entry_point == "mainVS":
        return "vertex"
    if entry_point == "mainPS":
        return "pixel"
    if entry_point == "main":
        return "present-blit"
    if entry_point.startswith("mlfi"):
        return "compute-neural"
    if entry_point.startswith("CS"):
        return "compute-classic"
    return "compute"


def dxil_bytecode_version(dxil):
    """Record (never branch on) the DXIL bytecode version from the DXIL part header."""
    if len(dxil) < 8:
        return None
    return u32(dxil, 0)



def inventory(image):
    if image.sha256 != DLL_SHA256:
        raise ValueError("Frame Generation DLL identity mismatch")
    sections = image.sections
    containers = {}
    offset = 0
    total_occurrences = 0
    while True:
        offset = image.data.find(b"DXBC", offset)
        if offset < 0:
            break
        total_occurrences += 1
        start_offset = offset
        parsed = parse_container(image.data, offset)
        offset += 1
        if parsed is None:
            # A coincidental 'DXBC' inside a string/other table, not a real header.
            # parse_container enforces the failure mode for genuine malformed blobs.
            continue
        blob, parts = parsed
        matches = [
            (sec["name"], sec["rva"] + start_offset - sec["raw_offset"])
            for sec in sections
            if sec["raw_offset"] <= start_offset
            and start_offset + len(blob) <= sec["raw_offset"] + sec["raw_size"]
        ]
        if len(matches) != 1:
            raise ValueError(
                f"DXBC at {start_offset:#x} is not inside exactly one PE section"
            )
        section_name, container_rva = matches[0]
        tags = Counter(tag.decode("latin-1", "replace") for tag, _ in parts)
        # Fail on unsupported layouts rather than skipping them.
        for required in CORE_CHUNK_TAGS:
            if tags.get(required, 0) != 1:
                raise ValueError(
                    f"DXBC at {start_offset:#x} lacks a single {required} chunk"
                )
        has_immediate = any(tags.get(t, 0) >= 1 for t in IMMEDIATE_TAGS)
        has_classic_sig = any(tags.get(t, 0) >= 1 for t in CLASSIC_SIG_TAGS)
        if not has_immediate and not has_classic_sig:
            raise ValueError(
                f"DXBC at {start_offset:#x} has neither an immediate shader graph "
                "(ISG1/OSG1) nor a classic root signature (FCSR/RSIG): unsupported"
            )
        psv0 = next(p for t, p in parts if t == b"PSV0")
        try:
            entry = read_entry_point(psv0)
        except ValueError as error:
            raise ValueError(
                f"DXBC at {start_offset:#x} has an unsupported PSV0 layout: {error}"
            )
        stage = classify_stage(entry)
        digest = hashlib.sha256(blob).hexdigest()
        location = {
            "file_offset": start_offset,
            "rva": hex(container_rva),
            "section": section_name,
        }
        record = containers.get(digest)
        if record is None:
            containers[digest] = {
                "container_sha256": digest,
                "size": len(blob),
                "entry_point": entry,
                "shader_stage": stage,
                "chunk_tags": dict(tags),
                "immediate_shader_graph": has_immediate,
                "classic_root_signature": has_classic_sig,
                "dxil_bytecode_version": dxil_bytecode_version(
                    next(p for t, p in parts if t == b"DXIL")
                ),
                "locations": [location],
            }
        else:
            if (record["shader_stage"] != stage
                    or record["entry_point"] != entry):
                raise ValueError(
                    f"container {digest[:12]} deduped with inconsistent stage/entry"
                )
            record["locations"].append(location)
    if not containers:
        raise ValueError("no Frame Generation DXBC containers found")
    stage_counter = Counter(row["shader_stage"] for row in containers.values())
    entry_counter = Counter(
        str(row["entry_point"]) for row in containers.values()
    )
    immediate_only = all(
        row["immediate_shader_graph"] for row in containers.values()
    )
    return {
        "schema": 1,
        "tool_version": TOOL_VERSION,
        "component": COMPONENT,
        "sdk_version": SDK_VERSION,
        "source": image.path.name,
        "source_sha256": image.sha256,
        "image_base": hex(image.image_base),
        "sections": [
            {"name": s["name"], "rva": hex(s["rva"]),
             "raw_offset": hex(s["raw_offset"]), "raw_size": s["raw_size"]}
            for s in sections
        ],
        "dxbc_signatures_seen": total_occurrences,
        "unique_container_count": len(containers),
        "total_container_locations": sum(
            len(r["locations"]) for r in containers.values()
        ),
        "immediate_shader_graph_only": immediate_only,
        "shader_stage_census": dict(sorted(stage_counter.items())),
        "entry_point_census": dict(sorted(entry_counter.items())),
        "shaders": sorted(
            containers.values(),
            key=lambda row: row["locations"][0]["file_offset"],
        ),
    }



def extract_dxbc(image, output_dir):
    """Write one file per unique container, keyed by entry point + container hash.

    Shares the same validated reader as inventory(); the identity gate runs first,
    so a wrong or unsupported image never produces a partial tree.
    """
    output = Path(output_dir)
    if output.exists():
        raise ValueError("use a fresh extraction output directory")
    output.mkdir(parents=True)
    written = {}
    offset = 0
    seen = {}
    while True:
        offset = image.data.find(b"DXBC", offset)
        if offset < 0:
            break
        start_offset = offset
        parsed = parse_container(image.data, offset)
        offset += 1
        if parsed is None:
            continue
        blob, parts = parsed
        digest = hashlib.sha256(blob).hexdigest()
        if digest in seen:
            continue
        seen[digest] = True
        psv0 = next(p for t, p in parts if t == b"PSV0")
        entry = read_entry_point(psv0) or "unsupported"
        safe = re.sub(r"[^A-Za-z0-9_]", "_", entry)
        name = f"{start_offset:08x}-{safe}-{digest[:16]}.dxbc"
        (output / name).write_bytes(blob)
        written[name] = {
            "entry_point": entry,
            "container_sha256": digest,
            "file_offset": start_offset,
            "size": len(blob),
        }
    return written


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dll", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--extract-dxbc", type=Path,
        help="write one file per unique container under this fresh directory")
    parser.add_argument("--expect-unique", type=int)
    parser.add_argument("--expect-signatures", type=int)
    parser.add_argument("--expect-vertex", type=int)
    parser.add_argument("--expect-pixel", type=int)
    args = parser.parse_args(argv)
    try:
        image = Image(args.dll)
        if args.extract_dxbc:
            extract_dxbc(image, args.extract_dxbc)
        result = inventory(image)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    checks = {
        "--expect-unique": result["unique_container_count"],
        "--expect-signatures": result["dxbc_signatures_seen"],
        "--expect-vertex": result["shader_stage_census"].get("vertex", 0),
        "--expect-pixel": result["shader_stage_census"].get("pixel", 0),
    }
    for flag, actual in checks.items():
        requested = getattr(args, flag[2:].replace("-", "_"), None)
        if requested is not None and requested != actual:
            parser.error(f"{flag}: expected {requested}, got {actual}")
    report = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.exists() and args.output.read_text() != report:
            parser.error("output exists with different contents; choose a fresh path")
        args.output.write_text(report)
    else:
        sys.stdout.write(report)
    print(
        f"Frame Generation {COMPONENT} {SDK_VERSION}: "
        f"{result['unique_container_count']} unique containers from "
        f"{result['dxbc_signatures_seen']} DXBC signatures; "
        f"stages={result['shader_stage_census']}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
