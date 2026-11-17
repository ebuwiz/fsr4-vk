#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 fsr4-vk contributors.
# See LICENSE in the project root.

"""Inventory DXBC containers in the pinned FSR 4.1.2.4555 DLL."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import struct

SDK_SHA256 = "aad4fccccb29084a655f9a4f9f613bd22c8c03fb4a0f8840148cc6543b4ccace"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[2] / "build/inventory-4.1.2.4555.json"


def u32(data, offset):
    return struct.unpack_from("<I", data, offset)[0]


def container(data, offset):
    if offset + 32 > len(data):
        return None
    size, count = struct.unpack_from("<II", data, offset + 24)
    if size < 32 or offset + size > len(data) or count > (size - 32) // 4:
        return None
    chunks = []
    for i in range(count):
        start = u32(data, offset + 32 + i * 4)
        if start < 32 + 4 * count or start + 8 > size:
            return None
        length = u32(data, offset + start + 4)
        if start + 8 + length > size:
            return None
        tag = data[offset + start : offset + start + 4]
        chunks.append((tag, data[offset + start + 8 : offset + start + 8 + length]))
    if sum(tag == b"DXIL" for tag, _ in chunks) != 1:
        return None
    return data[offset : offset + size], chunks


def entry_point(chunks):
    psvs = [data for tag, data in chunks if tag == b"PSV0"]
    if len(psvs) != 1:
        raise ValueError("Expected exactly one PSV0 chunk")
    data = psvs[0]
    if len(data) < 60 or u32(data, 0) != 52:
        raise ValueError("Unsupported PSV0 runtime info layout")
    name_offset = u32(data, 52)
    count = u32(data, 56)
    pos = 60
    if count:
        if pos + 4 > len(data):
            raise ValueError("Truncated PSV0 resource stride")
        stride = u32(data, pos)
        pos += 4 + count * stride
    if pos + 4 > len(data):
        raise ValueError("Truncated PSV0 resource table")
    length = u32(data, pos)
    pos += 4
    if pos + length > len(data) or name_offset >= length:
        raise ValueError("Invalid PSV0 string table")
    end = data.find(b"\0", pos + name_offset, pos + length)
    if end == -1:
        raise ValueError("Unterminated PSV0 entry point")
    return data[pos + name_offset : end].decode("ascii")


def pe_sections(data):
    if len(data) < 64 or data[:2] != b"MZ":
        raise ValueError("Not a PE image")
    pe = u32(data, 0x3C)
    if pe + 24 > len(data) or data[pe : pe + 4] != b"PE\0\0":
        raise ValueError("Invalid PE header")
    count = struct.unpack_from("<H", data, pe + 6)[0]
    optional_size = struct.unpack_from("<H", data, pe + 20)[0]
    start = pe + 24 + optional_size
    if start + count * 40 > len(data):
        raise ValueError("Truncated PE section table")
    sections = []
    for i in range(count):
        entry = start + i * 40
        name = data[entry : entry + 8].rstrip(b"\0").decode("ascii")
        rva, raw_size, raw = struct.unpack_from("<III", data, entry + 12)
        sections.append((name, rva, raw, raw_size))
    return sections


def inventory(data):
    digest = hashlib.sha256(data).hexdigest()
    if digest != SDK_SHA256:
        raise ValueError("Unsupported DLL: expected FSR 4.1.2.4555 SHA-256 " + SDK_SHA256)
    sections = pe_sections(data)
    shaders = {}
    offset = 0
    while True:
        offset = data.find(b"DXBC", offset)
        if offset < 0:
            break
        parsed = container(data, offset)
        if parsed is not None:
            blob, chunks = parsed
            matches = [(name, rva + offset - raw) for name, rva, raw, size in sections
                       if raw <= offset and offset + len(blob) <= raw + size]
            if len(matches) != 1:
                raise ValueError(f"DXBC at {offset:#x} is not in exactly one PE section")
            name, rva = matches[0]
            sha256 = hashlib.sha256(blob).hexdigest()
            if sha256 not in shaders:
                hashes = [part for tag, part in chunks if tag == b"HASH"]
                if len(hashes) != 1 or len(hashes[0]) != 20:
                    raise ValueError(f"Invalid HASH chunk at {offset:#x}")
                shaders[sha256] = {
                    "sha256": sha256,
                    "size": len(blob),
                    "entry_point": entry_point(chunks),
                    "dxil_hash": hashes[0][4:].hex(),
                    "chunks": [tag.decode("ascii") for tag, _ in chunks],
                    "locations": [],
                }
            shaders[sha256]["locations"].append(
                {"file_offset": offset, "rva": rva, "section": name}
            )
        offset += 1
    if not shaders:
        raise ValueError("No DXIL containers found")
    return {
        "sdk_version": "4.1.2.4555",
        "source_sha256": digest,
        "container_count": sum(len(row["locations"]) for row in shaders.values()),
        "unique_count": len(shaders),
        "entry_points": dict(sorted(Counter(
            row["entry_point"] for row in shaders.values()
        ).items())),
        "shaders": sorted(shaders.values(), key=lambda row: row["locations"][0]["file_offset"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dll", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    try:
        result = inventory(args.dll.read_bytes())
        report = json.dumps(result, indent=2) + "\n"
        if args.output.exists():
            if args.output.read_text() != report:
                raise ValueError("Output already exists with different contents; choose a fresh path")
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(report)
    except (ValueError, OSError, UnicodeError, struct.error) as error:
        parser.error(str(error))
    print(f"Wrote {result['unique_count']} unique DXIL shaders "
          f"({result['container_count']} containers) to {args.output}")


if __name__ == "__main__":
    main()
