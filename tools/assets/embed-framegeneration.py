#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 fsr4-vk contributors.
# See LICENSE in the project root.

"""Create one indexed binary plus ELF incbin for Frame Generation assets."""

import argparse
import hashlib
import json
import struct
from pathlib import Path

MAGIC = b"FFG4AS1\0"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pack", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.pack / "manifest.json").read_text())
    if manifest.get("schema") != "fgvk-pack-v1":
        parser.error("not a Frame Generation fgvk-pack-v1 pack")
    graph = manifest.get("mlfi_graph") or {}
    model = manifest.get("mlfi_model") or {}
    if not graph.get("file") or not model.get("file"):
        parser.error("pack is missing reverse-engineered ML graph/model")
    files = {
        graph["file"],
        model["file"],
        "manifest.json",
        *(f"modules/{name}" for name in manifest["modules"]),
    }
    rows = []
    payload = bytearray()
    for name in sorted(files):
        path = args.pack / name
        if not path.is_file():
            parser.error("pack file missing: " + name)
        data = path.read_bytes()
        name_bytes = name.encode("utf-8")
        rows.append((name_bytes, len(payload), len(data)))
        payload.extend(data)
    header = bytearray(MAGIC)
    header += struct.pack("<II", len(rows), len(payload))
    for name, offset, size in rows:
        header += struct.pack("<HQQ", len(name), offset, size) + name
    blob = bytes(header + payload)
    args.output.mkdir(parents=True, exist_ok=True)
    binary = args.output / "framegeneration-assets.bin"
    binary.write_bytes(blob)
    assembly = args.output / "framegeneration-assets.S"
    assembly.write_text(
        ".section .rodata.fsr4fg_assets,\"a\",@progbits\n"
        ".p2align 4\n"
        ".global fsr4fg_embedded_assets_begin\n"
        ".global fsr4fg_embedded_assets_end\n"
        "fsr4fg_embedded_assets_begin:\n"
        f".incbin \"{binary.as_posix()}\"\n"
        "fsr4fg_embedded_assets_end:\n"
        ".section .note.GNU-stack,\"\",@progbits\n"
    )
    print(
        f"Embedded Frame Generation assets: files={len(rows)} bytes={len(blob)} "
        f"sha256={hashlib.sha256(blob).hexdigest()}"
    )


if __name__ == "__main__":
    main()
