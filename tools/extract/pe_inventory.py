# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 fsr4-vk contributors.
# See LICENSE in the project root.

"""Discover the FSR 4.1.2 shader records and DXBC containers in the shipping DLL.

Every address used here is derived from the PE image itself: no section offset,
record address or table address is inherited from an earlier SDK build.  The
4.1.2 upscaler keeps its per-variant shader records in `.data` as 0x130-byte
objects (`u32 DXBC size`, `u64 DXBC virtual address`, then six 0x28-byte
parameter groups) and the dispatch code copies those groups directly, so the
records have to be located by structure rather than by a hardcoded table.
"""

import argparse
import hashlib
import json
import re
import struct
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DLL_PATH = ROOT / "dlls/4.1.2.4555/amd_fidelityfx_upscaler_dx12.dll"
DLL_SHA256 = "aad4fccccb29084a655f9a4f9f613bd22c8c03fb4a0f8840148cc6543b4ccace"
SDK_VERSION = "4.1.2.4555"
RECORD_STRIDE = 0x130
RECORD_GROUPS = tuple(range(0x18, RECORD_STRIDE, 0x28))
NAME_PATTERN = re.compile(
    rb"(fsr4[A-Za-z0-9_]*|fsr_[A-Za-z0-9_]+|brw[A-Za-z0-9_]+|[a-z][A-Za-z0-9_]{5,}_texture)"
)


class Image:
    """Minimal PE/COFF reader: RVAs, file offsets and raw reads, never executed."""

    def __init__(self, path=None, expect_sha256=DLL_SHA256):
        self.path = Path(path or DLL_PATH)
        self.data = self.path.read_bytes()
        digest = hashlib.sha256(self.data).hexdigest()
        self.sha256 = digest
        if expect_sha256 is not None and digest != expect_sha256:
            raise ValueError(
                f"unsupported DLL: expected {expect_sha256[:12]}..., got {digest[:12]}..."
            )
        pe = struct.unpack_from("<I", self.data, 0x3C)[0]
        count = struct.unpack_from("<H", self.data, pe + 6)[0]
        optional = pe + 24
        optional_size = struct.unpack_from("<H", self.data, pe + 20)[0]
        self.image_base = struct.unpack_from("<Q", self.data, optional + 24)[0]
        self.sections = []
        for index in range(count):
            entry = optional + optional_size + index * 40
            name = self.data[entry : entry + 8].rstrip(b"\0").decode("latin-1")
            virtual_size, rva, raw_size, raw_offset = struct.unpack_from("<4I", self.data, entry + 8)
            self.sections.append({"name": name, "rva": rva, "virtual_size": virtual_size,
                                  "raw_offset": raw_offset, "raw_size": raw_size})

    def offset(self, rva, size=1):
        for section in self.sections:
            if section["rva"] <= rva < section["rva"] + section["raw_size"]:
                raw = section["raw_offset"] + rva - section["rva"]
                if raw + size <= section["raw_offset"] + section["raw_size"]:
                    return raw
        raise ValueError(f"unmapped RVA: {rva:#x}/{size}")

    def rva(self, offset):
        for section in self.sections:
            if section["raw_offset"] <= offset < section["raw_offset"] + section["raw_size"]:
                return section["rva"] + offset - section["raw_offset"]
        return None

    def section_name(self, rva):
        for section in self.sections:
            if section["rva"] <= rva < section["rva"] + section["raw_size"]:
                return section["name"]
        return "?"

    def read(self, rva, size):
        return self.data[self.offset(rva, size) : self.offset(rva, size) + size]

    def uint(self, rva, size=4):
        return struct.unpack_from("<" + {4: "I", 8: "Q"}[size], self.read(rva, size))[0]


def containers(image):
    """Every validated DXBC container in the image, with its entry-point names."""
    found = {}
    position = 0
    while True:
        position = image.data.find(b"DXBC", position + 1)
        if position < 0 or position + 28 > len(image.data):
            break
        size = struct.unpack_from("<I", image.data, position + 24)[0]
        if not 0x100 < size <= 0x4000000 or position + size > len(image.data):
            continue
        blob = image.data[position : position + size]
        rva = image.rva(position)
        if rva is None:
            continue
        names = sorted({name.decode("latin-1") for name in NAME_PATTERN.findall(blob)})
        found[position] = {
            "offset": position, "rva": rva, "section": image.section_name(rva),
            "size": size, "sha256": hashlib.sha256(blob).hexdigest(), "names": names,
        }
    return found


def records(image, blobs):
    """0x130 shader records: DXBC size + DXBC VA + six 0x28 parameter groups."""
    by_offset = {blob["offset"]: blob for blob in blobs.values()}
    result = {}
    for offset in range(0, len(image.data) - RECORD_STRIDE, 8):
        size = struct.unpack_from("<I", image.data, offset)[0]
        pointer = struct.unpack_from("<Q", image.data, offset + 8)[0]
        if not 0x100 < size <= 0x4000000:
            continue
        target = pointer - image.image_base
        try:
            blob_offset = image.offset(target, size)
        except ValueError:
            continue
        blob = by_offset.get(blob_offset)
        if blob is None or blob["size"] != size:
            continue
        rva = image.rva(offset)
        if rva is None:
            continue
        groups = []
        for group in RECORD_GROUPS:
            values = [struct.unpack_from("<Q", image.data, offset + group + 8 * index)[0]
                      for index in range(1, 5)]
            groups.append({"offset": group,
                           "count": struct.unpack_from("<I", image.data, offset + group)[0],
                           "pointers": [value - image.image_base for value in values if value]})
        result[rva] = {"rva": rva, "offset": offset, "section": image.section_name(rva),
                       "dxbc_rva": target, "size": size, "sha256": blob["sha256"],
                       "names": blob["names"], "groups": groups}
    return result


def arrays(table):
    """Maximal runs of consecutive records: one record array per variant table."""
    groups, current = [], []
    for rva in sorted(table):
        if current and rva != current[-1] + RECORD_STRIDE:
            groups.append(current)
            current = []
        current.append(rva)
    if current:
        groups.append(current)
    result = []
    for index, group in enumerate(groups):
        slots = [{"index": slot, "rva": rva, "name": (table[rva]["names"] or ["<unnamed>"])[0],
                  "dxbc_rva": table[rva]["dxbc_rva"], "size": table[rva]["size"],
                  "sha256": table[rva]["sha256"], "group_counts":
                  [group_["count"] for group_ in table[rva]["groups"]]}
                 for slot, rva in enumerate(group)]
        result.append({"index": index, "rva": group[0], "count": len(group),
                       "section": table[group[0]]["section"],
                       "names": sorted({slot["name"] for slot in slots}),
                       "blobs": sorted({slot["sha256"][:16] for slot in slots}),
                       "slots": slots})
    return result


def summarize(image, blobs, table, table_arrays):
    print(f"DLL: {image.path} (sha256 {hashlib.sha256(image.data).hexdigest()[:16]}...)")
    print("sections: " + ", ".join(f"{s['name']}@{s['rva']:#x}+{s['raw_size']:#x}"
                                  for s in image.sections))
    print(f"DXBC containers: {len(blobs)}")
    print(f"shader records ({RECORD_STRIDE:#x}): {len(table)}")
    histogram = Counter(entry["count"] for entry in table_arrays)
    print("record arrays: " + " ".join(f"{count}x{size}" for size, count in sorted(histogram.items())))
    per_name = defaultdict(lambda: {"arrays": set(), "records": 0, "blobs": set()})
    for entry in table_arrays:
        for slot in entry["slots"]:
            state = per_name[slot["name"]]
            state["arrays"].add(entry["index"])
            state["records"] += 1
            state["blobs"].add(slot["sha256"])
    print(f"entry names: {len(per_name)}")
    for name, state in sorted(per_name.items(), key=lambda kv: -kv[1]["records"]):
        print(f"  {name:44} records={state['records']:4} arrays={len(state['arrays']):3} "
              f"blobs={len(state['blobs']):4}")
    return per_name


def write_blobs(image, table_arrays, output):
    """Write the DXBC container of every record slot, keyed by entry name and table."""
    output = Path(output)
    if output.exists():
        raise ValueError("use a fresh inventory output directory")
    output.mkdir(parents=True)
    files = {}
    for entry in table_arrays:
        for slot in entry["slots"]:
            path = (output / "dxbc" / slot["name"]
                    / f"{entry['index']:03d}-{slot['index']:03d}.dxbc")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(image.read(slot["dxbc_rva"], slot["size"]))
            files[path.relative_to(output).as_posix()] = {
                "name": slot["name"], "array": entry["index"], "slot": slot["index"],
                "record_rva": entry["rva"] + slot["index"] * RECORD_STRIDE,
                "dxbc_rva": slot["dxbc_rva"], "size": slot["size"], "sha256": slot["sha256"],
            }
    manifest = {
        "schema": 1, "sdk_version": SDK_VERSION, "source": image.path.name,
        "source_sha256": hashlib.sha256(image.data).hexdigest(),
        "record_stride": RECORD_STRIDE,
        "record_groups": [group for group in RECORD_GROUPS],
        "files": files,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return files


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dll", type=Path, default=DLL_PATH)
    parser.add_argument("--json", type=Path, help="write the full inventory JSON here")
    parser.add_argument("--extract-dxbc", type=Path,
                        help="write every record DXBC under this directory")
    args = parser.parse_args(argv)
    try:
        image = Image(args.dll)
        blobs = containers(image)
        table = records(image, blobs)
        table_arrays = arrays(table)
        summarize(image, blobs, table, table_arrays)
        if args.json:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(json.dumps({
                "schema": 1, "sdk_version": SDK_VERSION, "source": image.path.name,
                "source_sha256": hashlib.sha256(image.data).hexdigest(),
                "image_base": image.image_base,
                "sections": image.sections,
                "containers": sorted(blobs.values(), key=lambda b: b["rva"]),
                "records": sorted(table.values(), key=lambda r: r["rva"]),
                "arrays": table_arrays,
            }, indent=1) + "\n")
            print("inventory JSON:", args.json)
        if args.extract_dxbc:
            files = write_blobs(image, table_arrays, args.extract_dxbc)
            print(f"DXBC written: {len(files)} files -> {args.extract_dxbc}")
    except (ValueError, OSError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
