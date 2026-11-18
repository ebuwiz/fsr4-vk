#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 fsr4-vk contributors.
# See LICENSE in the project root.

"""Account for every shader record table in the image by how it gets selected.

`dispatch.py` decodes the two indexed model switches.  `fp8.py` decodes a third
dispatcher that reaches records through a base-register offset (`INDEXED_RECORD`)
or through a literal address (`DIRECT_RECORD`), neither of which the indexed scan
sees, so a table can look "unreached" while the FP8 matrix dispatch reads it every
day.  A table that none of the three reaches is a table this repository has no way
to select, which is the only honest way to tell "another kit's table" apart from
"a model variant we do not extract" - the distinction the `Fsr4LiteInt8UpscalerModel`
and `Fsr4Int8navi2UpscalerModel` class names in the same image keep colliding with.

    tools/extract/modeltables.py --dll amd_fidelityfx_upscaler_dx12.dll --expect-unreached 0

On the pinned 4.1.2.4555 image it measures: 459 record arrays, 58 of them model shaped
(>= 8 records), and slot-verified record selections - `indexed` 324, `base-pair` 200,
`base-relative` 12, `direct` 125 sites - reach all 58 of them, so nothing of model shape
sits unowned.  Accepting any register pairing, `.text` holds 16 `cmp`-guarded switches:
the two 27-case indexed switches `dispatch.py` pins, one more 27-case switch at 0x10a628
(the FP8 matrix dispatch `fp8.py` decodes), and 13 fourteen-case-or-smaller switches
belonging to non-model kits.  Those three are the only model dispatchers in the image,
and this repository covers all three; a third DXIL program version (62000500) does
appear, only in nameless tables reached through `base-pair` loads, and nothing in this
repository pins or extracts it.
"""
import argparse
import bisect
import re
from collections import Counter
from pathlib import Path

from sdk import RECORD_STRIDE, SDK
import dispatch
import fp8

# A base-register record load with any register pairing: `[rcx + r8 + 0x36c3490]`
# names the record itself, while `fp8.py`'s `INDEXED_RECORD` only matches the two
# orderings it happens to have seen and misses the rest of the image's dispatchers.
RECORD_PAIR = re.compile(r"\[(?:\w+ \+ \w+|rip \+ \w+) \+ (0x[0-9a-f]{5,})\]")
# A base-register index-table load: `[rcx + 4*rax + 0x36c0de0]`.
INDEXED_PAIR = re.compile(r"\[(?:\w+ \+ )?4\*\w+ \+ (0x[0-9a-f]{5,})\]")


def spans(arrays):
    """The record span of each array, so any record address normalises to its table."""
    rows = sorted((array["slots"][0]["rva"],
                   array["slots"][-1]["rva"] + RECORD_STRIDE,
                   array["rva"])
                  for array in arrays if array["slots"])
    return rows, [row[0] for row in rows]


def table_of(rows, starts, address):
    index = bisect.bisect_right(starts, address) - 1
    if index < 0:
        return None
    low, high, rva = rows[index]
    return rva if low <= address < high else None


def selections(sdk):
    """Every record selection site in `.text`, labelled by the pattern that found it.

    The compiler names a record in three shapes: a `rip`-relative indexed table
    lookup, an indexed lookup through a base register plus an index register
    (`[rcx + r8 + 0x36c3490]`, which `fp8.py`'s r14-only pattern misses), and a
    literal address.  Record addresses land mid-array as often as at the start and
    the base-register form points either at the record or at its container pointer
    eight bytes later, so every address is mapped onto the table that contains it
    rather than being assumed to be a table pointer.
    """
    found = set()
    slots = sdk.record_slots()

    def keep(label, address):
        """Record the site only when the address really lands on a shader record."""
        for shift in (0, 8, -8):
            if slots.get(address + shift) is not None:
                found.add((label, address))
                return
    for _, _, array in sdk.dispatch_pairs():
        keep("indexed", array)
    for _, text in sdk.instructions():
        direct = fp8.DIRECT_RECORD.fullmatch(text)
        if direct:
            keep("direct", int(direct.group(1), 16) - sdk.image_base)
        for pattern, label in ((fp8.INDEXED_RECORD, "base-relative"),
                               (RECORD_PAIR, "base-pair"),
                               (INDEXED_PAIR, "base-indexed")):
            match = pattern.search(text)
            if match is None:
                continue
            value = int(match.group(1), 16)
            keep(label, value - sdk.image_base if value >= sdk.image_base else value)
    return found


def analyze(sdk, arrays, covered):
    """One row per table wide enough to be a model table: versions, naming, reach."""
    rows = []
    for array in arrays:
        if array["count"] < 8:
            continue
        versions = Counter()
        for slot in array["slots"]:
            try:
                versions[sdk.dxil_version(slot["dxbc_rva"], slot["size"]).hex()] += 1
            except ValueError:
                versions["no-dxil"] += 1
        named = sum(1 for slot in array["slots"] if slot["name"] != "<unnamed>")
        mechanisms = sorted(covered.get(array["rva"], ()))
        rows.append({"rva": array["rva"], "count": array["count"],
                     "versions": versions, "named": named,
                     "sizes": Counter(slot["size"] // 1024 for slot in array["slots"]),
                     "mechanisms": mechanisms})
    return sorted(rows, key=lambda row: row["rva"])


def label(row):
    versions = row["versions"]
    return ("%s,%s" % (",".join(f"{name} x{count}"
                                for name, count in versions.most_common()),
                       "named" if row["named"] else "nameless"),
            "+".join(row["mechanisms"]) if row["mechanisms"] else "unreached")


def switches_any(sdk):
    """Every pass dispatcher in `.text`, accepting any register pairing.

    `sdk.dispatch_switches` only reads the `rip`-relative shape, and `fp8.dispatcher`
    only the `[r14 + 4*rax + off]` shape, so between them they can leave a model switch
    invisible; a switch shows here when a `cmp <reg>, 0xN` guard is followed within 12
    instructions by an indexed load of either pairing and all N+1 cases land in `.text`.
    """
    rows = sdk.instructions()
    low, high = sdk.section_bounds(".text")
    found = {}
    for index, (_, text) in enumerate(rows):
        load = INDEXED_PAIR.search(text)
        if load is None:
            continue
        table = int(load.group(1), 16)
        for back in range(max(0, index - 12), index):
            guard = re.match(r"cmp\s+\w+, (0x[0-9a-f]+)", rows[back][1])
            if guard is None:
                continue
            count = int(guard.group(1), 16) + 1
            if not 2 <= count <= 64:
                continue
            try:
                cases = sorted(sdk.uint(table + 4 * step) for step in range(count))
            except ValueError:
                continue
            if not all(low + 4 <= case < high for case in cases):
                continue
            found[(table, count)] = cases
    return found


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dll", type=Path, required=True)
    parser.add_argument("--min-slots", type=int, default=8,
                       help="tables at least this wide are model-shaped (default 8)")
    parser.add_argument("--expect-unreached", type=int, default=None,
                       help="exit non-zero unless exactly this many tables are unreached")
    args = parser.parse_args(argv)

    sdk = SDK(args.dll)
    arrays = sdk.record_arrays()
    rows, starts = spans(arrays)
    sites = selections(sdk)
    covered = {}
    for mechanism, address in sites:
        rva = table_of(rows, starts, address)
        if rva is not None:
            covered.setdefault(rva, set()).add(mechanism)
    tables = analyze(sdk, arrays, covered)
    switches = sdk.dispatch_switches()
    matrix_table, matrix_cases = fp8.dispatcher(sdk)
    any_switches = switches_any(sdk)
    known = {spec["jump_table"] for spec in dispatch.families(sdk).values()}
    known.add(matrix_table)

    indexed = "; ".join("%d cases at 0x%x" % (len(switch["cases"]), switch["table"])
                        for switch in switches)
    print("record arrays: %d, of them %d model-shaped (>= %d records)"
          % (len(arrays),
             sum(1 for table in tables if table["count"] >= args.min_slots),
             args.min_slots))
    print("selection sites by pattern:",
          dict(Counter(mechanism for mechanism, _ in sites)))
    print("indexed model switches:", indexed)
    print("base-relative model switch: %d cases at 0x%x"
          % (len(matrix_cases), matrix_table))
    print("switches accepting any register pairing: %d" % len(any_switches))
    print("  case-count histogram:",
          dict(Counter(count for _, count in any_switches)))
    extra = [(table, count) for (table, count) in any_switches if table not in known]
    print("  switches no extractor in this repo covers: %d" % len(extra))
    for table, count in sorted(extra):
        print("    table 0x%x cases=%d" % (table, count))

    classes = Counter()
    unreached = []
    for table in tables:
        if table["count"] < args.min_slots:
            continue
        key, how = label(table)
        classes[(key, how)] += 1
        if how == "unreached":
            unreached.append(table)

    print(f"\nmodel-shaped tables (>= {args.min_slots} records): "
          f"{sum(classes.values())}")
    print(f"{'shape':<40} {'selected_by':<24} tables")
    for (key, how), count in classes.most_common():
        print(f"{key:<40} {how:<24} {count}")

    print(f"\nunreached model-shaped tables: {len(unreached)}")
    for table in unreached:
        print(f"  {table['rva']:#x} records={table['count']:<4} "
              f"named={table['named']:<4} size_kb={dict(table['sizes'].most_common(3))}")

    if args.expect_unreached is not None and len(unreached) != args.expect_unreached:
        raise SystemExit(f"expected {args.expect_unreached} unreached tables, "
                         f"found {len(unreached)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
