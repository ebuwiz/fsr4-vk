#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 fsr4-vk contributors.
# See LICENSE in the project root.
"""Promote an AMD-OBSERVED Frame Generation dispatch ORDER into the pass plan (GPU-free).

The ORDER is NEVER derived here: it is read verbatim from the artifact capture-diff.py --emit-order produces, whose
first_seen_index is AMD's OWN vkCmdDispatch sequence (identity-checked against the pack by SPIR-V digest). This maps
each observed module (64-hex digest) to its plan row (via modules.tsv name<->digest) and records that schedule as an
authoritative `amd_order` on the plan + order_source=amd-runtime-capture. Without a COMPLETE capture it refuses to
invent an order (--require-complete fails; the committed plan stays grouped-scaffold). Chain that closes the dataflow.py
"ML order not statically derivable" boundary ONLY via a genuine capture:
  capture-shim -> capture.tsv ; capture-diff.py --pack <pack> --capture capture.tsv --emit-order <pack>/fg-order-capture.tsv
  apply-capture-order.py --pack <pack> --capture <pack>/fg-order-capture.tsv --require-complete --in-place"""
import argparse
import csv
import json
from pathlib import Path


def read_modules(pack):
    sha_to_name, name_to_sha = {}, {}
    with (pack / "modules.tsv").open(newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            if row["stage"].startswith("compute"):
                sha_to_name[row["spirv_sha256"]] = row["name"]
                name_to_sha[row["name"]] = row["spirv_sha256"]
    return sha_to_name, name_to_sha


def read_order(path):
    with path.open(newline="") as fh:
        return [(int(r["first_seen_index"]), r["sha"]) for r in csv.DictReader(fh, delimiter="\t")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--capture", required=True, help="the capture-diff.py --emit-order TSV AMD produced")
    ap.add_argument("--require-complete", action="store_true")
    ap.add_argument("--in-place", action="store_true", help="rewrite <pack>/fg-plan.{tsv,json}")
    ap.add_argument("--out-prefix", type=Path, default=None, help="write fg-plan.{tsv,json} here instead")
    args = ap.parse_args()
    pack = Path(args.pack)
    if not (pack / "modules.tsv").is_file() or not (pack / "fg-plan.tsv").is_file():
        print("SKIP apply-capture-order: pack modules.tsv/fg-plan.tsv absent"); return 0
    cap = Path(args.capture)
    if not cap.is_file():
        print("FAIL apply-capture-order: no AMD order capture at", cap); return 2
    sha_to_name, name_to_sha = read_modules(pack)   # sha<->name; 474 names alias onto 408 distinct shas
    captured = read_order(cap)
    problems, seen_i, seen_s, unknown = [], set(), set(), []
    for (i, s) in captured:
        if s not in sha_to_name:
            unknown.append(s); continue
        if i in seen_i: problems.append(f"duplicate first_seen_index {i}")
        if s in seen_s: problems.append(f"module captured twice: {s[:16]}")
        seen_i.add(i); seen_s.add(s)
    if unknown:
        problems.append(f"{len(unknown)} capture sha(s) NOT pack modules (identity mismatch): {unknown[0][:16]}...")
    if sorted(seen_i) != list(range(len(captured))):
        problems.append(f"first_seen_index not a contiguous 0..{len(captured)-1} total order")
    header, plan_rows = [], []
    with (pack / "fg-plan.tsv").open(newline="") as fh:
        rd = csv.DictReader(fh, delimiter="\t"); header = rd.fieldnames; plan_rows = list(rd)
    # The DISPATCH identity is the distinct SPIR-V DIGEST; plan ROWS alias onto it. Order attaches per digest and
    # every plan row inherits its digest's first_seen, so aliased rows share one amd_order (they ARE one program).
    compute_shas = {name_to_sha[r["name"]] for r in plan_rows
                    if r["stage"].startswith("compute") and r["name"] in name_to_sha}
    n_shas = len(compute_shas)
    captured_shas = {s for (_, s) in captured if s in sha_to_name}
    missing = [s for s in compute_shas if s not in captured_shas]
    if args.require_complete and missing:
        problems.append(f"capture covers only {len(captured_shas)}/{n_shas} compute digests (unobserved {missing[0][:16]}) "
                        "-- refusing to claim a complete AMD order")
    if problems:
        print("FAIL apply-capture-order: " + "; ".join(problems[:4])); return 1
    idx_by_sha = {s: i for (i, s) in captured if s in sha_to_name}
    complete = not missing
    order_source = "amd-runtime-capture" if complete else "amd-runtime-capture-partial"
    if not args.in_place and args.out_prefix is None:
        print(f"apply-capture-order: DRY RUN (add --in-place or --out-prefix). coverage={len(captured_shas)}/{n_shas} "
              f"complete={complete} order_source={order_source} (order READ from AMD trace, never derived)")
        return 0
    out = pack if args.in_place else Path(args.out_prefix)
    tsv, js = out / "fg-plan.tsv", out / "fg-plan.json"
    with tsv.open("w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(list(header) + ["amd_order"])
        for r in plan_rows:
            sha = name_to_sha.get(r["name"], "")
            w.writerow([r.get(c, "") for c in header] + [idx_by_sha.get(sha, "")])
    summary = json.loads((pack / "fg-plan.json").read_text())
    summary["order_source"] = order_source
    summary["amd_order_coverage"] = [len(captured_shas), n_shas]
    summary["amd_execution_order"] = {s: i for (i, s) in sorted(captured, key=lambda t: t[0])}
    with js.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True); fh.write("\n")
    print(f"apply-capture-order: promoted AMD-captured order -> {out}: coverage={len(captured_shas)}/{n_shas} "
          f"complete={complete} order_source={order_source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
