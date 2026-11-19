#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 fsr4-vk contributors.
# See LICENSE in the project root.
"""Diff an intercepted Vulkan compute trace (from AMD's Frame Generation provider at runtime) against the
pass PLAN we reconstructed from the pack, WITHOUT ever asserting the ML order ourselves.

A capture is a file of compute dispatch records, each carrying the bound pipeline's SPIR-V digest. We look
that digest up in the pack's modules.tsv: this confirms WHICH plan module each observed dispatch is (a per-
module FIRST-OBSERVED position + call count) and flags any dispatch that does NOT correspond to a known pack
module (extra). It reports plan modules never observed in this capture (missing) but, for a partial capture,
missing is informational, NOT a failure. The ORDER is only ever read off AMD's trace; the tool never claims a
canonical schedule (dataflow.py proves none is statically derivable).

Run:  python3 capture-diff.py --pack assets/frame-generation/pack --capture capture.json [--emit-order o.tsv]
Modes: --expect-plan (extra must be 0), --complete (also missing must be 0), --selftest (synth round-trip)."""
import argparse
import json
import hashlib
import sys
from pathlib import Path


def load_plan(pack_dir):
    """Return {sha: (stage, entry, wgs)} for every COMPUTE module (the 474)."""
    plan = {}
    tsv = Path(pack_dir) / "modules.tsv"
    if not tsv.is_file():
        return plan
    lines = tsv.read_text().splitlines()
    for line in lines[1:]:
        p = line.split("\t")
        if len(p) < 5:
            continue
        stage, entry, wgs, sha = p[1], p[2], p[3], p[4]
        if stage.startswith("compute"):
            plan[sha] = (stage, entry, wgs)
    return plan


def read_capture(path):
    """Return a list of (sha, dims) dispatch records from JSON or TSV."""
    text = Path(path).read_text()
    if str(path).endswith(".json"):
        data = json.loads(text)
        return [(r["spirv_sha256"], r.get("dims")) for r in data["dispatches"]]
    out = []
    for line in text.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        p = line.split("\t")
        out.append((p[0], p[1] if len(p) > 1 else None))
    return out


def analyze(plan, records):
    first_seen, counts, unknown = {}, {}, []
    for pos, (sha, dims) in enumerate(records):
        if sha in plan:
            first_seen.setdefault(sha, pos)
            counts[sha] = counts.get(sha, 0) + 1
        else:
            unknown.append((pos, sha))
    observed = set(first_seen)
    extra = sorted({s for _, s in unknown})           # dispatches with no matching pack module
    missing = sorted(set(plan) - observed)            # plan modules not exercised by this capture
    return first_seen, counts, extra, missing, observed


def emit(path, plan, first_seen, counts):
    rows = sorted(first_seen.items(), key=lambda kv: kv[1])
    with Path(path).open("w") as f:
        f.write("first_seen_index\tcall_count\tstage\tentry\tsha\n")
        for sha, pos in rows:
            stage, entry, _ = plan[sha]
            f.write(f"{pos}\t{counts[sha]}\t{stage}\t{entry}\t{sha}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack")
    ap.add_argument("--capture")
    ap.add_argument("--emit-order")
    ap.add_argument("--expect-plan", action="store_true")
    ap.add_argument("--complete", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        if not args.pack or not Path(args.pack).is_dir():
            print("selftest needs --pack with a populated pack"); return 2
        plan = load_plan(args.pack)
        if not plan:
            print("SKIP selftest: no compute modules in pack"); return 0
        # Good capture: every compute module once, ordered by entry (deterministic; NOT a claimed schedule).
        good = [(sha, plan[sha][2]) for sha in sorted(plan, key=lambda s: plan[s][1])]
        _, _, extra, missing, _ = analyze(plan, good)
        if extra or missing:
            print(f"selftest FAIL round-trip extra={len(extra)} missing={len(missing)}"); return 1
        # Bad capture: inject one unknown digest -> must be detected as extra.
        bad = list(good); bad[0] = ("deadbeef" * 8, "8,8,1")
        _, _, extra2, _, _ = analyze(plan, bad)
        if not extra2:
            print("selftest FAIL: unknown digest not flagged as extra"); return 1
        print(f"selftest PASS: round-trip clean (compute={len(plan)}); "
              f"unknown-digest drift detected extra={len(extra2)}")
        return 0

    if not args.capture or not args.pack:
        print("need --pack and --capture (or --selftest)"); return 2
    plan = load_plan(args.pack)
    if not plan:
        print("SKIP: no compute modules in pack"); return 0
    recs = read_capture(args.capture)
    fs, cnt, extra, missing, obs = analyze(plan, recs)
    if args.emit_order:
        emit(args.emit_order, plan, fs, cnt)
    print(f"capture dispatches={len(recs)} observed_compute={len(obs)} plan_compute={len(plan)} "
          f"extra={len(extra)} missing={len(missing)}")
    if extra:
        print("EXTRA (no matching pack module): " + ", ".join(extra[:6]))
    if args.expect_plan and extra:
        print("FAIL: capture dispatches do not all map to the plan"); return 1
    if args.complete and missing:
        print("FAIL: capture did not exercise every plan compute module"); return 1
    print("capture-diff: order read from AMD's trace (identity-checked, never self-derived)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
