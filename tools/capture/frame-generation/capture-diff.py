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
      Works on EITHER pack kind: the FG pack (modules.tsv) or the UPSCALER pack-unified (manifest.json -> its .spv payload digests).
Modes: --expect-plan (extra must be 0), --complete (also missing must be 0), --selftest (synth round-trip)."""
import argparse
import json
import hashlib
import sys
from pathlib import Path


def load_plan(pack_dir):
    """Return (plan, named). plan = {sha: (stage, entry, wgs)} over every COMPUTE/SPIR-V module; named =
    [(pack-relative .spv path, wgs)] used by the path-mode selftest. Detects two pack kinds: the FG pack (has
    modules.tsv) and the UPSCALER pack pack-unified (has manifest.json whose .spv payloads carry the digests)."""
    pack_dir = Path(pack_dir)
    plan, named = {}, []
    tsv = pack_dir / "modules.tsv"
    if tsv.is_file():
        for line in tsv.read_text().splitlines()[1:]:
            p = line.split("\t")
            if len(p) < 5:
                continue
            name, stage, entry, wgs, sha = p[0], p[1], p[2], p[3], p[4]
            if stage.startswith("compute"):
                plan[sha] = (stage, entry, wgs)
                named.append((f"modules/{name}", wgs))  # FG name already ends in .spv; file lives at modules/<name>
        return plan, named
    man = pack_dir / "manifest.json"
    if man.is_file():
        for r in json.loads(man.read_text())["payloads"]:
            f = str(r["file"])
            if f.endswith(".spv"):
                plan[r["sha256"]] = ("upscaler", Path(f).name, None)
                named.append((f, None))
        return plan, named
    return plan, named


def resolve_sha(token, base):
    """A capture identity token is a 64-hex SPIR-V digest, or a path to a dumped .spv file (hash its bytes)."""
    import re
    if re.fullmatch(r"[0-9a-fA-F]{64}", token):
        return token.lower()
    p = Path(token)
    cand = p if p.is_absolute() else base / token
    if cand.is_file():
        return hashlib.sha256(cand.read_bytes()).hexdigest()
    return token  # unresolvable -> surfaces as `extra`


def read_capture(path):
    """Return a list of (identity_token, dims) dispatch records from JSON or TSV, plus the base dir."""
    text = Path(path).read_text()
    base = Path(path).resolve().parent
    out = []
    if str(path).endswith(".json"):
        for r in json.loads(text)["dispatches"]:
            out.append((r.get("spirv_sha256") or r.get("spirv_ref"), r.get("dims")))
    else:
        for line in text.splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            p = line.split("\t")
            out.append((p[0], p[1] if len(p) > 1 else None))
    return out, base


def analyze(plan, records, base):
    first_seen, counts, unknown = {}, {}, []
    for pos, (token, dims) in enumerate(records):
        sha = resolve_sha(token, base)
        if sha in plan:
            first_seen.setdefault(sha, pos)
            counts[sha] = counts.get(sha, 0) + 1
        else:
            unknown.append((pos, sha))
    observed = set(first_seen)
    extra = sorted({s for _, s in unknown})           # dispatches not matching any pack module
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
        plan, named = load_plan(args.pack)
        if not plan:
            print("SKIP selftest: no compute modules in pack"); return 0
        prob = []
        # Mode A: raw-digest capture (every compute SHA once), ordered by entry (deterministic; NOT a schedule).
        good = [(sha, plan[sha][2]) for sha in sorted(plan, key=lambda s: plan[s][1])]
        _, _, extra, missing, _ = analyze(plan, good, Path("."))
        if extra or missing:
            prob.append(f"digest round-trip extra={len(extra)} missing={len(missing)}")
        bad = list(good); bad[0] = ("deadbeef" * 8, "8,8,1")
        _, _, extra2, _, _ = analyze(plan, bad, Path("."))
        if not extra2:
            prob.append("digest drift (unknown digest) not detected")
        # Mode B: shim-style PATH capture — hash each pack SPIR-V file's bytes to the plan digest. FG files live
        # at modules/<name>.spv; the upscaler pack's live at their manifest-relative paths (both uncompressed on disk).
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            root = Path(args.pack).resolve()
            lines = [f"{root}/{rel}\t{wgs or ''}" for (rel, wgs) in sorted(named, key=lambda x: x[0])]
            cap = Path(td) / "capture.tsv"; cap.write_text("\n".join(lines) + "\n")
            recs, bse = read_capture(cap)
            _, _, ex, mi, _ = analyze(plan, recs, bse)
            if ex or mi:
                prob.append(f"path round-trip extra={len(ex)} missing={len(mi)}")
            lines[0] = "/nope/missing.spv\t8,8,1"
            cap.write_text("\n".join(lines) + "\n")
            recs, bse = read_capture(cap)
            _, _, ex3, _, _ = analyze(plan, recs, bse)
            if not ex3:
                prob.append("path drift (unknown digest) not detected")
        if prob:
            print("selftest FAIL: " + "; ".join(prob)); return 1
        print(f"selftest PASS: digest + .spv-path round-trips clean (compute={len(plan)}); "
              "unknown-digest drift detected both modes")
        return 0

    if not args.capture or not args.pack:
        print("need --pack and --capture (or --selftest)"); return 2
    plan, _ = load_plan(args.pack)
    if not plan:
        print("SKIP: no compute modules in pack"); return 0
    recs, base = read_capture(args.capture)
    fs, cnt, extra, missing, obs = analyze(plan, recs, base)
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
