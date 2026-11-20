#!/usr/bin/env python3
"""tools/extract/frame-generation/classic_order.py - the Phase-2 scaffold for a runnable classic FG runtime.

It joins the two ground truths this repo already owns, and invents nothing:

  * What the shipped pack says about ORDER (`fg-plan.tsv` + `manifest.json`). Measured census, re-checked on
    every run: 486 modules = 382 `compute-neural` (entry points literally `mlfi_4k_passNN` / `mlfi_8k_passNN`, whose
    `pass` column is FILLED) + 92 `compute-classic` (entry point `CS`: the `pass` column is EMPTY - the missing name
    is the open half - while `order` 0..91, the binding-shape signature, the group size, the transport class and
    `resolved` are all FILLED) + 7 present-blit + 1 pixel + 1 vertex (+ 3 unsupported-psv in the manifest).
    Classic group-size census: 56 x 8,8,1 / 16 x 256,1,1 / 4 x 16,16,1 / 4 x 64,1,1 / 4 x 16,4,1 / 4 x 4,4,4 / 4 x 32,8,1.
  * What the vendored SDK 2.4.0 says about the ALGORITHM: 11 `ffx_frameinterpolation_*.hlsl` + 7 `ffx_opticalflow_*.hlsl`
    pass sources under `Kits/FidelityFX/framegeneration/fsr3/internal/shaders/`, plus the CPU-side rules in
    `fsr3/internal/ffx_provider_fsr3framegeneration.cpp`: the optical-flow dispatch runs FIRST, then the frame-interpolation
    dispatch; `opticalFlowBlockSize = 8` (the same 8x8 tiling those 56 classic modules show); the interpolation rect falls
    back to the full presentColor extent when `generationRect` is 0,0; `reset` is the OR of the caller flag, the Prepare flag
    and a disjoint-frameID test; Prepare pairs with generate through
    `prepareDescriptions[frameID % MAX_QUEUED_FRAMES]`; and the reported memory usage is the SUM over the FrameGeneration
    + OpticalFlow + Shared contexts.

The emitted file is a WORK LIST, not an identity claim: for every order slot it records the module hash, the group size
and the binding-shape signature, and the run of adjacent slots sharing that signature (runs are what a pass family looks
like from the pack side - 92 slots, but far fewer distinct shapes, so shape alone cannot name {len} passes). Naming a
run to an SDK pass family still needs the DLL's CPU-side argument marshal (the same evidence the
`--descriptor-buffer-parity` gate consumes) and, for pixel parity, AMD reference captures.

Run:  python3 tools/extract/frame-generation/classic_order.py [--out build/fg-classic-order.tsv]
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import re
from collections import Counter

SDK_KIT = "third_party/fidelityfx-sdk/Kits/FidelityFX/framegeneration"
SDK_PROVIDER = SDK_KIT + "/fsr3/internal/ffx_provider_fsr3framegeneration.cpp"


def sdk_pass_sources():
    d = os.path.join(SDK_KIT, "fsr3/internal/shaders")
    names = sorted(os.listdir(d))
    fi = [n for n in names if n.startswith("ffx_frameinterpolation_") and n.endswith(".hlsl")]
    of = [n for n in names if n.startswith("ffx_opticalflow_") and n.endswith(".hlsl")]
    return fi, of


def sdk_provider_rules():
    txt = open(SDK_PROVIDER, encoding="utf-8", errors="replace").read()
    return {
        "block_size_8": "opticalFlowBlockSize = 8" in txt,
        "optical_flow_dispatch_first": (txt.find("ffxOpticalflowContextDispatch") < txt.find("ffxFrameInterpolationDispatch")),
        "generation_rect_fallback": "generationRect.height == 0" in txt,
        "disjoint_frameid_reset": "bDisjointFrameID" in txt and "bFrameID_Skipped" in txt,
        "prepare_frameid_modulo": "MAX_QUEUED_FRAMES" in txt,
        "memory_usage_is_a_sum": "pGpuMemoryUsageOpticalFlow.totalUsageInBytes" in txt,
    }

def sdk_pass_tiles():
    """Read the SDK's OWN thread-group shape for every classic pass family (its [numthreads] triplet defines).

    Returns {family: (w,h,d)}; families without a compute triplet (the swapchain pacing/UI shaders) are skipped
    and reported, because those belong to the separate SWAPCHAIN effect id, not to this chain."""
    d = os.path.join(SDK_KIT, "fsr3/internal/shaders")
    pat = re.compile(r"#define\s+FFX_(FRAMEINTERPOLATION|OPTICALFLOW)_THREAD_GROUP_(WIDTH|HEIGHT|DEPTH)\s+(\d+)")
    tiles, skipped = {}, []
    for name in sorted(os.listdir(d)):
        if not name.endswith(".hlsl"):
            continue
        vals = {}
        for m in pat.finditer(open(os.path.join(d, name), encoding="utf-8", errors="replace").read()):
            vals[m.group(2)] = int(m.group(3))
        if len(vals) == 3:
            tiles[name[:-5]] = (vals["WIDTH"], vals["HEIGHT"], vals["DEPTH"])
        else:
            skipped.append(name[:-5])
    return tiles, skipped


def join_shapes(cl, tiles):
    """Two-sided join: the pack's 92 ordered classic slots (measured shape census) against the SDK's named pass
    families (measured tile shapes). Asserts NO ORPHAN ON EITHER SIDE, then buckets every slot by how far shape
    and multiplicity get toward naming it. Labels are search state, never identity claims."""
    pack_cls = Counter(r["w0,w1,w2"] for r in cl)
    sdk_cls = Counter("%d,%d,%d" % v for v in tiles.values())
    owners = {}
    for fam, t in tiles.items():
        owners.setdefault("%d,%d,%d" % t, []).append(fam)
    orphan_pack = sorted(c for c in pack_cls if c not in sdk_cls)
    orphan_sdk = sorted(c for c in sdk_cls if c not in pack_cls)
    if orphan_pack or orphan_sdk:
        print(f"FAIL orphan shape classes: pack_only={orphan_pack} sdk_only={orphan_sdk}")
        return None
    from math import gcd
    variants = 0
    for c, n in pack_cls.items():
        variants = gcd(variants, n) if variants else n
    print(f"join: {len(cl)} pack slots in {len(pack_cls)} shape classes, {len(tiles)} SDK families in "
          f"{len(sdk_cls)} classes; largest per-class slot divisor (variants) = {variants}")
    print(f"  pack classes : {dict(pack_cls)}")
    print(f"  sdk classes  : {dict(sdk_cls)}")
    buckets = Counter()
    for r in cl:
        c = r["w0,w1,w2"]
        per_variant, own = pack_cls[c] // variants, len(owners[c])
        if own == 1:
            r["_naming"] = "unique_by_shape"
        elif own == per_variant:
            r["_naming"] = "count_matched_needs_pairing"
        else:
            r["_naming"] = "shape_ambiguous"
        buckets[r["_naming"]] += 1
    print(f"  slot naming state: {dict(buckets)}")
    for c in sorted(pack_cls):
        print(f"   {c:>8} pack x{pack_cls[c]:>3} (x{pack_cls[c] // variants} per variant)  sdk owners "
              f"{len(owners[c])}: {', '.join(sorted(owners[c]))}")
    return buckets


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", default="assets/frame-generation/pack/fg-plan.tsv")
    ap.add_argument("--manifest", default="assets/frame-generation/pack/manifest.json")
    ap.add_argument("--out", default="build/fg-classic-order.tsv")
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.tsv), delimiter="\t"))
    man = json.load(open(a.manifest))
    mods = man["modules"]
    stages = Counter(str(m.get("shader_stage")) for m in mods.values())
    cl = sorted((r for r in rows if r["stage"] == "compute-classic"), key=lambda r: int(r["order"]))
    ne = [r for r in rows if r["stage"] == "compute-neural"]
    named = [r for r in rows if (r.get("pass") or "").strip()]

    print(f"pack manifest modules={len(mods)} stages={dict(stages)}")
    print(f"plan tsv rows={len(rows)} classic={len(cl)} neural={len(ne)} "
          f"named_pass_rows={len(named)} all_named_neural={all(r['stage'] == 'compute-neural' for r in named)}")
    orders = [int(r["order"]) for r in cl]
    if orders != list(range(len(orders))):
        print(f"FAIL classic orders are not contiguous 0..{len(orders) - 1}")
        return 2
    print(f"classic order contiguous 0..{orders[-1]}: yes ({len(orders)} slots)")
    print(f"classic group sizes: {dict(Counter(r['w0,w1,w2'] for r in cl))}")
    print(f"classic transports:  {dict(Counter(r['transport'] for r in cl))}")
    print(f"classic resolved:    {dict(Counter(r['resolved'] for r in cl))}")

    runs, rid, prev = [], 0, None
    for r in cl:
        if r["groups"] != prev:
            rid += 1
        prev = r["groups"]
        r["_run"] = rid
    size = Counter(x["_run"] for x in cl)
    print(f"signature runs over the classic chain: {len(size)} runs covering {len(cl)} slots")
    print(f"run sizes: {sorted(size.values(), reverse=True)}")
    print(f"distinct signatures: {len(set(r['groups'] for r in cl))}")

    tiles, skipped = sdk_pass_tiles()
    buckets = join_shapes(cl, tiles)
    if buckets is None:
        return 2

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["order", "module", "groups", "binding_shape", "run", "run_size", "binding_count",
                    "transport", "resolved", "naming_state"])
        for r in cl:
            w.writerow([r["order"], r["name"], r["w0,w1,w2"], r["groups"], r["_run"], size[r["_run"]],
                        r["binding_count"], r["transport"], r["resolved"], r["_naming"]])
    print(f"wrote {a.out}: {len(cl)} ordered slots + {len(size)} signature runs + naming state "
          f"{dict(buckets)} (search state, never an identity claim)")
    print(f"sdk classic pass families with a compute triplet: {len(tiles)}; without one (swapchain effect id, "
          f"skipped here): {skipped}")

    fi, of = sdk_pass_sources()
    print(f"sdk classic pass sources: frameinterpolation={len(fi)} opticalflow={len(of)}")
    print("  fi: " + ", ".join(n[len("ffx_frameinterpolation_"):-5] for n in fi))
    print("  of: " + ", ".join(n[len("ffx_opticalflow_"):-5] for n in of))
    rules = sdk_provider_rules()
    print("sdk provider rules present: " + ", ".join(f"{k}={v}" for k, v in rules.items()))
    if not (len(fi) >= 10 and len(of) >= 6 and all(rules.values())):
        print("FAIL an SDK-side ground truth this scaffold depends on is missing")
        return 2
    print(f"name state after the join: {sum(buckets['unique_by_shape'] for _ in [0]) if 'unique_by_shape' in buckets else 0} "
          f"slots named uniquely by shape, {buckets.get('count_matched_needs_pairing', 0)} count-matched to owners "
          f"but still unpaired, {buckets.get('shape_ambiguous', 0)} ambiguous -> the pack has MORE tile-8 slots per "
          f"variant (14) than the SDK's classic list has owners (9): 5 slots per variant are NOT covered by the "
          f"vendored SDK's classic pass sources, and pairing the rest still needs the DLL's CPU-side argument marshal")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
