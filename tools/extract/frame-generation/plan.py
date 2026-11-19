#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 fsr4-vk contributors.
# See LICENSE in the project root.

"""Frame Generation pass-plan reconstruction (roadmap Phase 0).

The Frame Generation asset pack stores every program content-addressed (a hashed
`<hash>.spv`) with only its SPIR-V entry name and Vulkan pipeline layout kept in
`pack/modules.tsv` + `pack/pipelines.tsv`. That tells us WHICH modules exist and
HOW each one is bound, but it does NOT assemble them into an ordered runtime plan,
and it does NOT say what intermediate render targets the plan needs. The provider
currently refuses `ffxDispatch(Prepare|Frame Generation)` precisely because that
plan substrate is absent (see README host-integration boundary).

This tool builds that substrate as an explicit, machine-checked artefact:

  * `fg-plan.tsv`  one row per compute module (the 474 dispatchable ones: 382
    `compute-neural` mlfi inference + 92 `compute-classic` interpolation). Carries
    the module's de-duplicated Vulkan binding census, serialized with the EXACT
    grammar `tools/framegeneration-exec.cpp` uses to build real descriptor-set
    layouts, so a downstream C++ cross-check can prove the two independent parsers
    agree (the drift gate, mirroring the "one helper, two call paths" rule used by
    cap-probe).

  * `fg-plan.json`  the summary the provider and the parity gate read: coverage
    counts, per-module transport class, a grouped execution scaffold (an ORDER,
    NOT a claim to match AMD's DLL order), the effect-owned intermediate
    render-target model + its byte total (what the provider will allocate, and
    therefore what `ffxQuery(…GetGPUMemoryUsage)` will honestly report in Phase 3), and a
    CooperativeMatrix census — per module a `coop` column of its de-duplicated `CooperativeMatrixKHR`
    operand shapes (compType:scope:rows:cols:K), plus the aggregate distinct-shape/`modules_using`/
    per-tier counts. This recovers the DLL's "CsTensorSizes" (the tile geometry the 156 ML neural
    modules run on) as a machine-checked fact from the shipped SPIR-V, independently re-derived in C++.

HONEST BOUNDARY: the ORDER is a reconstructed, deterministic grouping (classic
passes first, then mlfi tier-grouped with pass indices ascending). It is validated
for structure (bijection, tier grouping, determinism), NOT for pixel-level equality
with AMD's Frame Generation DLL, which is not verifiable on a host without reference
captures (the 12 graphics/present + PSV modules are not compute-dispatchable either).
Recovering the CooperativeMatrix operand shapes (CsTensorSizes) makes the ML graph's
TILE GEOMETRY a known, gated fact — but it is NOT the same as recovering the graph's
dependency order, the intermediate-target allocation or the swapchain pacing, all of
which the provider still requires before it can honestly run `ffxDispatch`; it continues
to refuse dispatch (runtime error) until those are reconstructed and reference-validated.
"""

import argparse
import collections
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

# The descriptor-kind abbreviations the executor's vkTypeFor() recognizes; any other
# token kind (or a token with fewer than three fields) means "unresolved" ('U'), and
# a (set,binding) slot claimed by two DIFFERENT kinds is a cross-kind collision that
# needs the buffer-device-address unified transport.
KINDS = {"SB", "UB", "CIS", "SI", "SMP"}

ENTRY_MLFI = re.compile(r"mlfi_(.+)_pass(\d+)$")

# CooperativeMatrix operand-shape grammar (the DLL's "CsTensorSizes"). SPIRV-Tools prints
#   OpTypeCooperativeMatrixKHR %<comp> %uint_<scope> %uint_<rows> %uint_<cols> %uint_<K>
# where scope/rows/cols/K are %uint_ integer literals and <comp> is a scalar shorthand (%half,
# %float, %uchar ...). Capturing these shapes turns "which tile geometry the ML graph runs on"
# from an assumption into a recovered, machine-checkable fact.
COOP_SHAPE = re.compile(
    r"OpTypeCooperativeMatrixKHR\s+%(\w+)\s+%uint_(\d+)\s+%uint_(\d+)\s+%uint_(\d+)\s+%uint_(\d+)")


def resolve_spirv_dis():
    env = __import__("os").environ.get("FSR4_SPIRV_DIS")
    if env and Path(env).is_file():
        return env
    which = shutil.which("spirv-dis")
    if which:
        return which
    root = Path(__file__).resolve().parents[3]
    cand = root / "build/tools/SPIRV-Tools/bin/spirv-dis"
    return str(cand) if cand.is_file() else None


def coop_shapes(asm_text):
    """Canonical sorted DISTINCT shape tokens 'comp:scope:rows:cols:K' for one module."""
    return sorted({"%s:%s:%s:%s:%s" % sh for sh in COOP_SHAPE.findall(asm_text)})


def parse_layout(layout):
    """De-duplicate a pipelines.tsv `layout` string into binding groups.

    Grammar (identical to tools/framegeneration-exec.cpp): tokens are ';'-separated;
    each is ':'-separated `set:binding:KIND[:a]`. Return (groups, order, unresolved,
    unified) where groups is a dict {(set,binding): (kind, arrayed)} and `order` is
    the first-seen (set,binding) sequence.
    """
    groups = {}
    order = []
    unresolved = False
    unified = False
    for token in layout.split(";"):
        if not token:
            continue
        fields = token.split(":")
        if len(fields) < 3 or fields[2] not in KINDS:
            # 'U'/short token: the layout is not statically resolvable here.
            unresolved = True
            continue
        sset, bnd, kind = fields[0], fields[1], fields[2]
        arrayed = len(fields) >= 4 and fields[3] == "a"
        key = (int(sset), int(bnd))
        if key in groups:
            if groups[key][0] != kind:
                unified = True
            groups[key] = (kind, groups[key][1] or arrayed)
        else:
            groups[key] = (kind, arrayed)
            order.append(key)
    return groups, order, unresolved, unified


def canonical_groups(groups, order):
    """Serialize the de-duplicated groups into a canonical ';'-joined string.

    Sort the group keys by (set, binding) so the two parsers (this Python one and
    the executor's C++ one) compare on a stable order, not file order.
    """
    parts = []
    for (sset, bnd) in sorted(order):
        kind, arrayed = groups[(sset, bnd)]
        tok = f"{sset}:{bnd}:{kind}" + (":a" if arrayed else "")
        parts.append(tok)
    return ";".join(parts)


def bytes_per_texel(fmt):
    """Best-effort texel size for the reference back-buffer format (design only)."""
    table = {
        "VK_FORMAT_R8G8B8A8_UNORM": 4, "VK_FORMAT_B8G8R8A8_UNORM": 4,
        "VK_FORMAT_R8G8B8A8_SRGB": 4, "VK_FORMAT_R8G8B8_UNORM": 3,
        "VK_FORMAT_R16G16B16A16_SFLOAT": 8, "VK_FORMAT_R16G16B16A16_UNORM": 8,
        "VK_FORMAT_R11G11B10_FLOAT": 4, "VK_FORMAT_R10G10B10A2_UNORM": 4,
        "VK_FORMAT_R16G16_SFLOAT": 4, "VK_FORMAT_R32G32_SFLOAT": 8,
        "VK_FORMAT_R32G32B32A32_SFLOAT": 16, "VK_FORMAT_BC7_UNORM": 1,
    }
    return table.get(fmt, 4)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pack", type=Path, required=True,
                    help="dir holding pack/modules.tsv + pack/pipelines.tsv")
    ap.add_argument("--max-render", type=int, nargs=2, metavar=("W", "H"),
                    default=(1920, 1080), help="reference max render size")
    ap.add_argument("--max-upscale", type=int, nargs=2, metavar=("W", "H"),
                    default=(3840, 2160), help="reference max upscale size")
    ap.add_argument("--backbuffer-format", default="VK_FORMAT_B8G8R8A8_UNORM")
    ap.add_argument("--out-prefix", type=Path, default=None,
                    help="base path for fg-plan.{tsv,json} (default <pack>/fg-plan)")
    args = ap.parse_args(argv)

    pack = args.pack
    pipelines = pack / "pipelines.tsv"
    modules = pack / "modules.tsv"
    if not pipelines.is_file() or not modules.is_file():
        sys.exit(f"plan.py: {pipelines} or {modules} missing (run the FG converter first)")

    # modules.tsv: name stage entry w0,w1,w2 spirv_sha256
    workgroup = {}
    with modules.open(encoding="utf-8") as fh:
        mrows = fh.read().splitlines()
    for line in mrows[1:]:
        cols = line.split("\t")
        if len(cols) < 4:
            continue
        wparts = cols[3].split(",")
        while len(wparts) < 3:
            wparts.append("1")
        workgroup[cols[0]] = (wparts[0], wparts[1], wparts[2])

    # pipelines.tsv: name stage entry push_bytes collisions standardizable layout transport
    recs = []
    with pipelines.open(encoding="utf-8") as fh:
        prows = fh.read().splitlines()
    for line in prows[1:]:
        cols = line.split("\t")
        if len(cols) < 8:
            continue
        name, stage, entry, transport = cols[0], cols[1], cols[2], cols[7]
        if stage not in ("compute-neural", "compute-classic"):
            continue  # 12 graphics/present/PSV modules are not compute-dispatchable
        groups, order, unresolved, unified = parse_layout(cols[6])
        if stage == "compute-neural":
            m = ENTRY_MLFI.search(entry)
            if not m:
                sys.exit(f"plan.py: neural entry does not match mlfi_<tier>_pass<NNN>: {entry}")
            tier, index = m.group(1), int(m.group(2))
        else:
            tier, index = "classic", None
        wg = workgroup.get(name, ("32", "1", "1"))
        recs.append({
            "name": name, "stage": stage, "entry": entry, "tier": tier, "pass": index,
            "push_bytes": cols[3], "transport": transport,
            "resolved": (not unresolved and not unified),
            "unresolved": unresolved, "unified": unified,
            "groups": canonical_groups(groups, order),
            "binding_count": len(groups),
            "w0": wg[0], "w1": wg[1], "w2": wg[2],
        })

    # Grouped execution scaffold (an ORDER, not an AMD-order claim). Classic first
    # in pack order, then mlfi grouped by tier (tier order = first pack occurrence)
    # with pass index ascending within a tier.
    classic = [r for r in recs if r["stage"] == "compute-classic"]
    neural = [r for r in recs if r["stage"] == "compute-neural"]
    tiers_seen = {}
    for r in neural:
        tiers_seen.setdefault(r["tier"], len(tiers_seen))
    neural_sorted = sorted(neural, key=lambda r: (tiers_seen[r["tier"]], r["pass"]))
    ordered = classic + neural_sorted
    for i, r in enumerate(ordered):
        r["order"] = i

    # Recover the CooperativeMatrix operand shapes the DLL's ML inference runs on (its "CsTensorSizes")
    # per compute module, straight from the shipped SPIR-V. GPU-free: no DLL, no device, no
    # reference capture. This is a recovered substrate fact that is machine-checked against an
    # independent C++ re-derivation (framegeneration-plan-test); on its own it does NOT enable
    # ffxDispatch -- the ML-graph dependency/order, intermediate-target allocation and swapchain
    # pacing are still unreconstructed, so the provider continues to refuse dispatch honestly.
    spirv_dis = resolve_spirv_dis()
    coop_modules_using = 0
    coop_classic_using = 0
    distinct = collections.Counter()
    per_mod_count = collections.Counter()
    tier_using = collections.Counter()
    tier_geometry = collections.defaultdict(list)
    if spirv_dis:
        for r in ordered:
            shader = pack / "modules" / r["name"]
            try:
                asm = subprocess.run([spirv_dis, str(shader)], capture_output=True,
                                     text=True, check=True).stdout
            except Exception:
                r["coop"] = "<DISFAIL>"
                continue
            shapes = coop_shapes(asm)
            r["coop"] = ";".join(shapes)
            if shapes:
                coop_modules_using += 1
                per_mod_count[len(shapes)] += 1
                for sh in shapes:
                    distinct[sh] += 1
                if r["tier"] == "classic":
                    coop_classic_using += 1
                else:
                    tier_using[r["tier"]] += 1
                    tier_geometry[r["tier"]].append(";".join(shapes))
    else:
        for r in ordered:
            r["coop"] = ""

    render_w, render_h = args.max_render
    up_w, up_h = args.max_upscale
    bpp = bytes_per_texel(args.backbuffer_format)
    # Effect-owned intermediate render targets the provider allocates (design; Phase
    # 1 allocates these, Phase 3 sums them into the honest GPU-memory-usage report).
    targets = [
        ("optical_flow", render_w, render_h, 4),
        ("scene_change_detection", render_w, render_h, 4),
        ("scene_color_intermediate", render_w, render_h, bpp),
        ("history_color", render_w, render_h, bpp),
        ("upscale_intermediate", up_w, up_h, bpp),
    ]
    intermediate = [{"role": r, "width": w, "height": h, "bytes_per_texel": b,
                     "bytes": w * h * b} for (r, w, h, b) in targets]
    intermediate_bytes = sum(t["bytes"] for t in intermediate)

    prefix = args.out_prefix or (pack / "fg-plan")
    tsv = Path(str(prefix) + ".tsv")
    with tsv.open("w", encoding="utf-8") as fh:
        fh.write("name\tstage\tentry\ttier\tpass\torder\ttransport\tresolved\t"
                 "binding_count\tgroups\tw0,w1,w2\tcoop\n")
        for r in ordered:
            fh.write("\t".join([
                r["name"], r["stage"], r["entry"], r["tier"],
                "" if r["pass"] is None else str(r["pass"]), str(r["order"]),
                r["transport"], "1" if r["resolved"] else "0", str(r["binding_count"]),
                r["groups"], f'{r["w0"]},{r["w1"]},{r["w2"]}', r["coop"],
            ]) + "\n")

    summary = {
        "component": "frame-generation",
        "schema": 1,
        "compute_modules": len(recs),
        "neural": len(neural),
        "classic": len(classic),
        "distinct_binding_groups_total": sum(r["binding_count"] for r in recs),
        "resolved": sum(1 for r in recs if r["resolved"]),
        "unified": sum(1 for r in recs if r["unified"]),
        "unresolved": sum(1 for r in recs if r["unresolved"]),
        "transport": dict(collections.Counter(r["transport"] for r in recs)),
        "tiers": dict(collections.Counter(r["tier"] for r in neural)),
        "order_basis": "grouped-scaffold (classic-first, mlfi tier-grouped pass-ascending); "
                       "NOT asserted to match AMD Frame Generation DLL execution order",
        "reference_max_render": [render_w, render_h],
        "reference_max_upscale": [up_w, up_h],
        "backbuffer_format": args.backbuffer_format,
        "intermediate_targets": intermediate,
        "intermediate_bytes_total": intermediate_bytes,
        "cooperative_extractor": spirv_dis,
        "cooperative_matrices": ({
            "modules_using": coop_modules_using,
            "classic_using": coop_classic_using,
            "distinct_shapes": [{"shape": k, "modules": v}
                                for (k, v) in sorted(distinct.items(),
                                                     key=lambda kv: (-kv[1], kv[0]))],
            "per_module_distinct_shape_count": {str(k): v
                                                for (k, v) in sorted(per_mod_count.items())},
            "neural_tiers_using": {t: tier_using[t] for t in sorted(tier_using)},
            "tier_geometry_identical_across_tiers": (
                len(tier_geometry) >= 2 and
                all(sorted(tier_geometry[t]) ==
                    sorted(tier_geometry[sorted(tier_geometry)[0]]) for t in tier_geometry)),
        } if spirv_dis else None),
    }
    js = Path(str(prefix) + ".json")
    with js.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
        fh.write("\n")

    print("fg-plan: compute=%d neural=%d classic=%d groups=%d resolved=%d unified=%d "
          "unresolved=%d" % (
              len(recs), len(neural), len(classic),
              summary["distinct_binding_groups_total"], summary["resolved"],
              summary["unified"], summary["unresolved"]))
    print("fg-plan: intermediate_bytes_total=%d (%d targets)" % (
        intermediate_bytes, len(intermediate)))
    if spirv_dis:
        print("fg-plan: cooperative-matrix modules=%d (classic=%d) distinct_shapes=%d tiers=%s "
              "per-module-shape-counts=%s" % (
                  coop_modules_using, coop_classic_using, len(distinct),
                  dict(tier_using), dict(per_mod_count)))
    else:
        print("fg-plan: cooperative-matrix extraction SKIPPED (spirv-dis not found)")
    print("fg-plan: wrote %s and %s" % (tsv, js))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


