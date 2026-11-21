#!/usr/bin/env python3
"""tools/extract/frame-generation/classic_shapes.py - what each classic FG pass DECLARES per binding slot.

Phase-2c's marshalling contract, read out of the shipped SPIR-V and inventing nothing. `fg-plan.tsv` gives
each module's (set,binding,KIND,arrayed) - the LAYOUT half. It says NOTHING about the RESOURCE half: the
element type, the image dimension, the texel FORMAT, whether a slot is sampled or storage, and whether a
slot is a RUNTIME-SIZED (unbounded) array a pass can index past whatever the runtime serialized. Those
decide whether a dispatch is legal at all: the substrate binds EVERY image slot as R32G32B32A32_SFLOAT today,
so a slot whose shader declares a `%uint` image is bound to a format that CANNOT satisfy it, and every
bindless array slot the pass indexes is a slot this runtime has never filled. Measured per slot, for all 92.

Method: run the pinned SPIRV-Tools `spirv-dis` (the resolver `plan.py` uses: FSR4_SPIRV_DIS, then PATH, then
the vendored build) and read what it prints:

    %10 = OpTypeImage %float 2D 0 0 0 1 Unknown   <- <componentType> <dim> <arrayed> <MS> <depth> <sampled> <format>
    %29 = OpTypeSampler
    %28 = OpVariable %_ptr_Uniform__runtimearr_BindlessCBV Uniform
    OpDecorate %28 DescriptorSet 5   ;   OpDecorate %28 Binding 0

FORMAT is `Unknown` when the shader is BINDLESS about that image (only the descriptor the runtime writes can
supply it) or a literal like `R32ui` when the shader FIXED it (then the runtime MUST bind exactly that format).
A buffer slot appears as an unbounded array of a named struct (`__runtimearr_BindlessCBV`) - the bindless
indirection the pack's classic passes are built on. Image use-sites (`OpImageFetch/Sample/Read/Write`) are
counted, so a slot's role is MEASURED rather than inferred from the layout.

Run: python3 tools/extract/frame-generation/classic_shapes.py [--out build/fg-classic-shapes.tsv]
     [--expect-modules=92 --expect-slots=N --expect-typed-slots=N --expect-uint-slots=N ...]
GPU-free; prints SKIP and exits 0 when the pack modules or spirv-dis are absent.
"""
from __future__ import annotations
import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PACK = ROOT / "assets/frame-generation/pack"
PLAN_TSV = PACK / "fg-plan.tsv"
MODULES_DIR = PACK / "modules"

# OpTypeImage %<id> %<componentType> <dim> <arrayed> <MS> <depth> <sampled> <format>
IMAGE_RE = re.compile(r"^(%\S+) = OpTypeImage (%\S+) (\w+) (\d+) (\d+) (\d+) (\d+) (\S+)$")
SAMPLER_RE = re.compile(r"^%\S+ = OpTypeSampler$")
VAR_RE = re.compile(r"^(%\S+) = OpVariable (%\S+) (\w+)(?: (%\S+))?$")
PTR_RE = re.compile(r"^(%\S+) = OpTypePointer (\w+) (%\S+)$")
DEC_RE = re.compile(r"^OpDecorate (%\S+) (\w+) ?(\S*)?")
RTARR_RE = re.compile(r"^(%\S+) = OpTypeRuntimeArray (%\S+)$")
ARR_RE = re.compile(r"^(%\S+) = OpTypeArray (%\S+) (%\S+)$")
STRUCT_RE = re.compile(r"^(%\S+) = OpTypeStruct (.*)$")
USE_RE = re.compile(r"= (OpImageFetch|OpImageSample\w*|OpImageRead|OpImageWrite)\b")
FLAGS = ("modules", "slots", "typed_slots", "bindless_slots", "uint_slots", "float_slots",
         "unbounded_slots", "write_modules", "plan_shader_agreement", "mismatched_format_slots")


def parse_slots(text):
    """Every (set,binding) the SHADER declares, plus the type it says that slot must hold.

    A slot the plan declares but the shader does not is a slot NO binding can be legal for; the other way
    round is a plan that under-declares. Both directions are counted and reported, neither smoothed over.
    """
    types, ptrs, vars_, uses = {}, {}, {}, Counter()
    for raw in text.splitlines():
        line = raw.strip()
        m = IMAGE_RE.match(line)
        if m:
            types[m.group(1)] = {"class": "image", "elem": m.group(2), "dim": m.group(3), "arrayed": m.group(4),
                                 "ms": m.group(5), "depth": m.group(6), "sampled": m.group(7), "format": m.group(8)}
            continue
        if SAMPLER_RE.match(line):
            types[line.split(" = ")[0]] = {"class": "sampler"}
            continue
        m = STRUCT_RE.match(line)
        if m:
            types[m.group(1)] = {"class": "struct", "members": m.group(2)}
            continue
        m = RTARR_RE.match(line)
        if m:
            types[m.group(1)] = {"class": "runtime_array", "of": m.group(2)}
            continue
        m = ARR_RE.match(line)
        if m:
            types[m.group(1)] = {"class": "array", "of": m.group(2), "n": m.group(3)}
            continue
        m = PTR_RE.match(line)
        if m:
            ptrs[m.group(1)] = {"storage": m.group(2), "of": m.group(3)}
            continue
        m = VAR_RE.match(line)
        if m:
            vars_.setdefault(m.group(1), {}).update({"pointer": m.group(2), "storage": m.group(3)})
            continue
        m = DEC_RE.match(line)
        if m and m.group(2) in ("DescriptorSet", "Binding"):
            key = "set" if m.group(2) == "DescriptorSet" else "binding"
            vars_.setdefault(m.group(1), {})[key] = int(m.group(3))
            continue
        m = USE_RE.search(line)
        if m:
            uses[m.group(1)] += 1
    out = {}
    for var, d in vars_.items():
        if "set" not in d or "binding" not in d:
            continue  # Private/Function variables and decoration groups are not bound slots
        p = ptrs.get(d.get("pointer"))
        if not p:
            continue
        pointee, chain, seen = p["of"], [], set()
        while pointee in types and pointee not in seen:
            seen.add(pointee)
            t = types[pointee]
            chain.append(t)
            if t["class"] in ("runtime_array", "array"):
                pointee = t["of"]
            else:
                break
        head = chain[-1] if chain else {}
        kind = {"image": "image", "sampler": "sampler"}.get(head.get("class"), "buffer")
        out[(d["set"], d["binding"])] = {
            "storage": p["storage"], "kind": kind,
            "unbounded": any(t["class"] == "runtime_array" for t in chain),
            "const_n": next((t["n"] for t in chain if t["class"] == "array"), "-"),
            "dim": head.get("dim", "-"), "elem": head.get("elem", "-"),
            "sampled": head.get("sampled", "-"), "format": head.get("format", "-"),
            "struct": (next((t["of"] for t in chain if t["class"] in ("runtime_array", "array")), None)
                       or (pointee if kind == "buffer" else "-")),
        }
    return out, uses


def plan_slots(row):
    """The plan's OWN layout half for one module, in the grammar plan.py serializes and the provider parses."""
    out = {}
    for token in (row["groups"] or "").split(";"):
        if not token:
            continue
        parts = token.split(":")
        if len(parts) < 3:
            continue
        out[(int(parts[0]), int(parts[1]))] = {"kind": parts[2], "arrayed": len(parts) > 3}
    return out




def resolve_spirv_dis():
    env = os.environ.get("FSR4_SPIRV_DIS")
    if env and Path(env).is_file():
        return env
    which = shutil.which("spirv-dis")
    if which:
        return which
    cand = ROOT / "build/tools/SPIRV-Tools/bin/spirv-dis"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="build/fg-classic-shapes.tsv")
    for flag in FLAGS:
        ap.add_argument(f"--expect-{flag.replace('_', '-')}", type=int, default=None,
                        help=f"pinned expectation for the measured {flag} count")
    a = ap.parse_args()
    if not (PLAN_TSV.is_file() and MODULES_DIR.is_dir()):
        print("SKIP classic-shapes: pack plan or module bytes absent (run the asset workflow first)")
        return 0
    dis = resolve_spirv_dis()
    if dis is None:
        print("SKIP classic-shapes: no spirv-dis (set FSR4_SPIRV_DIS or install SPIRV-Tools)")
        return 0
    rows = list(csv.DictReader(PLAN_TSV.open(newline=""), delimiter="\t"))
    cl = sorted((r for r in rows if r["stage"] == "compute-classic"), key=lambda r: int(r["order"]))
    census, m = [], Counter()
    for r in cl:
        path = MODULES_DIR / r["name"]
        if not path.is_file():
            print(f"FAIL classic module bytes absent: {r['name']}")
            return 2
        text = subprocess.run([dis, str(path)], capture_output=True, text=True).stdout
        declared, uses = parse_slots(text)
        planned = plan_slots(r)
        m["modules"] += 1
        m["plan_shader_agreement"] += sum(1 for k in planned if k in declared)
        m["plan_only"] += sum(1 for k in planned if k not in declared)
        m["shader_only"] += sum(1 for k in declared if k not in planned)
        if uses.get("OpImageWrite", 0) or uses.get("OpImageRead", 0):
            m["write_modules"] += 1
        for d in declared.values():
            m["slots"] += 1
            if d["format"] not in ("-", "Unknown"):
                m["typed_slots"] += 1
            elif d["format"] == "Unknown":
                m["bindless_slots"] += 1
                if d["kind"] == "image" and d["elem"] in ("%uint", "%int"):
                    # THE bug this census exists to catch: an image slot typed %uint/%int can NEVER be
                    # satisfied by the R32G32B32A32_SFLOAT backing the substrate allocates for EVERY image
                    # slot today, and this is exactly the class of binding the plan's own kind token cannot
                    # tell apart (both are `SI`/`CIS`).
                    m["mismatched_format_slots"] += 1
            if d["elem"] in ("%uint", "%int"):
                m["uint_slots"] += 1
            elif d["elem"] == "%float":
                m["float_slots"] += 1
            if d["unbounded"]:
                m["unbounded_slots"] += 1
        for key in sorted(set(planned) | set(declared)):
            p, d = planned.get(key), declared.get(key)
            row = {"order": r["order"], "module": r["name"], "set": key[0], "binding": key[1],
                   "plan_kind": (p or {}).get("kind", "-"),
                   "plan_arrayed": "a" if (p or {}).get("arrayed") else "-",
                   "spv_kind": (d or {}).get("kind", "-"),
                   "spv_storage": (d or {}).get("storage", "-"),
                   "dim": (d or {}).get("dim", "-"), "elem": (d or {}).get("elem", "-"),
                   "sampled": (d or {}).get("sampled", "-"), "format": (d or {}).get("format", "-"),
                   "unbounded": ("runtime" if (d or {}).get("unbounded") else
                                 (f"const{(d or {}).get('const_n')}"
                                  if (d or {}).get("const_n") not in ("-", None) else "-")),
                   "struct": (d or {}).get("struct", "-"),
                   "fetch": uses.get("OpImageFetch", 0),
                   "sample": sum(v for k, v in uses.items() if k.startswith("OpImageSample")),
                   "read": uses.get("OpImageRead", 0), "write": uses.get("OpImageWrite", 0)}
            fmt = row["format"]
            row["format_state"] = ("literal" if fmt not in ("-", "Unknown") else
                                   "bindless" if fmt == "Unknown" else "-")
            census.append(row)
    return report(a, census, m)

    return str(cand) if cand.is_file() else None


def report(a, census, m):
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["order", "module", "set", "binding", "plan_kind", "plan_arrayed", "spv_kind", "spv_storage",
            "dim", "elem", "sampled", "format", "format_state", "unbounded", "struct", "fetch", "sample",
            "read", "write"]
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        w.writerows(census)
    print(f"classic modules covered: {m['modules']}   declared (set,binding) slots: {m['slots']}")
    print(f"slot join vs the plan: (set,binding) agree {m['plan_shader_agreement']} "
          f"plan-only {m['plan_only']} shader-only {m['shader_only']}")
    print(f"format contract: literal {m['typed_slots']} bindless(Unknown) {m['bindless_slots']} "
          f"| element %float {m['float_slots']} %uint|int {m['uint_slots']}")
    print(f"unbounded(runtime-array) slots: {m['unbounded_slots']}")
    print(f"bindless %uint|int IMAGE slots an SFLOAT backing CANNOT satisfy: {m['mismatched_format_slots']}")
    print(f"modules with a read or write image op: {m['write_modules']}")
    print(f"formats seen: {dict(Counter(r['format'] for r in census if r['format'] != '-'))}")
    print(f"image dims: {dict(Counter(r['dim'] for r in census if r['spv_kind'] == 'image'))}")
    print(f"sampled field: {dict(Counter(r['sampled'] for r in census if r['spv_kind'] == 'image'))}")
    print(f"kinds: {dict(Counter(r['spv_kind'] for r in census))}")
    print(f"op totals: fetch={sum(r['fetch'] for r in census)} sample={sum(r['sample'] for r in census)} "
          f"read={sum(r['read'] for r in census)} write={sum(r['write'] for r in census)}")
    print(f"wrote {out}: {len(census)} per-slot rows")
    for flag in FLAGS:
        want = getattr(a, f"expect_{flag}")
        if want is not None and want != m[flag]:
            print(f"FAIL {flag}: measured {m[flag]}, expected {want}")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

