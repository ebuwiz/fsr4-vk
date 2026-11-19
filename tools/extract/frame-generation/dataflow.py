#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 fsr4-vk contributors.
# See LICENSE in the project root.

"""Frame Generation cooperative-dataflow BOUNDARY probe (roadmap: ML graph ORDER).

The provider refuses ffxDispatch(Prepare|Frame Generation) because the ML graph execution ORDER and
per-node wiring are not reconstructed. This probe asks the sharpest question possible WITHOUT the
DLL's CPU-side descriptor writes: from the SHIPPED SPIR-V alone, can a cooperative operand's source/
target descriptor slot be traced by static pointer provenance, and do consecutive passes share a
write->read slot (a real edge)?

Pinned finding: the three cooperative descriptor sets (weights set 1, activation set 4, CBV set 5) ARE
statically visible as OpVariables + OpDecorate, but each OpCooperativeMatrixLoad/StoreKHR pointer operand
is a BINDLESS dynamically-computed buffer address (no OpAccessChain rooted at a DescriptorSet OpVariable).
So statically-resolvable cooperative reads/writes = 0 and statically-derivable pass->pass edges = 0: the
ORDER genuinely requires the DLL's CPU-side descriptor writes / a DX12 reference capture. GPU-free.
--expect-* asserts the pinned expectation so a pack/toolchain change that flips it is flagged, not faked.
"""

import argparse
import json
import re
import subprocess
from collections import Counter
from pathlib import Path


def resolve_operands(asm):
    var, sets, bindings = {}, {}, {}
    for ln in asm.splitlines():
        m = re.match(r"\s*%(\S+) = (Op\w+)(?:\s+(.*))?", ln)
        if m:
            var[m.group(1)] = (m.group(2), (m.group(3) or "").split())
        d = re.search(r"OpDecorate %(\S+) DescriptorSet (\d+)", ln)
        if d:
            sets[d.group(1)] = int(d.group(2))
        b = re.search(r"OpDecorate %(\S+) Binding (\d+)", ln)
        if b:
            bindings[b.group(1)] = int(b.group(2))
    deco = {v: (sets[v], bindings[v]) for v in sets if v in bindings}

    def base(pid, seen):
        if pid in seen or pid not in var:
            return None
        seen = seen | {pid}
        op, args = var[pid]
        if op == "OpVariable":
            return pid if pid in deco else None
        if op.startswith("OpType"):
            return None
        for a in args[1:]:
            if re.fullmatch(r"%\w+", a) and not a.startswith("%uint_"):
                r = base(a, seen)
                if r is not None:
                    return r
        return None

    reads, writes = set(), set()
    for _, (op, args) in var.items():
        if op not in ("OpCooperativeMatrixLoadKHR", "OpCooperativeMatrixStoreKHR"):
            continue
        cands = [a for a in args if re.fullmatch(r"%\w+", a) and not a.startswith("%uint_")]
        ptr = (cands[1] if op.endswith("LoadKHR") and len(cands) >= 2 else cands[0]) if cands else None
        root = base(ptr, set()) if ptr else None
        if root:
            (reads if op.endswith("LoadKHR") else writes).add(deco[root])
    return reads, writes, set(deco.values())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pack", type=Path, required=True)
    ap.add_argument("--spirv-dis", default="spirv-dis")
    ap.add_argument("--expect-resolved", type=int, default=0)
    ap.add_argument("--expect-edges", type=int, default=0)
    args = ap.parse_args()
    pack = args.pack
    graph = json.loads((pack / "mlfi-graph.json").read_text())
    mod_of = {e["entry"]: pack / "modules" / e["module"]
              for t in ("4k", "8k") for e in graph["chains"][t]}
    resolved = 0
    edges = 0
    declared = Counter()
    per_tier = {}
    for t in ("4k", "8k"):
        for e in graph["chains"][t]:
            if not e.get("cooperative_matrix"):
                continue
            asm = subprocess.run([args.spirv_dis, str(mod_of[e["entry"]])],
                                 capture_output=True, text=True, check=True).stdout
            r, w, dec = resolve_operands(asm)
            declared.update(dec)
            if r or w:
                resolved += 1
            per_tier.setdefault(t, []).append((r, w))
    for t in ("4k", "8k"):
        seq = per_tier[t]
        for i in range(len(seq)):
            for j in range(i + 1, len(seq)):
                if seq[i][1] & seq[j][0]:
                    edges += 1
                    break
    print(f"coop passes scanned: {sum(len(v) for v in per_tier.values())}")
    print(f"statically-resolvable cooperative passes: {resolved}")
    print(f"statically-derivable forward dataflow edges: {edges}")
    print(f"declared cooperative descriptor sets (set,binding->count): {dict(declared)}")
    ok = (resolved == args.expect_resolved and edges == args.expect_edges
          and all((s, 0) in declared for s in (1, 4, 5)))
    print("dataflow probe: " + ("OK (order NOT statically derivable -> ffxDispatch stays refused)"
                                if ok else "DRIFT/FAILED (see values above)"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
