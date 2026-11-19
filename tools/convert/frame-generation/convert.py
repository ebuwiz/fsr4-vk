#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 fsr4-vk contributors.
# See LICENSE in the project root.

"""Convert the pinned Frame Generation DXBC into SPIR-V and derive the Vulkan
execution contract.

This is an independent Frame Generation conversion (it does not reuse the FSR
upscaler convert paths). It drives three pinned tools in sequence:

  1. tools/extract/frame-generation/inventory.py validates the exact SHA-256,
     pins SDK version 4.0.2.4555, fails on any unsupported container layout,
     deduplicates the immediate shader-graph containers and writes each unique
     container next to a manifest.
  2. tools/shader-inspect.py (extended, not forked) translates every unique
     container with the pinned dxil-spirv and records SPIR-V capabilities,
     workgroup size, subgroup operations, atomics, workgroup storage and the
     Vulkan memory model per shader, as JSON + CSV + a human summary.
  3. this file aggregates the per-shader facts into one algorithm-level contract.

Translation flags are derived from what Frame Generation containers actually are
(Shader Model 6.6 immediate shader graphs) rather than inherited from the upscaler
model-keyed defaults. If any module fails to translate, the build fails clearly and
preserves the exact translator diagnostic; nothing is silently skipped.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SDK_VERSION = "4.0.2.4555"
COMPONENT = "frame-generation"
SOURCE_SHA256 = "e6843252f58b0341a0d80f6a20bc0d840643ee95bc48c22cac08745344577bb6"
INVENTORY_TOOL = ROOT / "tools/extract/frame-generation/inventory.py"
INSPECT_TOOL = ROOT / "tools/shader-inspect.py"

# The containers are Shader Model 6.6 immediate shader graphs (ISG1/OSG1). Measured
# on the pinned DLL they translate with only the base bindless flags; they carry
# no upscaler model-keyed matrix/dot primitives that would need --full-wmma or
# --enable-shader-i8-dot. This list is what the shader signatures demand, verified
# by running the translator over every module (a failure below preserves the
# translator diagnostic and fails the build).
DXIL_FLAGS = []


def run(command):
    return subprocess.run(
        [str(item) for item in command], capture_output=True, text=True, check=False
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dll", type=Path, required=True)
    parser.add_argument("--dxbc", type=Path, help="fresh directory for unique containers")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--json", type=Path, help="per-shader capability manifest (JSON)")
    parser.add_argument("--csv", type=Path, help="per-shader capability manifest (CSV)")
    parser.add_argument("--contract", type=Path, help="aggregated execution contract (JSON)")
    parser.add_argument("--spv-dir", type=Path,
        help="persist translated SPIR-V modules here (Frame Generation pack)")
    parser.add_argument("--pack", type=Path,
        help="write the reproducible Frame Generation asset pack manifest here")
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args(argv)

    scratch = args.dxbc
    spv_dir = args.spv_dir or (ROOT / "build/fg-convert/spv")
    pack_path = args.pack
    manifest_path = args.manifest or (ROOT / "build/fg-capabilities/inventory.json")
    per_shader_json = args.json or (ROOT / "build/fg-capabilities/shaders.json")
    per_shader_csv = args.csv or (ROOT / "build/fg-capabilities/shaders.csv")
    contract_path = args.contract or (ROOT / "build/fg-capabilities/execution-contract.json")
    for path in (manifest_path, per_shader_json, per_shader_csv, contract_path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)

    if scratch and scratch.exists():
        parser.error("--dxbc must be a fresh directory")

    # Step 1: validate + inventory + extract.
    inventory_cmd = [sys.executable, INVENTORY_TOOL, "--dll", args.dll]
    if scratch:
        inventory_cmd += ["--extract-dxbc", scratch]
    inventory_cmd += ["--output", manifest_path]
    inv = run(inventory_cmd)
    if inv.returncode != 0:
        sys.stderr.write((inv.stderr or "") + "\n")
        parser.error("Frame Generation inventory failed (identity/layout gate)")
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest["sdk_version"] != SDK_VERSION or manifest["source_sha256"] != SOURCE_SHA256:
        parser.error("inventory manifest does not carry the pinned Frame Generation identity")

    dxil_root = Path(scratch) if scratch else None
    if not dxil_root or not dxil_root.is_dir():
        parser.error("--dxbc is required (an extraction directory of unique containers)")

    # Step 2: translate + inspect every unique module (extended inspector),
    # persisting each translated SPIR-V under its source-identity stem.
    inspect_cmd = [sys.executable, INSPECT_TOOL, str(dxil_root),
                   "--dxil-root", str(dxil_root), "--dxil-flags", *DXIL_FLAGS,
                   "--component", COMPONENT, "--sdk-version", SDK_VERSION,
                   "--output-spv-dir", spv_dir,
                   "--output-json", per_shader_json, "--output-csv", per_shader_csv,
                   "--jobs", args.jobs, "--summary"]
    insp = run(inspect_cmd)
    sys.stderr.write(insp.stderr or "")
    report = json.loads(Path(per_shader_json).read_text())
    failures = [row for row in report["shaders"] if row.get("status") != "ok"]



    # Step 3: aggregate the per-shader facts into one algorithm-level contract.
    caps = Counter()
    reqs = Counter()
    subgroup_ops = Counter()
    atomics = 0
    workgroup_sizes = defaultdict(int)
    vk_versions = Counter()
    workgroup_shared = 0
    layout_unified_required = 0
    layout_collapsible = 0
    layout_representable = 0
    layout_push_known = 0
    layout_bindless_image = 0
    collision_slot_total = 0
    cross_kind_slot_total = 0
    ok_rows = 0
    for row in report["shaders"]:
        if row.get("status") != "ok":
            continue
        ok_rows += 1
        for capability in row.get("capabilities", []):
            caps[capability] += 1
        for key, value in (row.get("requirements") or {}).items():
            if value:
                reqs[key] += 1
        for op in row.get("subgroup_ops", []):
            subgroup_ops[op] += 1
        atomics += len(row.get("atomic_ops", []))
        if row.get("workgroup_shared"):
            workgroup_shared += 1
        if row.get("workgroup_size"):
            workgroup_sizes[tuple(row["workgroup_size"])] += 1
        vk_versions[row.get("spirv_version")] += 1
        dl = row.get("descriptor_layout") or {}
        if dl.get("unified_transport_required"):
            layout_unified_required += 1  # cross-kind slot: needs buffer-device-address unification
        elif dl.get("collapsible_layout"):
            layout_collapsible += 1       # same-kind duplicate slots: plain layout after de-dup
        else:
            layout_representable += 1     # no duplicate (set,binding) slots at all
        if dl.get("push_constant_bytes"):
            layout_push_known += 1
        if dl.get("uses_runtime_image_array"):
            layout_bindless_image += 1
        collision_slot_total += dl.get("binding_collisions", 0)
        cross_kind_slot_total += dl.get("cross_kind_collisions", 0)

    contract = {
        "schema": 1,
        "component": COMPONENT,
        "sdk_version": SDK_VERSION,
        "source_sha256": manifest["source_sha256"],
        "module_count": report["shader_count"],
        "unique_containers": manifest["unique_container_count"],
        "translation_failures": len(failures),
        "spirv_versions": dict(sorted(vk_versions.items(), key=lambda kv: -kv[1])),
        "required_capabilities_any": sorted(caps),
        "capability_shader_counts": dict(sorted(caps.items(), key=lambda kv: -kv[1])),
        "feature_requirements_any": sorted(reqs),
        "requirement_shader_counts": dict(sorted(reqs.items(), key=lambda kv: -kv[1])),
        "subgroup_operations": dict(sorted(subgroup_ops.items(), key=lambda kv: -kv[1])),
        "shaders_using_atomics": atomics,
        "shaders_using_workgroup_storage": workgroup_shared,
        "workgroup_sizes": {str(k): v for k, v in sorted(workgroup_sizes.items(), key=lambda kv: -kv[1])},
        "layout_census": {
            "total_modules": ok_rows,
            "plain_layout_modules": layout_representable,
            "collapsible_layout_modules": layout_collapsible,
            "unified_transport_required": layout_unified_required,
            "collision_slots_total": collision_slot_total,
            "cross_kind_collision_slots_total": cross_kind_slot_total,
            "push_constant_sized_modules": layout_push_known,
            "bindless_runtime_image_array_modules": layout_bindless_image,
        },
        "blockers": [
            {"source": row.get("source"), "error": row.get("error")}
            for row in failures
        ],
    }
    Path(contract_path).write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")

    # Step 4: build the reproducible Frame Generation asset pack. Each translated
    # SPIR-V is copied beside a manifest that ties it to its pinned-DLL identity
    # (container hash + file offset + entry point + stage) and to the per-module
    # Vulkan facts the runtime reads (workgroup size, capabilities, bindings).
    # This is generated locally, carries the source SHA in every record, and is
    # never committed (assets/ is gitignored).
    pack_dir = args.pack or (ROOT / "assets/frame-generation/pack")
    stage_by_offset = {}
    for record in manifest["shaders"]:
        for location in record["locations"]:
            stage_by_offset[location["file_offset"]] = (
                record["entry_point"], record["shader_stage"], record["container_sha256"]
            )
    modules = {}
    for row in report["shaders"]:
        if row.get("status") != "ok":
            continue
        stem = Path(row["source"]).stem
        parts = stem.split("-", 2)
        offset = int(parts[0], 16)
        entry, stage, container = stage_by_offset.get(offset, ("?", "?", "?"))
        spv_source = Path(spv_dir) / f"{stem}.spv"
        if not spv_source.is_file():
            continue
        spv_sha = hashlib.sha256(spv_source.read_bytes()).hexdigest()
        name = f"{container[:16]}.spv"
        (pack_dir / "modules").mkdir(parents=True, exist_ok=True)
        (pack_dir / "modules" / name).write_bytes(spv_source.read_bytes())
        requirements = {k: v for k, v in (row.get("requirements") or {}).items() if v}
        dl = row.get("descriptor_layout") or {}
        modules[name] = {
            "entry_point": entry,
            "shader_stage": stage,
            "container_sha256": container,
            "file_offset": offset,
            "spirv_sha256": spv_sha,
            "spirv_version": row.get("spirv_version"),
            "workgroup_size": row.get("workgroup_size"),
            "required_capabilities": row.get("capabilities", []),
            "requirements": requirements,
            "descriptor_bindings": (row.get("resources") or {}).get("descriptor_bindings"),
            "push_constants": (row.get("resources") or {}).get("push_constants"),
            "push_constant_bytes": dl.get("push_constant_bytes", 0),
            "binding_collisions": dl.get("binding_collisions", 0),
            "cross_kind_collisions": dl.get("cross_kind_collisions", 0),
            "standardizable_layout": not dl.get("unified_transport_required", False),
            "transport": ("unified" if dl.get("unified_transport_required")
                          else "collapsible" if dl.get("collapsible_layout")
                          else "plain"),
            "slots": dl.get("slots", []),
        }
    (pack_dir).mkdir(parents=True, exist_ok=True)
    pack = {
        "schema": "fgvk-pack-v1",
        "tool_version": "framegeneration-pack-v1",
        "component": COMPONENT,
        "sdk_version": SDK_VERSION,
        "source_sha256": manifest["source_sha256"],
        "module_count": len(modules),
        "execution_contract": contract,
        "modules": dict(sorted(modules.items())),
    }
    (pack_dir / "manifest.json").write_text(json.dumps(pack, indent=1, sort_keys=True) + "\n")
    # A line-oriented index the C++ runtime can read without a JSON parser.
    # (Push-constant byte sizes and per-descriptor-set structure are not captured
    #  by the inspector, so they are intentionally omitted here rather than guessed.)
    lines = ["name\tstage\tentry\tw0,w1,w2\tspirv_sha256"]
    for name, record in modules.items():
        wg = record.get("workgroup_size") or [1, 1, 1]
        lines.append(
            "\t".join(
                [name, str(record.get("shader_stage") or "?"),
                 str(record.get("entry_point") or "?"),
                 ",".join(str(v) for v in wg),
                 str(record.get("spirv_sha256") or "?")]
            )
        )
    (pack_dir / "modules.tsv").write_text("\n".join(lines) + "\n")
    # A pipeline-layout feasibility index: name \t stage \t entry \t push_bytes \t
    # collisions \t standardizable(1|0) \t layout \t transport. The layout column encodes
    # each resolved slot as ``set:binding:TYPE[:a]`` (SB=storage-buffer, UB=uniform-buffer,
    # CIS=combined-image, SI=storage-image, SMP=sampler, U=unknown; trailing ``a`` = bindless
    # whole-array). The transport column is the harness's instruction for that module:
    # ``plain`` (no duplicate slots), ``collapsible`` (duplicate (set,binding) slots that
    # are all the SAME descriptor kind -> de-duplicate into one binding; a plain
    # VkPipelineLayout still suffices), or ``unified`` (a slot claimed by DIFFERENT
    # kinds -> genuinely needs the FG buffer-device-address unified transport).
    pipeline_lines = ["name\tstage\tentry\tpush_bytes\tcollisions\tstandardizable\tlayout\ttransport"]
    kind_code = {"storage-buffer": "SB", "uniform-buffer": "UB",
                 "combined-image": "CIS", "storage-image": "SI",
                 "sampler": "SMP", "unknown": "U"}
    def fmt_layout(slots):
        parts = []
        for sl in sorted(slots, key=lambda x: (x["set"], x["binding"])):
            code = kind_code.get(sl["kind"], "U")
            parts.append(f"{sl['set']}:{sl['binding']}:{code}" + (":a" if sl.get("arrayed") else ""))
        return ";".join(parts)
    for name, record in modules.items():
        pipeline_lines.append(
            "\t".join([name, str(record.get("shader_stage") or "?"),
                      str(record.get("entry_point") or "?"),
                      str(record.get("push_constant_bytes") or 0),
                      str(record.get("binding_collisions") or 0),
                      "1" if record.get("standardizable_layout") else "0",
                      fmt_layout(record.get("slots", [])),
                      str(record.get("transport") or "plain")])
        )
    (pack_dir / "pipelines.tsv").write_text("\n".join(pipeline_lines) + "\n")
    print(
        f"Frame Generation: {report['shader_count']} modules inspected, "
        f"{len(failures)} translation failures -> {contract_path}; "
        f"pack={pack_dir} modules={len(modules)} "
        f"layout_census plain={layout_representable} collapsible={layout_collapsible} "
        f"unified={layout_unified_required} collision_slots={collision_slot_total} "
        f"cross_kind_slots={cross_kind_slot_total}"
    )
    if failures:
        # Fail the asset build clearly; preserve the exact translator diagnostics.
        for failure in failures[:8]:
            sys.stderr.write(
                f"UNTRANSLATED {failure.get('source')}: "
                f"{(failure.get('error') or '')[:200]}\n"
            )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
