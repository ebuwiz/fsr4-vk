#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 fsr4-vk contributors.
# See LICENSE in the project root.

"""Inspect reconstructed SPIR-V and optionally translate source DXIL."""

import argparse
import csv
import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import re
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent / "convert"))
from toolchain import find_tool  # noqa: E402


TYPE_FLOAT = {
    8: "fp8",
    16: "fp16",
    32: "fp32",
    64: "fp64",
}
TYPE_INTEGER = {
    (8, 1): "int8",
    (8, 0): "uint8",
    (16, 1): "int16",
    (16, 0): "uint16",
    (32, 1): "int32",
    (32, 0): "uint32",
    (64, 1): "int64",
    (64, 0): "uint64",
}
ARITHMETIC = {
    "float": re.compile(r"\bOpF(?:Add|Sub|Mul|Div|Rem|Mod|Negate|Fma|mad)\b"),
    "integer": re.compile(r"\bOpI(?:Add|Sub|Mul|Div|Rem|Mod|Negate|MulExtended)\b"),
    "bitwise": re.compile(r"\bOp(?:Bitwise|Shift|BitField|Not|BitCount|ReverseBits)"),
    "conversion": re.compile(r"\bOp(?:SConvert|UConvert|FConvert|Convert[FSU]|QuantizeToF16)\b"),
    "bitcast": re.compile(r"\bOpBitcast\b"),
    "cooperative_matrix": re.compile(r"\bOpCooperativeMatrix"),
    "dot_product": re.compile(r"\bOp[US]?Dot\b"),
    "group": re.compile(r"\bOpGroup"),
    "memory": re.compile(r"\bOp(?:Load|Store|ImageRead|ImageWrite|ImageFetch|ImageSample)"),
    "barrier": re.compile(r"\bOp(?:ControlBarrier|MemoryBarrier|Atomic)"),
}

# Bytes per SPIR-V scalar in a push-constant struct (used to derive the exact push
# constant range size a Vulkan pipeline layout needs).
SCALAR_BYTES = {
    "%_bool": 4,
    "bool": 4,
    "fp8": 1, "fp16": 2, "fp32": 4, "fp64": 8,
    "int8": 1, "int16": 2, "int32": 4, "int64": 8,
    "uint8": 1, "uint16": 2, "uint32": 4, "uint64": 8,
    "float": 4, "float-other": 4, "integer-other": 4,
}


def analyze_descriptor_layout(assembly: str) -> dict:
    """Read the Vulkan pipeline-layout facts a module actually declares, off the
    disassembly only. Two things decide whether a raw SPIR-V module maps to a
    plain ``VkPipelineLayout``:

    * whether one ``(set, binding)`` slot is claimed by MORE THAN ONE descriptor
      variable (HLSL re-binds a single resource-heap slot under different types).
      A standard ``VkDescriptorSetLayoutBinding`` has exactly one descriptor type
      per slot, so such a module needs the unified descriptor-buffer transport.
    * the exact ``PushConstant`` block size, needed to set the push-constant range.

    Never guesses: if a push member is a nested struct/array we cannot size, the
    push size is reported as unknown (0) rather than invented.
    """
    uniform_var = re.compile(r"^\s*%(\S+) = OpVariable %(\S+) UniformConstant")
    set_deco = re.compile(r"OpDecorate %(\S+) DescriptorSet (\d+)")
    bind_deco = re.compile(r"OpDecorate %(\S+) Binding (\d+)")

    slot_of: dict[str, tuple[int, int]] = {}
    for line in assembly.splitlines():
        if (m := set_deco.match(line.strip())):
            slot_of[m[1]] = (int(m[2]), slot_of.get(m[1], (-1, -1))[1])
        if (m := bind_deco.match(line.strip())):
            slot_of[m[1]] = (slot_of.get(m[1], (-1, -1))[0], int(m[2]))

    resolved = resolve_descriptor_slots(assembly)
    slot_counter: Counter = Counter((s["set"], s["binding"]) for s in resolved)
    slots_multi = {k for k, c in slot_counter.items() if c > 1}
    # A colliding slot only needs the FG unified (buffer-device-address) transport if
    # the variables at that (set,binding) claim DIFFERENT descriptor kinds. If they are
    # all the SAME kind (the common HLSL resource-heap case: several views into one
    # runtime array), a single same-type VkDescriptorSetLayoutBinding de-duplicates them
    # -- a plain VkPipelineLayout suffices.
    kinds_by_slot: dict[tuple[int, int], set[str]] = {}
    for s in resolved:
        kinds_by_slot.setdefault((s["set"], s["binding"]), set()).add(s["kind"])
    cross_kind_slots = {k for k in slots_multi if len(kinds_by_slot[k]) > 1}
    same_kind_slots = slots_multi - cross_kind_slots

    # Bindless resource arrays: a UniformConstant runtime array of images.
    img_arr = any(s["kind"] in ("combined-image", "storage-image") and s["arrayed"] for s in resolved)

    # Push-constant size: find the PushConstant variable -> its struct -> members.
    push_var = re.search(r"^\s*%\S+ = OpVariable (%\S+) PushConstant", assembly, re.M)
    push_bytes = 0
    if push_var:
        ptr_target = re.search(r"%\S+ = OpTypePointer PushConstant (%\S+)", assembly)
        struct_id = ptr_target[1] if ptr_target else None
        struct_def = re.search(re.escape(struct_id) + r" = OpTypeStruct(.*)", assembly) if struct_id else None
        if struct_def:
            members = struct_def[1].split()
            offsets = {i: None for i in range(len(members))}
            for om in re.finditer(
                r"OpMemberDecorate " + re.escape(struct_id) + r" (\d+) Offset (\d+)", assembly
            ):
                if int(om[1]) < len(members):
                    offsets[int(om[1])] = int(om[2])
            total = 0
            ok = True
            for index, member_id in enumerate(members):
                if offsets.get(index) is None:
                    ok = False
                    break
                size = scalar_byte_size(member_id, assembly)
                if size is None:
                    ok = False
                    break
                total = max(total, offsets[index] + size)
            push_bytes = total if ok else 0

    return {
        "binding_slots": len(slot_counter),
        "binding_collisions": len(slots_multi),
        "collision_slots": sorted(f"{s}:{b}" for (s, b) in slots_multi),
        "cross_kind_collisions": len(cross_kind_slots),
        "unified_transport_required": bool(cross_kind_slots),
        "collapsible_layout": bool(same_kind_slots) and not cross_kind_slots,
        "uses_runtime_image_array": img_arr,
        "push_constant_bytes": push_bytes,
        "slots": resolved,
    }


def resolve_descriptor_slots(assembly: str) -> list[dict]:
    # Resolve each descriptor variable to a Vulkan descriptor TYPE + arrayness from
    # the pointer storage class the fork uses: a *StorageBuffer* OpVariable is a
    # storage buffer, *Uniform* a uniform buffer, *UniformConstant* a (possibly
    # bindless / runtime-array) image or sampler. arrayed iff the pointer target
    # is an OpTypeRuntimeArray (bindless whole-array). Returns one entry per
    # declared slot; an unresolved kind is tagged "unknown" so the caller can tell
    # a genuinely plain-layout module from one needing further binding work.
    ptrdef: dict[str, tuple[str, str]] = {}
    runtimearr: dict[str, str] = {}
    image_sampled: dict[str, int] = {}
    sampler_ids: set[str] = set()
    for line in assembly.splitlines():
        line = line.strip()
        if (m := re.match(r"(%\S+) = OpTypePointer (\S+) (%\S+)", line)):
            ptrdef[m[1]] = (m[2], m[3])
        if (m := re.match(r"(%\S+) = OpTypeRuntimeArray (%\S+)", line)):
            runtimearr[m[1]] = m[2]
        if (m := re.match(r"(%\S+) = OpTypeSampler", line)):
            sampler_ids.add(m[1])
        if (m := re.match(r"(%\S+) = OpTypeImage (.*)", line)):
            toks = m[2].split()
            # sampled flag is the token immediately before the image format field.
            if len(toks) >= 2:
                try:
                    image_sampled[m[1]] = int(toks[-2])
                except ValueError:
                    image_sampled[m[1]] = 1
    slot_of: dict[str, tuple[int, int]] = {}
    for line in assembly.splitlines():
        if (m := re.match(r"OpDecorate %(\S+) DescriptorSet (\d+)", line.strip())):
            slot_of[m[1]] = (int(m[2]), slot_of.get(m[1], (-1, -1))[1])
        if (m := re.match(r"OpDecorate %(\S+) Binding (\d+)", line.strip())):
            slot_of[m[1]] = (slot_of.get(m[1], (-1, -1))[0], int(m[2]))

    resolved: list[dict] = []
    for line in assembly.splitlines():
        if not (m := re.match(r"%(\S+) = OpVariable (%\S+) (\S+)", line.strip())):
            continue
        var_id, ptr_id, space = m[1], m[2], m[3]
        if space not in ("StorageBuffer", "Uniform", "UniformConstant"):
            continue
        slot = slot_of.get(var_id)
        if not slot or slot[0] < 0 or slot[1] < 0:
            continue
        target = ptrdef.get(ptr_id, (space, ""))[1]
        arrayed = target in runtimearr
        elem = runtimearr.get(target, target)
        if space == "StorageBuffer":
            kind = "storage-buffer"
        elif space == "Uniform":
            kind = "uniform-buffer"
        else:  # UniformConstant -> image (or sampler)
            if elem in image_sampled:
                kind = "combined-image" if image_sampled[elem] == 1 else "storage-image"
            elif elem in sampler_ids:
                kind = "sampler"
            elif elem and elem not in runtimearr:
                kind = "unknown"
            else:
                kind = "unknown"
        resolved.append({"set": slot[0], "binding": slot[1], "kind": kind, "arrayed": arrayed,
                        "space": space, "pointee": elem})
    return resolved


def scalar_byte_size(type_id: str, assembly: str) -> int | None:
    if type_id in scalar_types_map(assembly):
        name = scalar_types_map(assembly)[type_id]
        return SCALAR_BYTES.get(name)
    if re.search(re.escape(type_id) + r" = OpTypeVector %\S+ (\d+)", assembly):
        scalar = re.search(re.escape(type_id) + r" = OpTypeVector (%\S+) (\d+)", assembly)
        inner = scalar_byte_size(scalar[1], assembly)
        return inner * int(scalar[2]) if inner is not None else None
    if re.search(re.escape(type_id) + r" = OpTypeMatrix %\S+ (\d+)", assembly):
        m = re.search(re.escape(type_id) + r" = OpTypeMatrix (%\S+) (\d+)", assembly)
        inner = scalar_byte_size(m[1], assembly)
        return inner * int(m[2]) if inner is not None else None
    return None


def scalar_types_map(assembly: str) -> dict[str, str]:
    table: dict[str, str] = {}
    for line in assembly.splitlines():
        found = classify_type(line)
        if found:
            table[found[0]] = found[1]
    return table



def classify_type(line: str) -> tuple[str, str] | None:
    scalar_float = re.search(r"(?P<id>%\w+)\s+=\s+OpTypeFloat\s+(?P<bits>\d+)", line)
    if scalar_float:
        return scalar_float["id"], TYPE_FLOAT.get(int(scalar_float["bits"]), "float-other")
    scalar_int = re.search(
        r"(?P<id>%\w+)\s+=\s+OpTypeInt\s+(?P<bits>\d+)\s+(?P<signed>[01])", line
    )
    if scalar_int:
        key = (int(scalar_int["bits"]), int(scalar_int["signed"]))
        return scalar_int["id"], TYPE_INTEGER.get(key, "integer-other")
    boolean = re.search(r"(?P<id>%\w+)\s+=\s+OpTypeBool\b", line)
    if boolean:
        return boolean["id"], "bool"
    return None


def model_family(path: Path) -> str:
    parts = path.parts
    return next((part for part in ("fp8", "int8") if part in parts), "common")


def shader_bucket(path: Path) -> str:
    return next((part for part in ("1080", "2160", "4320") if part in path.parts), "shared")


def inspect_assembly(path: Path, assembly: str, source: Path | None = None) -> dict:
    lines = assembly.splitlines()
    version = next(
        (line.split(":", 1)[1].strip() for line in lines if line.startswith("; Version:")),
        "unknown",
    )
    capabilities = sorted(
        match[1] for line in lines if (match := re.match(r"\s*OpCapability\s+(\S+)", line))
    )
    extensions = sorted(
        match[1].strip('"')
        for line in lines
        if (match := re.match(r'\s*OpExtension\s+"([^"]+)"', line))
    )
    local = re.search(r"OpExecutionMode\s+%\w+\s+LocalSize\s+(\d+)\s+(\d+)\s+(\d+)", assembly)
    workgroup_size = list(map(int, local.groups())) if local else None
    types: set[str] = set()
    scalar_types: dict[str, str] = {}
    for line in lines:
        found = classify_type(line)
        if found:
            scalar_types[found[0]] = found[1]
            types.add(found[1])
    packed_types = sorted(
        set(re.findall(r"\bOpTypeVector\s+([%]\w+)\s+\d+", assembly))
    )
    for scalar_id in packed_types:
        if scalar_id in scalar_types:
            types.add(scalar_types[scalar_id] + "-vector")
    operations = {
        name: len(pattern.findall(assembly)) for name, pattern in ARITHMETIC.items()
    }
    # The same inspection has to serve the Frame Generation capability contract,
    # so record the optional mechanisms a module actually reaches (read off the
    # assembly, never guessed): subgroup ops, atomics, workgroup storage and the
    # Vulkan memory model.
    subgroup_ops = sorted(set(re.findall(r"\bOp(?:GroupNonUniform|Subgroup)\w+\b", assembly)))
    atomic_ops = sorted(set(re.findall(r"\bOpAtomic\w+\b", assembly)))
    workgroup_shared = bool(re.search(r"%_ptr_Workgroup_", assembly))
    memory_model = "VulkanMemoryModel" in capabilities
    decorators = {
        "descriptor_sets": len(re.findall(r"OpDecorate\s+%\w+\s+DescriptorSet\b", assembly)),
        "descriptor_bindings": len(
            re.findall(r"OpDecorate\s+%\w+\s+Binding\b", assembly)
        ),
        "push_constants": assembly.count(" PushConstant"),
    }
    requirements = {
        "shader_float16": "Float16" in capabilities,
        "shader_int8": "Int8" in capabilities,
        "shader_int16": "Int16" in capabilities,
        "storage_buffer_8bit": "StorageBuffer8BitAccess" in capabilities,
        "storage_image_read_without_format": "StorageImageReadWithoutFormat"
        in capabilities,
        "storage_image_write_without_format": "StorageImageWriteWithoutFormat"
        in capabilities,
        "runtime_descriptor_array": "RuntimeDescriptorArray" in capabilities,
        "buffer_device_address": "PhysicalStorageBufferAddresses" in capabilities,
        "cooperative_matrix": "CooperativeMatrixKHR" in capabilities,
        "shader_float8": "Float8EXT" in capabilities,
        "integer_dot_product": "DotProduct" in capabilities,
        "compute_derivative_linear": "ComputeDerivativeGroupLinearKHR" in capabilities,
        "quad_subgroup": "GroupNonUniformQuad" in capabilities,
        "group_non_uniform": "GroupNonUniform" in capabilities,
        "subgroup_size_control": "SubgroupSizes" in capabilities,
        "vulkan_memory_model": memory_model,
        "float_controls": "DenormPreserve" in capabilities,
        "storage_buffer_16bit": "StorageBuffer16BitAccess" in capabilities,
    }
    pass_match = re.search(r"(?:pass[-_]?)(\d+)", path.name)
    descriptor_layout = analyze_descriptor_layout(assembly)
    return {
        "file": path.as_posix(),
        "source": source.as_posix() if source else None,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "status": "ok",
        "error": None,
        "model_family": model_family(path),
        "resolution_bucket": shader_bucket(path),
        "pass_index": int(pass_match[1]) if pass_match else None,
        "spirv_version": version,
        "workgroup_size": workgroup_size,
        "types": sorted(types),
        "capabilities": capabilities,
        "extensions": extensions,
        "requirements": requirements,
        "resources": decorators,
        "descriptor_layout": descriptor_layout,
        "instruction_counts": operations,
        "subgroup_ops": subgroup_ops,
        "atomic_ops": atomic_ops,
        "workgroup_shared": workgroup_shared,
    }


def inspect_spirv(disassembler: str, path: Path, source: Path | None = None) -> dict:
    try:
        result = subprocess.run(
            [disassembler, str(path), "-o", "-"],
            check=True,
            capture_output=True,
            text=True,
        )
        return inspect_assembly(path, result.stdout, source)
    except (OSError, subprocess.CalledProcessError) as error:
        detail = getattr(error, "stderr", None) or str(error)
        return {
            "file": path.as_posix(),
            "source": source.as_posix() if source else None,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "status": "error",
            "error": detail.strip(),
            "model_family": model_family(path),
            "resolution_bucket": shader_bucket(path),
            "pass_index": None,
        }


def translate_dxil(
    translator: str, disassembler: str, source: Path, destination: Path,
    flags: list[str] | None = None,
) -> dict:
    if flags is not None:
        # An explicit flag set (Frame Generation derives its own) overrides the
        # model-keyed defaults entirely.
        extra = list(flags)
    else:
        extra = (
            ["--full-wmma", "1", "0"]
            if "fp8" in source.parts
            else ["--enable-shader-i8-dot", "--assume-ssbo-32bit-wrapping"]
        )
    try:
        subprocess.run(
            [
                translator,
                str(source),
                "--output",
                str(destination),
                "--bindless",
                "--ssbo-uav",
                "--ssbo-srv",
                *extra,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return inspect_spirv(disassembler, destination, source)
    except (OSError, subprocess.CalledProcessError) as error:
        detail = getattr(error, "stderr", None) or str(error)
        return {
            "file": destination.as_posix(),
            "source": source.as_posix(),
            "sha256": None,
            "status": "error",
            "error": detail.strip(),
            "model_family": model_family(source),
            "resolution_bucket": shader_bucket(source),
            "pass_index": None,
        }


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = (
        "file",
        "source",
        "sha256",
        "status",
        "error",
        "model_family",
        "resolution_bucket",
        "pass_index",
        "spirv_version",
        "workgroup_size",
        "types",
        "capabilities",
        "extensions",
        "requirements",
        "resources",
        "instruction_counts",
        "subgroup_ops",
        "atomic_ops",
        "workgroup_shared",
    )
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: json.dumps(row.get(field), sort_keys=True)
                    if isinstance(row.get(field), (dict, list))
                    else row.get(field)
                    for field in fields
                }
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="SPIR-V file or directory")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--output-spv-dir", type=Path,
        help="persist translated SPIR-V modules here (Frame Generation pack)")
    parser.add_argument("--dxil-root", type=Path, help="DXBC/DXIL root to translate and inspect")
    parser.add_argument(
        "--dxil-flags", nargs="*", action="append",
        help="override the model-keyed translator flags (Frame Generation passes its own)",
    )
    parser.add_argument("--component", default="fsr4-upscaler",
        help="report component identity (e.g. frame-generation)")
    parser.add_argument("--sdk-version", default="4.1.2.4555",
        help="report SDK version (e.g. 4.0.2.4555)")
    parser.add_argument("--spirv-dis")
    parser.add_argument("--dxil-spirv")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    # --dxil-flags appends once per occurrence. When present, the last group wins
    # even if it is empty (an empty group means "base flags only", which is what
    # the Frame Generation immediate shader graphs need). When absent, the model-
    # keyed defaults still apply so the upscaler inspection is unchanged.
    dxil_flags = args.dxil_flags[-1] if args.dxil_flags is not None else None
    try:
        disassembler = find_tool("spirv-dis", args.spirv_dis)
        translator = find_tool("dxil-spirv", args.dxil_spirv, "DXIL_SPIRV") if args.dxil_root else None
    except ValueError as error:
        parser.error(str(error))

    shaders = sorted(
        {
            path
            for item in args.inputs
            for path in ([item] if item.is_file() else item.rglob("*.spv"))
            if path.is_file()
        }
    )
    if not shaders and not args.dxil_root:
        parser.error("No SPIR-V modules found")

    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        rows = list(pool.map(lambda path: inspect_spirv(disassembler, path), shaders))

    if args.dxil_root:
        dxil = (
            [args.dxil_root]
            if args.dxil_root.is_file()
            else sorted(
                path
                for path in args.dxil_root.rglob("*")
                if path.suffix.lower() in (".dxbc", ".dxil")
            )
        )
        if not dxil:
            parser.error("No DXIL/DXBC modules found under " + str(args.dxil_root))
        with tempfile.TemporaryDirectory(prefix="fsr4vk-inspect-") as temporary:
            directory = Path(temporary)
            for offset in range(0, len(dxil), max(1, args.jobs)):
                batch = dxil[offset : offset + max(1, args.jobs)]

                def translated(item: tuple[int, Path]) -> dict:
                    index, source = item
                    destination = directory / f"{index}-{source.stem}.spv"
                    row = translate_dxil(
                        translator, disassembler, source, destination, dxil_flags
                    )
                    # Optionally persist the translated SPIR-V beside its source
                    # identity so a later runtime stage can consume it without
                    # re-translating (the Frame Generation asset pack path).
                    if args.output_spv_dir and row.get("status") == "ok":
                        args.output_spv_dir.mkdir(parents=True, exist_ok=True)
                        (args.output_spv_dir / f"{source.stem}.spv").write_bytes(
                            destination.read_bytes()
                        )
                    return row

                with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
                    rows.extend(
                        pool.map(
                            translated,
                            enumerate(batch, start=offset),
                        )
                    )

    report = {
        "schema": 1,
        "component": args.component,
        "version": args.sdk_version,
        "shader_count": len(rows),
        "error_count": sum(row["status"] != "ok" for row in rows),
        "shaders": rows,
    }
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2) + "\n")
    else:
        print(json.dumps(report, indent=2))
    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        write_csv(args.output_csv, rows)

    if args.summary:
        print(
            f"Inspected {len(rows)} shaders: {report['shader_count'] - report['error_count']} ok, "
            f"{report['error_count']} errors"
        )
        capabilities = Counter(
            capability for row in rows for capability in row.get("capabilities", [])
        )
        print("Capabilities:", ", ".join(f"{name}={count}" for name, count in capabilities.most_common()))
    return 1 if report["error_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
