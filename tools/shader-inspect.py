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
    }
    pass_match = re.search(r"(?:pass[-_]?)(\d+)", path.name)
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
        "instruction_counts": operations,
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
    translator: str, disassembler: str, source: Path, destination: Path
) -> dict:
    flags = (
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
                *flags,
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
    parser.add_argument("--dxil-root", type=Path, help="DXBC/DXIL root to translate and inspect")
    parser.add_argument("--spirv-dis")
    parser.add_argument("--dxil-spirv")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
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
                    return translate_dxil(translator, disassembler, source, destination)

                with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
                    rows.extend(
                        pool.map(
                            translated,
                            enumerate(batch, start=offset),
                        )
                    )

    report = {
        "schema": 1,
        "version": "4.1.2.4555",
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
