# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 fsr4-vk contributors.

"""Prepare the FSR 4.1.2 and Frame Generation 4.0.2 Vulkan assets."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE / "assets"))
sys.path.insert(0, str(HERE / "convert"))
import workflow
import build_int8
from toolchain import find_library, find_tool, patching_tools

UPSCALER_DLL = "amd_fidelityfx_upscaler_dx12.dll"
FRAMEGEN_DLL = "amd_fidelityfx_framegeneration_dx12.dll"
FRAMEGEN_SHA256 = "e6843252f58b0341a0d80f6a20bc0d840643ee95bc48c22cac08745344577bb6"


def provider_dlls(directory):
    if directory is None:
        return {}
    if not directory.exists():
        raise ValueError("DLL directory does not exist: " + str(directory))
    if not directory.is_dir():
        raise ValueError(
            "--dll must name a directory containing the pinned provider DLLs, not a file: "
            + str(directory)
        )
    dlls = {
        "upscaler": directory / UPSCALER_DLL,
        "frame-generation": directory / FRAMEGEN_DLL,
    }
    missing = [path.name for path in dlls.values() if not path.is_file()]
    if missing:
        raise ValueError(
            "DLL directory is missing required provider binary: " + ", ".join(missing)
        )
    framegeneration_sha256 = hashlib.sha256(
        dlls["frame-generation"].read_bytes()
    ).hexdigest()
    if framegeneration_sha256 != FRAMEGEN_SHA256:
        raise ValueError(
            "Unsupported Frame Generation DLL: expected SHA-256 "
            f"{FRAMEGEN_SHA256}, got {framegeneration_sha256}"
        )
    return dlls


def conversion_tools(args):
    tools = patching_tools(args.spirv_tools)
    os.environ["FSR4_SPIRV_TOOLS"] = str(tools)
    os.environ["PATH"] = str(tools) + os.pathsep + os.environ.get("PATH", "")
    translator = find_tool("dxil-spirv", args.translator, "DXIL_SPIRV")
    library = find_library(args.library, translator)
    return tools, translator, library


def prepare_int8(args, extraction, pack):
    if pack.exists():
        manifest = json.loads((pack / "manifest.json").read_text())
        if manifest.get("sdk_version") != "4.1.2.4555":
            raise ValueError("Existing asset pack is not from SDK 4.1.2.4555")
        build_int8.validate_endpoints({
            name: value for name, value in manifest["aliases"].items()
            if name.startswith(f"int8/shaders/{build_int8.VERSION}/color-endpoints/")
        })
        workflow.run(HERE / "assets/manage.py", "verify", pack)
        return

    tools, translator, library = conversion_tools(args)
    (ROOT / "build").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="setup-int8-", dir=ROOT / "build") as directory:
        work = Path(directory)
        exposure = work / "auto-exposure.spv"
        workflow.run(HERE / "convert/auto_exposure.py",
                     extraction / "auto-exposure/auto-exposure.dxbc", exposure,
                     "--library", library)
        subprocess.run([str(tools / "spirv-val"), str(exposure)], check=True)
        workflow.run(HERE / "convert/int8.py", extraction / "int8", work / "int8",
                     "--dxil-spirv", translator, "--auto-exposure", exposure)
        workflow.run(HERE / "convert/endpoints.py", extraction / "endpoints", work / "endpoints",
                     translator, "--family", "int8")
        workflow.run(HERE / "convert/rcas.py", extraction / "rcas", work / "rcas-raw",
                     "--translator", translator)
        for source in sorted((work / "rcas-raw").glob("*.spv")):
            workflow.run(HERE / "patches/shared/rcas/tile_4x64.py",
                         source, work / "rcas" / source.name)
        workflow.run(HERE / "assets/build_int8.py", "--int8", work / "int8",
                     "--endpoints", work / "endpoints", "--rcas", work / "rcas",
                     "--output", pack)
    workflow.run(HERE / "assets/manage.py", "verify", pack)


def validate_framegeneration_pack(pack):
    manifest_path = pack / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("Frame Generation asset pack is missing manifest.json")
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("component") != "frame-generation"
        or manifest.get("sdk_version") != "4.0.2.4555"
        or manifest.get("source_sha256") != FRAMEGEN_SHA256
    ):
        raise ValueError("Existing Frame Generation pack is not the pinned 4.0.2.4555 asset set")
    modules = manifest.get("modules") or {}
    if manifest.get("module_count") != len(modules):
        raise ValueError("Frame Generation pack module count does not match its manifest")
    for name, record in modules.items():
        path = pack / "modules" / name
        if not path.is_file():
            raise ValueError("Frame Generation pack is missing module " + name)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != record.get("spirv_sha256"):
            raise ValueError("Frame Generation pack changed: " + name)
    model = manifest.get("mlfi_model") or {}
    graph = manifest.get("mlfi_graph") or {}
    model_path = pack / model.get("file", "")
    graph_path = pack / graph.get("file", "")
    if not model_path.is_file() or not graph_path.is_file():
        raise ValueError("Frame Generation pack is missing ML model/graph data")
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != model.get("sha256"):
        raise ValueError("Frame Generation ML model changed")
    if hashlib.sha256(graph_path.read_bytes()).hexdigest() != graph.get("sha256"):
        raise ValueError("Frame Generation ML graph changed")


def prepare_framegeneration(args, dlls, pack):
    if pack.exists():
        try:
            validate_framegeneration_pack(pack)
            print("Verified Frame Generation assets:", pack.resolve())
            return
        except ValueError:
            if not dlls:
                raise
    if not dlls:
        raise ValueError(
            "Provide --dll /path/to/dlls for the first Frame Generation extraction"
        )

    pack.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="setup-framegeneration-", dir=args.assets) as directory:
        work = Path(directory)
        workflow.run(
            HERE / "convert/frame-generation/convert.py",
            "--dll", dlls["frame-generation"],
            "--dxbc", work / "dxbc",
            "--manifest", work / "inventory.json",
            "--json", work / "shaders.json",
            "--csv", work / "shaders.csv",
            "--contract", work / "execution-contract.json",
            "--spv-dir", work / "spv",
            "--pack", work / "pack",
            "--jobs", 4,
        )
        validate_framegeneration_pack(work / "pack")
        if pack.exists():
            shutil.rmtree(pack)
        (work / "pack").rename(pack)
    print("Prepared Frame Generation 4.0.2 assets:", pack.resolve())


def setup(args):
    dlls = provider_dlls(args.dll)
    extraction = args.assets / "extracted"
    pack = args.assets / "pack"
    unified = args.assets / "pack-unified"
    framegeneration_pack = args.assets / "frame-generation" / "pack"
    args.assets.mkdir(parents=True, exist_ok=True)
    if dlls:
        sdk = workflow.load("setup_sdk", HERE / "extract/sdk.py")
        sdk.SDK(dlls["upscaler"])
    if not extraction.exists():
        if not dlls:
            raise ValueError("Provide --dll /path/to/dlls for the first extraction")
        workflow.run(
            HERE / "extract/extract.py",
            "--dll", dlls["upscaler"],
            "--output", extraction
        )
    workflow.validate_extraction(extraction)
    extraction_manifest = json.loads((extraction / "manifest.json").read_text())
    if not unified.exists() and not extraction_manifest.get("fp8_matrix_model"):
        raise ValueError("FP8 matrix shaders are missing from the extraction; "
                         "rerun setup with --dll and a fresh --assets directory")
    int8_manifest = json.loads((extraction / "int8" / "manifest.json").read_text())
    build_int8.validate_weights(extraction / "int8", int8_manifest["weights"])
    prepare_int8(args, extraction, pack)
    prepare_framegeneration(args, dlls, framegeneration_pack)

    if unified.exists():
        manifest = json.loads((unified / "manifest.json").read_text())
        if manifest.get("sdk_version") != "4.1.2.4555" or not manifest.get("fp8_matrix_model"):
            raise ValueError("Existing unified pack is not the 4.1.2 FP8 matrix model")
        workflow.run(HERE / "assets/manage.py", "verify", unified)
        print("Verified INT8 and FP8 assets:", unified.resolve())
        return
    tools, translator, library = conversion_tools(args)
    with tempfile.TemporaryDirectory(prefix="setup-fp8-", dir=args.assets) as directory:
        work = Path(directory)
        workflow.run(HERE / "convert/fp8.py", extraction / "fp8/model", work / "model",
                     "--translator", translator, "--library", library,
                     "--spirv-val", tools / "spirv-val")
        workflow.run(HERE / "convert/endpoints.py", extraction / "fp8/endpoints",
                     work / "endpoints", translator, "--family", "fp8")
        workflow.run(HERE / "assets/build_fp8.py", "--base", pack, "--model", work / "model",
                     "--endpoints", work / "endpoints", "--output", work / "pack")
        workflow.run(HERE / "assets/manage.py", "verify", work / "pack")
        (work / "pack").rename(unified)
    print("Prepared experimental INT8 and FP8 assets:", unified.resolve())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dll",
        type=Path,
        help=(
            "directory containing amd_fidelityfx_upscaler_dx12.dll and "
            "amd_fidelityfx_framegeneration_dx12.dll"
        ),
    )
    parser.add_argument("--assets", type=Path, default=ROOT / "assets")
    parser.add_argument("--translator", type=Path)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--spirv-tools", type=Path)
    args = parser.parse_args()
    try:
        setup(args)
    except (ValueError, RuntimeError, OSError, KeyError, subprocess.CalledProcessError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
