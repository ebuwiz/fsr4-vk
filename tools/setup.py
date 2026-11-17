# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 fsr4-vk contributors.

"""Prepare the experimental FSR 4.1.2 INT8 and FP8 Vulkan assets."""

import argparse
import json
import os
from pathlib import Path
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


def setup(args):
    extraction = args.assets / "extracted"
    pack = args.assets / "pack"
    unified = args.assets / "pack-unified"
    if args.dll:
        sdk = workflow.load("setup_sdk", HERE / "extract/sdk.py")
        sdk.SDK(args.dll)
    if not extraction.exists():
        if not args.dll:
            raise ValueError("Provide --dll for the first extraction")
        workflow.run(HERE / "extract/extract.py", "--dll", args.dll, "--output", extraction)
    workflow.validate_extraction(extraction)
    extraction_manifest = json.loads((extraction / "manifest.json").read_text())
    if not unified.exists() and not extraction_manifest.get("fp8_matrix_model"):
        raise ValueError("FP8 matrix shaders are missing from the extraction; "
                         "rerun setup with --dll and a fresh --assets directory")
    int8_manifest = json.loads((extraction / "int8" / "manifest.json").read_text())
    build_int8.validate_weights(extraction / "int8", int8_manifest["weights"])
    prepare_int8(args, extraction, pack)

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
    parser.add_argument("--dll", type=Path, help="FSR 4.1.2.4555 DLL")
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
