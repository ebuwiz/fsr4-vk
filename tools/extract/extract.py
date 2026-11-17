#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Extract the FSR 4.1.2 model weights and shaders from the pinned SDK DLL."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

import endpoints
import fp8
import int8
from sdk import SDK, extract_auto_exposure, extract_rcas

STAGES = ("model", "int8", "endpoints", "rcas", "auto-exposure")



def file_digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract(dll, output, model="all"):
    sdk = SDK(dll)
    output = Path(output)
    if output.exists():
        raise ValueError("Use a fresh extraction directory")
    output.mkdir(parents=True)
    stages = [model] if model != "all" else list(STAGES)
    skipped = []
    for name in stages:
        if name == "model":
            fp8.extract(dll, output / "fp8")
        elif name == "int8":
            keys = sdk.keys()
            if keys.get("dispatchers") and keys.get("int8_weights"):
                int8.extract(dll, output / "int8")
            else:
                skipped.append(
                    f"int8: keys/{sdk.version}.json lacks the quantized model pins: "
                    "run tools/extract/dispatch.py --dll <dll> --write and "
                    "tools/extract/int8.py --dll <dll> --write first"
                )
        elif name == "endpoints":
            if sdk.keys().get("dispatchers"):
                endpoints.extract(dll, output / "endpoints")
            else:
                skipped.append(
                    f"endpoints: keys/{sdk.version}.json lacks the \"dispatchers\" pins: "
                    "run tools/extract/dispatch.py --dll <dll> --write first"
                )
        elif name == "rcas":
            extract_rcas(sdk, output / "rcas")
        elif name == "auto-exposure":
            extract_auto_exposure(sdk, output / "auto-exposure")
    files = {
        path.relative_to(output).as_posix(): file_digest(path)
        for path in sorted(output.rglob("*")) if path.is_file()
    }
    manifest = {
        "schema": "fsr4vk-extraction-v1",
        "sdk_version": sdk.version,
        "extracted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": str(dll),
        "source_sha256": sdk.sha256,
        "model": model,
        "complete": not skipped,
        "fp8_matrix_model": model in ("all", "model"),
        "skipped": skipped,
        "files": files,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {len(manifest['files'])} files to {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dll", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="all", choices=("all",) + STAGES,
                        help="extract a single 4.1.2 asset family")
    args = parser.parse_args()
    try:
        extract(args.dll, args.output, args.model)
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
