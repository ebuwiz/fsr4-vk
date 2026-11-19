#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 fsr4-vk contributors.
# See LICENSE in the project root.

"""Frame Generation audit + ABI tests. Deterministic; no GPU required for the
inventory/identity tests, and the ABI test self-skips if the .so is not built.

They check the load-bearing invariants, not constants-for-theirs-sake:
  - the exact SHA-256 / pinned SDK version gates extraction;
  - a wrong or truncated image is rejected, never silently skipped;
  - dedup + stage census of the immediate shader-graph containers;
  - a --expect gate fails on drift;
  - the built provider .so exports exactly the FFX generic API entry points
    and nothing else that is text-visible.
"""

import hashlib
import collections
import json
import re
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
INVENTORY = ROOT / "tools/extract/frame-generation/inventory.py"
CACHE_HEADER = ROOT / "src/framegeneration/pipeline-cache.hpp"
PACK = ROOT / "assets/frame-generation/pack"
DLL = next(
    (
        path
        for path in (
            ROOT / "dlls/amd_fidelityfx_framegeneration_dx12.dll",
            ROOT / "amd_fidelityfx_framegeneration_dx12.dll",
        )
        if path.is_file()
    ),
    ROOT / "dlls/amd_fidelityfx_framegeneration_dx12.dll",
)
EXPECTED_SHA = "e6843252f58b0341a0d80f6a20bc0d840643ee95bc48c22cac08745344577bb6"
EXPECTED_MODEL_SIZE = 9437184  # 0x900000: the FG 4.0.2 ML model blob length (0xF00000..0x1800000 in the DLL)
EXPECTED_MODEL_SHA = "6345f9ef4bbead6a942a7379d17caeaaf01d6142aa3dccae4695e56f68627dbe"
PROVIDER_SO = ROOT / "build/linux/amd_fidelityfx_framegeneration_vk.so"
EXEC_EXE = ROOT / "build/linux/framegeneration-exec"
FFX_EXPORTS = {"ffxCreateContext", "ffxDestroyContext", "ffxConfigure", "ffxQuery", "ffxDispatch"}


def run(command):
    return subprocess.run([str(c) for c in command], capture_output=True, text=True)


def test_identity_gate():
    if not DLL.is_file():
        print("SKIP identity: Frame Generation DLL absent"); return
    if hashlib.sha256(DLL.read_bytes()).hexdigest() != EXPECTED_SHA:
        raise AssertionError("local DLL is not the pinned identity")
    result = run([sys.executable, INVENTORY, "--dll", DLL,
                  "--output", tempfile.NamedTemporaryFile(suffix=".json").name])
    if result.returncode != 0:
        raise AssertionError("pinned DLL rejected: " + result.stderr)


def test_wrong_image_rejected():
    if not DLL.is_file():
        print("SKIP wrong-image: DLL absent"); return
    data = DLL.read_bytes()
    with tempfile.NamedTemporaryFile(suffix=".dll") as handle:
        handle.write(data[: len(data) // 2])  # a truncated copy
        handle.flush()
        result = run([sys.executable, INVENTORY, "--dll", handle.name])
    if result.returncode == 0:
        raise AssertionError("a truncated/other image was accepted")
    if "SHA-256" not in result.stderr:
        raise AssertionError("rejection did not name the identity gate: " + result.stderr)


def test_not_pe_rejected():
    with tempfile.NamedTemporaryFile(suffix=".dll") as handle:
        handle.write(b"NOTAPEFILE")
        handle.flush()
        result = run([sys.executable, INVENTORY, "--dll", handle.name])
    if result.returncode == 0 or "PE" not in result.stderr:
        raise AssertionError("a non-PE image was not rejected as a PE failure: " + result.stderr)


def test_dedup_and_census():
    if not DLL.is_file():
        print("SKIP census: DLL absent"); return
    out = ROOT / "build/fg-audit-test.json"
    result = run([sys.executable, INVENTORY, "--dll", DLL, "--output", out,
                  "--expect-signatures", 487, "--expect-unique", 486])
    if result.returncode != 0:
        raise AssertionError("census gate failed: " + result.stderr)
    manifest = json.loads(out.read_text())
    if not manifest["immediate_shader_graph_only"]:
        raise AssertionError("not all containers are immediate shader graphs")
    census = manifest["shader_stage_census"]
    for stage in ("compute-neural", "compute-classic", "vertex", "pixel"):
        if stage not in census:
            raise AssertionError(f"missing stage class {stage} in {census}")
    # A deliberately wrong --expect must fail (drift detection).
    bad = run([sys.executable, INVENTORY, "--dll", DLL, "--output", out,
               "--expect-unique", 1])
    if bad.returncode == 0:
        raise AssertionError("--expect-unique drift was not caught")


def test_abi_exports():
    if not PROVIDER_SO.is_file():
        print("SKIP ABI: provider .so not built"); return
    listing = run(["nm", "-D", "--defined-only", PROVIDER_SO])
    exported = set()
    for line in listing.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[1] in ("T", "i", "W"):
            exported.add(parts[2])
    ffx = {name for name in exported if name.startswith("ffx")}
    if ffx != FFX_EXPORTS:
        raise AssertionError(f"exported ffx symbols {sorted(ffx)} != {sorted(FFX_EXPORTS)}")
    cpp = {name for name in exported if name.startswith("_ZN") or "::" in name}
    if cpp:
        raise AssertionError(f"C++ symbols leaked: {sorted(cpp)[:5]}")


def test_pipeline_cache_namespace_isolation():
    # Mirror src/framegeneration/pipeline-cache.hpp's container layout so a cache
    # built by the *upscaler* (or another GPU/driver) is proven to never be reused
    # by the frame-generation provider, and vice versa. Deterministic, no GPU.
    if not CACHE_HEADER.is_file():
        print("SKIP cache: pipeline-cache.hpp absent"); return
    source = CACHE_HEADER.read_text()
    import re
    magic = re.search(r"kCacheMagic\[\d+\]\s*=\s*\{([^}]*)\}", source)
    if not magic:
        raise AssertionError("kCacheMagic array definition missing from header")

    def cval(token: str) -> int:
        token = token.strip()
        if token[0] == "'":
            inner = token[1:-1]
            if inner in ("\\0", "\\00", "\\000"):
                return 0
            if len(inner) == 1:
                return ord(inner)
            if inner.startswith("\\"):
                return int(inner[1:], 8)
            return ord(inner)
        return int(token, 0)

    magic_bytes = bytes(cval(t) for t in magic.group(1).split(",") if t.strip())
    if not (b"FFG4PC1" in magic_bytes and magic_bytes[-1] == 0):
        raise AssertionError(f"kCacheMagic is not the FFG4PC1\\0 container tag: {magic_bytes!r}")
    if "fourcc('f', 'g', '4')" not in source:
        raise AssertionError("frame-generation cache namespace tag missing from header")

    def fourcc(a, b, c):
        return ord(a) | (ord(b) << 8) | (ord(c) << 16)
    fg4, fs4 = fourcc("f", "g", "4"), fourcc("f", "s", "4")
    if fg4 == fs4:
        raise AssertionError("frame-generation and upscaler namespaces collide")

    magic = b"FFG4PC1\x00"
    fmt = 1

    def write_blob(identity, blob):
        name = identity["device"].encode()
        out = bytearray(magic)
        for v in (fmt, identity["algorithm"], identity["driver"], identity["api"],
                  identity["vendor"], identity["device_id"], identity["salt"]):
            out += struct.pack("<I", v)
        out += struct.pack("<I", len(name)) + name
        out += struct.pack("<I", len(blob)) + blob
        return out

    def read_blob(data, identity):
        # Returns the driver blob only if EVERY pinned identity field matches.
        if len(data) < len(magic) + 6 * 4 or data[:len(magic)] != magic:
            return b""
        off = len(magic)
        vals = {}
        for key in ("fmt", "algorithm", "driver", "api", "vendor", "device_id", "salt"):
            vals[key] = struct.unpack_from("<I", data, off)[0]; off += 4
        if vals["fmt"] != fmt:
            return b""
        name_len = struct.unpack_from("<I", data, off)[0]; off += 4
        name = data[off:off + name_len]; off += name_len
        blob_len = struct.unpack_from("<I", data, off)[0]; off += 4
        if len(data) < off + blob_len:
            return b""
        if (vals["algorithm"] != identity["algorithm"] or name != identity["device"].encode()
                or vals["driver"] != identity["driver"] or vals["api"] != identity["api"]
                or vals["vendor"] != identity["vendor"] or vals["device_id"] != identity["device_id"]):
            return b""
        return data[off:off + blob_len]

    ident = {"algorithm": fg4, "device": "AMD Radeon RX 9070 XT", "driver": 1234,
             "api": 4206694, "vendor": 0x1002, "device_id": 0x42, "salt": 0}
    blob = bytes(range(36))
    data = write_blob(ident, blob)
    if read_blob(data, ident) != blob:
        raise AssertionError("same-identity cache failed to reload")
    # Foreign ALGORITHM (upscaler blob) must be refused by the FG reader.
    foreign = dict(ident); foreign["algorithm"] = fs4
    if read_blob(write_blob(foreign, blob), ident) != b"":
        raise AssertionError("FG reader accepted an upscaler-namespace cache")
    # Foreign GPU/driver must be refused too.
    other_gpu = dict(ident); other_gpu["device_id"] = 0x99
    if read_blob(write_blob(other_gpu, blob), ident) != b"":
        raise AssertionError("FG reader accepted a different-GPU cache")


def test_pack_manifest_shape():
    # The reproducible Frame Generation asset pack (tools/convert/frame-generation).
    # Self-skips when the pack has not been generated in this workspace.
    if not (PACK / "manifest.json").is_file():
        print("SKIP pack: not generated"); return
    pack = json.loads((PACK / "manifest.json").read_text())
    if pack["schema"] != "fgvk-pack-v1":
        raise AssertionError("unexpected pack schema: " + pack["schema"])
    if pack["component"] != "frame-generation" or pack["sdk_version"] != "4.0.2.4555":
        raise AssertionError("pack identity not pinned to frame-generation 4.0.2.4555")
    if pack["source_sha256"] != EXPECTED_SHA:
        raise AssertionError("pack does not carry the pinned DLL source SHA")
    model = pack.get("mlfi_model") or {}
    model_path = PACK / model.get("file", "")
    if not model_path.is_file():
        raise AssertionError("pack lacks the reverse-engineered ML model")
    if model.get("size") != 9437184:
        raise AssertionError("unexpected ML model size")
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != model.get("sha256"):
        raise AssertionError("ML model hash drifted")
    graph = pack.get("mlfi_graph") or {}
    graph_path = PACK / graph.get("file", "")
    if not graph_path.is_file():
        raise AssertionError("pack lacks the ML graph")
    graph_data = json.loads(graph_path.read_text())
    for tier in ("4k", "8k"):
        if len(graph_data.get("chains", {}).get(tier, [])) != 191:
            raise AssertionError(f"{tier} ML graph is not a complete 191-pass chain")
    contract = pack["execution_contract"]
    if contract.get("blockers"):
        raise AssertionError(f"pack contract has unexpected blockers: {contract['blockers']}")
    modules = pack["modules"]
    if len(modules) != pack["module_count"]:
        raise AssertionError("pack module_count mismatch")
    stages = set()
    for name, rec in modules.items():
        for field in ("shader_stage", "entry_point", "workgroup_size", "spirv_sha256"):
            if field not in rec:
                raise AssertionError(f"module {name} missing {field}")
        if not rec["spirv_sha256"] or (len(rec["spirv_sha256"]) != 64):
            raise AssertionError(f"module {name} lacks a 64-hex SPIR-V sha")
        stages.add(rec["shader_stage"])
    # Both Frame Generation compute families must be present in the pack.
    if not {"compute-classic", "compute-neural"}.issubset(stages):
        raise AssertionError(f"missing a Frame Generation compute family in {sorted(stages)}")
    # The line index the C++ runtime reads must be well-formed (5 columns).
    tsv = (PACK / "modules.tsv").read_text().splitlines()
    if tsv[0].split("\t") != ["name", "stage", "entry", "w0,w1,w2", "spirv_sha256"]:
        raise AssertionError("modules.tsv header drift")
    if any(len(line.split("\t")) != 5 for line in tsv):
        raise AssertionError("modules.tsv column-count drift")
    # The pipeline-layout index the compile census reads is 7 columns; every layout
    # token must match ``set:binding:TYPE[:a]`` with TYPE in SB|UB|CIS|SI|SMP|U, plus a
    # transport column (plain|collapsible|unified).
    import re as _re
    ptsv = (PACK / "pipelines.tsv").read_text().splitlines()
    if ptsv[0].split("\t") != ["name", "stage", "entry", "push_bytes", "collisions",
                               "standardizable", "layout", "transport"]:
        raise AssertionError("pipelines.tsv header drift")
    token_re = _re.compile(r"\d+:\d+:(SB|UB|CIS|SI|SMP|U)(:a)?")
    transport_seen = set()
    layout_seen = 0
    collapsible_with_dupslot = 0
    for line in ptsv[1:]:
        cols = line.split("\t")
        if len(cols) != 8:
            raise AssertionError("pipelines.tsv column-count drift")
        transport = cols[7]
        if transport not in ("plain", "collapsible", "unified"):
            raise AssertionError(f"pipelines.tsv bad transport value {transport!r} in {cols[0]}")
        transport_seen.add(transport)
        # tokens are "set:binding:TYPE[:a]" -> the (set,binding) key is the first two fields
        keys = []
        for tok in (cols[6].split(";") if cols[6] else []):
            if not token_re.fullmatch(tok):
                raise AssertionError(f"pipelines.tsv bad layout token {tok!r} in {cols[0]}")
            layout_seen += 1
            setb = ":".join(tok.split(":")[:2])
            keys.append(setb)
        dup = len(keys) != len(set(keys))
        if transport == "collapsible" and dup:
            collapsible_with_dupslot += 1
    if layout_seen == 0:
        raise AssertionError("pipelines.tsv layout column is empty across the whole pack")
    # The Frame Generation DLL has collapsible (same-kind duplicate) slots; they must be
    # labelled collapsible, and must actually contain duplicate set:binding keys.
    if "collapsible" not in transport_seen or collapsible_with_dupslot == 0:
        raise AssertionError("pack did not classify same-kind duplicate slots as collapsible")


def test_framegeneration_exec_selftest():
    # The execution + pipeline-cache benchmark harness must self-test cleanly with
    # no GPU, and advertise the FG-only cache namespace (never the upscaler's).
    if not EXEC_EXE.is_file():
        print("SKIP exec: framegeneration-exec not built"); return
    result = run([EXEC_EXE, "--self-test"])
    if result.returncode != 0:
        raise AssertionError("exec self-test exited " + str(result.returncode) + ": " + result.stdout)
    out = result.stdout
    if "pipeline-cache-namespace=fg4" not in out:
        raise AssertionError("exec self-test did not advertise the fg4 cache namespace: " + out)
    if "fs4" in out:
        raise AssertionError("exec self-test references the upscaler cache namespace")


def test_descriptor_slot_resolver():
    # The compile census depends on the inspector resolving every descriptor variable
    # to a Vulkan descriptor kind + arrayness from the pointer storage class the fork
    # uses (StorageBuffer/Uniform/UniformConstant+image), not only UniformConstant.
    inspector = ROOT / "tools/shader-inspect.py"
    if not inspector.is_file():
        print("SKIP slot resolver: shader-inspect.py absent"); return
    import importlib.util
    spec = importlib.util.spec_from_file_location("fg_si2", inspector)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    lines = [
        "%uint = OpTypeInt 32 0",
        "%float = OpTypeFloat 32",
        "%_runtimearr_uint = OpTypeRuntimeArray %uint",
        "%SSBO = OpTypeStruct %_runtimearr_uint",
        "%_runtimearr_SSBO = OpTypeRuntimeArray %SSBO",
        "%_ptr_StorageBuffer__runtimearr_SSBO = OpTypePointer StorageBuffer %_runtimearr_SSBO",
        "%13 = OpVariable %_ptr_StorageBuffer__runtimearr_SSBO StorageBuffer",
        "%BindlessCBV = OpTypeStruct %_runtimearr_uint",
        "%_runtimearr_BindlessCBV = OpTypeRuntimeArray %BindlessCBV",
        "%_ptr_Uniform__runtimearr_BindlessCBV = OpTypePointer Uniform %_runtimearr_BindlessCBV",
        "%21 = OpVariable %_ptr_Uniform__runtimearr_BindlessCBV Uniform",
        "%10 = OpTypeImage %float 2D 0 0 0 1 Unknown",
        "%_runtimearr_10 = OpTypeRuntimeArray %10",
        "%_ptr_UniformConstant__runtimearr_10 = OpTypePointer UniformConstant %_runtimearr_10",
        "%31 = OpVariable %_ptr_UniformConstant__runtimearr_10 UniformConstant",
        "%14 = OpTypeImage %float 2D 0 0 0 2 R32f",
        "%_runtimearr_14 = OpTypeRuntimeArray %14",
        "%_ptr_UniformConstant__runtimearr_14 = OpTypePointer UniformConstant %_runtimearr_14",
        "%41 = OpVariable %_ptr_UniformConstant__runtimearr_14 UniformConstant",
        "%29 = OpTypeSampler",
        "%_runtimearr_29 = OpTypeRuntimeArray %29",
        "%_ptr_UniformConstant__runtimearr_29 = OpTypePointer UniformConstant %_runtimearr_29",
        "%51 = OpVariable %_ptr_UniformConstant__runtimearr_29 UniformConstant",
        "%RootConstants = OpTypeStruct %uint %uint %uint %uint %uint %uint %uint %uint",
        "%_ptr_PushConstant_RootConstants = OpTypePointer PushConstant %RootConstants",
        "%registers = OpVariable %_ptr_PushConstant_RootConstants PushConstant",
        "OpDecorate %13 DescriptorSet 4", "OpDecorate %13 Binding 0",
        "OpDecorate %21 DescriptorSet 5", "OpDecorate %21 Binding 0",
        "OpDecorate %31 DescriptorSet 0", "OpDecorate %31 Binding 0",
        "OpDecorate %41 DescriptorSet 3", "OpDecorate %41 Binding 0",
        "OpDecorate %51 DescriptorSet 1", "OpDecorate %51 Binding 0",
    ] + [f"OpMemberDecorate %RootConstants {i} Offset {i * 4}" for i in range(8)]
    census = module.analyze_descriptor_layout("\n".join(lines))
    kinds = {s["kind"]: s for s in census["slots"]}
    by_slot = {(s["set"], s["binding"]): s["kind"] for s in census["slots"]}
    if by_slot != {(4, 0): "storage-buffer", (5, 0): "uniform-buffer",
                   (0, 0): "combined-image", (3, 0): "storage-image",
                   (1, 0): "sampler"}:
        raise AssertionError(f"slot kind mapping wrong: {by_slot}")
    if not all(s["arrayed"] for s in census["slots"]):
        raise AssertionError("runtime-array (bindless whole-array) slots must report arrayed")
    if census["binding_slots"] != 5 or census["binding_collisions"] != 0:
        raise AssertionError("5 distinct slots, no collisions expected")
    if census["unified_transport_required"]:
        raise AssertionError("all-different-slot kinds must be standardizable")
    if census["push_constant_bytes"] != 32:
        raise AssertionError("push should be 32 bytes")

    # Same-kind duplicate at one slot: HLSL resource-heap views into one runtime array.
    # Must be flagged COLLAPSIBLE (one same-type binding de-duplicates them), NOT unified.
    dup = "\n".join([
        "%float = OpTypeFloat 32",
        "%14 = OpTypeImage %float 2D 0 0 0 2 Unknown",
        "%18 = OpTypeImage %float 2D 0 0 0 2 Unknown",
        "%_runtimearr_14 = OpTypeRuntimeArray %14",
        "%_runtimearr_18 = OpTypeRuntimeArray %18",
        "%_ptr_UniformConstant__runtimearr_14 = OpTypePointer UniformConstant %_runtimearr_14",
        "%_ptr_UniformConstant__runtimearr_18 = OpTypePointer UniformConstant %_runtimearr_18",
        "%17 = OpVariable %_ptr_UniformConstant__runtimearr_14 UniformConstant",
        "%21 = OpVariable %_ptr_UniformConstant__runtimearr_18 UniformConstant",
        "OpDecorate %17 DescriptorSet 3", "OpDecorate %17 Binding 0",
        "OpDecorate %21 DescriptorSet 3", "OpDecorate %21 Binding 0",
    ])
    c = module.analyze_descriptor_layout(dup)
    if c["binding_collisions"] != 1 or c["cross_kind_collisions"] != 0:
        raise AssertionError(f"same-kind dup must give 1 collision, 0 cross-kind: {c}")
    if not c["collapsible_layout"] or c["unified_transport_required"]:
        raise AssertionError("same-kind duplicate must be collapsible, not unified")

    # Cross-kind duplicate at one slot (storage image + uniform buffer at 3:0): genuinely
    # needs the FG buffer-device-address unified transport.
    mix = "\n".join([
        "%float = OpTypeFloat 32",
        "%uint = OpTypeInt 32 0",
        "%14 = OpTypeImage %float 2D 0 0 0 2 Unknown",
        "%_runtimearr_14 = OpTypeRuntimeArray %14",
        "%_ptr_UniformConstant__runtimearr_14 = OpTypePointer UniformConstant %_runtimearr_14",
        "%17 = OpVariable %_ptr_UniformConstant__runtimearr_14 UniformConstant",
        "%_runtimearr_uint = OpTypeRuntimeArray %uint",
        "%UB = OpTypeStruct %_runtimearr_uint",
        "%_runtimearr_UB = OpTypeRuntimeArray %UB",
        "%_ptr_Uniform__runtimearr_UB = OpTypePointer Uniform %_runtimearr_UB",
        "%21 = OpVariable %_ptr_Uniform__runtimearr_UB Uniform",
        "OpDecorate %17 DescriptorSet 3", "OpDecorate %17 Binding 0",
        "OpDecorate %21 DescriptorSet 3", "OpDecorate %21 Binding 0",
    ])
    c2 = module.analyze_descriptor_layout(mix)
    if c2["cross_kind_collisions"] != 1 or not c2["unified_transport_required"]:
        raise AssertionError(f"cross-kind must give unified: {c2}")
    if c2["collapsible_layout"]:
        raise AssertionError("cross-kind must NOT be collapsible")


def test_layout_census_analyzer():
    # Deterministic, GPU-free: exercise the inspector's descriptor-layout census
    # on the exact shape a small Frame Generation interpolation CS disassembles to
    # (an 8x-uint RootConstants push block + a 3:0 slot claimed by two descriptor
    # variables). Assert it reports the collision and the exact push size.
    inspector = ROOT / "tools/shader-inspect.py"
    if not inspector.is_file():
        print("SKIP layout census: shader-inspect.py absent"); return
    import importlib.util
    spec = importlib.util.spec_from_file_location("fg_shader_inspect", inspector)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    analyze = module.analyze_descriptor_layout
    assembly = "\n".join([
        "OpDecorate %13 DescriptorSet 0",
        "OpDecorate %13 Binding 0",
        "OpDecorate %17 DescriptorSet 3",
        "OpDecorate %17 Binding 0",
        "OpDecorate %21 DescriptorSet 3",
        "OpDecorate %21 Binding 0",
        "OpDecorate %28 DescriptorSet 5",
        "OpDecorate %28 Binding 0",
        "%uint = OpTypeInt 32 0",
        "%13 = OpVariable %_ptr_UniformConstant__runtimearr_10 UniformConstant",
        "%17 = OpVariable %_ptr_UniformConstant__runtimearr_14 UniformConstant",
        "%21 = OpVariable %_ptr_UniformConstant__runtimearr_18 UniformConstant",
        "%28 = OpVariable %_ptr_UniformConstant__runtimearr_22 UniformConstant",
        "%RootConstants = OpTypeStruct %uint %uint %uint %uint %uint %uint %uint %uint",
        "%_ptr_PushConstant_RootConstants = OpTypePointer PushConstant %RootConstants",
        "%registers = OpVariable %_ptr_PushConstant_RootConstants PushConstant",
    ] + [f"OpMemberDecorate %RootConstants {i} Offset {i * 4}" for i in range(8)])
    census = analyze(assembly)
    if census["binding_collisions"] != 1:
        raise AssertionError("expected one 3:0 collision, got " + str(census["binding_collisions"]))
    if census["collision_slots"] != ["3:0"]:
        raise AssertionError(f"collision_slots: {census['collision_slots']}")
    if not census["collapsible_layout"]:
        raise AssertionError("same-slot same-kind multi-variable layout must be collapsible")
    if census["cross_kind_collisions"] != 0 or census["unified_transport_required"]:
        raise AssertionError("this case is same-kind (collapsible), not cross-kind (unified)")
    if census["push_constant_bytes"] != 32:
        raise AssertionError("push_constant_bytes should be 32, got " + str(census["push_constant_bytes"]))
    if census["binding_slots"] != 3:
        raise AssertionError("expected 3 distinct descriptor slots, got " + str(census["binding_slots"]))


def test_mlfi_graph_chain_matches_pack():
    # The Frame Generation reverse-engineering report (`mlfi-graph.json`, produced from the pinned
    # DLL) must agree, shader-for-shader and byte-for-byte, with the SHIPPED pack (modules.tsv) and
    # with the extracted model blob -- a three-way identity gate: DLL-RE artifact <-> pack <-> model
    # bytes -- and the cooperative-matrix count it carries must equal the SPIR-V census (156). This
    # grounds the 191+191 ML chain and its per-pass shader identity WITHOUT asserting any execution
    # order or enabling dispatch. GPU-free (reads only generated pack files); self-skips when the
    # pack/graph has not been produced on this checkout.
    graph_path = PACK / "mlfi-graph.json"
    modules_path = PACK / "modules.tsv"
    if not (graph_path.is_file() and modules_path.is_file()):
        print("SKIP mlfi-chain: FG graph/pack not generated")
        return
    graph = json.loads(graph_path.read_text())
    chains = graph.get("chains") or {}
    if sorted(chains) != ["4k", "8k"]:
        raise AssertionError(f"ML chains != 4k/8k: {sorted(chains)}")
    for tier, passes in chains.items():
        if len(passes) != 191:
            raise AssertionError(f"{tier} chain is not a complete 191-pass sequence: {len(passes)}")
    # modules.tsv: name stage entry w0,w1,w2 spirv_sha256 -- index the neural (mlfi) entries.
    pack_by_entry = {}
    for ln in modules_path.read_text().splitlines()[1:]:
        c = ln.split("\t")
        if len(c) >= 5 and c[1] == "compute-neural":
            pack_by_entry[c[2]] = (c[0], c[4])
    if len(pack_by_entry) != 382:
        raise AssertionError(f"pack neural entries != 382: {len(pack_by_entry)}")
    coop_entries = set()
    for tier in ("4k", "8k"):
        for index, e in enumerate(chains[tier]):
            expected_entry = "mlfi_%s_pass%d" % (tier, index)
            entry = e.get("entry")
            if entry != expected_entry:
                raise AssertionError(f"{tier}[{index}] entry != {expected_entry} (got {entry})")
            packrow = pack_by_entry.get(entry)
            if packrow is None:
                raise AssertionError(f"graph pass {entry} has no matching pack neural module")
            if e.get("module") != packrow[0]:
                raise AssertionError(
                    f"module-hash drift for {entry}: graph {e.get('module')} vs pack {packrow[0]}")
            if e.get("spirv_sha256") != packrow[1]:
                raise AssertionError(f"SPIR-V digest drift for {entry}: {entry}")
            if e.get("cooperative_matrix"):
                coop_entries.add(entry)
    if len(coop_entries) != 156:
        raise AssertionError(f"graph cooperative-matrix modules != 156: {len(coop_entries)}")
    model = graph.get("model") or {}
    if model.get("size") != EXPECTED_MODEL_SIZE:
        raise AssertionError(f"model size {model.get('size')} != {EXPECTED_MODEL_SIZE}")
    if model.get("sha256") != EXPECTED_MODEL_SHA:
        raise AssertionError("graph model sha256 != the pinned FG 4.0.2 model identity")
    model_bytes = PACK / "mlfi-model.bin"
    if model_bytes.is_file():
        raw = model_bytes.read_bytes()
        if (len(raw) != EXPECTED_MODEL_SIZE or
                hashlib.sha256(raw).hexdigest() != EXPECTED_MODEL_SHA):
            raise AssertionError("on-disk mlfi-model.bin != the DLL-extracted model blob (identity)")


def test_mlfi_chain_structure_and_model_tiling():
    # Grounded from the DLL-RE graph + the shipped pack, NO execution order/DAG claimed. Verifies:
    #  (1) every mlfi pass' descriptor-set participation in mlfi-graph.json EQUALS the pack
    #      pipelines.tsv bound sets (DLL-RE <-> pack conversion: two independent producers);
    #  (2) the 4k and 8k chains are STRUCTURALLY TIER-INVARIANT: an identical per-index signature
    #      sequence for positions 0..190;
    #  (3) the 156 cooperative-matrix passes are exactly the passes binding descriptor sets 1,4,5
    #      (activation + model weights), and the whole 382-pass signature multiset matches the pinned
    #      FG 4.0.2 census;
    #  (4) the 9,437,184-byte model blob aligns to the pinned 16x16 cooperative tiling for every
    #      recovered component dtype (fp16/fp32/int8 -> a whole number of tiles).
    # This grounds roadmap blocker #2's premise (chain structure / per-pass resource participation) and
    # the model<->tile grid (toward #1 CsTensorSizes and #3 allocation). It does NOT recover the graph
    # dependency ORDER or per-node offsets (not derivable from these signatures) and the provider still
    # refuses dispatch. GPU-free; self-skips when the generated pack/graph is absent.
    graph_path = PACK / "mlfi-graph.json"
    pipelines_path = PACK / "pipelines.tsv"
    if not (graph_path.is_file() and pipelines_path.is_file()):
        print("SKIP mlfi-structure: FG graph/pack not generated")
        return
    graph = json.loads(graph_path.read_text())
    chains = graph["chains"]
    if sorted(chains) != ["4k", "8k"]:
        raise AssertionError(f"chains != 4k/8k: {sorted(chains)}")
    if len(chains["4k"]) != 191 or len(chains["8k"]) != 191:
        raise AssertionError("ML chains not a complete 191+191")

    # (1) pack bound descriptor-set signature per neural entry (set ids from set:binding:KIND tokens).
    kinds = {"SB", "UB", "CIS", "SI", "SMP"}
    pack_sig = {}
    for ln in pipelines_path.read_text().splitlines()[1:]:
        c = ln.split("\t")
        if len(c) < 8 or c[1] != "compute-neural":
            continue
        s = set()
        for tok in c[6].split(";"):
            f = tok.split(":")
            if len(f) >= 3 and f[2] in kinds:
                s.add(int(f[0]))
        pack_sig[c[2]] = tuple(sorted(s))
    graph_sig = {}
    for tier in ("4k", "8k"):
        for e in chains[tier]:
            graph_sig[e["entry"]] = tuple(sorted(e.get("descriptor_sets", [])))
    if set(graph_sig) != set(pack_sig):
        raise AssertionError("graph vs pack neural entry sets differ")
    drift = [k for k in graph_sig if graph_sig[k] != pack_sig[k]]
    if drift:
        raise AssertionError(f"descriptor-set participation drift on {len(drift)} passes, e.g. {drift[:3]}")

    # (2) tier-invariant per-index signature sequence (4k[i] == 8k[i] for all i).
    for i in range(191):
        a = tuple(sorted(chains["4k"][i]["descriptor_sets"]))
        b = tuple(sorted(chains["8k"][i]["descriptor_sets"]))
        if a != b:
            raise AssertionError(f"4k[{i}] sig {a} != 8k[{i}] sig {b}")

    # (3) cooperative passes and the pinned signature multiset.
    coop = [e for tier in ("4k", "8k") for e in chains[tier] if e.get("cooperative_matrix")]
    if len(coop) != 156:
        raise AssertionError(f"cooperative passes != 156: {len(coop)}")
    coop_sig = collections.Counter(tuple(sorted(e["descriptor_sets"])) for e in coop)
    if coop_sig != collections.Counter({(1, 4, 5): 156}):
        raise AssertionError(f"cooperative passes must all bind sets 1,4,5; got {dict(coop_sig)}")
    sig_multiset = collections.Counter(
        tuple(sorted(e["descriptor_sets"])) for tier in ("4k", "8k") for e in chains[tier])
    expected = collections.Counter({
        (4, 5): 186, (1, 4, 5): 156, (1,): 32,
        (0, 2, 4, 5): 4, (3, 4, 5): 2, (0, 2, 3, 4, 5): 2,
    })
    if sig_multiset != expected:
        raise AssertionError(f"signature multiset != pinned FG 4.0.2 census: {dict(sig_multiset)}")

    # (4) model blob <-> 16x16 cooperative tiling for every recovered dtype.
    model = graph["model"]["size"]
    if model != 9437184:
        raise AssertionError(f"model size {model} != 9437184")
    for footprint, tiles, dtype in ((512, 18432, "fp16"), (1024, 9216, "fp32"), (256, 36864, "int8")):
        if model % footprint != 0 or model // footprint != tiles:
            raise AssertionError(f"model {model}B is not {tiles} whole 16x16 {dtype} tiles")


def test_sdk_ground_truth():
    # AMD's OWN CPU control flow, now read straight from the vendored FidelityFX SDK 2.4.0 submodule
    # (the Redstone line, pinned 7539ae8b). This REPLACES blind DLL disassembly with ground-truth source
    # for the FrameGeneration provider: the ffxDispatch Prepare orchestration, the shared FSR3<->FG resource
    # coupling, the GetGPUMemoryUsage answer model (FI + OpticalFlow + Shared), the FrameInterpolation
    # 48-id intermediate-resource table (roadmap blocker #3 -- now RESOLVED-by-source), and the Vulkan
    # present/pacing/UI-composition host-integration sources (previously an open gap). It ALSO PROVES the
    # MLFI ONNX neural net -- the 382 mlfi_* inference passes -- is NOT shipped in the open SDK, so the
    # ML pass->pass execution ORDER remains the single genuinely closed boundary and ffxDispatch's ML
    # path stays honestly refused. GPU-free (reads SDK source only); self-skips if the SDK submodule is
    # not populated on this checkout.
    fg = ROOT / "third_party/fidelityfx-sdk/Kits/FidelityFX/framegeneration"
    if not fg.is_dir():
        print("SKIP sdk: FFX SDK submodule not populated")
        return

    # (1) FrameInterpolation intermediate-resource model == AMD's 48-id table + 2 constant buffers.
    res_h = fg / "fsr3/include/gpu/frameinterpolation/ffx_frameinterpolation_resources.h"
    if not res_h.is_file():
        raise AssertionError("SDK FrameInterpolation resources header not found (SDK layout drift)")
    txt = res_h.read_text(errors="ignore")
    if not re.search(r"RESOURCE_IDENTIFIER_COUNT\s+48\b", txt):
        raise AssertionError("FrameInterpolation resource COUNT != 48 (drift vs pinned SDK)")
    if not re.search(r"CONSTANTBUFFER_COUNT\s+2\b", txt):
        raise AssertionError("FrameInterpolation CONSTANTBUFFER_COUNT != 2 (drift vs pinned SDK)")
    for rid in ("OUTPUT", "DILATED_DEPTH", "DILATED_MOTION_VECTORS", "DISOCCLUSION_MASK",
                "OPTICAL_FLOW_SCENE_CHANGE_DETECTION", "INPAINTING_PYRAMID_MIPMAP_12",
                "PRESENT_BACKBUFFER", "COUNTERS", "DISTORTION_FIELD"):
        if f"FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_{rid} " not in txt:
            raise AssertionError(f"FrameInterpolation resource id {rid} absent from SDK source")

    # (2) The provider's CPU control flow (ffxDispatch / Query) facts must still match AMD's source.
    prov_path = fg / "fsr3/internal/ffx_provider_fsr3framegeneration.cpp"
    if not prov_path.is_file():
        raise AssertionError("SDK ffxDispatch provider source not found")
    prov = prov_path.read_text(errors="ignore")
    for fact in (
        "MAX_QUEUED_FRAMES = 2",
        "ffxFrameInterpolationContextGetGpuMemoryUsage",
        "ffxOpticalflowContextGetGpuMemoryUsage",
        "ffxSharedContextGetGpuMemoryUsage",
        "ffxFrameInterpolationPrepare(",
        "FFX_FSR3_RESOURCE_IDENTIFIER_DILATED_DEPTH_0",
        "FFX_FSR3_RESOURCE_IDENTIFIER_DILATED_MOTION_VECTORS_0",
        "FFX_FSR3_RESOURCE_IDENTIFIER_RECONSTRUCTED_PREVIOUS_NEAREST_DEPTH_0",
        "prerequisite for MLFI provider enablement",
        "numGeneratedFrames = desc->numInterpolatedFrames",
    ):
        if fact not in prov:
            raise AssertionError(f"ffxDispatch provider fact drifted from pinned SDK: {fact}")

    # (3) The Vulkan present/pacing/UI-composition half is AMD source now, not a reconstruction gap.
    for vk in ("FrameInterpolationSwapchainVK.cpp", "FrameInterpolationSwapchainVK_DebugPacing.cpp",
               "FrameInterpolationSwapchainVK_UiComposition.cpp", "FrameInterpolationSwapchainVK_Helpers.cpp"):
        if not (fg / "fsr3/vk" / vk).is_file():
            raise AssertionError(f"SDK Vulkan present source missing: {vk}")

    # (4) The MLFI neural net (382 mlfi_* ONNX passes) is NOT in the open SDK -- the ML ORDER boundary
    # is real, not a failure to look. Assert zero neural-model bytes anywhere under the FG SDK tree.
    mlfi = re.compile(rb"mlfi_[0-9A-Za-z]|\.onnx", re.I)
    offenders = []
    for p in fg.rglob("*"):
        if p.is_file() and mlfi.search(p.read_bytes()):
            offenders.append(str(p.relative_to(fg)))
    if offenders:
        raise AssertionError(f"unexpected MLFI/neural bytes in open SDK; re-check the ML boundary: {offenders[:5]}")


def test_sdk_classic_controlflow_order():
    # MINE (reverse) AMD's CLASSIC FrameGeneration CPU control flow straight from the pinned v2.4.0 SDK
    # provider source -- the actual scheduleDispatch / CreateComputePipeline call sequence -- and PIN it
    # as a drift gate. This recovers roadmap blocker #2's ORDER for the *classic* half (the only CPU
    # orchestration AMD ships in source); it does NOT and cannot recover the 382 ML `mlfi_*` node order,
    # which stays the closed boundary (proven absent from the SDK by test_sdk_ground_truth). If a future
    # SDK reorders/renames a classic pass this fails loudly. GPU-free; self-skips if the SDK is absent.
    fg = ROOT / "third_party/fidelityfx-sdk/Kits/FidelityFX/framegeneration"
    if not fg.is_dir():
        print("SKIP sdk-order: FFX SDK submodule not populated")
        return
    of = (fg / "fsr3/internal/ffx_opticalflow.cpp").read_text(errors="ignore")
    fi = (fg / "fsr3/internal/ffx_frameinterpolation.cpp").read_text(errors="ignore")

    # Creation inventory (AMD's own pass-name enum), in declared order.
    disp = re.compile(r"scheduleDispatch\((?:[^&)]*?&context(?:Private)?->pipeline([A-Za-z0-9_]+))")
    created = re.compile(r"CreateComputePipeline\((FFX_(?:FRAMEINTERPOLATION|OPTICALFLOW)_PASS_[A-Z0-9_]+)")
    of_disp = disp.findall(of)
    fi_disp = disp.findall(fi)
    fi_created = created.findall(fi)

    # OpticalFlow dispatch order (PrepareLuma, then the input pyramid, then the SCD + search chain).
    if of_disp != ["PrepareLuma", "GenerateOpticalFlowInputPyramid", "GenerateSCDHistogram", "ComputeSCDDivergence",
                   "ComputeOpticalFlowAdvancedV5", "FilterOpticalFlowV5", "ScaleOpticalFlowAdvancedV5"]:
        raise AssertionError(f"OpticalFlow dispatch order drifted from pinned SDK: {of_disp}")
    # FrameInterpolation: Prepare step = FiReconstructAndDilate (first submit), then the Generate chain in
    # this order. The two pyramid submits are wrapped in a local lambda (scheduleDispatchGameVectorField...),
    # so they are not asserted by token position; only the canonical Generate chain is checked as an
    # ORDERED subsequence. If AMD reorders any of these the gate fails.
    if not fi_disp or fi_disp[0] != "FiReconstructAndDilate":
        raise AssertionError(f"FrameInterpolation Prepare step is not ReconstructAndDilate: {fi_disp[:3]}")
    fi_core = ["FiSetup", "FiReconstructPreviousDepth", "FiGameMotionVectorField",
               "FiOpticalFlowVectorField", "FiDisocclusionMask", "FiScfi",
               "InpaintingPyramid", "Inpainting"]
    pos = -1
    for step in fi_core:
        try:
            pos = fi_disp.index(step, pos + 1)
        except ValueError:
            raise AssertionError(f"FrameInterpolation Generate order drifted from pinned SDK: {fi_disp}")
    # The declared CLASSIC pass inventory must equal AMD's 11 FrameInterpolation passes.
    fi_passes = [c for c in fi_created if "FRAMEINTERPOLATION" in c]
    expected_fi_passes = [
        "FFX_FRAMEINTERPOLATION_PASS_RECONSTRUCT_AND_DILATE", "FFX_FRAMEINTERPOLATION_PASS_SETUP",
        "FFX_FRAMEINTERPOLATION_PASS_RECONSTRUCT_PREV_DEPTH", "FFX_FRAMEINTERPOLATION_PASS_GAME_MOTION_VECTOR_FIELD",
        "FFX_FRAMEINTERPOLATION_PASS_OPTICAL_FLOW_VECTOR_FIELD", "FFX_FRAMEINTERPOLATION_PASS_DISOCCLUSION_MASK",
        "FFX_FRAMEINTERPOLATION_PASS_INTERPOLATION", "FFX_FRAMEINTERPOLATION_PASS_INPAINTING_PYRAMID",
        "FFX_FRAMEINTERPOLATION_PASS_INPAINTING", "FFX_FRAMEINTERPOLATION_PASS_GAME_VECTOR_FIELD_INPAINTING_PYRAMID",
        "FFX_FRAMEINTERPOLATION_PASS_DEBUG_VIEW",
    ]
    if fi_passes != expected_fi_passes:
        raise AssertionError(f"FrameInterpolation declared pass inventory drifted: {fi_passes}")


def test_sdk_opticalflow_binding_table():
    # MINE AMD's OpticalFlow descriptor-set layout (the named binding-identifier table the CPU binds
    # uav/srv resources into before each OF pass) from the pinned v2.4.0 SDK -- the ground truth a
    # classic-path implementation must reproduce. Asserts the canonical named bindings, the 7-level
    # optical-flow input pyramid, the SCD (scene-change-detection) binding set, the shared bindings, the
    # 2 optical-flow constant buffers, and that the CPU actually WIRES resources->uav/srv bindings.
    # GPU-free; self-skips if the SDK submodule is not populated.
    fg = ROOT / "third_party/fidelityfx-sdk/Kits/FidelityFX/framegeneration"
    if not fg.is_dir():
        print("SKIP sdk-of-bindings: FFX SDK submodule not populated")
        return
    of = (fg / "fsr3/internal/ffx_opticalflow.cpp").read_text(errors="ignore")

    for binding in (
        "FFX_OF_BINDING_IDENTIFIER_INPUT_COLOR",
        "FFX_OF_BINDING_IDENTIFIER_OPTICAL_FLOW_INPUT",
        "FFX_OF_BINDING_IDENTIFIER_OPTICAL_FLOW_PREVIOUS_INPUT",
        "FFX_OF_BINDING_IDENTIFIER_OPTICAL_FLOW",
        "FFX_OF_BINDING_IDENTIFIER_OPTICAL_FLOW_PREVIOUS",
        "FFX_OF_BINDING_IDENTIFIER_OPTICAL_FLOW_NEXT_LEVEL",
        "FFX_OF_BINDING_IDENTIFIER_OPTICAL_FLOW_SCD_HISTOGRAM",
        "FFX_OF_BINDING_IDENTIFIER_OPTICAL_FLOW_SCD_PREVIOUS_HISTOGRAM",
        "FFX_OF_BINDING_IDENTIFIER_OPTICAL_FLOW_SCD_TEMP",
        "FFX_OF_BINDING_IDENTIFIER_OPTICAL_FLOW_SCD_OUTPUT",
        "FFX_OF_BINDING_IDENTIFIER_SHARED_OPTICAL_FLOW_VECTOR",
        "FFX_OF_BINDING_IDENTIFIER_SHARED_OPTICAL_FLOW_SCD_OUTPUT",
    ):
        if binding not in of:
            raise AssertionError(f"OpticalFlow binding identifier absent from SDK: {binding}")

    # The optical-flow input pyramid is 7 levels deep (base + LEVEL_1..LEVEL_6).
    if not all(f"FFX_OF_BINDING_IDENTIFIER_OPTICAL_FLOW_INPUT_LEVEL_{n}" in of for n in range(1, 7)):
        raise AssertionError("OpticalFlow input pyramid is not the 7-level (base + 1..6) chain")

    # The binding count enum must be declared (bounds every OF descriptor slot).
    if "FFX_OF_BINDING_IDENTIFIER_COUNT" not in of:
        raise AssertionError("FFX_OF_BINDING_IDENTIFIER_COUNT not declared in the SDK OF source")

    # 2 optical-flow constant buffers (cbOF + cbOF_SPD) per the OF resource header.
    res = (fg / "fsr3/include/gpu/opticalflow/ffx_opticalflow_resources.h").read_text(errors="ignore")
    if "FFX_OPTICALFLOW_CONSTANTBUFFER_IDENTIFIER_SPD" not in res:
        raise AssertionError("OpticalFlow SPD constant buffer (cbOF_SPD) identifier missing from SDK")

    # The CPU must actually wire resources -> uav/srv bindings (descriptor-set construction).
    if "uavBindings[FFX_OF_BINDING_IDENTIFIER_OPTICAL_FLOW_SCD_HISTOGRAM]" not in of \
            or "srvBindings[FFX_OF_BINDING_IDENTIFIER_INPUT_COLOR]" not in of:
        raise AssertionError("OpticalFlow CPU does not wire resources into named uav/srv bindings as pinned")


def test_sdk_shared_fsr3_fg_resources():
    # MINE the SHARED FSR3<->FrameGeneration resource pool the FG provider consumes from the upscaler
    # (this is the exact coupling that lets FG reuse FSR3's dilated depth/motion + reconstructed prev-depth
    # intermediates, and it explains the provider's `..._0 + (frameToggle * UPSCALED_COUNT)` indexing).
    # Asserts AMD's 17-id pool: the 2 shared OF outputs + 4 per-frame slots of the dilated/reconstructed
    # triple, `COUNT`=17, `UPSCALED_COUNT`=3. This closes the "shared resource allocation" part of roadmap
    # blocker #3 for the classic path. GPU-free; self-skips when the SDK submodule is not populated.
    fg_res = ROOT / "third_party/fidelityfx-sdk/Kits/FidelityFX/upscalers/fsr3/include/gpu/fsr3/ffx_fsr3_resources.h"
    if not fg_res.is_file():
        print("SKIP sdk-shared-res: FFX SDK submodule not populated")
        return
    txt = fg_res.read_text(errors="ignore")
    if not re.search(r"FFX_FSR3_RESOURCE_IDENTIFIER_COUNT\s+17\b", txt):
        raise AssertionError("shared FSR3 resource pool COUNT != 17 (drift vs pinned SDK)")
    if not re.search(r"FFX_FSR3_RESOURCE_IDENTIFIER_UPSCALED_COUNT\s+3\b", txt):
        raise AssertionError("shared FSR3 UPSCALED_COUNT (per-frame stride) != 3 (drift vs pinned SDK)")
    for shared in ("OPTICAL_FLOW_VECTOR", "OPTICAL_FLOW_SCD_OUTPUT"):
        if f"FFX_FSR3_RESOURCE_IDENTIFIER_{shared} " not in txt:
            raise AssertionError(f"shared FSR3 resource {shared} absent from SDK")
    # 4 frame slots x the dilated-depth / dilated-motion / reconstructed-prev-nearest-depth triple.
    for i in range(4):
        for triple in ("DILATED_DEPTH", "DILATED_MOTION_VECTORS", "RECONSTRUCTED_PREVIOUS_NEAREST_DEPTH"):
            if f"FFX_FSR3_RESOURCE_IDENTIFIER_{triple}_{i} " not in txt:
                raise AssertionError(f"shared FSR3 slot {triple}_{i} absent from SDK")


def test_sdk_spd_inpainting_pyramid_setup():
    # MINE AMD's inpainting-pyramid SPD (single-pass downsampling) dispatch-sizing ground truth: the
    # CPU calls ffxSpdSetup(rectInfo) to derive dispatchThreadGroupCountXY / workGroupOffset /
    # numWorkGroupsAndMips, copies them into the INPAINTING_PYRAMID constant buffer (cbInpaintingPyramid),
    # then dispatches the two pyramid passes (game-vector-field on render size, inpainting on display size).
    # Pinning the exact formula (64px work-group offset, numWorkGroups = dtgc.x*dtgc.y, mips capped at 12)
    # so a classic-path implementation that reproduces the pyramid geometry is provably aligned with AMD.
    # GPU-free; self-skips when the SDK submodule is not populated.
    fg = ROOT / "third_party/fidelityfx-sdk/Kits/FidelityFX/framegeneration/fsr3"
    if not fg.is_dir():
        print("SKIP sdk-spd: FFX SDK submodule not populated")
        return
    spd = (fg / "include/gpu/spd/ffx_spd.h").read_text(errors="ignore")
    fi = (fg / "internal/ffx_frameinterpolation.cpp").read_text(errors="ignore")

    # ffxSpdSetup signature + the exact sizing math.
    if "ffxSpdSetup(" not in spd:
        raise AssertionError("ffxSpdSetup not defined in SDK SPD header")
    for rx in (r"workGroupOffset\[0\]\s*=\s*rectInfo\[0\]\s*/\s*64",
               r"numWorkGroupsAndMips\[0\]\s*=\s*\(dispatchThreadGroupCountXY\[0\]\)\s*\*",
               r"FfxFloat32\(12\)",
               r"floor\(log2\("):
        if not re.search(rx, spd):
            raise AssertionError(f"ffxSpdSetup sizing formula token missing/drifted: {rx}")

    # FG drives it twice (one per pyramid pass) and fills the inpainting-pyramid constant buffer.
    if fi.count("ffxSpdSetup(") < 2:
        raise AssertionError(f"expected >=2 ffxSpdSetup drives in FG cpp, found {fi.count('ffxSpdSetup(')}")
    for tok in ("inpaintingPyramidContants.numworkGroups", "inpaintingPyramidContants.mips",
                "inpaintingPyramidContants.workGroupOffset",
                "FFX_FRAMEINTERPOLATION_PASS_INPAINTING_PYRAMID",
                "FFX_FRAMEINTERPOLATION_PASS_GAME_VECTOR_FIELD_INPAINTING_PYRAMID"):
        if tok not in fi:
            raise AssertionError(f"FG inpainting-pyramid constant-buffer/pass token missing: {tok}")


def test_sdk_present_swapchain_contract():
    # MINE AMD's Vulkan present/pacing/UI-composition host-integration algorithm from
    # fsr3/vk/FrameInterpolationSwapchainVK.* -- the exact present behaviour the classic path must
    # reproduce at Present (previously only a 'no provider' return-code was knowable). Pins the two
    # composition modes, the game<->present queue-family ownership transfer, the UNDEFINED->PRESENT_SRC
    # layout transition, the premul-alpha UI flag, and the UI-resource / maximum-frame-latency VK APIs.
    # GPU-free; self-skips when the SDK submodule is not populated.
    vk = ROOT / "third_party/fidelityfx-sdk/Kits/FidelityFX/framegeneration/fsr3/vk"
    if not vk.is_dir():
        print("SKIP sdk-present: FFX SDK submodule not populated")
        return
    h = (vk / "FrameInterpolationSwapchainVK.h").read_text(errors="ignore")
    c = (vk / "FrameInterpolationSwapchainVK.cpp").read_text(errors="ignore")
    for tok in ("enum class FGSwapchainCompositionMode", "IFrameInterpolationSwapChainVK",
                "FrameInterpolationSwapChainVK", "class FrameInterpolationSwapChainVK"):
        if tok not in h:
            raise AssertionError(f"VK swapchain type/enum missing from SDK header: {tok}")
    for tok in ("eComposeOnPresentQueue", "eComposeOnGameQueue",
                "queueFamilyOwnershipTransferGameToPresent",
                "VK_IMAGE_LAYOUT_PRESENT_SRC_KHR",
                "FFX_FRAMEGENERATION_UI_COMPOSITION_FLAG_USE_PREMUL_ALPHA",
                "ffxRegisterFrameinterpolationUiResourceVK", "ffxGetMaximumFrameLatencyVK",
                "ffxSetFrameGenerationConfigToSwapchainVK"):
        if tok not in c:
            raise AssertionError(f"VK present/pacing contract token missing from SDK source: {tok}")
    # The UI-composition (incl. premul-alpha) + debug-pacing present VS/PS programs are AMD source
    # (HLSL-in-C blob headers under fsr3/vk). Count, not exact spelling, so a rename can't slip past.
    names = [p.name for p in vk.iterdir() if p.suffix == ".h"]
    ui = [n for n in names if "UiComposition" in n and ("VS" in n or "PS" in n)]
    dbg = [n for n in names if "DebugPacing" in n and ("VS" in n or "PS" in n)]
    if len(ui) < 4:   # UiComposition VS/PS + UiCompositionPremul VS/PS
        raise AssertionError(f"SDK VK UI-composition present programs < 4: {ui}")
    if len(dbg) < 2:  # DebugPacing VS/PS
        raise AssertionError(f"SDK VK debug-pacing present programs < 2: {dbg}")


def test_sdk_framegeneration_api_contract():
    # Pin the public FG host-integration contract our provider is written against (the SAME shape Hookshot
    # drives the AMD DLL with) from the vendored ffx_frame_generation.h: the present callback carries
    # currentBackBuffer/currentUI/outputSwapChainBuffer + an isGeneratedFrame flag; ffxConfigure
    # requires presentCallback (+ user ctx) AND frameGenerationCallback (+ user ctx) + the app swapChain;
    # and the dispatch caps generation at outputs[4]/numGeneratedFrames -> AT MOST 4 presented frames
    # (<=3 interpolated -> a 4x ceiling). This grounds roadmap "not-a-small-patch" reality: the API SHAPE
    # is fixed + already mirrored by ffgcheck::configureDescOk / generateDescOk; what remains is the
    # (large) present/pacing + interpolation runtime, NOT a different API. GPU-free; self-skips if absent.
    hdr = ROOT / "third_party/fidelityfx-api/framegeneration/include/ffx_frame_generation.h"
    if not hdr.is_file():
        print("SKIP sdk-api-contract: fidelityfx-api FG header not present")
        return
    t = hdr.read_text(errors="ignore")
    for tok in ("struct FfxApiResource outputs[4];", "uint32_t              numGeneratedFrames;",
                "FfxApiPresentCallbackFunc presentCallback;", "void* presentCallbackUserContext;",
                "FfxApiFrameGenerationDispatchFunc frameGenerationCallback;", "void* frameGenerationCallbackUserContext;",
                "struct FfxApiResource currentBackBuffer;", "struct FfxApiResource currentUI;",
                "struct FfxApiResource outputSwapChainBuffer;", "bool isGeneratedFrame;"):
        if tok not in t:
            raise AssertionError(f"FG API contract token drifted in vendored header: {tok}")
    # Our provider's own check must stay pinned to the 1..3 generated-frame range the header implies.
    chk = ROOT / "src/framegeneration/framegeneration-checks.hpp"
    if "numGeneratedFrames must be 1..3" not in chk.read_text(errors="ignore"):
        raise AssertionError("provider generate check no longer pins the 1..3 generated-frame contract")


def test_sdk_present_pacing_threading():
    # MINE the present/pacing THREADING substrate from AMD's Vulkan swapchain source -- the exact
    # semaphore/queue/worker choreography a classic-4x Present() would implement. Pins: the 8-signal
    # acquire buffer ceiling, the game/interpolation/composition/replacement-buffer/ frame-rendered
    # semaphore family, the dedicated presenter thread (spawn/kill + handle), the game<->present
    # queue-FAMILY ownership transfer backed by family-indexed command-buffer pools, the MIXED timeline
    # + binary semaphore choreography, and the swapchain-REPLACEMENT entry points (acquireNextImage/
    # getSwapchainImages/queuePresent). This is SUBSTRATE (ground truth for the next implementation step)
    # -- it does NOT run or claim any present behaviour. GPU-free; self-skips when the SDK is absent.
    vk = ROOT / "third_party/fidelityfx-sdk/Kits/FidelityFX/framegeneration/fsr3/vk"
    if not vk.is_dir():
        print("SKIP sdk-present-threading: FFX SDK submodule not populated")
        return
    blob = ""
    for f in ("FrameInterpolationSwapchainVK.h", "FrameInterpolationSwapchainVK.cpp",
              "FrameInterpolationSwapchainVK_Helpers.h", "FrameInterpolationSwapchainVK_Helpers.cpp"):
        p = vk / f
        if p.is_file():
            blob += p.read_text(errors="ignore")
    if not blob:
        raise AssertionError("no Vulkan swapchain source found under fsr3/vk")

    if not re.search(r"MAX_ACQUIRE_SEMAPHORE_COUNT\s+8\b", blob):
        raise AssertionError("acquire-semaphore buffer ceiling != 8 in the pinned SDK present source")
    for tok in ("gameSemaphore", "interpolationSemaphore", "compositionSemaphore",
                "replacementBufferSemaphore", "frameRenderedSemaphores",
                "spawnPresenterThread", "killPresenterThread", "presenterThreadHandle",
                "queueFamilyOwnershipTransferGameToPresent", "SubmissionSemaphores",
                "acquireNextImage", "getSwapchainImages", "queuePresent", "familyIndex"):
        if tok not in blob:
            raise AssertionError(f"present/pacing threading token missing from SDK source: {tok}")
    # Mixed timeline + binary choreography must both be present (pacing uses timeline values,
    # per-buffer frame-rendered tracking uses binary semaphores).
    if not ("VK_SEMAPHORE_TYPE_TIMELINE" in blob and "VkTimelineSemaphoreSubmitInfo" in blob
            and "vkGetSemaphoreCounterValue" in blob and "frameRenderedSemaphores" in blob):
        raise AssertionError("present/pacing mixed timeline+binary semaphore choreography drifted")


def test_sdk_swapchain_replacement_provider():
    # Ground the user-pasted "Frame Generation (Swapchain)" doc as machine-checkable: the SDK ships a
    # SEPARATE provider (fsr3/vk/ffx_provider_framegenerationswapchain_vk.cpp) that REPLACES the app's
    # VkSwapchainKHR by handing back replacement function pointers, and -- crucially -- DELEGATES the
    # interpolation workload to the HOST (frameGenerationCallback / InterpolationCommandListVK +
    # InterpolationTextureVK) rather than computing it. So the swapchain present/pacing/UI product is
    # implementable WITHOUT the closed 382-pass ML and WITHOUT fabricating any image. Pins the whole
    # documented API surface. GPU-free; self-skips when the SDK is not populated.
    fg = ROOT / "third_party/fidelityfx-sdk/Kits/FidelityFX/framegeneration"
    prov = fg / "fsr3/vk/ffx_provider_framegenerationswapchain_vk.cpp"
    if not prov.is_file():
        print("SKIP sdk-swapchain-provider: FFX SDK submodule not populated")
        return
    t = prov.read_text(errors="ignore")
    for tok in (
        # The VkSwapchainKHR replacement function-pointer surface handed back via ffxQuery.
        "SwapchainReplacementFunctionsVK",
        "pOutCreateSwapchainFFXAPI", "pOutDestroySwapchainFFXAPI",
        "pOutGetSwapchainImagesKHR", "pOutAcquireNextImageKHR", "pOutQueuePresentKHR",
        "pOutSetHdrMetadataEXT", "pOutGetLastPresentCountFFXAPI",
        # Creation + mode + version descs.
        "CreateContextDescFrameGenerationSwapChainVK",
        "CreateContextDescFrameGenerationSwapChainVersionVK",
        "CreateContextDescFrameGenerationSwapChainModeVK", "composeOnPresentQueue",
        # Interpolation is HOST-supplied (the swapchain does not itself run the FI/ML interface).
        "InterpolationCommandListVK", "InterpolationTextureVK",
        # Present lifecycle + memory queries + pinned swapchain version id.
        "WaitForPresentsVK", "SwapChainGetGPUMemoryUsageVK",
        "FFX_FRAMEGENERATION_SWAPCHAIN_VK_VERSION",
    ):
        if tok not in t:
            raise AssertionError(f"documented Swapchain-provider API token missing from SDK source: {tok}")
    # The before-destroy lifecycle + V2 memory query + async-workload opt-in gate also exist.
    aux = (fg / "fsr3/vk/FrameInterpolationSwapchainVK.cpp").read_text(errors="ignore")
    if "SwapChainGetGPUMemoryUsageVKV2" not in aux:
        raise AssertionError("SwapChain GetGPUMemoryUsageVKV2 not present in the VK swapchain source")
    fi = (fg / "fsr3/internal/ffx_provider_fsr3framegeneration.cpp").read_text(errors="ignore")
    if "ENABLE_ASYNC_WORKLOAD_SUPPORT" not in fi:
        raise AssertionError("FFX_FRAMEGENERATION_ENABLE_ASYNC_WORKLOAD_SUPPORT opt-in flag absent from SDK")


def test_sdk_interpolation_api_contract():
    # Pin the AMD "Frame Generation (interpolation)" API doc the user supplied against the 2.4.0 header our
    # provider compiles against, so the CONTRACT our ffxCreateContext/Configure/Query/Dispatch must honour
    # is machine-checked ground truth (NOT the pixel/ML reference -- those stay out). Pins: the 7
    # FFX_FRAMEGENERATION_ENABLE_* create flags, the version-link desc + FFX_FRAMEGENERATION_VERSION, the
    # PrepareV2 path + the deprecated unused_reset it replaces, both GetGPUMemoryUsage V1 + V2, the
    # Hudless create block, the UI-composition flags (premul + internal double-buffering), the four
    # DRAW_DEBUG_* view flags, FLAG_NO_SWAPCHAIN_CONTEXT_NOTIFY, and the configure fields generationRect
    # / HUDLessColor / onlyPresentGenerated. GPU-free; self-skips if the SDK is not populated.
    hdr = ROOT / "third_party/fidelityfx-sdk/Kits/FidelityFX/framegeneration/include/ffx_framegeneration.h"
    if not hdr.is_file():
        print("SKIP sdk-interp-api: FFX SDK FG header not present")
        return
    t = hdr.read_text(errors="ignore")
    for tok in (
        # 7 create flags + the shared version negotiation.
        "FFX_FRAMEGENERATION_ENABLE_ASYNC_WORKLOAD_SUPPORT",
        "FFX_FRAMEGENERATION_ENABLE_DISPLAY_RESOLUTION_MOTION_VECTORS",
        "FFX_FRAMEGENERATION_ENABLE_MOTION_VECTORS_JITTER_CANCELLATION",
        "FFX_FRAMEGENERATION_ENABLE_DEPTH_INVERTED",
        "FFX_FRAMEGENERATION_ENABLE_DEPTH_INFINITE",
        "FFX_FRAMEGENERATION_ENABLE_HIGH_DYNAMIC_RANGE",
        "FFX_FRAMEGENERATION_ENABLE_DEBUG_CHECKING",
        "FFX_FRAMEGENERATION_VERSION", "CreateContextDescFrameGenerationVersion",
        # Prepare V2 (reset effective) supersedes the deprecated Prepare/unused_reset.
        "DispatchDescFrameGenerationPrepareV2", "unused_reset",
        # Memory usage: post-context (V1) AND the pre-context/NULL-context V2 with ffxOverrideVersion.
        "QueryDescFrameGenerationGetGPUMemoryUsage", "QueryDescFrameGenerationGetGPUMemoryUsageV2",
        # Optional HUD-less create block + UI-composition lifetime/blend flags.
        "CreateContextDescFrameGenerationHudless",
        "FFX_FRAMEGENERATION_UI_COMPOSITION_FLAG_USE_PREMUL_ALPHA",
        "FFX_FRAMEGENERATION_UI_COMPOSITION_FLAG_ENABLE_INTERNAL_UI_DOUBLE_BUFFERING",
        # Debug-view flags (tear lines / reset indicators / view / pacing lines) + no-swapchain notify.
        "FFX_FRAMEGENERATION_FLAG_DRAW_DEBUG_TEAR_LINES",
        "FFX_FRAMEGENERATION_FLAG_DRAW_DEBUG_RESET_INDICATORS",
        "FFX_FRAMEGENERATION_FLAG_DRAW_DEBUG_VIEW",
        "FFX_FRAMEGENERATION_FLAG_DRAW_DEBUG_PACING_LINES",
        "FFX_FRAMEGENERATION_FLAG_NO_SWAPCHAIN_CONTEXT_NOTIFY",
        # Configure fields the doc pins (frameID +1, letterbox rect, hudless surface, generated-only).
        "generationRect", "HUDLessColor", "onlyPresentGenerated",
    ):
        if tok not in t:
            raise AssertionError(f"documented interpolation-API token missing from 2.4.0 header: {tok}")
    # PrepareCameraInfo is deprecated in favour of embedding camera data in PrepareV2.
    if "DispatchDescFrameGenerationPrepareCameraInfo" not in t:
        raise AssertionError("PrepareCameraInfo (deprecated) desc not found as pinned")


def test_sdk_framegeneration_descriptor_layouts():
    # PROVE the descriptor type-id constants our FG provider reads match AMD's 2.4.0 SDK exactly -- the
    # four the older ffx_frame_generation.h lacks (PrepareV2 0x0c, GetGPUMemoryUsageV2 0x0b, present-premul
    # 0x0d, create-version 0x0e) plus the existing configure/generate ids -- so the C++ `kEffectId | 0xNN`
    # composition in framegeneration.hpp cannot silently drift from the reference. Also pins the FRAME
    # GENERATION SWAPCHAIN descs hang off a DIFFERENT effect id (so they are a separate provider, NOT read
    # here). GPU-free; self-skips when the SDK is not populated.
    inc = ROOT / "third_party/fidelityfx-sdk/Kits/FidelityFX/framegeneration/include"
    hdr = (inc / "ffx_framegeneration.h")
    vk = (inc / "vk/ffx_api_framegeneration_vk.h")
    hpp = ROOT / "src/framegeneration/framegeneration.hpp"
    if not (hdr.is_file() and vk.is_file() and hpp.is_file()):
        print("SKIP sdk-desc-layouts: FFX SDK FG headers not present")
        return
    t = hdr.read_text(errors="ignore")

    # Each interpolation desc's sub-id under FFX_API_EFFECT_ID_FRAMEGENERATION must match.
    for macro, subid in (
        ("FFX_API_CONFIGURE_DESC_TYPE_FRAMEGENERATION", "0x02"),
        ("FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION", "0x03"),
        ("FFX_API_QUERY_DESC_TYPE_FRAMEGENERATION_GPU_MEMORY_USAGE_V2", "0x0b"),
        ("FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION_PREPARE_V2", "0x0c"),
        ("FFX_API_CALLBACK_DESC_TYPE_FRAMEGENERATION_PRESENT_PREMUL_ALPHA", "0x0d"),
        ("FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATION_VERSION", "0x0e"),
    ):
        if not re.search(re.escape(macro) + r"\s+FFX_API_MAKE_EFFECT_SUB_ID\(FFX_API_EFFECT_ID_FRAMEGENERATION,\s*" + subid + r"\s*\)", t):
            raise AssertionError(f"2.4.0 header sub-id != {subid} for {macro}")

    # Our provider's C++ constants must carry the SAME composition.
    our = hpp.read_text(errors="ignore")
    for const, subid in (("kDispatchPrepareV2Type", "0x0c"), ("kQueryGPUUsageV2Type", "0x0b"),
                         ("kCreateVersionDescType", "0x0e"), ("kPresentPremulCallbackDescType", "0x0d")):
        if not re.search(re.escape(const) + r"\s*=\s*kEffectId\s*\|\s*" + subid + r"\s*;", our):
            raise AssertionError(f"framegeneration.hpp {const} not set to (kEffectId | {subid}) as AMD's 2.4.0 requires")

    # The swapchain replacement-functions query is a SEPARATE effect id -> not this provider's to read.
    v = vk.read_text(errors="ignore")
    if not re.search(r"FFX_API_QUERY_DESC_TYPE_FGSWAPCHAIN_FUNCTIONS_VK\s+FFX_API_MAKE_EFFECT_SUB_ID\(FFX_API_EFFECT_ID_FRAMEGENERATIONSWAPCHAIN_VK,\s*0x05\s*\)", v):
        raise AssertionError("swapchain replacement-functions query is not on the SWAPCHAIN effect id (0x05) as pinned")


def test_sdk_fg_version_negotiation():
    # The working FSRFG_Dx12 injector reference proves how a real host drives FG: enumerate the DLL's
    # algorithm versions with ffxQueryDescGetVersions (createDescType/outputCount/versionIds/versionNames),
    # then SELECT classic vs ML by linking ffxOverrideVersion.versionId (its note: 0=non-ML 3.1.7, 1=ML 4.0.0)
    # -- the DLL bundles the ML order, so NO pass-order reconstruction is needed to DRIVE it -- and the ML
    # path is further gated by linking a non-zero PrepareCameraInfo (all-zero camera -> the DLL falls back to
    # FSR-SRFG 3). Pins that this negotiation contract exists in the header AND that our provider already
    # answers GET_VERSIONS + models the override block. GPU-free; self-skips if the header is absent.
    api = ROOT / "third_party/fidelityfx-api/api/include/ffx_api.h"
    prov = ROOT / "src/framegeneration/provider.cpp"
    hpp = ROOT / "src/framegeneration/framegeneration.hpp"
    if not (api.is_file() and prov.is_file() and hpp.is_file()):
        print("SKIP sdk-version-negotiation: FFX api header/provider not present")
        return
    t = api.read_text(errors="ignore")
    for tok in ("FFX_API_QUERY_DESC_TYPE_GET_VERSIONS", "ffxQueryDescGetVersions",
                "FFX_API_DESC_TYPE_OVERRIDE_VERSION", "ffxOverrideVersion",
                "versionIds", "versionNames", "createDescType", "outputCount"):
        if tok not in t:
            raise AssertionError(f"version-negotiation contract token missing from api header: {tok}")
    p = prov.read_text(errors="ignore")
    if "FFX_API_QUERY_DESC_TYPE_GET_VERSIONS" not in p or "versionIds[0]" not in p:
        raise AssertionError("provider does not answer ffxQueryDescGetVersions as the injector host expects")
    h = hpp.read_text(errors="ignore")
    if "kOverrideVersionDescType" not in h or "CreateVersionOverride" not in h or "findVersionOverride" not in h:
        raise AssertionError("provider does not model ffxOverrideVersion selection")


def main():
    failures = []
    for test in (test_identity_gate, test_wrong_image_rejected, test_not_pe_rejected,
                 test_dedup_and_census, test_abi_exports, test_pipeline_cache_namespace_isolation,
                 test_pack_manifest_shape, test_framegeneration_exec_selftest,
                 test_descriptor_slot_resolver, test_layout_census_analyzer,
                 test_mlfi_graph_chain_matches_pack,
                 test_mlfi_chain_structure_and_model_tiling,
                 test_sdk_ground_truth,
                 test_sdk_classic_controlflow_order,
                 test_sdk_opticalflow_binding_table,
                 test_sdk_shared_fsr3_fg_resources,
                 test_sdk_spd_inpainting_pyramid_setup,
                 test_sdk_present_swapchain_contract,
                 test_sdk_framegeneration_api_contract,
                 test_sdk_present_pacing_threading,
                 test_sdk_swapchain_replacement_provider,
                 test_sdk_interpolation_api_contract,
                 test_sdk_framegeneration_descriptor_layouts,
                 test_sdk_fg_version_negotiation):
        name = test.__name__
        try:
            test()
            print(f"ok   {name}")
        except AssertionError as error:
            print(f"FAIL {name}: {error}")
            failures.append(name)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
