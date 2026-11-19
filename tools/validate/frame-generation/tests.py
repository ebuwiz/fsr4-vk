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
import json
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


def main():
    failures = []
    for test in (test_identity_gate, test_wrong_image_rejected, test_not_pe_rejected,
                 test_dedup_and_census, test_abi_exports, test_pipeline_cache_namespace_isolation,
                 test_pack_manifest_shape, test_framegeneration_exec_selftest,
                 test_descriptor_slot_resolver, test_layout_census_analyzer):
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
