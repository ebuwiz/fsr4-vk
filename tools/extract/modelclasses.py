#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 fsr4-vk contributors.
# See LICENSE in the project root.

"""How the Fsr4 model class names and the ffxProvider_ variants look in the image.

`modeltables.py` settles which shader record tables exist. The class names
`Fsr4LiteInt8UpscalerModel` and `Fsr4Int8navi2UpscalerModel` in the same image were the
only handles anyone had for the *selection* question - which variant makes the SDK run a
Lite or Navi2 model - and this tool's name census answers it at the level the names are
actually keyed on:

    tools/extract/modelclasses.py --dll amd_fidelityfx_upscaler_dx12.dll \
        --expect-model-names 5 --expect-rtti-names 33 --expect-provider-variants 7 \
        --expect-referenced-descriptors 0 --expect-referenced-strings 5 \
        --expect-labeled-vftables 0

The 33 `.?AV` names of this image include seven `ffxProvider_` algorithm variants -
`_FSR4Lite_Int8`, `_FSR4_Int8_Navi2`, `_FSR4_Int8`, `_FSR4`, `_FSR4_Common`, `_FSR3Upscale`
and `_FSR2` - over the `ffxProvider`/`ffxProviderExternal` bases. Lite and Navi2 are therefore
*provider* variants keyed by algorithm, which is exactly the axis this repository already
models in `src/provider/provider.cpp`, not a fourth family of model assets. The five
upscaler-model names (`fsr4_common::UpscalerModel` interface, `Fsr4UpscalerModel` base, and
the plain/Lite/Navi2 INT8 classes) are the implementations behind those providers, and no
FP8-named model class exists in the image at all: FP8 is an asset and capability contract
here, not a compiled class in that DLL.

Linkage from names to code is absent, and that is the second half of the result: no u64
anywhere in the file equals one of those names' addresses or an address 4, 8, 16, 24 or
32 bytes before it - the offsets at which an x64 TypeDescriptor could begin - so the RTTI
records are bytes with no inbound pointer, which is why `dynamic_cast`/`typeid` is provably
not how the SDK selects a variant.

Not-a-registry, the third angle on the same question: outside the RTTI names, 1616 strings
in the image name a variant token (both quantizations, the two disputed variants, the four
FSR algorithms, the upscale verb), and exactly five of them have any inbound pointer -
`cbFSR2`, `cbFSR3Upscaler`, `r_internal_upscaled_color`, `rw_upscaled_output` and
`rw_internal_upscaled_color` - all DXBC signature semantics, i.e. c-buffer and shader
resource names, not provider lookup keys. So the SDK has no string-keyed provider table to
read a variant key out of; its variant choice is compiled constants. The two c-buffer names
are a useful side result: they are the FSR2 and FSR3 kits' own signatures, which is what the
`ffxProvider_FSR2` and `ffxProvider_FSR3Upscale` classes imply, and it matches the 15 non-model
switches `modeltables.py` counts.

Constructors: the data sections hold 17 vftable-shaped runs of `.text` pointers (entry
counts 8, 5 or 3), and `.text` installs 15 of them at 22 sites through rip-relative loads,
so the constructor sites are enumerable as addresses without touching `.reloc` - but zero
runs carry an RTTI static-type pointer, consistent with the 0 referenced names, so no
constructor can be labelled by reading this file; the names give the vocabulary, and only
`.reloc`-aware disassembly or Windows runtime observation would tie a site to a variant.
"""

import argparse
import re
import struct
import sys
from collections import Counter
from pathlib import Path

from sdk import RIP_POINTER, SDK

RTTI_PREFIX = b".?AV"
# An x64 TypeDescriptor stores its name after 0-32 bytes of pointer/hash fields, so
# every offset a static-type pointer could plausibly hold is probed rather than one
# guessed layout being assumed correct.
DESCRIPTOR_SLOTS = (0, 4, 8, 16, 24, 32)
# Three consecutive function pointers is the minimum credible vftable: two could be a
# callback pair in a data table, three is what every real MSVC vftable in this image has.
VFTABLE_RUN_MIN = 3


def rtti_names(sdk):
    """Every .?AV-prefixed type name in the image, as (rva, name).

    The prefix marks an MSVC TypeDescriptor name field, so its presence says the string
    is RTTI data rather than a logging key - and the absence of any further structure
    is what the linkage scan then measures.
    """
    data = sdk.data
    found = set()
    pos = data.find(RTTI_PREFIX)
    while pos != -1:
        end = data.find(b"\x00", pos + 4)
        if end != -1 and 0 < end - pos <= 256:
            found.add((sdk.image().rva(pos), data[pos:end].decode("latin-1", "replace")))
        pos = data.find(RTTI_PREFIX, pos + 1)
    return sorted(found)


def pointer_hits(sdk, value):
    """File offsets whose 8-byte word equals one pointer value."""
    packed = struct.pack("<Q", value)
    offsets = []
    pos = sdk.data.find(packed)
    while pos != -1:
        offsets.append(pos)
        pos = sdk.data.find(packed, pos + 1)
    return offsets


def referenced_names(sdk, names):
    """Which RTTI names have an inbound pointer at the name or a slot before it.

    This is the whole RTTI question in one number: a referenced name means a TypeDescriptor
    is reachable, which would make the class traceable to the code that uses it.
    """
    referenced = []
    for rva, name in names:
        hits = sum(len(pointer_hits(sdk, sdk.image_base + rva - shift))
                   for shift in DESCRIPTOR_SLOTS)
        if hits:
            referenced.append((name, rva, hits))
    return referenced

def data_sections(sdk):
    """The initialised data areas, as (name, rva, file offset, virtual size).

    Vftables live in `.rdata`/`.data*`, never in `.text`, which is what lets the run
    scan below treat a run of `.text` pointers as a table rather than as code.
    """
    out = []
    for name, (start, raw, length) in zip(sdk.section_names, sdk.sections):
        if name.startswith(".rdata") or name.startswith(".data"):
            out.append((name, start, raw, length))
    return out


def label_slots(names):
    """RVAs a static-type pointer would have to hold to identify a class.

    The pointer sits at the TypeDescriptor start, and the name field is 0-32 bytes into
    that structure, so a name is identified by `name_rva - shift` for any shift probed.
    This is exactly the inverse of what `referenced_names` scans for, so both halves
    of the test use one definition of "linked".
    """
    return {rva - shift for rva, _ in names for shift in DESCRIPTOR_SLOTS}


def vftable_runs(sdk, names):
    """Runs of `.text` pointers in the data areas, plus their static-type slot.

    In MSVC x64 the u64 just before a vftable is the RTVQ pointer to the class's
    TypeDescriptor. If no run has such a pointer, the run is an unlabelled function table
    and the code that installs it can never be attributed to a class name.
    """
    low, high = sdk.section_bounds(".text")
    slots = label_slots(names)
    runs = []
    for name, start, raw, length in data_sections(sdk):
        limit = min(length, len(sdk.data) - raw)
        pos = raw
        while pos + 8 <= raw + limit:
            entry = int.from_bytes(sdk.data[pos:pos + 8], "little") - sdk.image_base
            if not low <= entry < high:
                pos += 8
                continue
            begin = pos
            count = 0
            while pos + 8 <= raw + limit:
                entry = int.from_bytes(sdk.data[pos:pos + 8], "little") - sdk.image_base
                if not low <= entry < high:
                    break
                count += 1
                pos += 8
            if count < VFTABLE_RUN_MIN:
                continue
            table = start + (begin - raw)
            static = (int.from_bytes(sdk.data[begin - 8:begin], "little") - sdk.image_base
                      if begin - 8 >= raw else None)
            runs.append({"table": table, "entries": count, "section": name,
                         "static_type": static, "labelled": static in slots})
    return runs


def installers(sdk, runs):
    """`.text` rip-relative loads of each vftable: the constructor sites, unlabelled.

    objdump resolves the `[rip + off]` target into a trailing `# 0x...` comment, which
    `RIP_POINTER` reads, so a constructor that installs a table is visible without
    walking `.reloc` - it simply carries no name.
    """
    wanted = {run["table"] for run in runs}
    hits = Counter()
    for _, text in sdk.instructions():
        match = RIP_POINTER.search(text)
        if match is not None and int(match.group(1), 16) - sdk.image_base in wanted:
            hits[int(match.group(1), 16) - sdk.image_base] += 1
    return hits


# One word at a time, anchored on a variant token rather than scanning every printable
# run in a 60 MiB image: this pattern hits 1616 strings on the pinned image, of which
# five have an inbound pointer. The tokens are the ones an FFX provider key would use -
# the two disputed variants, both quantizations, the four FSR algorithms and the upscale
# verb of the FSR3 provider - so a miss here would be a spelling the SDK does not use.
VOCABULARY = re.compile(rb"(?i)[A-Za-z0-9_.:@-]{0,64}(navi2|lite|int8|fp8|float8|"
                       rb"fsr[1234]|upscal)[A-Za-z0-9_.:@-]{0,64}")


def variant_strings(sdk):
    """Plain (non-RTTI) strings naming a variant, and whether anything points at them.

    A provider registry would key variants by string id, which is the second place a
    Lite/Navi2 key could live after the RTTI names. A run only counts as a key if some
    u64 in the image equals its start address, the way a table of string pointers looks.
    """
    data = sdk.data
    found = set()
    for match in VOCABULARY.finditer(data):
        begin = data.rfind(b"\x00", 0, match.start()) + 1
        end = data.find(b"\x00", match.start())
        if end == -1:
            end = len(data)
        text = data[begin:end]
        if len(text) > 200 or text.startswith(RTTI_PREFIX):
            continue
        found.add((sdk.image().rva(begin), text.decode("latin-1", "replace")))
    referenced = []
    for rva, text in sorted(found):
        hits = len(pointer_hits(sdk, sdk.image_base + rva))
        if hits:
            referenced.append((text, rva, hits))
    return sorted(found), referenced


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dll", type=Path, required=True)
    parser.add_argument("--expect-model-names", type=int, default=None,
                       help="exit non-zero unless this many upscaler model names exist")
    parser.add_argument("--expect-rtti-names", type=int, default=None,
                       help="exit non-zero unless this many .?AV names exist")
    parser.add_argument("--expect-referenced-descriptors", type=int, default=None,
                       help="exit non-zero unless this many RTTI names have an inbound "
                       "pointer")
    parser.add_argument("--expect-labeled-vftables", type=int, default=None,
                       help="exit non-zero unless this many vftable runs carry an RTTI "
                       "static-type pointer")
    parser.add_argument("--expect-provider-variants", type=int, default=None,
                       help="exit non-zero unless this many ffxProvider_ algorithm variant "
                       "names exist")
    parser.add_argument("--expect-referenced-strings", type=int, default=None,
                       help="exit non-zero unless this many non-RTTI variant-vocabulary "
                       "strings have an inbound pointer")
    args = parser.parse_args(argv)

    sdk = SDK(args.dll)
    names = rtti_names(sdk)
    models = [(rva, name) for rva, name in names if "UpscalerModel" in name]
    print(f"image: FSR {sdk.version} sha256={sdk.sha256[:12]} sections={len(sdk.sections)}")
    print(f"RTTI (.?AV) names: {len(names)}, of them upscaler models: {len(models)}")
    for rva, name in models:
        print(f"  {rva:#x}  {name}")
    print("all RTTI names in the image:")
    for rva, name in names:
        print(f"  {rva:#x}  {name}")
    named_fp8 = [name for _, name in names
                 if any(token in name for token in ("FP8", "Float8", "fp8"))]
    print(f"names naming an FP8 model class: {len(named_fp8)}")
    providers = [name for _, name in names if "ffxProvider" in name]
    variants = sorted(name for name in providers if name.startswith(".?AVffxProvider_"))
    print(f"ffxProvider* names: {len(providers)}, of them algorithm variants: {len(variants)}")
    for name in variants:
        print(f"  {name}")

    referenced = referenced_names(sdk, names)
    print("\nRTTI names with an inbound pointer at the name or a slot before it: "
          f"{len(referenced)}")
    for name, rva, hits in referenced:
        print(f"  {name} at {rva:#x}: {hits} pointer(s)")

    strings, referenced_strings = variant_strings(sdk)
    print(f"\nnon-RTTI strings naming a variant token: {len(strings)}")
    print(f"  of those, pointed at by a u64: {len(referenced_strings)}")
    for text, rva, hits in referenced_strings:
        print(f"    {text!r} at {rva:#x} referenced {hits} time(s)")
    print("  meaning: the pointed-at strings are DXBC signature semantics (cbFSR2, cbFSR3Upscaler,"
          " the upscaled-colour resources), not provider keys, so no string-keyed provider registry"
          " exists to read a variant key out of; the FSR2 and FSR3 c-buffer names also confirm this"
          " image carries those kits beside the FSR4 ones.")

    runs = vftable_runs(sdk, names)
    sizes = Counter(run["entries"] for run in runs)
    print(f"\nvftable-shaped runs in the data sections: {len(runs)}")
    print(f"  entry-count histogram: {dict(sizes.most_common(12))}")
    print("  runs carrying an RTTI static-type pointer: "
          f"{sum(1 for run in runs if run['labelled'])}")

    hits = installers(sdk, runs)
    print(f"\nruns installed by a rip-relative load in .text: {len(hits)} of {len(runs)}, "
          f"{sum(hits.values())} sites")
    for table, count in sorted(hits.items(), key=lambda pair: -pair[1])[:12]:
        entries = [run["entries"] for run in runs if run["table"] == table]
        print(f"  table {table:#x} entries={entries} installed at {count} site(s)")
    print("  meaning: the variant vocabulary is in the file - ffxProvider_FSR4Lite_Int8 and "
          "_FSR4_Int8_Navi2 name the Lite and Navi2 providers - and the constructor sites are "
          "enumerable, but nothing links a name to a site, so no key-to-constructor map can be "
          "read out of it; which model this project runs is decided by gpu-capabilities.cpp and "
          "FSR4_VK_MODEL, and modeltables.py is what says which assets exist at all.")

    checks = {"upscaler model names": (len(models), args.expect_model_names),
              "RTTI names": (len(names), args.expect_rtti_names),
              "referenced descriptors": (len(referenced), args.expect_referenced_descriptors),
              "labeled vftables": (sum(1 for run in runs if run["labelled"]),
                                   args.expect_labeled_vftables),
              "provider variants": (len(variants), args.expect_provider_variants),
              "referenced strings": (len(referenced_strings), args.expect_referenced_strings)}
    for what, (found, expected) in checks.items():
        if expected is not None and found != expected:
            raise SystemExit(f"expected {expected} {what}, found {found}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
