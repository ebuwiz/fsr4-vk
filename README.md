# fsr4-vk

An independent Vulkan implementation of FSR 4.1.2 FP8 and INT8 for Windows
and Linux, using AMD's model weights and shaders translated to SPIR-V.

This project is not affiliated with or endorsed by AMD.

## Requirements

- Python 3.10+, CMake 3.22.1+ and a C++20 compiler.
- `llvm-objdump`.
- Python 3.14+ for built-in Zstandard support, or the `zstandard` package
  (`python3 -m pip install zstandard`).

## Prepare assets

Get the following provider DLLs from
[AMD's FidelityFX SDK 2.4.0](https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK/releases/tag/v2.4.0)
and place them together in one directory:

- `amd_fidelityfx_upscaler_dx12.dll`
- `amd_fidelityfx_framegeneration_dx12.dll`

The tools require upscaler 4.1.2.4555 and Frame Generation 4.0.2.4555 and verify
both DLL hashes automatically.
A third-party mirror works too, because the hash is checked either way:
https://www.dllme.com/dll/files/amd_fidelityfx_loader_dx12/08561837c75e2f9374e48b2320f473b2
2.3.0.4555 - e2191d3c0828be45eb8affd4e3ec0378f2293adc345b772a7170edcf3b021ecb

https://www.dllme.com/dll/files/amd_fidelityfx_upscaler_dx12/c9b73b38f9f2744c7096de53429db6eb
4.1.2.4555 - aad4fccccb29084a655f9a4f9f613bd22c8c03fb4a0f8840148cc6543b4ccace

https://www.dllme.com/dll/files/amd_fidelityfx_framegeneration_dx12/3f3592db077231c91ae505e4e93efcb3
4.0.2.4555 - e6843252f58b0341a0d80f6a20bc0d840643ee95bc48c22cac08745344577bb6

https://www.dllme.com/dll/files/amdvkffx64/9a68fd5b91acddc58b6e6b845784b5b5
2.4.0.4541 - 179bf7e1c0fd3824a57f1f674924163ce6ce5b4cc11e2594cf773c10175efaf2


```sh
git submodule update --init --recursive
python3 tools/setup.py --dll /path/to/dlls
```

`--dll` must be a directory, not a path to one DLL. Setup extracts the upscaler
models, converts the Frame Generation shader set, translates the shaders and
applies the upscaler patches under `assets/`. These generated assets are not
included in the repository. Later runs can omit `--dll` to verify and reuse both
generated asset packs. Required shader tools are built automatically from the
pinned submodules into `build/tools/`.

## Build

Prepare the assets before building. The scripts configure CMake and build a
Release library with the models and compressed shaders embedded, so the
`assets/` folder is not needed at runtime. Rerun the same command after source
changes for an incremental build.

### Linux

Build the native shared library using the system C++ compiler:

```sh
bash tools/build/linux.sh
```

Output: `build/linux/amd_fidelityfx_upscaler_vk.so`.

### Windows

With Visual Studio C++ support installed, run from a developer PowerShell.
The script configures an x64 build and builds the Release configuration:

```powershell
.\tools\build\windows.ps1
```

Output: `build/windows-native/Release/amd_fidelityfx_upscaler_vk.dll`.

### Windows from Linux

Cross-compile using MinGW-w64. The script expects
`x86_64-w64-mingw32-g++`, `x86_64-w64-mingw32-windres` and
`x86_64-w64-mingw32-dlltool`:

```sh
bash tools/build/windows.sh
```

Output: `build/windows/amd_fidelityfx_upscaler_vk.dll`.
Shader tools run on the build host, a separate Windows Vulkan SDK is not required.
If the tools are in another directory, pass it with `--mingw-location`:

```sh
bash tools/build/windows.sh --mingw-location /path/to/mingw/bin
```

## Runtime levers

Every lever is read once per device at startup and parsed by exact spelling: a value
the parser does not recognise decides nothing, and the log prints what it was handed
(`device_class=refused(Rdna2)`, `unsupported FSR4_VK_MODEL value=…`) so a typo is
visible instead of silently picking a model or a packaging shape for the operator.

| Variable | Accepted values | What it decides |
| --- | --- | --- |
| `FSR4_VK_MODEL` | `auto` or `safe`; `int8` or `1`; `fp8` or `2` | Which model family serves the frame. `auto` and `safe` walk the same validated ladder, INT8 → FP8, without converting either representation: INT8 stays the first rung, and FP8 is reached only where the driver refuses INT8's own SPIR-V capabilities — which is what a low-entry AMD APU does. A device that refuses both is refused, with both blocker lists in the reason. An explicit request is strict: it is refused naming its own blockers, never swapped for the other family behind the caller's back. A version id that names a model (this provider's own INT8/FP8 ids, in the table below) outranks this variable; the reference DLL's own id names no model, so this variable still decides there; and an id no provider advertises is rejected outright. |
| `FSR4_VK_DESCRIPTOR_BACKEND` | `auto`; `sets` or `descriptor-set`; `buffers` or `descriptor-buffer` | Selects conventional descriptor sets or `VK_EXT_descriptor_buffer`. `auto` prefers descriptor buffers only when support and device entry points are available, otherwise it selects sets. Manual choices still pass the same capability and entry-point checks. The legacy `FSR4_DESCRIPTOR_BACKEND` spelling is also accepted. |
| `FSR4_VK_PIPELINE_CACHE_PATH` | file path | Stores Vulkan pipeline-cache data under a GPU/driver-specific default filename, or at this exact path. The Vulkan cache header performs the final UUID and driver-compatibility check. Set `FSR4_VK_PIPELINE_CACHE_DISABLED=1` to keep the cache in memory only. |
| `FSR4_VK_BENCHMARK` | `1` / `0` | Enables the existing GPU timestamp profiler at every frame and reports average, median, P95 and P99 after a warm-up. `FSR4_VK_BENCHMARK_WARMUP` and `FSR4_VK_BENCHMARK_FRAMES` default to 30 and 300. |
| `FSR4_VK_BENCHMARK_PASSES` | `1` / `0` | Adds per-dispatch and post-dispatch barrier timings to benchmark CSV output. |
| `FSR4_VK_DEVICE_CLASS` | `rdna2`, `apu`, `integrated` pin RDNA2 packaging; `not-rdna2`, `modern` lift it | The RDNA2 verdict feeds exactly two things: the INT8 wave32 pin and the AMD shader-pack allowance. The device-ID table in `src/runtime/resources.hpp` is RDNA2-only by design — RDNA1, RDNA3 and RDNA4 run 64-lane wavefronts and "wave32 was slightly faster" was measured on RDNA2 — so this lever is how an operator covers an AMD device whose ID that release has not measured. It adds no ID to the table, and it cannot make a non-AMD device RDNA2. |
| `FSR4_VK_FORCE_WINDOWS_AMD_SHADERS` | unset; `0` off; anything else on | Requests the AMD-optimised Windows shader packs without waiting for the AMD driver check. Native-asset builds have no such packs, which the same log line reports as `windows_shaders_available=no`, so forcing changes the request, not the availability. |

### Version ids

A host can also pin a model by id, through the provider override, so what an id *means*
is part of the interface. Every id this provider knows packs its build half as 4.1.2 —
`(4 << 22) | (1 << 12) | 2` — and differs only in the high half, the FFX API's module
number. The table lives in `src/runtime/gpu-capabilities.hpp` next to `ModelChoice`, and
both the provider and `cap-probe` read it, so neither can drift from the other.

| id | asks for |
| --- | --- |
| `0xF5A5CA1D_01001002` | this provider's auto entry: no model, the ladder decides |
| `0xF5A5CA1F_01001002` | strict INT8 — refused naming its own blockers if the device cannot run it |
| `0xF5A5CA20_01001002` | strict FP8 — refused the same way |
| `0xF5A5CA1E_01001002` | the **reference DLL's own published id**: no model, the ladder decides |

That last row is a compatibility note rather than a cosmetic one. Measured on the pinned
DLL, `0xF5A5CA1E01001002` is the *only* FFX-packed version id in the file (a single 8-byte
word at `.text 0x10b0`; no other prefix in that family occurs anywhere), while the same
image carries both quantizations' weight banks and the INT8, Lite and Navi2 provider classes
`tools/extract/modelclasses.py` censuses. So that id names "the 4.1.2 FSR4 upscaler" and
cannot mean "FP8" — reading it as a strict FP8 pin is what denied a device its upscaler: a
host pinning that id on a driver that runs INT8 but exports no float8 path got `NO_PROVIDER`
even though INT8 was available. If you pinned `…1E` to force FP8, pin `…20`, or set
`FSR4_VK_MODEL=fp8`. `[model] selection source=…` prints which of the three it was
(`loader-vendor-id`, `loader-auto-id`, `loader`) so the reading is visible in a ticket.

Two log lines carry the whole decision for support tickets. `[capabilities]` prints the
device identity, the model support flags, `integrated_amd=` and the driver's own
graphics-IP token as `gfx_ip=` (`GFX1201` for a card RADV names "(RADV GFX1201)",
`unreported` where the driver prints no such token) — reported, never branched on, because
the AMD proprietary driver prints none and a rule built on it would be a guess.
`[device-policy]` prints vendor, device ID, driver, the RDNA2 verdict, how it was reached
(`device_class=auto|pinned-rdna2(…)|refused(…)`) and the wave/allowance outcome. When the
device is AMD, reports itself integrated and matches no table ID, it warns with the
exact `device_id=` the table would need — that is the value to report so the case
becomes a measured rule rather than a permanent override.

## Checks

The build itself verifies every embedded asset payload and alias. On top of that:

- `cmake --build build/linux --target format-check` checks owned C/C++ sources against
  `tools/format.py` without touching files; drop `-check` to apply. This needs a
  `clang-format-22` executable (the build looks that name up with `find_program`): other
  versions are refused rather than trusted, so a formatter upgrade cannot rewrite the
  tree behind a reviewer's back.
- `cmake --build build/linux --target cap-probe && ./build/linux/cap-probe` prints, per
  Vulkan device the loader can see, the profile, the INT8/FP8 blockers, and the ladder
  rung the provider would pick — then the synthetic gates: every ladder case (both
  rungs, either refusal, explicit requests), every `FSR4_VK_DEVICE_CLASS` spelling,
  and the RDNA2 identity edges including the RDNA4 card. It links no assets and no
  provider, so these gates run on a machine with no game. Exit code 0 means every
  gate held: the ladder cases, every `FSR4_VK_DEVICE_CLASS` spelling, the RDNA2 identity
  edges including the RDNA4 card, and every requested version id — including that the
  reference DLL's id names no model and so still upscales on a device that has INT8 but
  no float8 path, which is the case a strict reading used to refuse outright.
- `cmake --build build/linux --target descriptor-smoke && ./build/linux/descriptor-smoke`
  creates two real Vulkan devices on the available AMD INT8-capable GPU, one with
  `VK_EXT_descriptor_buffer` enabled and one without it, records and submits two
  deterministic INT8 and, when supported, FP8 frames through each backend, copies the outputs back and fails
  unless every output byte is identical. This validates the conventional
  descriptor-set fallback against the shipped shaders rather than only probing
  entry points. `FSR4_VK_SMOKE_FRAMES=N` extends the smoke run; enabling
  `FSR4_VK_BENCHMARK=1` uses the runtime's GPU timestamps and reports average,
  median, P95 and P99. `FSR4_VK_SMOKE_RENDER_SIZE=960x540` and
  `FSR4_VK_SMOKE_OUTPUT_SIZE=1920x1080` select production-sized inputs; per-pass
  and barrier CSV rows require `FSR4_VK_BENCHMARK_PASSES=1`. Devices without a
  suitable AMD GPU return exit code 77.
- `FSR4_VK_PIPELINE_CACHE_PATH` records a measured cold start on the first run and
  a warm start on later runs, while `FSR4_VK_BENCHMARK_WARMUP` and
  `FSR4_VK_BENCHMARK_FRAMES` control GPU timing. Cache data is accepted only by
  Vulkan's driver-specific pipeline-cache header and rebuilt on rejection.
- `python3 tools/extract/dispatch.py --dll <dll>` re-derives both model switches and
  the model variant map out of the DLL and fails on any drift from
  `tools/extract/keys/4.1.2.4555.json`, so a pin cannot silently describe a different
  build than the one being extracted.
- `python3 tools/extract/modeltables.py --dll <dll> --expect-unreached 0 --expect-weighted 0 --expect-part-shapes 2`
  accounts for every shader record table by the pattern that selects it, by whether any of
  its records points into a model weight bank, and by the DXBC part shape its containers
  carry. Minutes to run; it is the answer to "does this DLL hold model assets we do not extract".
- `python3 tools/extract/modelclasses.py --dll <dll> --expect-model-names 5 --expect-rtti-names 33 --expect-provider-variants 7 --expect-referenced-descriptors 0 --expect-referenced-strings 5 --expect-labeled-vftables 0`
  names every RTTI class in the image, checks whether any name links to code, and looks for
  a string-keyed provider registry: it finds the seven `ffxProvider_` algorithm variants
  (including `_FSR4Lite_Int8` and `_FSR4_Int8_Navi2`), the five upscaler-model classes
  behind them, no inbound pointer to any of them, and only DXBC signature semantics among
  the pointed-at variant strings — so Lite/Navi2 are provider variants keyed by compiled
  constants here, not a separate family of model assets and not a table you can read.
- Captures are compared frame by frame: `python3 tools/validate/ab-frames.py <capture_a> <capture_b>`.
  Add `--relax-variant` when the two runs differ by more than the descriptor transport,
  which is how an INT8/FP8 A/B is paired, since the two models select different wave
  sizes. Exit 2 means no pair formed — the comparison proves nothing rather than that
  the sides agree.
- `python3 tools/shader-inspect.py <spv-file-or-directory> --output-json report.json
  --output-csv report.csv --summary` records SPIR-V capabilities, workgroup size,
  scalar/vector types, instruction families, resources and parse errors. Add
  `--dxil-root <extracted-dxbc-directory>` to translate and inspect source DXIL/DXBC
  modules too; an undecodable module remains an error record and fails the command.

## Frame Generation (independent second provider)

`amd_fidelityfx_framegeneration_vk.so` is a second, independent provider family
added beside the upscaler. It implements the AMD FidelityFX **Frame Generation
4.0.2.4555** DX12 component (`amd_fidelityfx_framegeneration_dx12.dll`, SHA-256
`e6843252…77bb6`) as a caller-owned-Vulkan provider, extracted and translated
with the same pinned toolchain, but it does NOT share upscaler assets, model
logic, provider registration or runtime resources. The two outputs coexist.

Identity and audit (reproducible, no proprietary files in Git):

```sh
python3 tools/extract/frame-generation/inventory.py \
    --dll amd_fidelityfx_framegeneration_dx12.dll \
    --extract-dxbc build/fg-convert --output build/fg-capabilities/inventory.json
python3 tools/convert/frame-generation/convert.py \
    --dll amd_fidelityfx_framegeneration_dx12.dll --dxbc build/fg-convert
python3 tools/validate/frame-generation/tests.py
```

`convert.py` also **persists every translated SPIR-V** and emits a reproducible
Frame Generation **asset pack** under `assets/frame-generation/pack` (gitignored,
never committed): `modules/<container-hash>.spv`, a `manifest.json` that pins the
source SHA + embedded execution contract and per-module Vulkan facts (entry, stage,
workgroup size, SPIR-V SHA), and a 5-column `modules.tsv` line index the C++
runtime can read without a JSON parser.

`inventory.py` validates the PE image, the exact SHA-256 and pins SDK version
4.0.2.4555 before extracting anything, fails on unsupported container layouts
rather than skipping them, deduplicates containers by hash, and records each
one's stage. Measured on the pinned DLL: **487 DXBC signatures, 486 unique
containers, 0 rejected**, all Shader Model 6.6 *immediate shader graph* blobs
(`ISG1`/`OSG1`, no `FCSR`/`RSIG`). Stage census: 382 `mlfi_*` (ML-neural
inference), 92 classic interpolation `CS`, `mainVS`/`mainPS` + 7 `main` (the
swap-chain raster blit), 3 unsupported PSV. `convert.py` translates every module
with the pinned `dxil-spirv` (`--bindless --ssbo-uav --ssbo-srv`), validates
each, and aggregates a machine-readable execution contract (JSON/CSV/human).

Measured contract (`build/fg-capabilities/execution-contract.json`): all 486
translate with 0 failures; 330 modules are SPIR-V 1.3, 156 are SPIR-V 1.6.
`PhysicalStorageBufferAddresses` (486), `RuntimeDescriptorArray`+descriptor
indexing (484), `CooperativeMatrixKHR`+`VulkanMemoryModel`+`Int8`+8-bit storage
(156), `Float16` (174), `Int16` (48), `Int64`/`Int64Atomics` (1),
`StorageImageWriteWithoutFormat` (82), `GroupNonUniform` subgroup ops (146).

### Build

`bash tools/build/linux.sh` now builds both `amd_fidelityfx_upscaler_vk.so`
and `amd_fidelityfx_framegeneration_vk.so` (plus `framegeneration-probe`).
For an asset-free Frame Generation-only configure, use
`cmake -S . -B build/fg-only -DFSR4_BUILD_UPSCALER=OFF` and build
`amd_fidelityfx_framegeneration_vk`; this bypasses the unrelated upscaler asset
generation while preserving the normal combined build.
Verified exports (`nm -D`, `readelf -d`): exactly `ffxCreateContext`,
`ffxDestroyContext`, `ffxConfigure`, `ffxQuery`, `ffxDispatch`, SONAME
`amd_fidelityfx_framegeneration_vk.so`, no C++ leakage. Windows cross-build is
source-compatible with the same `__declspec`/`_WIN32` guards but was not run on
this host (no MinGW-w64 installed), so it is reported as unverified, not tested.

### Runtime levers

`FFG4_VK_EXTENSIVE_LOG=1` (verbose context/dispatch logging).
`FFG4_VK_PIPELINE_CACHE=<file>` selects the namespaced pipeline-cache container path
(default is identity-named `amd_fidelityfx_framegeneration_vk.pcache.<gpu>.<driver>` in
the working directory); `FFG4_VK_PIPELINE_CACHE_DISABLE=1` runs uncached;
`FFG4_VK_PROFILE=1` reports cache create/seed/save byte counts per context. Manual levers
must never bypass the capability gate; unknown values are diagnosed, not ignored.
The upscaler's `FSR4_VK_*` variables are independent and unchanged.

The Frame Generation create chain must contain its real effect descriptor,
type-3 Vulkan backend, and the provider's custom type-7
`CreateBackendVkEnabledFeaturesDesc` block. That last block declares the exact
feature bits the host enabled in `VkDeviceCreateInfo`; Vulkan cannot query the
enabled feature set from an existing `VkDevice`. Omitting it or claiming a feature
the physical device does not support refuses context creation.

The provider now owns a **namespaced, identity-validated `VkPipelineCache`**
(`src/framegeneration/pipeline-cache.hpp`): the driver's opaque cache blob is wrapped in
our own container whose header pins the algorithm namespace (`fg4`, distinct from the
upscaler's `fs4`) plus the exact GPU name + driver/api/vendor/device ids, so a cache
built by another effect, GPU or driver is never reused (it cold-starts instead).
Measured on RX 9070 XT: a cold context seeds `0` bytes, teardown serializes `36` bytes,
and the next context reloads that blob (`seedBytes=36`) — a warm start isolated from
the upscaler (verified by container magic `FFG4PC1` + algorithm tag `fg4`).

### Execution + pipeline-cache benchmark harness (`framegeneration-exec`)

`build/linux/framegeneration-exec` drives the provider's OWN namespaced cache (`fg4`)
with a **real compiled** compute pipeline (a synthetic substrate kernel embedded from
`tools/framegeneration-selftest.comp`, NOT a Frame Generation pass), measures cold vs warm
pipeline creation, runs a benchmark loop and per-stage profiling, and loads every pack
SPIR-V as a driver shader module. It shares only the header-only container format
(`src/framegeneration/pipeline-cache.hpp`) with the provider, so there is one cache
source of truth and no upscaler coupling.

Measured on RX 9070 XT (real cache entry, unlike the probe's header-only round trip):
**cold create 7.479 ms vs warm create 0.022 ms** (~340× faster — the cache hit is real),
the container grew from an empty seed to **1032 bytes** carrying a compiled pipeline,
`FFG4_VK_BENCHMARK=:8` repeated creation at sub-millisecond cache lookups, and **486/486
pack modules load on the target GPU** (a structural acceptance census, not a target
compilation). A further **target-compilation census** (`--compile-pack`, see below) creates
real `VkComputePipeline`s for the standardizable subset. Env: `FFG4_VK_BENCHMARK=N|:N|1`,
`FFG4_VK_PROFILE=1`, `FFG4_VK_PIPELINE_CACHE`, `FFG4_VK_PIPELINE_CACHE_DISABLE`, `FFG4_VK_PACK`,
`FFG4_VK_COMPILE_PACK`, `FFG4_VK_DB_PARITY`, `FFG4_VK_WRITE_DESC`. Exact-output value verification is reported as `pending-host-transport`,
not asserted.

**Why FG pipelines seemed blocked by a "unified transport" — and why it is NOT (measured):** the
translated shaders declare *multiple descriptor variables at one `set:binding` slot* (HLSL
re-binds a single resource-heap slot as several views). Crucially, measured across the
whole DLL, every such slot is claimed by variables of the **same** descriptor kind (several
views into one runtime array), NOT mixed kinds — so a single same-type
`VkDescriptorSetLayoutBinding` **de-duplicates** them and a plain `VkPipelineLayout`
suffices. The inspector now **resolves every
descriptor variable to a Vulkan descriptor kind + arrayness from the pointer storage
class the fork uses** (`StorageBuffer`→storage buffer, `Uniform`→uniform buffer,
`UniformConstant`+`OpTypeImage`→combined/storage image by the sampled flag; a runtime-array
pointer target → bindless whole-array). This corrected the earlier UniformConstant-only
census. The per-module layout (in `pipelines.tsv`, 8 columns: `… standardizable layout
transport`) measures, across all 486 modules: **403** have no duplicate `(set,binding)` slot
(`plain`), **83** have duplicate slots that are all the **same** kind and therefore **collapse
into one binding** (`collapsible` — the 87 duplicate slots are 40 `storage-image×2`, 29
`combined-image×2`, 14 `storage-buffer×2`, 4 `combined-image×3`), and **0** have a
cross-kind slot (`unified`): **`cross_kind_slots = 0`**. Descriptor-kind totals: 568 storage-buffer,
438 uniform-buffer, 135 combined-image, 143 storage-image, and 43 **bindless samplers**
(`OpTypeSampler` → `VK_DESCRIPTOR_TYPE_SAMPLER`) — **zero unresolved slots now** (the 43
formerly-unknown were all samplers). **Every** module declares an exactly **32-byte `RootConstants`**
push block (8×uint). At module level, **109 of 486** use bindless whole-array *images*, **470 of 486**
use bindless whole-array storage/uniform *buffers* (the ML neural kernels), and **43** use
bindless whole-array *samplers*. Compute modules: **382 neural + 92 classic = 474**.

**Measured target-compile census (`framegeneration-exec --compile-pack`, env
`FFG4_VK_COMPILE_PACK=1`):** the harness derives a plain `VkPipelineLayout` per
compute module from the pack's `layout` column (one set layout per declared
descriptor-set index — empty for gaps — **de-duplicating same-kind duplicate slots** into
one `VK_WHOLE_ARRAY`+`PARTIALLY_BOUND` binding, and a 32-byte push range) and **attempts
`vkCreateComputePipelines` on the live device**. On the primary target AMD RX 9070 XT
(RADV GFX1201): **attempted=474, created=474, failed=0, skipped=0** (82 via the
`collapsible` de-dup path) — **every FG compute module's** SPIR-V + derived layout compiles.
Running with `--descriptor-buffer-parity` (`FFG4_VK_DB_PARITY=1`) recreates those *identical*
layouts under `VK_PIPELINE_CREATE_DESCRIPTOR_BUFFER_BIT_EXT` (the fork's descriptor-buffer
transport, the flag the upscaler uses at `src/runtime/context.cpp`): also **474 created,
0 failed** — so **all 474 FG compute modules accept both descriptor-set and descriptor-buffer
transport** at pipeline-creation level. This is a target-*compilation* + transport-*parity*
acceptance census, explicitly **not** dispatch, execution, algorithm output, or image
quality. Consequence: Next #3 (descriptor-set ↔ descriptor-buffer parity **on real FG
shaders**) is now **measured for all 474** FG compute modules (mirroring the upscaler's
`descriptor-smoke` byte-identity), and the once-feared **Phase 1 unified descriptor-buffer
transport turns out NOT to be required** — no FG shader has a cross-kind slot. Pipeline
creation for the entire FG compute set is **solved with plain layouts**; the remaining
runtime work is **host descriptor population + dispatch** (running the interpolation `CS`
graph and the ML `mlfi_*` graph) and the swap-chain present. The upscaler still proves the
byte-level parity mechanism
(`descriptor-smoke`: both transports byte-identical on the same shaders), which the
FG runtime will adopt once its per-module bindings are known.

**Measured host-descriptor-population census (`--write-descriptors`, env
`FFG4_VK_WRITE_DESC=1`):** the harness additionally POPULATES one real entry into
every binding of every FG set layout (2692 sets across 474 modules, using real backing
buffers/images/sampler) and measures BOTH host transports. The harness's contract device now
enables `VK_EXT_descriptor_buffer` + `VkPhysicalDeviceDescriptorBufferFeaturesEXT` (what the host
runtime must request). On RX 9070 XT:

- **Conventional `VkDescriptorSet` path** (`vkAllocateDescriptorSets`): allocates for only **8**
  sets and **fails for 2684** (first failure on an `mlfi_*` module). FG's bindings are
  `VK_WHOLE_ARRAY` runtime arrays, which conventional descriptor sets structurally cannot size.
- **`VK_EXT_descriptor_buffer` path** (the transport that works here): the driver reports the real
  per-element byte costs the runtime sizes sets from — `vkGetDescriptorEXT` sizes **storage-buffer
  16 / uniform-buffer 16 / combined-image-sampler 64 / storage-image 32 / sampler 16** — and serializes
  **all 1208** FG descriptor bindings with **0 failures** (one `vkGetDescriptorEXT` per binding,
  25184 bytes total). Every whole-array binding — the exact ones classic sets cannot hold — is
  therefore host-populatable as `RESOURCE_BIT`-buffer descriptor bytes. `vkGetDescriptorEXT` is resolved
  through the device (it is not a global linker symbol), exactly as the upscaler runtime does
  (`dispatch.get_descriptor`). (`vkGetDescriptorSetLayoutSizeEXT` is also callable now and reports a
  non-zero base footprint for **1208** sets; it returns 0 only for a pure whole-array set, which is
  precisely why the per-element sizes above are the authoritative descriptor-buffer metric.)

This is the precise, measured reason the FG host runtime **must** use the `VK_EXT_descriptor_buffer`
transport (the upscaler's proven `descriptor-smoke` byte-identity path), and it is the gate before
any dispatch. **No dispatch is performed and no FG algorithm runs.**

### Feasibility boundary and honest scope (what is NOT claimed)

Frame Generation is **not** a single-dispatch compute effect. Measured on the
DLL it is an optical-flow classic-interpolation pipeline (92 `CS` passes) plus
the **ML Frame Generation** neural-inference graph (382 `mlfi_*` passes over
embedded neural weights) plus a **swap-chain present** (the `mainVS`/`mainPS`
raster blit) that depends on host presentation, pacing and queue ownership.
That presentation contract cannot be implemented inside a compute-only provider.

This change delivers and **tests**: the provider ABI, caller-owned-device
capability detection, version advertisement (`0xF600000001000002`, the DLL's
own published id), the measured execution contract, a reproducible Frame Generation
**asset pack**, a namespaced identity-validated **pipeline cache** (`fg4` vs the
upscaler's `fs4`), and the clean refusal of anything not yet implemented. It does **not**
yet implement or image-quality-validate the optical-flow / ML / present runtime, so
**`ffxDispatch` refuses with a named blocker rather than fabricating** a blend, a
motion-blur pass or a made-up optical-flow stand-in. Therefore `amd_fidelityfx_framegeneration_vk.so`
is **not** a completed Frame Generation equivalent: it is the verified ABI + capability
contract + asset pack + cache/host-integration boundary that a validated
interpolation/present runtime would plug into. Do not treat it as a drop-in
replacement for AMD's Frame Generation until the remaining phases — **running** (dispatching)
the interpolation and ML graphs from the pack, the FG-side descriptor-set ↔ descriptor-buffer
**output byte-identity at dispatch** (the analog of the upscaler `descriptor-smoke`, which needs
host binding), swap-chain present/pacing, benchmarking, and reference image-quality (PSNR/SSIM/flicker
vs the DX12 baseline) — are implemented and measured; note pipeline *creation* and both-transport
*acceptance* for **all 474** FG compute modules are already implemented and measured.

`tools/framegeneration-probe.cpp` produces the compatibility matrix on real
hardware. Measured on the primary target: AMD RX 9070 XT (RADV GFX1201) meets
the contract (create OK, advertises the id, dispatch refused pending runtime);
the Intel integrated (RPL-S) device is refused with
`missing enabled feature VK_KHR_cooperative_matrix (156 ML Frame Generation
neural modules use CooperativeMatrixKHR)`.
On the RX 9070 XT path the probe additionally runs a cold→warm pipeline-cache round
trip: the provider seeds an empty cache, serializes `36` bytes on teardown, reloads
them on the next create (`seedBytes=36`), and the probe verifies the on-disk container
carries the `FFG4PC1` magic and the `fg4` algorithm tag (never the upscaler's `fs4`).

## Documentation
FSR 4.1.2:
 - https://gpuopen.com/manuals/fsr_sdk/samples/super-resolution/
 - https://gpuopen.com/manuals/fsr_sdk/techniques/super-resolution-ml/
Framegeneration 4.0.2:
 - https://gpuopen.com/manuals/fsr_sdk/techniques/frame-interpolation-ml/

Another proyect doing the same:
 - https://github.com/DuIslingr/ffx-vk-linux


## Licensing

Code written for this project is licensed under [GPL-3.0-only](LICENSE).
Third-party code and assets retain their respective licences. See
[LICENSES/](LICENSES/) for the licence texts and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for attribution and details.
