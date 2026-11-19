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

## Upscaler API compliance (AMD FSR Upscaling 4.1.2)

This provider implements the documented `ffx_upscale.h` contract directly from the pinned
headers, so a host can pin the behavior against AMD's own `amd_fidelityfx_upscaler_vk.dll`
(which ffx-vk-linux shows runs on native Linux through AMD's Vulkan DLL).

- **Quality modes** (`ffxQueryDescUpscaleGetUpscaleRatioFromQualityMode` /
  `…GetRenderResolutionFromQualityMode`): NativeAA 1.0x, Quality 1.5x, Balanced 1.7x,
  Performance 2.0x, UltraPerformance 3.0x — the same ladder the asset pack carries
  (`native`/`quality`/`balanced`/`performance`/`ultraperf` presets).
- **Deterministic `ffxQuery` helpers are answered, not stubbed**: the upscaler ratio, the
  display→render resolution, `ffxQueryDescUpscaleGetJitterPhaseCount` and
  `…GetJitterOffset` (the R2 low-discrepancy sequence) are computed from a single
  shared source (`src/provider/upscaler-queries.hpp`) and pinned by the GPU-free
  `upscaler-query-test`. Our vendored `ffx_upscale.h` is AMD's newer (2025) revision — a
  superset of the 2024 header (it adds `NON_LINEAR_COLORSPACE 1<<8`, `DEBUG_VISUALIZATION
  1<<9`, `GPU_MEMORY_USAGE_V2 0x00010009`, `GET_RESOURCE_REQUIREMENTS 0x0001000a`); the
  GPU-memory-usage V1/V2 queries (`0x00010008`/`0x00010009`) are now **answered with the provider's
  true owned amount**: `fsr4query::gpuMemoryUsage()` returns `{0, 0}` because this provider retains
  no persistent GPU resources (every surface is host-supplied at dispatch and FSR4's intermediate
  targets are allocated through the caller's allocator) — the honest value, not a fabricated size.
  (The separate Frame Generation provider's GPU-memory-usage remains an honest refusal, because
  *that* algorithm is unimplemented and reporting a number would be fabrication.)
- **Create-context flags honored** (`ffxCreateContextDescUpscale.flags`): HDR, display-resolution
  motion vectors, motion-vector jitter cancellation, depth-inverted, depth-infinite, auto-exposure,
  dynamic-resolution, non-linear colorspace, debug-checking, debug-visualization.
- **Per-dispatch inputs honored** (`ffxDispatchDescUpscale`): color/depth/motion/exposure/reactive/
  transparency inputs, `jitterOffset`, `motionVectorScale`, `enableSharpening`/`sharpness` (RCAS),
  `preExposure` (non-positive → neutral 1.0 + a `[exposure]` warning), `reset`, `cameraNear`/`cameraFar`/
  `cameraFovAngleVertical`, and the colorspace flags `FFX_UPSCALE_FLAG_NON_LINEAR_COLOR_SRGB` /
  `…_PQ` (mutually exclusive, applied via `selectColorSpace`), plus `FFX_UPSCALE_FLAG_DRAW_DEBUG_VIEW`.
- **Debug checker** (`FFX_UPSCALE_ENABLE_DEBUG_CHECKING`): when the host requests it, the runtime
  emits the SDK's warnings through the app's `fpMessage`/`FFX_API_MESSAGE_TYPE_WARNING` channel (first
  dispatch, once, to avoid per-frame spam): the depth separation-artifact warning for
  `DEPTH_INVERTED` + `DEPTH_INFINITE` with a very low `cameraFar`, and the `frameTimeDelta < 1.0f`
  (should be milliseconds, ~16.6f for 60fps) warning. Off unless requested — release path unchanged.
- **Device gating** matches AMD's documented requirement (RX 9000/7000/6000 series or the AMD
  Ryzen AI/Z2 integrated-graphics processors): the model ladder walks INT8 → FP8 and refuses with
  named blockers where neither is offered (see `## Checks` and the runtime levers).

## Host integration (upscaler)

The provider is a **command-buffer recorder, not a scheduler**. Given a host-supplied
`VkCommandBuffer`, it records every FSR4 pass onto that one buffer and sequences them
itself — inserting `vkCmdPipelineBarrier2` / `vkCmdPipelineBarrier` between passes with
explicit source/destination stage and access masks (for the optional pixel readback, a
`TRANSFER_WRITE → HOST_READ` memory barrier), and performing the layout transitions on
the host's input images. Because the ordering lives inside the recorded buffer, the
sequence stays valid on whatever queue the host submits it to. The provider never calls
`vkCmd*/vkQueue*` submission, never chooses a queue, never touches cross-frame semaphores,
and never presents: **queue selection, `VkSubmitInfo`, frame-to-frame semaphores and
presentation are entirely the host's responsibility.**

Queueing guidance for a host embedding this (measurements from the experimental
[ffx-vk-linux](https://github.com/DuIslingr/ffx-vk-linux) project, *not* a claim that this
`.so` schedules work by itself):

- **Keep FSR4 on the graphics queue.** Measured there: FSR4 async-compute (moving the
  passes to a dedicated compute queue) gained ~0 on an RX-class card and *added* 0.1–0.2 ms
  of sync overhead. (FSR3 async gained ~+40%, but that does not carry to FSR4.) So run
  this provider's recorded buffer on the main graphics queue unless you have your own
  measurement showing otherwise.
- **If you reuse one command buffer across queues**, create it `VK_SHARING_MODE_CONCURRENT`
  and drive frame-to-frame ordering with **timeline semaphores** and **double-buffered per-frame
  inputs**; that is what keeps a shared buffer validation-clean when a queue consumes it.
- **Determinism check you can lean on:** this repo's `descriptor-smoke` target already
  proves — on the target GPU — that the two descriptor transports (conventional descriptor-set
  vs `VK_EXT_descriptor_buffer`) produce **byte-identical** INT8 (and FP8 where supported)
  output through the same shaders across repeated submits. A host can replay that as a
  3-frame byte-identical readback to catch any non-determinism in its own queueing/pacing.

## Checks

The build itself verifies every embedded asset payload and alias. On top of that:

- `ctest --test-dir build/linux --output-on-failure` is the standard discovery entry
  for the **compiled** invariant tests, complementing the bash gate. It runs
  `upscaler-debugcheck` (the FSR 4.1.2 debug-checker predicates), `upscaler-query-math`
  (the deterministic FSR helper math: quality→ratio, display→render resolution, jitter
  phase-count + R2 offset, from `src/provider/upscaler-queries.hpp`), `cap-probe-synthetic-gates`
  (the INT8 → FP8 ladder + every device-class spelling + RDNA2 identity edges),
  `descriptor-smoke-transports-byte-identical` (conventional descriptor-set ↔ `VK_EXT_descriptor_buffer`
  byte-identity through the real shaders), `frame-generation-validators`
  (`tools/validate/frame-generation/tests.py`: FG ABI + honest refusal),
  `framegeneration-check-order-camera` (`framegeneration-check-test`: the GPU-free
  Frame Generation mandatory-order + camera-info + create/generate predicates), and
  `framegeneration-compile-parity-474` (`framegeneration-exec --compile-pack
  --descriptor-buffer-parity --expect-compiled=474`: pins that **all 474 FG compute modules
  compile under BOTH descriptor transports**, exiting non-zero if a pack/SPIR-V regression
  drops any). It is registered only when the generated (gitignored) FG asset pack is present, and covers
   the two FG pass-plan gates: `framegeneration-plan-consistency`
   (`framegeneration-plan-test`: the GPU-free proof that an independent C++ re-parse of the
   pack's `pipelines.tsv` agrees, module-for-module, with the FG pass-plan that
   `tools/extract/frame-generation/plan.py` reconstructs: same 474 compute modules, same 1208
   de-duplicated binding groups, valid order scaffold) and `framegeneration-plan-drift`
   (`framegeneration-exec --compile-pack --descriptor-buffer-parity --expect-plan`: ties the
   driver-backed per-module binding census to that plan on a real device, measured
   `checked=474 missing=0 mismatch=0`) and `framegeneration-host-integration-ladder`
   (`framegeneration-probe`: on a device that offers the Frame Generation execution contract it now
   **asserts** — not merely prints — the entire documented host-integration return-code ladder
   (`ffxCreateContext`=OK → Prepare-before-Configure=parameter → GET_VERSIONS=pinned `0xF600000001000002`
   → GPU-memory-usage=honest runtime refusal → Configure=OK → Prepare=runtime-pending →
   Generate=runtime-pending → present-callback=no-provider → destroy=OK), and **skips** any device that
   does not meet the contract (e.g. one lacking `VK_KHR_cooperative_matrix`), so a provider routing/refusal
   drift fails the gate on a suitable AMD device while a device-less host stays green via exit 77.
   Measured on the AMD Radeon RX 9070 XT: every stage returned its documented code.
  The device-dependent entries auto-skip on exit code **77** — the tools' own "no suitable
  AMD device" signal — so `ctest` stays green on a GPU-less CI host instead of failing. Verified
  on the target device: **9/9 passed**.

- `bash tools/build/linux.sh` now ends by running `tools/build/check.sh`, a
  post-build validation gate in which every check asserts exit 0 (a red run is a
  real regression — invalid SPIR-V, a dropped asset payload, or a broken Frame
  Generation entry-point contract — not merely a compile failure). The gate runs
  the four checks feasible on any Linux host: `shaders.py` on the native pack
  (504 reconstructed modules), `elf-assets.py` (binary-level asset proof),
  `frame-generation/tests.py` (ABI + honest refusal), and the GPU-free
  `upscaler-debugcheck-test` (the FSR 4.1.2 debug-checker predicates) and then the compiled
  `ctest` set (so one command covers both the validators and the compiled invariant tests;
  GPU-dependent ctest entries auto-skip on exit 77). `windows-binary.py`
  (needs the Windows PE / MinGW-w64 build) and `ab-frames.py` (needs two live frame
  captures) are deliberately outside this Linux gate and run on the Windows/capture path
  instead — `cmake --build --target check-win`, defined only under `WIN32` (inert on a
  Linux configure), wraps `windows-binary.py` on the built `amd_fidelityfx_upscaler_vk.dll`
  plus the build's generated `embedding-selection.json`: the PE sibling of `elf-assets.py`
  (RCDATA payload hashes + exactly the five FFX API exports).
- The **binary-level** asset proof is `elf-assets.py`, and its `--storage-index`
  argument is required, not optional. Correct form:

  ```sh
  python3 tools/validate/elf-assets.py \
      build/linux/amd_fidelityfx_upscaler_vk.so assets/pack-unified \
      build/linux/generated/elf-asset-report.json \
      --storage-index build/linux/generated/embedded-storage.json
  ```

  It opens the `.so`'s `fsr4_embedded_assets_begin` ELF symbol, decompresses the
  embedded bytes with the same per-payload Zstandard index the build wrote
  (`generated/embedded-storage.json`), and hash-checks every payload against
  `assets/pack-unified/manifest.json` — expect `complete=true`, `payloads_verified=508`,
  `spirv_payloads=504`, `compressed_payloads=504`, `windows_payloads=0`,
  `omitted_shader_tiers=[]`, `read_only` and `private_data_symbol`. Without
  `--storage-index` it re-derives *uncompressed* sizes and reports a spurious
  "Embedded packed payload mismatch", so a healthy build looks red: always pass the
  index the build generated.

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
- `./build/linux/upscaler-debugcheck-test` is a GPU-free unit test for the FSR 4.1.2
  debug-checker predicates in `src/provider/upscaler-checks.hpp` — the same warnings the
  provider routes through `providerMessage` / `FFX_API_MESSAGE_TYPE_WARNING` on the first
  dispatch when the host sets `FFX_UPSCALE_ENABLE_DEBUG_CHECKING` (the SDK's depth-separation
  warning for `DEPTH_INVERTED`+`DEPTH_INFINITE` with a very low `cameraFar`, and the
  `frameTimeDelta < 1.0f` warning). It asserts the checker is silent unless requested, fires
  each warning on its own condition and both together, and stays silent for a well-formed
  frame — no Vulkan, no assets, no device, so the checker's decision logic cannot drift silently.
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

The extraction workflow additionally reverses the ML payload from the pinned PE:
`tools/extract/frame-generation/reverse.py` verifies two complete
`mlfi_<tier>_pass0..190` chains, extracts the aligned 9,437,184-byte model with
SHA-256 `6345f9ef4bbead6a942a7379d17caeaaf01d6142aa3dccae4695e56f68627dbe`,
and records PE record references in `mlfi-graph.json`. The Linux provider embeds
the verified SPIR-V pack, graph and model into
`amd_fidelityfx_framegeneration_vk.so`; generated bytes remain gitignored.

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
upscaler's `fs4`), and the clean refusal of anything not yet implemented. The SDK's
real Frame Generation descriptor types are now vendored (`ffx_frame_generation.h`:
create `0x00020001`, configure `0x00020002`, Prepare `0x00020004`, generate
`0x00020003`, linked camera-info `0x0002000a`) and the provider validates the create
descriptor's own arguments; under `FFX_FRAMEGENERATION_ENABLE_DEBUG_CHECKING` it enforces
the mandatory call order (Configure → Prepare → generate) and the orthonormal camera
basis, THEN still refuses the algorithm. It does **not**
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
*acceptance* for **all 474** FG compute modules are already implemented and measured. Two
  more boundaries are now delivered and machine-checked. `ffxConfigure`
  (`ffxConfigureDescFrameGeneration`, 0x00020002) validates **and captures** the full host-
  integration contract the present phase needs — swap chain, present + frame-generation
  callbacks (+ their user contexts), `HUDLessColor`, `onlyPresentGenerated`, `generationRect`,
  `frameID` — and the documented **Configure → Prepare → generate** order is enforced generally
  (not only under debug), pinned GPU-free by `framegeneration-check-test`. The reconstructed FG
  pass-plan (Phase 0) is cross-checked three ways (`framegeneration-plan-test` independent re-
  parse; `framegeneration-exec --expect-plan` driver-backed census; `plan.py`). The compute half
  — recording the 92 classic + 382 `mlfi` passes with their semantic buffer wiring — stays gated
  on a DLL-grounded FG dispatch/dataflow extractor (the analog of the upscaler's
  `tools/extract/{dispatch,endpoints}.py`), so `ffxQuery(…GetGPUMemoryUsage)` (0x00020007)
  remains an honest refusal until the provider actually allocates the plan's intermediate
  targets.

`tools/framegeneration-probe.cpp` produces the compatibility matrix on real
hardware. Measured on the primary target: AMD RX 9070 XT (RADV GFX1201) meets
the contract (real FG create descriptor accepted, advertises the id, a valid
 Prepare is accepted — order + camera basis — then the algorithm dispatch is refused
 pending runtime);
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
