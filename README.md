# fsr4-vk

An independent Vulkan implementation of FSR 4.1.2 FP8 and INT8 for Windows
and Linux, using AMD's model weights and shaders translated to SPIR-V.

This project is not affiliated with or endorsed by AMD.

## How to approach

The whole project is built on **measured ground truth, never inference**. Read this before any code change; every capability claim must trace to a runnable gate.

- **Pin the inputs.** Both source DLLs are pinned by SHA-256 - upscaler 4.1.2.4555 (`aad4...`) and Frame Generation 4.0.2.4555 (`e684...77bb6`) - and every extraction tool **fails on drift** rather than skipping a mismatch. AMD's open **FidelityFX SDK 2.4.0** (`third_party/fidelityfx-sdk`) is vendored as **CPU control-flow ground truth**: the *classic* interpolation pass order, resource tables, present/pacing/UI and descriptor layouts are **MINED from that readable source** and pinned by gates - not guessed out of the DLL.
- **Two independent providers.** The upscaler is **complete + validated**. The Frame Generation provider is the ABI + capability contract + host-integration surface + the **Phase-1 runtime substrate** (real device allocations, pipeline layouts, compute pipelines and descriptor serialization driven by the pack's own plan tables), and it **honestly refuses the not-yet-validated dispatch** instead of fabricating a blend or a made-up optical-flow stand-in. The two `.so` outputs must never share assets, model logic or runtime resources.
- **Verify before claiming.** GPU-free gates run on any host (`bash tools/build/check.sh`, `ctest`, and the 37 gates in `tools/validate/frame-generation/tests.py`); device gates **auto-skip with exit 77**, so a GPU-less CI host stays green instead of failing. A red run is a real regression, not a missing GPU. The suite now also contains **one gate that RUNS the harness app on a device** (`test_fgtest_host_app_runs_on_device`): every other gate is static or artifact-scoped, and a static gate CANNOT see a device-config regression — that is precisely how a bad extension request reached the user as a bare `NO_DEVICE` while 35/35 gates stayed green. It self-skips on rc 77 (no suitable device), on missing app artifacts, and on `FSR4_SKIP_GPU_GATES=1` for a deliberately device-free run. That gate also asserts the app's **measured motion-vector read-back is NON-ZERO**: an all-zero velocity field would mean the host never owed Frame Generation the input interpolation needs.
- **The app states plainly whether Frame Generation runs, in pixels.** The window pre-fills the provider's OWN `generate` output images with a sentinel and reads a centred 240×135 window back after every `generate()` call, so delivery is evidence rather than a return code: the HUD paints `FG NOT RUNNING` at 4× beside `OUT PX 0/32400 WRITTEN` and the measured `MOTION PX 633  MAXV 0.04023`, and the run ends with `FRAME GENERATION VERDICT: NOT RUNNING - 0 interpolated frames ... the missing piece is the interpolation runtime, NOT the measurement`. Two host obligations that made FG visibility impossible in this harness were fixed on the way: the rig's colour/depth/motion/output images are now a **ring of 2** (the SDK's `MAX_QUEUED_FRAMES == 2` means the provider samples the PREVIOUS frame while the next one renders — one image per role meant every hand-off pointed at an image the host was already overwriting), and the motion image is **WRITTEN per frame by a real velocity pass** (`velocity.frag`, UBO `prevModelMatrix`) instead of cleared once to zero. The arithmetic the verdict quotes is the effect's own: **2× inserts one generated frame per consecutive pair → N rendered frames become N + (N−1) shown (3 → 5); 4× inserts three per pair**.
- **Treat transport as a measured constraint.** Conventional `VkDescriptorSet` fails 2684 of 2692 FG sets because FG bindings are `VK_WHOLE_ARRAY` runtime arrays; the host runtime **must** use `VK_EXT_descriptor_buffer` (all 1208 FG bindings serialize with 0 failures). No FG shader has a cross-kind slot, so plain derived `VkPipelineLayout`s suffice - pipeline creation for all 474 compute modules is **solved**. The provider reproduces that on its own: its substrate serialized **304 of 304** classic-subset bindings with zero failures on RX 9070 XT.
- **Read the shaders for the half the plan cannot give you.** `tools/extract/frame-generation/classic_shapes.py` (37th gate) disassembles all 92 classic modules and pins what they DECLARE per binding slot: **304 slots join the plan PERFECTLY (0 orphans either way)**, all 304 are `OpTypeRuntimeArray`, **60 image slots fix a format in the shader** (`44 R32ui + 16 R32f`) while **116 are bindless**, **104 slots are `%uint`/`%int`-typed against 72 `%float`** — and the substrate binds `R32G32B32A32_SFLOAT` to every image slot, which **60 of those bindings cannot satisfy**. Decisively, **write=0**: no classic module issues an `OpImageWrite`, while every one carries `PhysicalStorageBufferAddresses` + a `BindlessCBV` array — the chain's writes go through addresses read out of those arrays, so a zero table is a write to address 0, which is the leading explanation for the cancelled-CS chain fault. Two naming routes are recorded as measured dead ends: SDK-source identifier bridge **0 of 92 strong matches** (4.0.2 shaders keep names the 2.4.0 sources do not share), and the SDK's classic `*_shaderblobs.cpp` hold **no blob bytes at all** to hash. The sanctioned order oracle therefore remains a capture of AMD's own provider: the rig is now complete (`tools/capture/frame-generation/host.cpp` + `build-host.sh` + `run-capture.sh`, and the shim interposes the **resolvers** so a pointer-resolved provider is recorded at all), and what is missing on this host is a Windows cross-compiler — `winegcc` measured as a dead end (it builds ELF images that cannot link Wine's Windows import libs, and Arch's `wine` ships no `wine/dlfcn.h`), while the toolchain IS an official package (`cachyos/llvm-mingw` 621 MiB, or `extra/mingw-w64-*`) that needs root, so `order_source` stays `grouped-scaffold` until the operator installs it.

  The CONSUMPTION half of that transport is now pinned as a measurement too: `tools/db-command-surface.cpp` resolves
  candidate command names through BOTH device and instance scope on the live driver, and this host's RADV implements
  `VK_EXT_descriptor_buffer` at **spec version 1 - the 0.14.2 draft generation** - exporting
  `vkCmdBindDescriptorBuffersEXT` (`VkDescriptorBufferBindingInfoEXT{address,usage}`),
  `vkCmdSetDescriptorBufferOffsetsEXT` (bindPoint, layout, firstSet, setCount, bufferIndices, offsets),
  `vkCmdBindDescriptorBufferEmbeddedSamplersEXT`, `vkGetPhysicalDeviceDescriptorSizeEXT` and
  `vkWriteResourceDescriptorsEXT` / `vkWriteSamplerDescriptorsEXT`, but **not** the final-revision
  `vkCmdPushPipelineDataEXT` / `vkCmdBindDescriptorRangesEXT` (ABSENT in both scopes). Two consequences for the
  runnable dispatch: the PINNED Vulkan-Headers already declare the names this driver speaks, so no hand-mirrored
  prototypes are needed; and every one of those commands resolves at **instance** scope while returning NULL at
  device scope - which is why a device-scope-only probe reads as "the driver has none of it", while the providers
  and the substrate's 304/304 serialization get them through the caller's resolver.
- **Respect the closed boundary — now with a corrected map of what is actually closed.** The 382 `mlfi_*` ML
  modules' **weights ARE extracted** (`pack/mlfi-model.bin`, 9,437,184 B at RVA `0xf00000`, `model.extracted=true`,
  entropy `7.403`, with per-record `.data` references) and their **chain order IS mined and gated**: `mlfi-graph.json`
  holds two chains of **191 entries each (`mlfi_4k_pass0..190` / `mlfi_8k_pass0..190`), and within a tier the
  execution order is fixed by the pass index** - and the plan's own `order` column agrees with a single linear
  sequence over all 474 compileable modules: **classic 0..91, then the 4k chain at 92..282, then the 8k chain at
  283..473**, each tier's passes contiguous. So the ML chain's order is NOT the open half; the anonymous
  `CS` **classic** slots are. Its own `order_basis` note says so, and adds that the *classic*
  optical-flow/interpolation ordering still needs control-flow correlation (which is what the new shape join advanced).
  What genuinely stays closed is (a) mapping each ML pass's chain edges onto the intermediate/resource objects and
  scratch sizing, and (b) **pixel parity**, which needs AMD reference captures. And the driver is NOT the blocker:
  measured on this host, `--compile-pack` reports `attempted=474 created=474 failed=0`, so all 156 `CooperativeMatrixKHR`
  modules and the 6 `ComputeDerivativeGroupLinearKHR` ones build as `VkComputePipeline`s on Mesa/RADV. The 12 modules
  outside that 474 are non-compute stages (7 present-blit + 1 pixel + 1 vertex + 3 unsupported-psv), not refusals. So the provider still stays refused *by design* until the runtime exists, but the closed half is
  narrowed to (a) the chain-edge → resource/scratch mapping and (b) reference captures — not to "unknowable order".

## What runs today (status matrix)

- **Upscaler `amd_fidelityfx_upscaler_vk.so` - runs.** Records every FSR4 pass onto the host's `VkCommandBuffer` and self-sequences the barriers; full `ffxQuery`/create/dispatch contract answered; validated **byte-identical across both descriptor transports** (`descriptor-smoke`).
- **Frame Generation `amd_fidelityfx_framegeneration_vk.so` - the boundary AND the Phase-1 substrate run; the algorithm refuses.**
  - *Runs (returns its documented code):* `ffxCreateContext` (capability contract + version advertisement + `fg4`-namespaced pipeline cache + warm reload); `ffxConfigure` capturing the full host contract; the **enforced** contract surface - all 7 create flags interpreted, mandatory Configure->Prepare->generate order, **version-negotiation refusal** of a non-classic `ffxOverrideVersion`, `PrepareV2` (`0x0c`) validated (order + arguments + EMBEDDED camera basis, via an ABI-mirror of the 2.4.0 struct), `numGeneratedFrames == 0` **cancel** -> `OK`; the offline tooling (inventory / convert / reverse / plan / vk-identity / capture-rig / classic-chain shape join) and the 36-gate python suite; `framegeneration-plan-test` + `framegeneration-check-test` (GPU-free) and `framegeneration-exec` (`--compile-pack`, `--descriptor-buffer-parity`, `--write-descriptors`, `--expect-plan`) + `framegeneration-probe` (device).
  - *Runs (allocates, since Phase 1):* at `Prepare` - where the host declares the render resolution - the provider builds the substrate the pack's plan tables describe: **5 intermediate targets (66,355,200 B at 1920x1080) + one 10,368 B descriptor pool + 528 set layouts + 92/92 real `VkComputePipeline`s** for the classic subset, and a **304/304** `vkGetDescriptorEXT` serialization census of that subset's bindings (fed by 7 deliberately tiny scratch resources, never bound to anything). `ffxQuery(GetGPUMemoryUsage)` (`0x07`) then answers **from those allocations**: `totalUsageInBytes=66,365,568`, `aliasableUsageInBytes=10,368`.
  - *Runs (device-free, since Phase 1):* `GetGPUMemoryUsageV2` (`0x0b`) is the pre-create sizing question, so it returns the pack's OWN byte model for the resolutions the CALLER names (measured: 640x360 -> 1280x720 = **7,372,800 B**), and the log line states `basis=plan-model(no allocation yet)` so the number can never be mistaken for an allocation.
  - *Honestly refused (`FFX_API_RETURN_ERROR_RUNTIME_ERROR`):* `ffxDispatch(Prepare | PrepareV2 | generate 1..3)` - the interpolation / ML / present runtime - and `ffxQuery(GetGPUMemoryUsage V1)` **before** that query has allocations behind it (no `Prepare` yet, or a host device without the `VK_EXT_descriptor_buffer` transport, which the refusal now names). The present-callback descriptor returns `NO_PROVIDER`.
  - *Runs (the moving-graphics host such a runtime would plug into, and the runnable mini app of this phase):*
    `tools/FGTest`. Offscreen (`bash tools/FGTest/build.sh`) it proves the REAL AMD FSR4 compute end-to-end
    (`rc=OK 3/3, output-changed 3/3, non-black 3/3`, per-frame checksums differ, `nonblack_px=518400` of the
    960x540 read-back) and prints the FG refusal ladder (`prepare=3 generate=3(1x) 3(2x) 3(3x) 3(4x) cancel=0`,
    i.e. documented codes, never a fabricated blend). As a window app (`bash tools/FGTest/run.sh`, 30 s by default,
    `Esc`/`q` to end) it blits that upscaled frame live - measured today: 552 presented frames at ~25 fps in
    1440x1080 - and paints a 5x7 bitmap-font HUD into every frame: `WIN:/RENDER:/UPSCALER VK:` resolutions top-left,
    `FPS` 4x and `MS FRAME` / `MS UPSCALER` / `MS FGMEM REFUSED` 2x top-right, and `GOT n X REQ n X` last, so a
    missing interpolation runtime is visible on the screen instead of claimed away. See `tools/FGTest/README.md`.
- **See `## Next steps`** for what turns these honest refusals into a running FG runtime.

## Requirements

- Python 3.10+, CMake 3.22.1+ and a C++20 compiler.
- `llvm-objdump`.
- For the optional `tools/FGTest` host app (the runnable mini graphic app of this phase): `glslc`, the Vulkan
  dev headers, and SDL2 via `sdl2-config` for the visible window. Without SDL2 the app still builds headless.
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
- **Second AMD source for the model blobs (Vulkan upscaler DLL).** `tools/validate/upscaler-vk-identity.py` byte-searches the pinned AMD *Vulkan* upscaler DLL (`amd_fidelityfx_upscaler_vk.dll`, SHA `4a7e895a...`, present in `dlls/`, gitignored) and proves all four shipped model weight banks - int8+fp8 `weights_standard.bin` and `weights_ultraperf.bin` - are **byte-identical inside AMD's own Vulkan provider** (fp8 standard x5 as the five-replication rule documents, fp8 ultraperf x1, int8 standard x10, int8 ultraperf x2), and that the fp8 digests equal the pinned `keys/4.1.2.4555.json` `standard_sha256`/`ultraperf_sha256`. It also confirms the `fsr_rcas_pass`, `auto_exposure`, `Navi2` and `Lite` variant names exist there. GPU-free, run by `check.sh`, self-skips when the DLL is absent. **What it does not add:** the `_vk` shader set is one compressed proprietary container (0 plaintext SPIR-V/DXBC), so the Vulkan DLL yields no extra extractable shaders - it *confirms* the model blobs + shader/variant identity we already ship, it does not fabricate new ones. It also ADJUDICATES whether the DLL hides any model weight bank beyond the 4 we ship and finds NONE: the +/-128 KB neighbour windows are pitch-boundary overlaps or single-occurrence artifacts (the real bank pitch is 128 KB plus 0-32 B of metadata drift), and `--expect` fails loudly if a genuinely replicated 128 KB bank ever appears, so a future AMD addition surfaces instead of being ignored or auto-injected.

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
  (`tools/validate/frame-generation/tests.py`: FG ABI, honest refusal, the DLL-inventory dedup + stage census, the pipeline-cache namespace
    isolation, and the FG model/graph/pipelines three-way identity gate — `test_mlfi_graph_chain_matches_pack`
    proves the 4k+8k 191-pass ML chains' per-pass shader identity (module hash + SPIR-V digest) matches
    the shipped `modules.tsv`, that the graph's cooperative-matrix count equals the SPIR-V census (156),
    and that the 9,437,184-byte `mlfi-model.bin` bytes hash to the DLL-extracted model SHA — all GPU-free — and `test_mlfi_chain_structure_and_model_tiling`, which pins the ML
    chain structure (the 4k and 8k per-index descriptor-set signatures are tier-invariant; the graph's
    per-pass sets equal the pack `pipelines.tsv`; the 156 cooperative passes bind sets 1,4,5; and the model
    blob is a whole number of 16x16 tiles for fp16/fp32/int8) WITHOUT asserting any dataflow order/DAG,
    and `test_sdk_ground_truth` (AMD's own CPU `ffxDispatch`/Query orchestration, the 48-id FrameInterpolation
    resource model, the GetGPUMemoryUsage answer model, the Vulkan present/pacing/UI sources and the
    MLFI-neural-not-in-open-SDK proof, all read from the pinned v2.4.0 SDK submodule)),
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
   de-duplicated binding groups, valid order scaffold) — that same GPU-free test ALSO re-derives every module's CooperativeMatrixKHR
    operand shapes (the DLL's CsTensorSizes) independently from the shipped SPIR-V via spirv-dis,
    agreeing with plan.py module-for-module and pinning the FG 4.0.2 census: 156 ML neural modules,
    6 distinct 16x16 tile shapes (float/half/uchar), identical 4k/8k geometry, 92 classic none. and `framegeneration-plan-drift`
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
  - `FG cooperative-dataflow boundary probe` (`tools/extract/frame-generation/dataflow.py`, run by
    `check.sh` when the FG pack exists): a GPU-free audit over all 156 ML cooperative passes that
    proves the resource SLOTS are statically visible (every coop pass declares weights @ `set 1`
    StorageBuffer + activation @ `set 4` StorageBuffer + CBV @ `set 5` Uniform, at `binding 0`) while each
    `OpCooperativeMatrixLoad/StoreKHR` operand pointer is a BINDLESS, dynamically computed buffer address.
    So ZERO pass→pass dataflow edges — and hence the ML execution ORDER — are derivable from the shipped
    SPIR-V alone. It pins `resolved=0`/`edges=0` as a drift gate: if a future pack/toolchain ever makes
    the order statically derivable it FAILS loudly (then dispatch work could proceed); until then it is
    EVIDENCE that `ffxDispatch` remains honestly refused pending the DLL's CPU-side descriptor writes plus
    a DX12 reference capture — a measured boundary, not an oversight.
  The device-dependent entries auto-skip on exit code **77** — the tools' own "no suitable
  AMD device" signal — so `ctest` stays green on a GPU-less CI host instead of failing. Verified
  on the target device: **9/9 passed**.

- **AMD's own SDK is now vendored and mined (ground truth, not blind DLL-RE).**
  `third_party/fidelityfx-sdk` is the pinned FidelityFX SDK **2.4.0** submodule (the
  Redstone line, commit 7539ae8b) and it ships the **readable CPU source** for the
  Frame Generation provider — the exact `ffxDispatch` control flow that was previously
  only blind-reverse-engineered from the DLL. `test_sdk_ground_truth` in
  `tools/validate/frame-generation/tests.py` reads that source and asserts the facts are
  intact: the provider's `MAX_QUEUED_FRAMES = 2`; the `GetGPUMemoryUsage` answer model
  (`ffxFrameInterpolation… + ffxOpticalflow… + ffxShared…` summed); the Prepare path's
  shared FSR3↔FG resource coupling (`FFX_FSR3_RESOURCE_IDENTIFIER_DILATED_DEPTH_0`,
  `…_DILATED_MOTION_VECTORS_0`, `…_RECONSTRUCTED_PREVIOUS_NEAREST_DEPTH_0`, with a
  per-frame toggle); the desc remap (`numInterpolatedFrames→numGeneratedFrames`); the
  **48-id `FFX_FRAMEINTERPOLATION_RESOURCE_IDENTIFIER_*`** intermediate-target table
  (+ 2 constant buffers) — roadmap blocker #3 is now *resolved by AMD source*, not inferred;
  and the presence of the full **Vulkan** present/pacing/UI-composition sources
  (`fsr3/vk/FrameInterpolationSwapchainVK{,_DebugPacing,_UiComposition,_Helpers}.cpp`).
  Two roadmap consequences follow: (1) the **classic** optical-flow + frame-interpolation
  CPU pass order is now **MINED (reversed) out of AMD's Process bodies and pinned** by
  `test_sdk_classic_controlflow_order`: OpticalFlow = PrepareLuma → GenerateOpticalFlowInputPyramid
  → GenerateSCDHistogram → ComputeSCDDivergence → ComputeOpticalFlowAdvancedV5 → FilterOpticalFlowV5
  → ScaleOpticalFlowAdvancedV5; FrameInterpolation = Prepare `ReconstructAndDilate`, then Generate
  Setup → ReconstructPreviousDepth → GameMotionVectorField → GameVectorFieldInpaintingPyramid →
  OpticalFlowVectorField → DisocclusionMask → Interpolation(SCFI) → InpaintingPyramid → Inpainting
  → GameVectorFieldInpaintingPyramid → DebugView (11 declared `FFX_FRAMEINTERPOLATION_PASS_*` + 7
  `FFX_OPTICALFLOW_PASS_*`); together with the 48-id resource model + present/pacing/UI, this closes
  roadmap blocker #2 for the CLASSIC half (and `test_sdk_opticalflow_binding_table` also pins
  AMD's **OpticalFlow descriptor layout** — the named binding-identifier table `r_input_color`
  → `r/rw_optical_flow_input` + the **7-level** input pyramid `…_LEVEL_1..6` → `r/rw_optical_flow
  (_previous/_next_level)` → the SCD set (`histogram/previous_histogram/temp/output`) → the shared
  vector/SCD bindings, plus the 2 constant buffers `cbOF`/`cbOF_SPD`, and that the CPU actually
  wires `resources→uav/srv bindings`); and `test_sdk_shared_fsr3_fg_resources` pins the **shared 17-id
  FSR3↔FG resource pool** the FG provider consumes from the upscaler — `OPTICAL_FLOW_VECTOR` + `OPTICAL_FLOW_SCD_OUTPUT`
  plus the 4 per-frame slots of the `DILATED_DEPTH_i`/`DILATED_MOTION_VECTORS_i`/`RECONSTRUCTED_PREVIOUS_NEAREST_DEPTH_i`
  triple with `COUNT`=17 and `UPSCALED_COUNT`=3 (which is exactly what the provider's `…_0 + (frameToggle *
  UPSCALED_COUNT)` indexing walks) — closing the shared-resource-allocation part of blocker #3 for the
  classic path; **`test_sdk_spd_inpainting_pyramid_setup`** pins the inpainting‑pyramid **SPD**
  (single‑pass downsampling) geometry AMD drives — `ffxSpdSetup(rectInfo{0,0,w,h})` yields
  `workGroupOffset = rectInfo/64`, `numWorkGroupsAndMips[0] = dtgcX*dtgcY`,
  `mips = min(floor(log2(resolution)), 12)` — called **twice** (game‑vector‑field pyramid on
  `renderSize`, inpainting pyramid on `displaySize`) and copied into the `cbInpaintingPyramid`
  constant buffer; and **`test_sdk_present_swapchain_contract`** mines the full **Vulkan present/pacing/UI**
  algorithm from `fsr3/vk/FrameInterpolationSwapchainVK.*` — the two `FGSwapchainCompositionMode` modes
  (`eComposeOnPresentQueue` / `eComposeOnGameQueue`), the game↔present **queue‑family ownership transfer**,
  the `UNDEFINED→VK_IMAGE_LAYOUT_PRESENT_SRC_KHR` transition, the `…UI_COMPOSITION_FLAG_USE_PREMUL_ALPHA`
  flag, the `ffxRegisterFrameinterpolationUiResourceVK` / `ffxGetMaximumFrameLatencyVK` / `ffxSetFrameGenerationConfigToSwapchainVK`
  APIs, and the UI‑composition(+premul) / debug‑pacing present VS/PS programs — the exact `Present` behaviour
  the classic path must reproduce (previously only a "no provider" return code was knowable); and
  **`test_sdk_present_pacing_threading`** mines the present/pacing **threading substrate** — the
  8‑signal acquire‑buffer ceiling (`MAX_ACQUIRE_SEMAPHORE_COUNT 8`), the game / interpolation /
  composition / replacement‑buffer / `frameRenderedSemaphores[]` semaphore family, a dedicated
  **presenter thread** (`spawnPresenterThread`/`killPresenterThread` + handle), the game↔present
  **queue‑family ownership transfer** over family‑indexed command‑buffer pools, and the **mixed
  timeline+binary** choreography (`VK_SEMAPHORE_TYPE_TIMELINE` + `VkTimelineSemaphoreSubmitInfo` +
  `vkGetSemaphoreCounterValue` for pacing, binary `frameRenderedSemaphores` per buffer). All of the
  SDK mining above is **substrate only**: the provider still refuses `ffxDispatch`/`ffxQuery` (no
  fabricated frames); wiring these mined pieces into a real **classic‑only 4×** runtime is the next
  implementation step, and it stays gated on AMD reference captures for validation, with the 382
  MLFI passes and any >4× still refused; (2) the same gate
  PROVES the **MLFI ONNX neural net** — the 382 `mlfi_*` inference passes — is *not*
  shipped in the open SDK, confirming the ML pass→pass execution ORDER is the single
  genuinely closed boundary; the ML `ffxDispatch` path therefore stays honestly refused
  exactly as before, now with source-level evidence rather than only a shader-side
  `edges=0` proof.
- **Frame Generation device scope.** Redstone/MLFI Frame Generation targets the AMD
  **RX 9000 series** (RDNA4): the FG host-integration ladder is exercised on the RX 9070
  XT, and any device that does not satisfy the Frame Generation execution contract
  (for example one lacking `VK_KHR_cooperative_matrix`) is *skipped* by
  `framegeneration-probe` (exit 77) rather than failed — this reflects the feature's
  supported-hardware boundary, not a gap in the provider.
- **The "AMD reference captures" blocker applies to the *interpolation output*, not to the Swapchain
  provider.** AMD's "Frame Generation (Swapchain)" doc + the SDK's
  `fsr3/vk/ffx_provider_framegenerationswapchain_vk.cpp` show the Swapchain is a **separate provider**
  that *replaces the app's `VkSwapchainKHR`* and **delegates the interpolation workload to the host**
  (`ffxConfigureDescFrameGeneration::frameGenerationCallback`, or the
  `ffxQueryDescFrameGenerationSwapChainInterpolationCommandListVK` + `…InterpolationTextureVK`
  query) — it does **not** itself run the Frame‑Interpolation/ML interface. Therefore a
  **present / pacing / UI‑composition provider is fully specified by this doc + source and can be
  built without the closed 382‑pass ML and without fabricating any image** (the host supplies the
  command buffer + destination texture that this provider merely presents and paces). What the doc
  does *not* give is *pixel* parity for the interpolation *result* — that still needs AMD reference
  captures. The whole surface is pinned by `test_sdk_swapchain_replacement_provider`. Doc facts this
  provider must honour: FG **4.0.0 requires AMD RX 9000 (RDNA4)+ or the API auto‑falls back to 3.1.7**
  (no hardware requirement); it creates **two CPU worker threads** (pacing + present/UI‑composition);
  **driver‑assisted frame pacing** (Adrenalin 26.10.1+, exclusive or borderless fullscreen, VSync
  off, presenting inside the display's VRR window) replaces the busy‑wait when available; FG **4.0.2
  targets an input rate of at least 30 FPS**; it needs a typed‑UAV‑load GPU with
  **`VK_FORMAT_R16G16B16A16_UNORM`** storage‑image support; the app gets the replacement `pOut…`
  pointers from `ffxQuery(ffxQueryDescSwapchainReplacementFunctionsVK)`; and
  `ffxDispatch(…WaitForPresentsVK)` must run before context destroy. So the concrete next implementation
  step is this **Swapchain provider** (a `VkSwapchainKHR` replacement + 2 threads + UI composition +
  pacing), which is now honest to build; the still‑closed parts remain the **ML order/weights** and
  any **>4×** (which the `outputs[4]` contract forbids).
- **The AMD *interpolation*‑API doc is now grounded too** (`test_sdk_interpolation_api_contract`), and
  it is the **contract** reference — not a pixel/ML reference. It confirms the API is *"common across
  all versions… algorithm differences on version‑specific pages,"* so it does **not** supply the 382‑pass
  ML order or output‑pixel parity (both still out). It pins the contract our provider must honour, of
  which these are **already honored** (`src/framegeneration/{provider,framegeneration-checks}.cpp/.hpp`):
  `ENABLE_DEBUG_CHECKING`, `ENABLE_DEPTH_INFINITE`, `generationRect` (non‑negative + `0`=whole‑frame
  sentinel), `HUDLessColor`, `onlyPresentGenerated`, `frameID`, `RegisterDistortionFieldResource`,
  `QueryDescFrameGenerationGetGPUMemoryUsage`. And these contract items are **now wired** by the contract-enforcement step (see `## How to approach` and the new gate `test_sdk_fg_host_contract_enforcement`): linking `ffxCreateContextDescFrameGenerationVersion` (`FFX_FRAMEGENERATION_VERSION`) in `pNext` at create is recognized; the **`PrepareV2`** path (`0x0c`) is accepted; the other create flags (`...DISPLAY_RESOLUTION_MOTION_VECTORS`, `...MOTION_VECTORS_JITTER_CANCELLATION`, `...DEPTH_INVERTED`, `...HIGH_DYNAMIC_RANGE`, `...ASYNC_WORKLOAD_SUPPORT`) are all interpreted; the **pre-context** `...GetGPUMemoryUsageV2` (`0x0b`) query is answered (honest refusal); and the documented `numGeneratedFrames == 0` **cancel** returns `OK`. And these are **still to wire** (honest gaps, all implementable from the header without the ML): the letterbox rule that `generationRect.width/height` must equal the upscale output size (only `left/top` may vary); the 3 UI paths + `RegisterUiResource...VK` and the `ENABLE_INTERNAL_UI_DOUBLE_BUFFERING` resource-lifetime rule; the `WaitForPresentsVK` teardown flush before context destroy; emitting the `unused_reset`->`reset` deprecation + honoring `reset` == non-contiguous `frameID`; and the documented **mutex-guard** list (the FG+SwapChain contexts are not thread-safe).
- **How a real host drives FG — grounded by the `FSRFG_Dx12` injector reference (3zwr1/AMDNR).** This
  working integration (the same family as Hookshot's own-thread present) shows the *exact* runtime sequence
  and, decisively, **how the ML path is reached without reconstructing any pass order**:
  `ffxQuery(ffxQueryDescGetVersions)` enumerates the DLL's algorithm versions (`versionIds[]`/`versionNames[]`),
  then the host **selects** the variant by linking `ffxOverrideVersion.versionId` (the injector's comment:
  *"0 is non-ML FG and 1 is ML FG"*) — the **DLL bundles the ML dispatch order**, so driving FG is
  *version negotiation*, not `mlfi` sequencing (this reframes roadmap #4 for *integration*; it does **not**
  give us the order for a *re-implementation*, which still lacks the ML compute). Further, MLFI is gated by
  linking a **non-zero** `PrepareCameraInfo` — the injector deliberately *omits* it when the camera is all-zero
  so *"the driver upgrade dll … falls back to FSR 3 FG."* The documented order the injector follows:
  `CreateContext(swapchain, …VersionDX12)` → `CreateContext(FG: backendDesc [+ optional Hudless] + …OverrideVersion`
  after `GetVersions` ) → `Configure(ffxConfigureDescFrameGeneration` with linked `RegisterDistortionField`, `RegisterUiResource`
  on the swapchain ctx, `HUDLessColor`, `flags`, `allowAsyncWorkloads`, `generationRect` (centered: `left/top=(buffer−interp)/2`,
  `width/height=interp size`) , `frameGenerationCallback`, `onlyPresentGenerated`, `frameID`, `swapChain` ) →
  `Configure(ffxConfigureDescGlobalDebug1)` → `Dispatch(ffxDispatchDescFrameGenerationPrepare + linked PrepareCameraInfo)`; the
  **generate** dispatch runs from inside `frameGenerationCallback` during present. Contract details this pins:
  `numGeneratedFrames = 0` **cancels** interpolation for that frame; a **duplicate `frameID`** cancels (`frameID`
  must advance by exactly 1); the injector even uses the **deprecated** `Prepare`+`CameraInfo` (still accepted). Our
  provider already answers `GetVersions` and models `OverrideVersion`, so it is negotiation-faithful; and it must
  offer only versions it can actually run — i.e. **classic only**, never advertise an ML id whose compute we lack.
- **Header finding (from the layout mining).** Our FG provider compiles against the older
  `third_party/fidelityfx-api` header, which **lacks** the 2.4.0 `PrepareV2` (`0x0c`),
  `GetGPUMemoryUsageV2` (`0x0b`), version‑link (`0x0e`) and present‑premul (`0x0d`) descs — so these are
  now declared in `framegeneration.hpp` via AMD's own `kEffectId | 0xNN` composition and proven equal to the
  2.4.0 SDK by `test_sdk_framegeneration_descriptor_layouts`. Relatedly, the **Frame Generation (Swapchain)**
  is a **separate effect id** (`FFX_API_EFFECT_ID_FRAMEGENERATIONSWAPCHAIN_VK`: create `0x01`, registerUI
  `0x02`, replacement‑functions query `0x05`, mode `0x10`, version `0x0a`, plus WaitForPresents /
  InterpolationCommandList / InterpolationTexture), so those descs reach this interpolation `.so` as
  `NO_PROVIDER` and belong to their **own** swapchain provider — they are not read here.
- **Why `ffxDispatch` / `ffxQuery` return an error today — by design, not a defect.** Until the
  Frame Generation compute (classic optical‑flow + dilate **and** the 382 ML inference passes)
  **and** the present/pacing runtime are implemented *and* image‑quality‑validated against AMD
  reference captures, the provider returns `FFX_API_RETURN_ERROR_RUNTIME_ERROR` (the GPU‑memory‑usage
  query stays honest) instead of fabricating frames — so a perfectly‑wired host **still** sees an
  error, which is the truthful result. The **API shape is already correct**: `ffxConfigure` requires the
  app `swapChain` + `presentCallback` + `frameGenerationCallback` (enforced by
  `ffgcheck::configureDescOk`) and `ffxDispatch(Frame Generation)` validates `numGeneratedFrames ∈ 1..3`
  (enforced by `ffgcheck::generateDescOk`), with the present callback carrying
  `isGeneratedFrame` / `currentBackBuffer` / `currentUI` / `outputSwapChainBuffer`. That is AMD's
  **frame‑generation swapchain‑replacement** model: FG drives presentation + pacing on **its own two
  CPU threads** and hands the app a per‑frame present callback — i.e. exactly the contract
  Hookshot already uses against the AMD DLL, **not a different one**. So there is no separate API to
  design for Hookshot; the unsolved work is the (large) present/pacing + interpolation runtime
  **behind** that contract. **Ceiling:** this FG 4.0.2 API tops out at `outputs[4]` → **4×**
  (≤3 interpolated frames, `frameID` advancing by exactly 1); "8×" is beyond FG 4.0.2 and is correctly
  refused, not a hook we are missing. This is pinned by `test_sdk_framegeneration_api_contract`.
- **Contract enforcement (now implemented, not just documented).** The provider now *acts* on this host‑
  integration surface instead of merely recording it: it reads **all 7** `FFX_FRAMEGENERATION_ENABLE_*`
  create flags into the context; it **honors version negotiation** by calling `ffgvk::findVersionOverride` and
  returning `RUNTIME_ERROR` if a host pins an `ffxOverrideVersion.versionId` this `.so` cannot run (classic
  only — the 382‑pass ML inference is not provided, so we *refuse* rather than silently substitute a different
  algorithm); it recognizes the 2.4.0‑only descriptor ids the older vendored header has **no struct for** — the
  create version‑link (`0x0e`), `PrepareV2` dispatch (`0x0c`, which now runs the same in‑order check then the
  honest `RUNTIME_ERROR` refusal) and `GetGPUMemoryUsageV2` (`0x0b`, refused exactly like V1) — matching them by
  `type` id only so the build stays safe; and it honors the documented `numGeneratedFrames == 0` **cancel** path
  by returning `OK` (nothing to fabricate) while still refusing `1..3` with `RUNTIME_ERROR`. Every change is
  **additive** to branches the on‑device ladder never enters (no override, `numGeneratedFrames = 1`, present‑
  callback), so the documented return‑code ladder is preserved by construction. Machine‑checked by the new
  gate `test_sdk_fg_host_contract_enforcement` (GPU‑free, pack‑free), bringing the FG gate count to **30**.
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
`FFG4_VK_SUBSTRATE=0` keeps the Phase-1 substrate off: the provider then allocates nothing
on the caller's device and `ffxQuery(GetGPUMemoryUsage)` keeps refusing instead of reporting
numbers - the default (unset) builds the substrate at `Prepare`.
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

### The Phase-1 runtime substrate (`src/framegeneration/fg-runtime.cpp`)

The provider reads the plan tables **the pack itself carries** (`fg-plan.tsv`,
`pipelines.tsv`, `fg-plan.json`, now embedded in the same `FFG4AS1` asset container as the
shaders, so a shipped build has them without a filesystem) and cross-checks them before
allocating anything: module count, stage split, binding totals, the resolved/unresolved
count and the intermediate byte sum must agree across the three tables, and any disagreement
marks the affected modules unresolved instead of picking a side. Only then, at
`Prepare` - the call where the host declares the render resolution, and never earlier -
does it build what those tables describe:

- the plan's intermediate targets, sized by comparing the reference extents against the
  plan's own `reference_max_render` / `reference_max_upscale` so a target follows
  *upscale* or *render* by the plan's arithmetic rather than by a role name guessed in code;
- one descriptor pool sized from the **driver's** reported element sizes
  (storage-buffer 16 B, uniform-buffer 16 B, combined-image 64 B, storage-image 32 B,
  sampler 16 B measured on RX 9070 XT), not from an assumed 8-byte descriptor;
- one `VkDescriptorSetLayout` per declared set - including an EMPTY layout for every gap in
  the set-index sequence, because `pSetLayouts` positions ARE set indices and a null entry
  there is invalid - plus one `VkPipelineLayout` and one real `VkComputePipeline` per
  classic-subset module (shader modules are destroyed immediately after creation, since
  the pipeline cache keeps the compiled state).
- a `vkGetDescriptorEXT` census of one descriptor per binding of every built module. That
  call returns void, so what is counted is whether the driver **wrote bytes** at its own
  element size; it is fed 7 deliberately tiny real resources (256 B buffers, 4x4 sampled +
  storage images with views, one sampler) because the driver dereferences its target:
  probing with null handles was measured to SIGSEGV inside `libvulkan_radeon`.

Measured on RX 9070 XT (`framegeneration-probe`, rc=0): `plan modules=474 classic=92
neural=382 groups=1208 intermediates=5/66355200B pool=10368B pipelines=92/92
layouts=528/92 descriptors=304/304 backings=7 census=run usage=66365568B`.
`GetGPUMemoryUsage` (`0x07`) reports exactly those numbers, `aliasableUsageInBytes` being
the pool; the same query still refuses before `Prepare`. `GetGPUMemoryUsageV2` (`0x0b`)
answers the pre-create question with the pack's byte model for the caller's own resolutions
(640x360 -> 1280x720 = 7,372,800 B) and logs `basis=plan-model(no allocation yet)`.

What this is **not**: `Prepare` itself still only allocates — the substrate never writes a
descriptor, binds a layout to a command buffer, records or queues a command, and never dispatches
as part of the shipped contract. Executing a pass is a separate **opt-in measurement entry**
(`PlanRuntime::dispatchOne`, reached only through the repo-internal query `0x000200f0`, which
`framegeneration-probe` asks for with `--dispatch-*`), and `ffxDispatch(generate)` still returns
`RUNTIME_ERROR` for the reason it has always given: which of the 92 anonymous `CS` classic modules
a pass is, in which order, with which targets, is still not derivable. What `dispatchOne` proves
is narrower and new: the plan's descriptors can be serialized, bound and **executed** on a real
device, and the pass's own writable outputs can be read back.

### Dispatching ONE pass on a real device (measured, RX 9070 XT / RADV GFX1201)

Sequence, all names resolved through the caller's resolver because a **device-scope**
`vkGetDeviceProcAddr` returns NULL for every one of these commands even where the
instance-scope one succeeds: `vkGetDescriptorEXT` per binding (at the driver's own element size
and binding offset) into a host buffer → `vkCmdUpdateBuffer` uploads into a pool allocated with
`VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT` → `vkCmdBindDescriptorBuffersEXT` →
`vkCmdSetDescriptorBufferOffsetsEXT` → `vkCmdBindPipeline` → `vkCmdDispatch` →
`vkQueueSubmit` → `vkQueueWaitIdle`, with `vkCmdCopyImageToBuffer` / `vkCmdCopyBuffer` read-backs
in the SAME submit.

| measured | value |
| --- | --- |
| plan-order reached execution | orders **0–11** each `attempted=1 bound=1 submitted=1 ran=1` at `canvas=128x64` (before the write census existed) and orders **0–4** the same at `canvas=64x64`; always one order per process, `context_lost=0`, no driver fault messages |
| pool the driver sized | `112–144 B` per module (6 sets, 3–5 binding groups), `descriptors=` equal bytes written |
| grid | `groups=8x8x1` for `canvas=64x64`, `workgroup=8x8x1`, plan's own max extent reported alongside as `3840x2160` |
| read-back census | `720 896 B` = 3 image copies (storage before/after + sampled after) + 2 buffer-back copies, `readback=1` |
| driver's set-size answer | `4294967232 = 2³² − 64` — a **wrapped** answer, i.e. "this set has a runtime-sized array, so it has no bounded size". The per-binding `offset + element` route is the bounded one; the code records which produced the number |
| first-execution cost | `ms_record≈0.3–0.5`, `ms_submit≈0.2`, **`ms_wait≈2003–2017`** for ONE dispatch |
| is that per-dispatch? | **No.** FOUR dispatches of the same module in one command buffer measured `ms_wait=2042` total — the extra 3 added ~30 ms. So `ms_wait / repeats` is a meaningless number and is not printed |
| chaining those dispatches | `repeats=4` produced `radv/amdgpu: The CS has been cancelled because the context is lost` while `vkQueueWaitIdle` still answered `VK_SUCCESS`; the next submit then returned `VK_ERROR_OTHER`. `repeats=1` produced **no** driver fault |
| why the chain faults | **HYPOTHESIS, not measured.** Two candidates fit the evidence equally well: (a) the dependency the extension requires between dispatches writing one pool is `VK_DEPENDENCY_BY_TRANSITION`, which is absent from the pinned headers, so only a plain barrier is expressible; (b) the shader READS a descriptor slot that was never BOUND — every array binding in these runs is an unbound whole-array slot, and an out-of-range read of that is exactly the kind of access that gets a CS cancelled. (b) is the more likely of the two given `poolSlack=zero(unbound_array_slots)`, and it would also explain why a PLAIN barrier did not help |
| how to tell (a) from (b) | three cheap GPU experiments, none run yet: `VK_LAYER_KHRONOS_validation=1` on the probe (it reports the read of an unbound descriptor, or it reports the missing by-transition dependency); `RADV_DEBUG=hang` (the driver names the faulting compute); and enabling `nullDescriptor` + `robustBufferAccess2` in the probe device — if the fault DISAPPEARS, it was an OOB read (b), not a dependency problem (a) |
| which of those exist on THIS host | `RADV_DEBUG=hang` **is** accepted (the `hang` knob is present in `/usr/lib/libvulkan_radeon.so`, Mesa 26.2.6; `nocompute` and `allbos` are too). The Khronos validation layer is **NOT installed here** — the only layer objects under `/usr/lib` are `libXrApiLayer_best_practices_validation.so`, `libXrApiLayer_core_validation.so` and MangoHud — so that half of the experiment needs the validation-layers package installed first. Do not report "validation found no error" before checking the layer was actually loaded |
| did the pass write? | `out_changed=0 wrote_sampled=0 sb_nonzeroB=0 ub_nonzeroB=0` — but for the `repeats=4` run this is **confounded** by the cancelled CS (the note says so in the output); a clean single-dispatch census has not been re-run since the census was widened |
| what those recorded runs PROVED | Only `ran=1`: that the pipeline, the bind, the dispatch, the submit and the wait all complete. They proved NOTHING about the pass, because EVERYTHING was zero-filled — `push=32B=zero(uninitialized)`, the whole-array slots unbound, both images cleared to zero. A flow pass handed zeros can WRITE zeros, leaving the after-hash IDENTICAL to the before-hash, so `out_changed=0` cannot distinguish "ran and produced nothing" from "did nothing at all" |
| the census can now tell those apart | the OUTPUT image is pre-filled with a NON-ZERO SENTINEL (the printed note carries `out_prefilled=sentinel(a_write_of_zeros_still_changes_hash)`), so ANY write — including a write of zeros — changes the after-hash. The INPUT image stays zero-filled, and the SB/UB backs stay zero-filled because their metric is a COUNT of NON-ZERO BYTES, which a non-zero fill would silently corrupt. **COMPILE-VERIFIED ONLY:** no GPU run has used the sentinel yet. Real push constants and real bound array resources are still missing, and that half is 2c's marshalling work, not a probe flag |

Command for that clean census, when you choose to touch the GPU again (one process, one
device, `repeats` defaults to 1):

```
./build/linux/framegeneration-probe --dispatch-one=0 --dispatch-canvas=64x64
```

**Do not loop the probe over modules in separate processes**: each process creates a
`VkInstance`/`VkDevice` on the live desktop session, and 92 of them disrupts the compositor.
`--dispatch-sweep=N` (N ≤ 92) is the single-process form; it stops on the first unusable device
and names the orders it did not measure.

### What the classic passes DECLARE per binding slot (measured, GPU-free, 37th gate)

`python3 tools/extract/frame-generation/classic_shapes.py` runs the pinned SPIRV-Tools `spirv-dis` over all 92
classic modules and reads the RESOURCE half of the marshalling contract — the half `fg-plan.tsv` cannot give,
because the plan records only `(set,binding,KIND,arrayed)`. Measured on the pinned pack:

| measured | value |
| --- | --- |
| plan ↔ shader slot join | **304** `(set,binding)` slots, **304 agree, 0 plan-only, 0 shader-only** — the plan's layout half is exactly what the shaders declare, so the substrate's `304/304 bindings serialized` number was never the problem |
| slot kinds | **176 IMAGE / 32 SAMPLER / 96 BUFFER**, every image `2D` |
| arrayness | **304 of 304 slots are `OpTypeRuntimeArray`** — not one bounded binding in the whole classic chain. That is why the driver answers every set size as the wrapped `2³²−64`, and why serializing ONE descriptor per slot leaves each pass free to index past it |
| format contract | **60 slots FIXED a format in the shader** (`44 R32ui + 16 R32f`, which the runtime MUST bind exactly that) and **116 are BINDLESS** (`Unknown`; the format exists only in the descriptor written at bind time) |
| component-type contract | **104 slots declare `%uint`/`%int`** vs **72 `%float`** — and the substrate allocates `R32G32B32A32_SFLOAT` for EVERY image slot, so **60 of the 304 bindings this harness makes cannot satisfy the shader's own element type**. The plan's `SI`/`CIS` kind token cannot tell those apart; only the SPIR-V can |
| writes | `fetch=2556 sample=260 read=660` **write=0** — **no classic module contains an `OpImageWrite` at all**, while 92 slots declare a storage (`sampled=2`) image and every module carries `OpCapability PhysicalStorageBufferAddresses` + `OpName %BindlessCBV` |

The last row is the one that changes the plan for 2c. A pass that never issues a descriptor-bound image write,
but does carry a bindless uniform array of a struct named `BindlessCBV` under
`PhysicalStorageBuffer64`, writes through **addresses read out of those runtime-sized arrays**. So the chain cannot
be made to run by binding more descriptors into a pool alone: the runtime has to fill the `BindlessCBV` tables with
**real device addresses** (`vkGetBufferDeviceAddress` of resources it actually allocated), because a zero read there
is a write to address `0`, which is precisely the access that gets a compute shader cancelled with
`The CS has been cancelled because the context is lost`. That is now the LEADING explanation for the chain fault
recorded above, and it does not need `VK_DEPENDENCY_BY_TRANSITION` to be true.

Two naming routes were tried on this host and MEASURED as dead ends, recorded so they are not re-attempted:

* **SDK-source identifier bridge:** for each of the 18 vendored classic pass sources
  (`Kits/FidelityFX/framegeneration/fsr3/internal/shaders/*_pass.hlsl`, `include/gpu/**` bodies), take the
  identifiers unique to that one file and search the 92 module disassemblies: **0 of 92 strong matches**. The pack's
  FG 4.0.2 shaders keep their OWN internal names (`frontier_phi_6.5.ladder.8.ladder`, `BindlessCBV`) which the
  2.4.0 open sources do not share, so the version drift kills the bridge.
* **SDK blob digest bridge:** `internal/ffx_frameinterpolation_shaderblobs.cpp` and
  `ffx_opticalflow_shaderblobs.cpp` contain **no byte arrays at all** (18 KB of registry code each: the
  `...GetPermutationBlobByIndex` declaration only) — the classic VK blobs are not vendored, so there is nothing
  to hash. (The pack's own `modules.tsv` digests ARE the file digests, verified 486/486, which is what
  `capture-diff.py` matches on.)

That leaves the sanctioned ORDER + marshal oracle as it has been: a **capture** of AMD's own provider run
(`tools/capture/frame-generation/`: `stage-vk-dlls.py` → the three pinned DLLs → `capture-shim.cpp` under
`FSR4_VK_CAPTURE_DIR` → `capture-diff.py --emit-order` → `apply-capture-order.py --require-complete`). The rig is
now COMPLETER than it was, and the reason the capture still has not run on this host is MEASURED, not assumed:

* `host.cpp` — a Windows-mode host that creates a device with `VK_EXT_descriptor_buffer` + the cooperative-matrix
  / buffer-address / synchronization-2 extensions, loads AMD's `amd_fidelityfx_framegeneration_vk.dll`, and drives
  create → GET_VERSIONS → configure(host contract) → Prepare(REAL depth + motion) → generate(presentColor + 3
  output backbuffers) in the SDK's order, printing the code of EVERY step. It also records its OWN capture: the host
  hands the provider `hostResolver` as `getDeviceProcAddr`, and that resolver returns the host's wrappers for
  `vkCreateShaderModule` (SPIR-V dump), `vkCreateComputePipelines` (pipeline→module link), `vkCmdBindPipeline`
  (per-command-buffer bind) and `vkCmdDispatch{,Base,Indirect}` (the ORDER rows) — because a provider resolved
  through function pointers never touches the symbol table, which is exactly what an LD_PRELOAD shim alone cannot
  catch. `build-host.sh` cross-compiles it only when a real Windows toolchain exists, else exits 77 with the reason.
* The shim gained the same half: it now interposes `vkGetDeviceProcAddr`/`vkGetInstanceProcAddr` and hands back its
  own implementations for the recorded names, so a pointer-resolved provider lands in `capture.tsv` too. The 37th
  gate now asserts those eight symbols are exported from a real `-shared` build of the shim, since an empty capture
  otherwise looks identical to a provider that simply refuses.
* MEASURED blocker for actually running it here, now pushed one step further: `winegcc` (wine 11.22) compiles and
  runs Windows programs — a hello-world worked, and Wine's own vulkan shim reached RADV inside that run — BUT winegcc
  builds a LINUX/ELF image in Wine's native mode, and Wine's Windows import libs cannot link into it (`ld:
  unrecognized relocation pe-x86-64 against elf64-x86-64`), and Arch's `wine` package ships no `wine/dlfcn.h`
  (`/usr/include/wine` has none), so the native-mode route cannot load a Windows `.dll` either. A REAL cross-build
  IS available WITHOUT root, and is now scripted (`fetch-mingw.sh`): `paru -Sp cachyos/llvm-mingw` prints the package
  URL with no privileges at all, the download is 96.69 MiB, `tar --zstd -xf` extracts a relocatable toolchain into
  `build/tools/mingw/opt/llvm-mingw`, and its `x86_64-w64-mingw32-clang++` COMPILES `host.cpp` into a real Windows
  image (a 521 KB `.exe` was produced and linked on this host). What is still open is the CRT pairing, measured both
  ways: Wine's `libmsvcrt.a`/`msvcrt.dll` provide the C stdio entry points the host calls but not the Microsoft-CRT
  startup internals (`__p___initenv`, `_crt_atexit`, `_set_invalid_parameter_handler`, `__p__fmode`, `__p__commode`,
  `_configthreadlocale`), while llvm-mingw's own `libmsvcrt.a` resolves those internals only by importing them from
  the split modern CRT DLLs (`api-ms-win-crt-stdio-l1-1-0.dll` and friends), which Wine does not provide. So the
  next step is one measured experiment: build the host with llvm-mingw's GCC/UCRT startup objects (`crt1u.o`,
  `crt2u.o`, shipped next to `crt1.o`/`crt2.o`) linked against Wine's `msvcrt.dll`. Until that links, NO AMD
  dispatch ORDER exists in this repo, and the pack's `order_source` stays `grouped-scaffold`.


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

## Reference capture path (ground truth for the algorithm)

The two remaining FG blockers - the 382-pass ML execution ORDER and output-pixel parity - are closed only by **observing AMD's own provider run**, not by decompiling it. The v2.4.0 tree already carries what a capture needs:

- **AMD's own reference provider + host, both native Vulkan** (no game; no Proton bridge in principle):
  - `Kits/FidelityFX/signedbin/amd_fidelityfx_framegeneration_vk.dll` (pinned SHA `d002dabf...`, FG 4.0.2.4555) plus its `_upscaler_vk.dll` / `_loader_vk.dll` siblings - AMD's reference binaries on the SAME API our `.so` implements.
  - `Samples/Upscalers/FidelityFX_FSR/vk/fsrapirendermodule.cpp` - a **native-Vulkan** FFX host (`ffx_api_framegeneration_vk.hpp`) that drives create -> the **4-queue Frame Generation swapchain** (`CreateContextDescFrameGenerationSwapChainVK`: game / asyncCompute / present / imageAcquire) -> configure -> prepare -> generate -> present. Readable host-integration ground truth for exactly the boundary we implement.
  - `dlls/amdvkffx64.dll` (pinned `179bf7e1...`) - the AMD Vulkan-FFX driver module (the `ffx-vk-linux` family).
- **What the `_vk` DLL gives statically, right now, GPU-free:** its readable shader **IDENTITY** - 382 `mlfi_{4k,8k}_passNNN` (pass0..190 per tier), 156 cooperative-matrix markers, and the classic interpolation names (`CsDilate`, `cbInpaintingPyramid`, `uOpticalFlowPyramidLevel`, `iInputLumaResolution`, `FfxFrameInterpolationDebugViewport`). `tools/extract/frame-generation/vk-identity.py` proves that set EQUALS our DX12-extracted pack (dll-only=0, pack-only=0) - a **second AMD-authored confirmation**, pinned by the new gate `test_vk_dll_shader_identity`.
- **What it does NOT give statically:** the `_vk` shader **bytes** are one compressed proprietary blob (no plaintext SPIR-V, no standard zstd/lz4 frames - the lone `18 4D 2E` marker decodes to a nonsense 3.7 GB size, so it is not a real zstd frame), and `passNNN` are **container labels, not the ML execution order** (already proven non-derivable). Identity is confirmed; the schedule still must be observed.
- **To recover the ML order + pixel parity:** build the `vk` sample (MIT), select the **ML FG variant** (not classic FSR3) on a device AMD's FG actually runs (RX 9000 / cooperative-matrix capable), and intercept the Vulkan compute the DLL issues - a thin `libvulkan.so` shim or the validation layer logging each `vkCmdDispatch*` -> (threadgroup dims, bound-pipeline SPIR-V hash, resolved descriptors) emitted in our existing `modules.tsv`/plan schema, so the captured order diffs straight against `framegeneration-exec --expect-plan`; read back the output textures for A/B via `tools/validate/ab-frames.py`.
- **Deps to stage in `dlls/`:** running the VK path needs `_loader_vk` (pinned `38a56a01...`) + `_upscaler_vk` (pinned `4a7e895a...`) alongside the already-present `amd_fidelityfx_framegeneration_vk.dll`. The DX12 alternative - capturing the pinned `_dx12` DLL (`e684...77bb6`) under Wine + vkd3d-proton - keeps the traced Vulkan calls aligned with the shader set you already extracted.

- **Now scaffolded, not just described (gates 27-30).** `tools/capture/frame-generation/stage-vk-dlls.py`
  stages the loader + upscaler + framegeneration `_vk` DLLs from the SDK `signedbin` into a runtime dir, verifying
  each against its pinned SHA (`d002 / 4a7e / 38a5`) and refusing a non-pinned version, so a capture rig never
  silently mixes versions; `tools/capture/frame-generation/capture-diff.py` diffs an intercepted `vkCmdDispatch`
  trace against the pack PLAN by SPIR-V digest - per-module first-seen position + call count, `extra` (a dispatch
  not in the plan) and `missing` (a plan module the capture did not exercise) - and NEVER derives the order itself.
  `capture-diff.py` also resolves the shim's OUTPUT form (per-dispatch path to a dumped `.spv` file) by hashing
  the bytes, which round-trips to pack identity (all 474 `modules/<name>.spv` files hash to their `modules.tsv` digest).
  The shim itself is now authored too - `tools/capture/frame-generation/capture-shim.cpp` is an LD_PRELOAD over
  `vkCreateShaderModule` (dumps SPIR-V) / `vkCreateComputePipelines` (links pipeline -> module file) /
  `vkCmdBindPipeline` + `vkCmdDispatch{,Base,Indirect}` (records order), built on a Vulkan-free `capture-ledger.hpp`.
  Six GPU-free gates cover the rig: `test_stage_vk_dll_pins_reference_rig`, `test_capture_diff_order_reader_roundtrip`
  (the FG pack's modules.tsv), `test_capture_diff_upscaler_pack_roundtrip` (the SAME rig accepts the UPSCALER
  pack-unified manifest's 504 .spv payload digests as its identity plan), `test_capture_ledger_unit_runs` (ledger
  dispatch attribution), `test_capture_shim_builds_against_vulkan_headers` and `test_capture_order_fusion`
  (`tools/extract/frame-generation/apply-capture-order.py` promotes an AMD-captured first-seen schedule into the
  plan as an authoritative `amd_order` + `order_source=amd-runtime-capture`; the gate proves the round-trip, refuses
  unknown-digest/duplicate-first_seen captures, and that with NO real capture the committed plan stays `grouped-scaffold`)
  (compiles vs the pinned fork, `pCode`). Nothing hand-written remains: the runtime-capture flow is stage the DLLs ->
  LD_PRELOAD the shim on the sample -> `capture-diff.py --expect-plan`, and the ORDER it yields is AMD's own observed one.

## Next steps

The Frame Generation algorithm is the only genuinely open work; everything else is buildable **without** touching the closed 382-pass ML boundary. In priority order:

1. **Build the Swapchain present/pacing provider** (a separate effect id, `FFX_API_EFFECT_ID_FRAMEGENERATIONSWAPCHAIN_VK`). It **replaces the app's `VkSwapchainKHR`** and **delegates the interpolation workload to the host** (`frameGenerationCallback`, or the `...InterpolationCommandListVK` / `...InterpolationTextureVK` query) - it does **not** run the ML interface itself. Fully specified by the SDK source + the present/pacing/UI + swapchain gates, so it needs **no** ML and **no** reference capture to build honestly: `VkSwapchainKHR` replacement + the 2 CPU worker threads (present + pacing) + UI composition (`...USE_PREMUL_ALPHA`) + the `WaitForPresentsVK` teardown flush + driver-assisted pacing. This is where the still-open contract items above (UI paths, `RegisterUiResource...VK`, the internal-double-buffering lifetime, the `generationRect` letterbox width/height-equality rule, and the mutex-guard list) land.
2. **Land a classic-only 4x interpolation runtime.** **Phase 1 is done**: the provider now reads the pack's own plan tables at `Prepare` and really allocates the plan's intermediates + a driver-sized descriptor pool, creates the 92 classic-subset `VkComputePipeline`s from the pack's SPIR-V, and serializes 304/304 of that subset's bindings through `VK_EXT_descriptor_buffer` - which is what makes `ffxQuery(GetGPUMemoryUsage)` report real allocation-backed bytes instead of refusing. What is still open is the *identity* half: which anonymous `CS` module is which pass, in what order, against which of the plan's targets, and then the recording dispatch itself (already gated by `--compile-pack` (474 created), `--descriptor-buffer-parity` (0 failed) and `--expect-plan` (checked=474 missing=0 mismatch=0)). Image parity still needs AMD reference captures, and it stays **<=4x** per the `outputs[4]` ceiling. The host-side half of this phase is DONE and runnable: `bash tools/FGTest/run.sh`
   opens the 30-second animated-triangle window (measured today: 552 presented frames at ~25 fps, AMD's own
   dispatch 0.6-0.8 ms of each), its HUD prints the resolutions, the FPS at 4x scale, the frame/upscaler/FG-dispatch
   clocks at 2x and `GOT n X REQ n X`, and `run.sh --headless` yields the identical proof plus the HUD corners as
   ASCII where there is no display - so the remaining gap (which CS module is which pass, and the dispatch itself)
   is something the app shows on screen rather than a claim. **Phase 2 has now landed its foundation, and it
   changed what is actually open.** `tools/extract/frame-generation/classic_order.py` (+ the 34th gate,
   `test_classic_chain_shape_join`, which writes `build/fg-classic-order.tsv`) joins the two ground truths the repo
   owns. Measured: the pack's classic subset is **92 modules whose plan ORDER is already recorded (contiguous
   0..91), all `resolved`, 64 collapsible / 28 plain, 6 distinct binding signatures, in exactly 7 thread-group
   classes** (56 x 8,8,1 / 16 x 256,1,1 and four per variant of 16,16,1, 64,1,1, 16,4,1, 4,4,4, 32,8,1 - i.e. **4
   variants of everything**), while their NAME column is empty; the vendored SDK names **18 classic pass families**
   (11 frame-interpolation + 7 optical-flow, each declaring its own thread-group shape: 9 at 8,8,1; 4 at 256,1,1; one
   each at 16,16,1 / 32,8,1 / 64,1,1 / 16,4,1 / 4,4,4), plus 2 swapchain shaders with no compute triplet that belong
   to the separate SWAPCHAIN effect id. The two sides **join with zero orphan shape classes on either side**, which
   leaves a measured naming state per slot: **20 unique-by-shape, 16 count-matched to exactly 4 named owners
   (needs pairing, not guessing), 56 ambiguous** - and crucially the pack carries **14 tile-8 slots per variant
   where the open SDK lists only 9 owners**, so 5 slots per variant are provably NOT covered by the SDK sources:
   those stay DLL-reversal work, now bounded to 20 of the 92 slots. The SDK provider source also fixes the runtime
   RULES (all six pinned by gates): the optical-flow dispatch runs BEFORE the interpolation dispatch, block size
   8 (the same 8x8 tiling those 56 slots show), a zero `generationRect` falls back to the full presentColor extent,
   `reset` is the OR of caller flag, Prepare flag and a disjoint-frameID test, Prepare pairs to generate through
   `frameID % MAX_QUEUED_FRAMES`, and reported memory usage is the SUM over FrameGeneration + OpticalFlow + Shared
   contexts. So the runtime half is now: allocate the SDK's shared-resource set with that arithmetic, bind the 92
   modules in recorded order from the plan's resolved layouts (Phase 1 already creates those pipelines and the
   plan-driven intermediates), record + submit, and let `generate` answer OK - leaving the 382 `mlfi_*` modules
   refused. **The record + submit half is no longer hypothetical:** `PlanRuntime::dispatchOne` serializes the plan's
   own descriptors with `vkGetDescriptorEXT`, binds the driver-sized pool, sets per-set offsets, binds the pipeline,
   really dispatches on the caller's device and reads back every writable output - one pass per device lifetime measured
   `ran=1` with no driver fault. What is NOT proven is a CHAIN (several dispatches over one pool fault here, see the
   measured section above). And the app closes the testability gap the README used to call "needs AMD captures":
   FGTest's scene is
   analytic (a triangle on a known translation/rotation schedule), so the classic chain's flow is predictable -
   a translation-only pair must yield a near-constant flow vector and an interpolated frame whose geometry sits
   between the two inputs, which is a real oracle for the classic path independent of pixel parity.
3. **The 382-pass ML (`mlfi`) chain stays blocked** on the DLL's closed CPU dispatch scheduler + AMD reference captures. Choose between: **(a) deep PE disassembly** of `amd_fidelityfx_framegeneration_dx12.dll` to recover the bindless descriptor writes / ML node order (high risk, unverifiable without captures), or **(b) hold** - accept FG as complete-as-documented with the honest, fully-gated substrate. **Recommended: do (1) and (2) first** - they deliver a real, honest FG experience without ever opening the closed boundary.

### Phase ladder to a running Frame Generation (measured)

Four phases make the **classic** chain generate frames and be testable in the app; a fifth makes it usable through
AMD's own swapchain path. The 382 `mlfi_*` ML modules are **2 further phases, not an impossible one** — their
weights (`pack/mlfi-model.bin`, 9,437,184 B), chain order (two 191-entry chains, order fixed by pass index) and
pipelines (474/474 created on this Mesa device, cooperative-matrix and all) are already in hand; what is open is
**M-a** the chain-edge → resource + scratch marshalling, and **M-b** validation (behaviour via the app's analytic
oracle, pixel parity via captures).

1. **2a - name the 92 ordered classic slots.** `build/fg-classic-order.tsv` already splits them: **20 unique by shape,
   16 count-matched to exactly 4 named owners (needs pairing by order + arguments), 56 tile-8 ambiguous** (the pack
   carries 14 tile-8 slots per variant, the open SDK only 9 owners - so **20 of the 92 have no owner in the SDK at all**
   and are the real DLL-reversal target). Evidence: the CPU-side argument marshal the `--write-descriptors` /
   `--expect-plan` / capture-rig tooling already consumes. Done when that table has a role for 92/92 slots, gated.
2. **2b - allocate the shared-resource set with the SDK's OWN arithmetic.** Optical-flow vector, SCD histogram + output,
   dilated depth, dilated motion vectors, reconstructed previous nearest depth (both frame-toggle sets), inpainting-pyramid
   staging, hudless colour. Done when `GetGPUMemoryUsage` (V1) grows from the allocation-backed
   `total=66,365,568 / aliasable=10,368` by exactly the added set and stays consistent with the plan-model V2 answer.
3. **2c - bind, record and submit in recorded order.** Per-slot bindings from the plan's 6 resolved signatures, 32-byte
   push constants per pass, usage flags, explicit barriers, optical-flow passes BEFORE interpolation passes, honouring the
   six rules the SDK pins (block size 8, zero-`generationRect` fallback, reset OR rule, `frameID % MAX_QUEUED_FRAMES`
   pairing, memory-usage sum, debug-view flags). Done when `ffxDispatch(generate 1..3)` returns `OK` for the classic
   chain with non-zero `generated` and real output images written, while `mlfi` stays `RUNTIME_ERROR`.
   **Measured on this host, and it constrains HOW:** ONE pass per device lifetime runs clean
   (`ran=1`, no driver fault) but **`repeats=4` dispatches of the same module over one pool inside one command
   buffer produced `radv/amdgpu: The CS has been cancelled because the context is lost` while
   `vkQueueWaitIdle` still answered `VK_SUCCESS`**. WHY is NOT settled: the missing-flag explanation
   (`VK_DEPENDENCY_BY_TRANSITION` is absent from the pinned headers) is ONE hypothesis, and an out-of-range
   read of an UNBOUND whole-array descriptor is at least as plausible given those runs bound no real resources
   at all. The two are separated by the validation layer / `RADV_DEBUG=hang` / `robustBufferAccess2`
   experiments recorded in the measurement table above, and until one of them runs the chain blocker should be
   read as UNSCOPED — it may evaporate the moment the dispatches get real inputs and real bindings. Two routes to try, in order: (i) **separate
   command buffers with one dispatch each** and a real `vkCmdPipelineBarrier` / event dependency between the
   submits (the earlier per-order runs show the fault tracks multi-dispatch-in-one-buffer, not the
   submit itself); (ii) **give each pass its OWN descriptor pool** so no dispatch writes a pool another
   dispatch is reading, which sidesteps the missing flag entirely at the cost of ~112-144 B per pass (the
   measured per-module pool size). Either way the ~2 s first-execution cost is per-device, not per-pass, so a
   92-pass chain is not 92 x 2 s.
   **UPDATED by the 37th gate, which reads the shaders and shrinks those hypotheses to one.** Over all 92 classic
   modules: **every one of the 304 `(set,binding)` slots is an `OpTypeRuntimeArray`** (so no run has ever bound a
   BOUNDED array — the `2³²−64` answer is structural, not a quirk), and **write=0**: no classic module contains an
   `OpImageWrite` at all, while every module carries `PhysicalStorageBufferAddresses` + a bindless uniform array
   named `BindlessCBV`. A pass that writes through addresses READ OUT of an unbound array is writing to the value
   the harness left there, and the harness left ZERO — a write to address 0 is the textbook shape of
   `The CS has been cancelled because the context is lost`. So route (ii) "give each pass its own pool" is
   insufficient on its own: **the bindless tables have to be filled with REAL device addresses** (`vkGetBufferDeviceAddress`
   of resources this runtime allocated), and the **60 format-fixed slots (`44 R32ui + 16 R32f`) and the 104
   `%uint`-typed vs 72 `%float`-typed slots must be bound by TYPE**, not with the single `R32G32B32A32_SFLOAT`
   backing the substrate uses for every image today. Both facts come from `build/fg-classic-shapes.tsv`
   (`classic_shapes.py`), which is where 2c's marshalling code should read them from — the plan's `SI`/`CIS` kind
   token cannot distinguish them.
4. **2d - prove it in the app, which is what the app was built for.** Measured with `--fg=2x` today:
   `host_frames=56 generated_frames=0 ok_calls=0 last_generate=3`, i.e. `GOT 1 X REQ 2 X` with the provider's
   own refusal on every frame — the honest baseline this step has to beat. Two harness fixes landed alongside
   the measurement: `ok_calls` now counts only calls that ANSWERED with frames (a `--fg=1x` run used to print
   `ok_calls=55` for a run in which nothing was ever requested — a hollow success read as a delivery), and the
   (`VK_xr_swapchain`). MEASURED NOW, and the first correction of that fix was itself wrong: `VK_KHR_swapchain` IS
   offered by the device and `vkCreateSwapchainKHR` RESOLVES with the extension enabled at DEVICE level ONLY — also
   requesting it at INSTANCE level made `vkCreateInstance` FAIL, which surfaced as a bare `NO_DEVICE` and turned the
   whole app off. The instance-level request is gone, and `initVulkan` now NAMES the failing step and retries each
   create WITHOUT the added extension, with stdout UNBUFFERED so a crash leaves its last line behind. The WSI path
   itself still crashes, now LOCALIZED to `vkQueueSubmit` of the first present-draw buffer (`wsi p1` prints, `wsi p2`
   does not), after two HOST bugs in it were found and fixed: a sampler write whose SAMPLER was `VK_NULL_HANDLE`
   (spec-legal, but `vkUpdateDescriptorSets` SIGSEGVed; a REAL sampler object is written now) and a Prepare that never
   LINKED the camera-info block (0x0002000a), which is why the provider refused it there while every other path in
   this tree was accepted. `vkAcquireNextImageKHR` is result-checked and its index bounded too, since a garbage index
   into `fences[]` is the delayed-SIGSEGV shape. So `--present` is a localized harness bug, not a driver verdict —
   and the validation layer installed on that host is the tool that can name the VUID. A gate
   (`test_harness_extension_names_are_real`) now rejects any extension name that is not defined by the pinned
   headers, which is exactly the class of silent capability loss that caused it.
   Then `bash tools/FGTest/run.sh --fg=2x` must show `GOT 2 X REQ 2 X` and a real `MS FGMEM` cost, and the analytic oracle on FGTest's own scene must hold: a
   translation-only frame pair yields a near-constant flow vector, the interpolated frame's geometry sits between the
   two inputs, and the disocclusion mask stays empty where nothing is revealed. Pixel parity against AMD stays
   capture-gated; this oracle does not need captures.
5. **3 - the Swapchain present/pacing provider** (separate effect id): replaces the app's `VkSwapchainKHR`, delegates
   interpolation to the host through `frameGenerationCallback`, owns the two CPU worker threads, UI composition, pacing
   and the `generationRect` letterbox rule. Specifiable from the SDK source with no ML and no captures.

**So: 4 phases to frames actually generated and testable here, 5 to a usable in-app Frame Generation, and 6 to a
full classic+ML runtime (2a-2d + M-a + M-b) - and the genuinely closed half of that is only the marshalling plus
parity evidence, not the pass order.**

One driver-side constraint belongs to 2c, not to the algorithm: RADV creates all 474 compute pipelines here, but
enabling `VK_EXT_descriptor_buffer` in the SAME device that hosts the app's graphics pipelines was measured to crash
inside the driver's own graphics pipeline creation (SIGSEGV in `setupPipeline`). So the FG runtime needs its own
`VkDevice` or that driver bug fixed - a bounded workaround, and never a reason to fake an interpolation. The
workaround is now IN PLACE in FGTest: `VkApp::fgDevice` is a second logical device over the same physical device,
created WITH the transport, and ONLY the Frame Generation backend block names it; the app keeps rendering on
`a.device`. Before that, the window refused the substrate on every frame with `caller device exports none of:
vkGetDescriptorEXT` - which was a HARNESS under-provisioning, not an FG limitation, and is why the probe could
build the substrate while the app could not. THAT FIX IS MEASURED, not expected: the window now prints
`window FG ffxCreateContext = 0 (0=OK)`, then on EVERY frame `phase1 substrate render=480x270 ... pipelines=92/92
... usage=35261568B (aliasable 10368B)` and `prepare substrate state: substrate built (usage reportable)`, with no
refusal line anywhere in the log, and a 5 s window run reported `host_frames=138` at ~27 fps under the honest FG
tally (`generated_frames=0 ok_calls=0 cancel_calls=0 empty_ok_calls=0 -> delivered 1X of requested 2X`). The byte
figure follows the extent the caller named (`47933568B` on the 1280x720 present rig, `66365568B` for the probe's
plan-sized query), so it is a function of the host shape, not a constant of the effect.

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
