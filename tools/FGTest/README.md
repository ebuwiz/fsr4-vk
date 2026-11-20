# tools/FGTest — headless "does it run?" probe for an animated-triangle Vulkan host on FSR4

A small, self-contained, **offscreen** (no window, no GLM, no Sascha framework) Vulkan 1.3 sample that
animates a moving/rotating colored triangle and drives BOTH FSR4 providers through their REAL, documented
FFX modular-API calls. It answers "does it run?" **per feature**, and never fabricates a result.

## Run it as a visible app (the deliverable of this phase)

```sh
bash tools/FGTest/run.sh          # 1440x1080 window for 30 s, HUD painted into every frame
```

`run.sh` compiles the SPIR-V + the app when anything is stale, points the providers at `assets/pack-unified`,
opens the window and prints one measured line per second (`t=Ns frames=.. fps=..`). `Esc`, `q` or closing the window
ends it early. Flags: `--seconds=N`, `--headless` (the same proof offscreen, HUD corners printed as ASCII — no
display needed), `--fg=2x|--fg=4x` (raises only the *requested* frame-generation multiplier the HUD reports),
`--present` (attempt the real Vulkan-WSI swapchain path), `--build` (compile only). Exit codes: **0** displayed;
**77** no display / no suitable device (run.sh then re-runs the identical proof offscreen and keeps 77 so a GPU-less
host stays honestly reported); **2** the repo's provider `.so` files are missing (`bash tools/build/linux.sh` builds
them).

## Build + run (headless, on a real driver)


```sh
bash tools/FGTest/build.sh
```

Requires the repo's built providers (`build/linux/amd_fidelityfx_{upscaler,framegeneration}_vk.so`),
`glslc`, `g++` (C++20), libvulkan, and a Vulkan device. Shaders (triangle.vert/frag) are compiled to SPIR‑V
by `build.sh`. PPM frames land in `tools/FGTest/build/`.

## Visible window (Wayland/X11) — optional

FGTest is headless by default. To actually WATCH the moving triangle in a window,
add `--window` (default 30 s; `--window=N` to change). It links **SDL2** (`sdl2-config`), and there are two window modes:

* `--window[=SECONDS]` (default, crash-free): each frame is rendered OFFSCREEN through the app's own
  headless `VkInstance` at 480x270, **upscaled by the REAL AMD FSR4 compute to 960x540**, read back into a CPU
  RGBA8 buffer, then blitted to the window with SDL's SOFTWARE renderer (960x540 → window size by SDL's scaler).
  A bitmap-font HUD (5x7 glyphs, no atlas, no shader) is baked into the buffer, and the FPS number is also printed
  once per second together with the average `ffxDispatch(upscale)` cost. The WSI swapchain is not touched by
  DEFAULT because the software blit is what makes the HUD reproducible with no compositor involved — NOT because
  this host lacks the entry point. An earlier revision of THIS harness claimed "the driver exposes no working
  `vkCreateSwapchainKHR`" and that claim WAS WRONG: the device asked for the name `VK_xr_swapchain`, which is not
  an extension at all, so `VK_KHR_swapchain` was never enabled and the entry point resolved to NULL. Both halves
  are fixed now (the extension is requested at instance level and at device level), so `--present` has a genuine
  chance of working on this Wayland host; it has NOT been re-measured since the fix. Measured on
  RX 9070 XT: FSR GPU 0.6 ms/frame, whole frame ≈ 25 ms (the 4 MB FP16 read-back + half decode dominates) → ~21 fps
  in a 1440x1080 window. If the upscaler context is unavailable the same window falls back to the native
  480x270 read-back, and it says so in the printed line. Measured today with `run.sh`: the 30 s window run presented
552 frames at a measured ~25 fps (~40 ms/frame end to end, of which AMD's own dispatch is 0.6-0.8 ms), and a 6 s
spot-check presented 137 frames at ~26-27 fps.
* `--window --present`: the real Vulkan-WSI path (SDL window surface → swapchain → AMD-FSR-upscaled blit
  1280x720 → 1440x1080). MEASURED NOW, STAGE BY STAGE, because the path prints every stage and stdout is
  unbuffered (a crash used to leave NO trace at all, which is why this bullet used to be speculation):
  `VK_KHR_swapchain` is offered by the device and `vkCreateSwapchainKHR` RESOLVES with the extension enabled at
  DEVICE level ONLY — the instance-level request added by the first attempt at this fix made `vkCreateInstance`
  FAIL, which is what produced a bare `NO_DEVICE` and killed the app entirely; that request is gone, and `initVulkan`
  now retries each create WITHOUT the added extension and prints WHICH step failed.
  What the run reaches: surface → 90 surface formats → 3 present modes → swapchain created (1440x1080, 3 images,
  fmt=58) → its images + views + depth → rig images + pipeline → clear pass → present pipeline → descriptor set.
  Two HOST bugs inside that block were found and fixed: the sampler write used `VK_NULL_HANDLE` as its SAMPLER
  (spec-legal, but `vkUpdateDescriptorSets` then SIGSEGVed, so a REAL `vkCreateSampler` object is written now), and
  this call site never LINKED the camera-info block (0x0002000a) after Prepare, so the provider refused Prepare
  HERE while every other path in the tree was accepted. `vkAcquireNextImageKHR` is now result-checked and its
  index bounded, since a garbage index into `fences[]` is the classic delayed-SIGSEGV shape.
  STILL MEASURED BROKEN: the first frame dies INSIDE `vkQueueSubmit` of the present-draw buffer — the log prints
  `wsi p1` (submitting) and never `wsi p2`. So `--present` is a LOCALIZED WSI bug in this harness, not a driver
  verdict, and it is NOT crash-safe today: the guards cover setup, not a death inside the driver. The validation
  layer is installed on this host now, so the next experiment is
  `VK_LAYER_KHRONOS_validation=1 build/FGTest … --present` read for the VUID it names. The DEFAULT window path
  touches none of this and works.

```sh
# after `bash tools/FGTest/build.sh` has produced the .so's + triangle.*.spv:
FSR4_VK_ASSET_ROOT=assets/pack-unified \
DISPLAY=:0 XDG_RUNTIME_DIR=/run/user/1000 \
build/FGTest "$REPO/build/linux/amd_fidelityfx_upscaler_vk.so" \
         "$REPO/build/linux/amd_fidelityfx_framegeneration_vk.so" \
         build/triangle.vert.spv build/triangle.frag.spv --window=30
```

The normal entry point is `bash tools/FGTest/run.sh` — 30-second window, HUD painted into every frame, `Esc`/`q`
to end it. run.sh flags: `--seconds=N`, `--headless` (identical proof offscreen + the HUD corners as ASCII),
`--fg=2x|4x` (raises only the requested multiplier the HUD reports), `--present`, `--build`. Exit codes: **0**
displayed, **77** no display/device (it re-runs the identical proof offscreen and keeps 77 so a GPU-less host
stays honestly reported), **2** the repo's provider `.so` files are missing. The

The `--present` loop prints `frame N: present=FSR(WxH) t=..s fps=.. interpolated=0 generated=0 [FG req=Nx]`, and
one `[FG] prepare=3 generate=3` line naming why: AMD's provider refuses `generate` ("refusing to fake frame
generation") even though it was handed real `depth`/`motionVectors`/`presentColor` images, so `generated` stays 0
and nothing is fabricated. `interpolated`/`generated` becoming non-zero requires the missing interpolation/ML/present
runtime, not a different call sequence here. If the machine's Vulkan loader does not expose a window surface
(e.g. headless CI) it prints `NO_SURFACE` and exits cleanly rather than fabricating anything.

## What it demonstrates (measured on AMD Radeon RX 9070 XT / Mesa RADV GFX1201)

- **Moving triangle host: RUNS.** 6/6 offscreen frames render non‑black and differ (rotation + translation)
  → the graphics host Frame Generation would plug into actually runs (verified by distinct frame hashes).
- **FSR4 upscaler: the REAL pixel dispatch RUNS end-to-end** (`FFX_API_DISPATCH_DESC_TYPE_UPSCALE` over this
  app's own `VkImage`s, recorded into a host command buffer, submitted, then copied back off the GPU):
  3/3 frames return `0 (OK)`; the 960x540 FP16 output is fully non-black, its `R` channel measures
  `min=0.000 max=0.996 mean=0.128` (i.e. the real scene, not a fill), the four PPM corners read
  `(25,25,50)` like the native RGBA8 read-back, 41k distinct colours, and **frame 0 vs frame 2 differ in
  323 520 of 1 555 200 bytes** — the AMD compute ran AND reacted to the moving geometry. GPU cost is
  **0.6 ms/frame** (14 ms for the first frame, which compiles the pipelines); the ~25 ms per frame in the
  window path is the CPU read-back (4 MB copy + FP16→RGBA8), not the upscaler.
- **Frame Generation provider: the full documented host-integration ladder runs WITH the images it owes, honestly:**
  - `ffxCreateContext(frame-generation) = 0` (OK) — the device meets the FG execution contract
    (cooperative matrix, 8/16-bit storage, runtime descriptors, etc.).
  - `ffxQuery(GET_VERSIONS)` advertises the real classic id `0xF600000001000002`.
  - `ffxQuery(GET_GPUMEMORYUSAGE) = 3 (RUNTIME_ERROR)` **in the ladder, and by design**: the ladder asks BEFORE any
    `Prepare`, and the substrate is allocated at `Prepare`, so there are no allocations behind the number yet and
    the provider refuses instead of inventing one. The DEVICE question is settled and MEASURED: the ladder's backend
    block names `a.fgDevice` (a second logical device created WITH `VK_EXT_descriptor_buffer`), and once `Prepare`
    runs the substrate IS built and usage becomes reportable — `phase1 substrate render=480x270 ... pipelines=92/92
    descriptors=304/304 ... usage=35261568B (aliasable 10368B)`, and `usage=47933568B` on the 1280x720 present rig.
    The byte total SCALES with the extent the caller names; the probe's `66,365,568 B` belongs to its plan-sized
    query, so no one of these is a constant of the algorithm.
  - `ffxConfigure(host contract) = 0` (OK) — swapchain/present/framegen callbacks accepted.
  - `ffxDispatch(Generate) BEFORE Prepare = 6` (PARAMETER) — the order guard is live, so the ladder below is
    not a no-op: `Configure → Prepare → generate` really is enforced.
  - `ffxDispatch(Prepare + REAL depth/motion + basis) = 3` — **"prepare accepted (order + camera valid) but
    dispatch pending"**: the app's REAL `D32_SFLOAT` depth (claimed with the `DEPTHTARGET` usage bit) and `RG16F`
    motion vectors are handed over, the linked orthonormal camera basis validates, and only the interpolation/ML
    runtime is refused.
  - `ffxDispatch(PrepareV2, embedded basis) = 3` — the 2.4.0 `0x0002000c` shape (camera basis EMBEDDED, read
    through the provider's `ffgcheck::PrepareV2Desc` ABI mirror) validates the same way.
  - `ffxDispatch(Generate 2 + presentColor + outputs) = 3` — **refused: "refusing to fake frame generation."**
    `presentColor` is the FSR-upscaled colour and `outputs[0..2]` are REAL display-size backbuffer images this
    host allocated, so the refusal is the provider's own, never a host-side parameter error.
  - `ffxDispatch(Generate cancel, 0 frames) = 0` (OK) — the provider's documented cancel branch answers honestly.

## HUD readout (painted into every presented frame)

The window draws a debug readout with **no text pipeline at all**: `tools/FGTest/main.cpp` carries a
hand-set **5x7 bitmap font** and blits the text into the SAME host-side RGBA8 buffer the FSR read-back
already produces, BEFORE that buffer is uploaded to SDL's software renderer — so the readout travels with
the pixels it labels and there is nothing to set up, bind, or dispatch for it.

**Top-left — the three resolutions actually in play** (each read back off the rig or off SDL, never assumed):

```
WIN: 1440x1080            SDL reports the live window extent every frame (the window is RESIZABLE)
RENDER: 480x270           the app's own offscreen render extent, read off FsrRig::rw/rh
UPSCALER VK: 960x540      what amd_fidelityfx_upscaler_vk rendered, read off FsrRig::ow/oh
```

**Top-right — the clocks**, with FPS painted at **4x** and everything else at 2x:

```
21 FPS                measured over the current second off PRESENTED frames
47.6 MS FRAME       measured frame-to-frame, off the presented frames (not SDL's 16 ms delay)
5.6 MS UPSCALER     this frame's ffxDispatch(upscale) + read-back cost (0.6 ms GPU dispatch + ~5 ms copy/decode)
0.0 MS FGMEM REFUSED  the ffxDispatch(generate) call cost, timed in this process
GOT 1 X REQ 2 X     the multiplier the provider DELIVERED vs the one --fg=2x/4x REQUESTED
FGEN 0/552 CODE 3   frames the provider GENERATED / host frames it was ASKED to interpolate / last return code
```

`FGEN` is the counter to watch. The window path now drives **Frame Generation every frame** —
`Prepare(frameID)` then `generate(frameID)`, paired the way the SDK pins it (`MAX_QUEUED_FRAMES = 2`, so
`frameID` alternates slots and Prepare is re-issued per frame rather than once), handing over this app's
real `D32` depth, `RG16F` motion vectors, FP16 present colour and three real display-size backbuffers, into
a command buffer the provider records in and the host submits. `generated` sums only what the provider WRITES
BACK in `numGeneratedFrames` after the call, so:

- while `generate` is refused, the screen reads **`FGEN 0/N CODE 3`** (N = presented host frames) — zero,
  not the requested multiplier;
- when the classic chain lands, the same code path reads **`FGEN 552/552 CODE 0`** and `GOT 2 X REQ 2 X`,
  with no change to how the number is produced. That is the milestone to watch for.

The same tallies print to stdout, once per second and at the end of the run — the SHAPE of those lines
(this path has not been executed since the counter landed; run `run.sh` to see the live numbers):

```
FG: frames_generated=0 host_frames_asked=552 last_generate_code=3 (0.018ms avg)
FRAME GENERATION tally: host_frames=552 generated_frames=0 ok_calls=0 cancel_calls=0 empty_ok_calls=0 last_generate=3 prepare=3 avg_cost=0.018ms -> delivered 1X of requested 2X (ok_calls counts DELIVERIES, never cancels)
```

`MS FGMEM` in the window is now that live average (per-frame call cost), not the headless ladder's one-shot
number, and `FGEN NO CONTEXT` prints when the FG provider could not create/configure a context on this host. The
HUD line is `FGEN n/m CAN k CODE c`: frames DELIVERED over generate calls made, calls that took the DOCUMENTED
cancel branch `numGeneratedFrames == 0` (which is what `--fg=1x` exercises every frame), and the last return code.
That split exists because a `--fg=1x` run used to print `ok_calls=55` for a run in which the host NEVER asked
for a single interpolated frame — a hollow success that read as a delivery. Cancels are now excluded from
`ok_calls`, and a call that returned OK while ANSWERING 0 despite being asked for frames lands in
`empty_ok_calls`, so a future fake success cannot hide in the delivery counter either.

`--fg=1x|2x|3x|4x` sets the REQUESTED value; because `generate` is refused here the screen keeps showing
`GOT 1 X` next to `REQ 2 X`, which is the point of printing it at all — a multiplier copied from the
request would imply an interpolation that never happened. Equally, the headless proof path has no presented
frames to time, so there the same font prints **`FPS NO PRESENTED`** / **`FRAME NO PRESENTED`** instead of a
derived pseudo-frame-rate; a number only appears where something was actually measured.

`--hud-dump` paints that same readout into the PPM proof frames (drawn AFTER the checksum, so the per-frame
proof is untouched) and prints the two corners as ASCII — that dump is how the glyph table's legibility is
checked without a display, and it is what was run to produce the numbers above:

```bash
FSR4_FGTEST_OUTDIR="$BUILD" ./build/FGTest --hud-dump <upscaler.so> <fg.so> <vert.spv> <frag.spv>
```

The GPU `--present` path is unchanged: it keeps `present.frag`'s own 5x3 digit readout, which needs no host
buffer because that path presents through the driver's swapchain rather than through SDL.

## The honest boundary (why FG does not "just work" here)

Frame Generation `generate` stays a data-blocked `RUNTIME_ERROR` in this provider slice: AMD's ML/graph
**dispatch order is not statically derivable** (see `tools/extract/frame-generation/dataflow.py`); only a live
`_vk` DLL capture closes it (`tools/capture/frame-generation/capture-diff.py --emit-order` →
`tools/extract/frame-generation/apply-capture-order.py`). This sample is the moving-graphics host such a runtime
would drive; it does NOT fake interpolated frames.

**What changed since Phase 1 (and what did not):** the provider now builds the substrate the pack's own plan
tables describe at `Prepare` — the plan's intermediate targets, one descriptor pool sized from the driver's
element costs, the 92 classic-subset `VkComputePipeline`s, and a `vkGetDescriptorEXT` census of that subset's
bindings — so `ffxQuery(GET_GPUMEMORYUSAGE)` reports those allocations instead of refusing **on devices that
enabled `VK_EXT_descriptor_buffer`**. This harness DOES now carry that transport — on a SECOND logical
device (`VkApp::fgDevice`, created over the SAME physical device) that ONLY the Frame Generation backend block
names. It is a separate device because turning the transport on in the device that builds this sample's own GRAPHICS
pipeline was measured to crash inside the driver (SIGSEGV in `libvulkan_radeon` from `setupPipeline`) — that half
is a driver-side fact and still stands, which is why the app keeps rendering on `a.device`. BOTH halves are now
MEASURED rather than expected: the window prints `window FG ffxCreateContext = 0 (0=OK)`, then on every frame
`phase1 substrate render=480x270 ... pipelines=92/92 ... usage=35261568B (aliasable 10368B)` and
`prepare substrate state: substrate built (usage reportable)`, with NO `exports none of` line anywhere; a 5-second
window run reported `host_frames=138 ... last_generate=3 prepare=3 ... delivered 1X of requested 2X` at ~27 fps.
Nothing here changed about `generate`: no descriptor write, no bind, no command recorded,
no dispatch — the refusal is the same honest one, one allocation layer deeper.

The **upscaler** provider genuinely creates a live context in-app: after attaching the `fsr4vk` backend(3) **+ instance(4)**
descriptor blocks and setting `FSR4_VK_ASSET_ROOT` to `assets/pack-unified`, `ffxCreateContext(upscale)` returns **0 (OK)**
on this device, and the `ffxQuery` helpers are answered by AMD's real math — `GET_UPSCALE_RATIO(BALANCED) = 1.70` and
`GET_RENDER_RESOLUTION(1440x810, BALANCED) -> 848x477`. The pixel upscale **DISPATCH is also driven here**, in `FsrRig`
/`fsrUpscaleFrame`, against the marshalling contract the provider itself enforces (`src/provider/provider.cpp`):

* `ffxDispatch` **records** compute work into the `commandList` the host supplies and never submits it — the host submits
  (and, in this sample, then issues the read-back barrier + `vkCmdCopyImageToBuffer` itself).
* `FfxApiResource.resource` is a **bare `VkImage` handle** (`reinterpret_cast<VkImage>(resource.resource)`), and
  `validate_resource` demands `type == TEXTURE2D`, `description.width/height ≥` the ACTIVE size, the role's claimed
  `format`, the role's `state`, and `FFX_API_RESOURCE_USAGE_UAV` whenever the state is `UNORDERED_ACCESS`.
* `inputColorVkFormat` accepts **only** `R16G16B16A16_FLOAT`, `R11G11B10_FLOAT` and `RGB9E5` — an RGBA8 colour claim
  (which the old `--present` code made) is rejected with `PARAMETER`, so that dispatch never ran; the rig's colour
  attachment is therefore FP16 and gets its own graphics pipeline.
* The depth role must **claim** `R32_FLOAT`; with `FFX_API_RESOURCE_USAGE_DEPTHTARGET` the provider binds `VK_FORMAT_D32_SFLOAT`
  + `DEPTH` aspect, which lets the app hand over its **real depth attachment** instead of a synthesized depth texture.
* Layouts follow this repo's own on-device gate `tools/descriptor-smoke.cpp`: inputs in `SHADER_READ_ONLY_OPTIMAL`,
  the storage output in `GENERAL`; `FSR4_VK_ASSET_ROOT` supplies the model sidecars and `FFX_UPSCALE_ENABLE_AUTO_EXPOSURE`
  lets the 1×1 exposure resource stay null.
The byte-identical descriptor-buffer vs descriptor-set comparison, int8+fp8 model walk and the memory-usage contract
stay covered by the green **`descriptor-smoke`** gate; FGTest proves the host-side ABI usage, not those internals.

## Gotchas found while wiring the image dispatch (each one cost a real failure mode)

1. **Colour format whitelist** — see above; a wrong claim is a `PARAMETER` refusal, silently leaving a black output.
2. **`setupPipeline` owns its shader modules**: it destroys `vs`/`fs` itself, so destroying them again in `setupFsrRig`
   produced a **double free** that surfaced as `double free or corruption (!prev)` deep inside `libvulkan_radeon`
   (found with `-fsanitize=address`, whose runtime has to be preloaded: `LD_PRELOAD=$(g++ -print-file-name=libasan.so)`).
3. **Per-frame `vkMapMemory`/`vkUnmapMemory` on a multi-MB staging buffer costs ~350 ms on this driver** — map it once
   for the session; and reading the mapping byte-by-byte is as slow, so bulk-copy into a local `std::vector` first.
4. **IEEE half exponent bias is 15, not 127**: decoding a half into a float re-biases to `e + 112`; using `e + 127`
   silently magnified everything by 2^15 and the read-back PPMs came out pure white (the pixel check — corner colours,
   distinct-colour count and frame-to-frame byte diff — is what catches that class of bug).
5. `memIndex` originally matched only `memoryTypeBits & 1` instead of bit `i`; that is fine only when memory type 0 happens
   to fit, and would hand the driver a bogus `memoryTypeIndex` otherwise.

## Files
- `main.cpp` — offscreen animated-triangle renderer + `dlopen` of both providers + the FFX call ladders, plus `FsrRig`
  /`FsrHost` (the app-owned FP16 colour + D32 depth + RG16F motion + FP16 storage images, their pipeline and the read-back staging buffer).
- `triangle.vert` / `triangle.frag` — GLSL compiled to SPIR-V by `build.sh`.
- `build.sh` — compile shaders + app, then run headless and print the per-feature status.
