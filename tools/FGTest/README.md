# tools/FGTest — headless "does it run?" probe for an animated-triangle Vulkan host on FSR4

A small, self-contained, **offscreen** (no window, no GLM, no Sascha framework) Vulkan 1.3 sample that
animates a moving/rotating colored triangle and drives BOTH FSR4 providers through their REAL, documented
FFX modular-API calls. It answers "does it run?" **per feature**, and never fabricates a result.

## Build + run (headless, on a real driver)

```sh
bash tools/FGTest/build.sh
```

Requires the repo's built providers (`build/linux/amd_fidelityfx_{upscaler,framegeneration}_vk.so`),
`glslc`, `g++` (C++20), libvulkan, and a Vulkan device. Shaders (triangle.vert/frag) are compiled to SPIR‑V
by `build.sh`. PPM frames land in `tools/FGTest/build/`.

## Visible window (Wayland/X11) — optional

FGTest is headless by default. To actually WATCH the moving triangle in a window,
add `--window` (optionally a duration). It links GLFW (`-lglfw`) and presents
the SAME renderer through core Vulkan WSI (GLFW window surface → swapchain →
present), logging one line per presented base frame:

```sh
# after `bash tools/FGTest/build.sh` has produced the .so's + triangle.*.spv:
FSR4_VK_ASSET_ROOT=assets/pack-unified \
DISPLAY=:0 XDG_RUNTIME_DIR=/run/user/1000 \
build/FGTest "$REPO/build/linux/amd_fidelityfx_upscaler_vk.so" \
         "$REPO/build/linux/amd_fidelityfx_framegeneration_vk.so" \
         build/triangle.vert.spv build/triangle.frag.spv --window=10
```

Each frame prints `frame N: present=base(WxH) t=..s interpolated=0 generated=0 [FG: REFUSED]`.
`interpolated`/`generated` stay 0 because AMD's provider refuses FG `generate` here (it
will not fake frames) and the FSR pixel-dispatch is not hand-run; the window shows the
NATIVE animated host at the compositor's frame rate. If the machine's Vulkan loader does
not expose a window surface (e.g. headless CI) it prints `NO_SURFACE` and exits cleanly
rather than fabricating anything.

## What it demonstrates (measured on AMD Radeon RX 9070 XT / Mesa RADV GFX1201)

- **Moving triangle host: RUNS.** 6/6 offscreen frames render non‑black and differ (rotation + translation)
  → the graphics host Frame Generation would plug into actually runs (verified by distinct frame hashes).
- **Frame Generation provider: the full documented host-integration ladder runs, honestly:**
  - `ffxCreateContext(frame-generation) = 0` (OK) — the device meets the FG execution contract
    (cooperative matrix, 8/16-bit storage, runtime descriptors, etc.).
  - `ffxQuery(GET_VERSIONS)` advertises the real classic id `0xF600000001000002`.
  - `ffxQuery(GET_GPUMEMORYUSAGE) = 3` (RUNTIME_ERROR) — not reportable until the runtime exists.
  - `ffxConfigure(host contract) = 0` (OK) — swapchain/present/framegen callbacks accepted.
  - `ffxDispatch(Prepare) = 3` — **the provider's own message: "prepare accepted (order + camera valid) but
    dispatch pending"** → args + camera basis validate; only the interpolation/ML/present runtime is refused.
  - `ffxDispatch(Generate 2) = 3` — **refused: "refusing to fake frame generation."**

## The honest boundary (why FG does not "just work" here)

Frame Generation `generate` stays a data-blocked `RUNTIME_ERROR` in this provider slice: AMD's ML/graph
**dispatch order is not statically derivable** (see `tools/extract/frame-generation/dataflow.py`); only a live
`_vk` DLL capture closes it (`tools/capture/frame-generation/capture-diff.py --emit-order` →
`tools/extract/frame-generation/apply-capture-order.py`). This sample is the moving-graphics host such a runtime
would drive; it does NOT fake interpolated frames.

The **upscaler** provider genuinely creates a live context in-app: after attaching the `fsr4vk` backend(3) **+ instance(4)** descriptor blocks and setting `FSR4_VK_ASSET_ROOT` to `assets/pack-unified`, `ffxCreateContext(upscale)` returns **0 (OK)** on this device, and the `ffxQuery` helpers are answered by AMD's real math — `GET_UPSCALE_RATIO(BALANCED) = 1.70` and `GET_RENDER_RESOLUTION(1440x810, BALANCED) -> 848x477`. The pixel upscale **DISPATCH** (feeding `VkImage` resources back through `ffxDispatchDescUpscale`) is deliberately NOT hand-rolled here — that is the runtime's `VkImageInfo` resource marshalling, and the upscaler's real on-GPU render → submit → read-back (descriptor-buffer vs descriptor-set byte-identical, int8 + fp8) is covered by the green **`descriptor-smoke`** on-device gate.

## Files
- `main.cpp` — offscreen animated-triangle renderer + `dlopen` of both providers + the FFX call ladders.
- `triangle.vert` / `triangle.frag` — GLSL compiled to SPIR-V by `build.sh`.
- `build.sh` — compile shaders + app, then run headless and print the per-feature status.
