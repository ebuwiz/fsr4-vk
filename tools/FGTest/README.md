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

The **upscaler** provider is driven too (`ffxCreateContext(upscale)` + `ffxQuery` ratio/render-resolution
helpers). Here `ffxCreateContext(upscale)` returns `3` because this probe attaches the *Frame Generation*
backend descriptor block; the `fsr4vk` upscaler expects its own backend+instance block (which the repo runtime
supplies). The upscaler's actual on-device run path lives in `descriptor-smoke` / the provider; this sample
deliberately does NOT blind-dispatch an image with a guessed resource layout (that would be unsafe, not a fake).

## Files
- `main.cpp` — offscreen animated-triangle renderer + `dlopen` of both providers + the FFX call ladders.
- `triangle.vert` / `triangle.frag` — GLSL compiled to SPIR-V by `build.sh`.
- `build.sh` — compile shaders + app, then run headless and print the per-feature status.
