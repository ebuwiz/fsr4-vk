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

Get `Samples/Upscalers/FidelityFX_FSR/dx12/x64/Release/amd_fidelityfx_upscaler_dx12.dll` from
[AMD's FidelityFX SDK 2.4.0](https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK/releases/tag/v2.4.0).
The tools require 4.1.2.4555 and verify the DLL hash automatically.
A third-party mirror works too, because the hash is checked either way:
https://www.dllme.com/dll/files/amd_fidelityfx_upscaler_dx12/c9b73b38f9f2744c7096de53429db6eb

```sh
git submodule update --init --recursive
python3 tools/setup.py --dll /path/to/amd_fidelityfx_upscaler_dx12.dll
```

Setup extracts the models, translates the shaders and applies the patches under
`assets/`. These generated assets are not included in the repository. Later runs
can omit `--dll` to reuse the extracted files. Required shader tools are built
automatically from the pinned submodules into `build/tools/`.

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
| `FSR4_VK_MODEL` | `auto`; `int8` or `1`; `fp8` or `2` | Which model family serves the frame. `auto` walks the ladder INT8 → FP8: INT8 stays the first rung, so nothing that runs INT8 today changes, and FP8 is reached only where the driver *refuses* INT8's own SPIR-V capabilities — which is what a low-entry AMD APU does. A device that refuses both is refused, with both blocker lists in the reason. An explicit request is strict: it is refused naming its own blockers, never swapped for the other family behind the caller's back. A version id that names a model (this provider's own INT8/FP8 ids, in the table below) outranks this variable; the reference DLL's own id names no model, so this variable still decides there; and an id no provider advertises is rejected outright. |
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

## Licensing

Code written for this project is licensed under [GPL-3.0-only](LICENSE).
Third-party code and assets retain their respective licences. See
[LICENSES/](LICENSES/) for the licence texts and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for attribution and details.
