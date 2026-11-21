#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
# tools/capture/frame-generation/build-host.sh - compile the capture host for the WINDOWS image that runs under
# Wine, so AMD's OWN provider (a Windows-only binary) can be driven and recorded.
#
# MEASURED on this host (CachyOS, wine 11.22), in this order:
#   * `winegcc` EXISTS and compiles + runs a Windows program (hello-world proved it, and Wine's own vulkan shim
#     reached RADV, printing the "radv is not a conformant Vulkan implementation" banner from inside the wine run);
#   * BUT winegcc builds a LINUX/ELF image in Wine's native mode, and Wine's Windows-side import libs
#     (/usr/lib/wine/x86_64-windows/libvulkan-1.a) CANNOT be linked into an ELF object - the linker says
#     "unrecognized relocation format pe-x86-64 against elf64-x86-64";
#   * and Arch's `wine` package does NOT ship `wine/dlfcn.h`, so the native-mode route cannot load a Windows
#     .dll either (measured: /usr/include/wine has no dlfcn.h).
# So a REAL Windows build is required, i.e. a cross-compiler: on this distro those are the OFFICIAL packages
# extra/mingw-w64-{binutils,crt,headers,gcc} (or cachyos/llvm-mingw, a self-contained LLVM one), and installing
# them needs root, which this session cannot obtain non-interactively (measured: `sudo -n true` fails). This script
# therefore uses a cross-compiler when present and otherwise stops with the exact command to run - it never
# pretends a host was built.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../../.." && pwd)"
out="${1:-$here/build}"
mkdir -p "$out"

cxx=""
# A cross-toolchain may be installed WITHOUT root: this repo supports a local, gitignored copy under
# build/tools/mingw (fetched with `paru -Sp cachyos/llvm-mingw` + `tar --zstd -xf`, 96.7 MiB download, measured
# working here), overridable with FSR4_MINGW_BIN. System installs are used when they exist.
for candidate in x86_64-w64-mingw32-clang++ x86_64-w64-mingw32-g++ clang++-mingw-w64-x86_64; do
    if command -v "$candidate" >/dev/null 2>&1; then cxx="$candidate"; break; fi
done
if [[ -z "$cxx" && -x "${FSR4_MINGW_BIN:-}/x86_64-w64-mingw32-clang++" ]]; then
    cxx="${FSR4_MINGW_BIN}/x86_64-w64-mingw32-clang++"
fi
if [[ -z "$cxx" ]]; then
    local_bin="$repo/build/tools/mingw/opt/llvm-mingw/bin"
    [[ -x "$local_bin/x86_64-w64-mingw32-clang++" ]] && cxx="$local_bin/x86_64-w64-mingw32-clang++"
fi
if [[ -z "$cxx" ]]; then
    local_bin="$repo/build/tools/mingw/opt/llvm-mingw/bin"
    [[ -x "$local_bin/x86_64-w64-mingw32-clang++" ]] && cxx="$local_bin/x86_64-w64-mingw32-clang++"
fi
if [[ -z "$cxx" ]]; then
    echo "SKIP build-host: no Windows cross-compiler (x86_64-w64-mingw32-{clang++,g++}) on this host." >&2
    echo "  measured: winegcc compiles ELF images that cannot link Wine's Windows import libs, and Arch's wine" >&2
    echo "  package ships no wine/dlfcn.h - so the native-mode route is a dead end, see the header of this script." >&2
    echo "  to unblock WITHOUT root (installs under build/tools, gitignored):" >&2
    echo "    mkdir -p build/tools/mingw /tmp/mwc && paru -Sp cachyos/llvm-mingw | xargs -I{} curl -fL -o /tmp/mwc/lm.zst {}" >&2
    echo "    tar --zstd -xf /tmp/mwc/lm.zst -C build/tools/mingw" >&2
    echo "  or system-wide: sudo pacman -S --needed cachyos/llvm-mingw  (or extra/mingw-w64-{binutils,crt,headers,gcc})" >&2
    exit 77
fi
echo "using $cxx"
# Wine's Windows-side Vulkan (winevulkan) forwards to the SAME driver the LINUX harness uses, so the host links
# vulkan-1 and gets real RADV underneath. Import libs come from Wine's own package directory.
wine_dir="/usr/lib/wine/x86_64-windows"
[[ -d "$wine_dir" ]] || { echo "SKIP build-host: no $wine_dir (wine not installed)"; exit 77; }
mingw_lib="$(dirname "$(dirname "$cxx")")/../x86_64-w64-mingw32/lib"
[[ -d "$mingw_lib" ]] || mingw_lib=""
"$cxx" -std=c++20 -Wall -O1 -DFSR4_WINDOWS_BUILD -o "$out/fgcap-host.exe" "$here/host.cpp" \
    -I"$repo/third_party/fidelityfx-api/api/include" \
    -I"$repo/third_party/fidelityfx-api/framegeneration/include" \
    -I"$repo/third_party/Vulkan-Headers/include" \
    ${mingw_lib:+-L"$mingw_lib"} -L"$wine_dir" -l vulkan-1 -l msvcrt -l c++abi
printf 'built %s -> run with: wine %s <staged dll> (FSR4_VK_CAPTURE_DIR=<dir>)\n' \
    "$out/fgcap-host.exe" "$out/fgcap-host.exe"

# THE HOST COMPILES FOR WINDOWS, and a real 521 KB Windows image linked on this host (measured). What is NOT yet
# done is the CRT PAIRING, probed three ways here, each result measured rather than assumed:
#   * Wine's OWN `libmsvcrt.a`/`msvcrt.dll` (a complete mingw-CRT implementation) supply the C stdio entry points
#     the host calls, but NOT the Microsoft-CRT startup internals llvm-mingw's default `crt2.o` references:
#     `__p___initenv`, `_crt_atexit`, `_set_invalid_parameter_handler`, `__p__fmode`, `__p__commode`,
#     `_configthreadlocale`;
#   * llvm-mingw's own `libmsvcrt.a` DOES resolve those, but imports them from the SPLIT MODERN CRT DLLS
#     (`api-ms-win-crt-stdio-l1-1-0.dll` and friends) that Wine does not provide - so that image would not LOAD
#     under Wine either;
#   * Wine DOES ship the unified CRT (`ucrtbase.dll` + `libucrtbase.a`) and llvm-mingw's UCRT startup objects are
#     `crt1u.o`/`crt2u.o`: passing BOTH yields duplicate `mainCRTStartup`/`atexit`, `crt2u.o` ALONE still references
#     Microsoft-CRT internals (`__wgetmainargs`, `_amsg_exit`, `_XcptFilter`, `__p___winitenv`, `__set_app_type`),
#     and Wine's `libucrtbase.a` drags a `crt_wwinmain.o` that wants a `wWinMain` entry this host does not define.
# So the last step is specific: build the host with its OWN ENTRY (`-nostartfiles` + `-Wl,--entry=...`, plus a
# `wWinMain` alias if that object is dragged), take the C runtime from Wine (`-l msvcrt -l ucrtbase`) and the C++
# runtime from llvm-mingw, and keep `hostResolver` as the only Vulkan surface the DLL can reach. Until it links, NO
# AMD dispatch ORDER exists in this repo: `order_source` stays `grouped-scaffold` and NOTHING is guessed about it.
