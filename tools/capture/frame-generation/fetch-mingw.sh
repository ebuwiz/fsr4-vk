#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
# tools/capture/frame-generation/fetch-mingw.sh - install a Windows cross-toolchain INTO build/tools WITHOUT root,
# which is what makes the ORDER capture possible on a host where `sudo` needs a password. MEASURED here (CachyOS,
# wine 11.22): `paru -Sp cachyos/llvm-mingw` prints the package URL with no root at all, the download is 96.69 MiB,
# `tar --zstd -xf` extracts a relocatable toolchain to build/tools/mingw/opt/llvm-mingw, and
# `x86_64-w64-mingw32-clang++` then COMPILES tools/capture/frame-generation/host.cpp into a real Windows image.
# The system-wide alternative (needs root) is `sudo pacman -S --needed cachyos/llvm-mingw`, or the split
# extra/mingw-w64-{binutils,crt,headers,gcc} set (1.17 GiB installed).
set -uo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
dest="$repo/build/tools/mingw"
cache="${FSR4_MINGW_CACHE:-/tmp/fsr4-mingw-cache}"
mkdir -p "$cache" "$dest"

if [[ -x "$dest/opt/llvm-mingw/bin/x86_64-w64-mingw32-clang++" ]]; then
    printf 'already installed: %s\\n' "$dest/opt/llvm-mingw/bin/x86_64-w64-mingw32-clang++"
    exit 0
fi

url="$(paru -Sp cachyos/llvm-mingw 2>/dev/null | head -1)"
if [[ -z "$url" ]]; then
    url="https://mirror.cachyos.org/repo/cachyos/x86_64/cachyos/llvm-mingw-23.1.0+20260826-1-x86_64.pkg.tar.zst"
    echo "paru could not resolve the package; falling back to the pinned mirror URL: $url" >&2
fi
pkg="$cache/llvm-mingw.pkg.tar.zst"
if [[ ! -s "$pkg" ]]; then
    curl -fL --retry 3 -o "$pkg" "$url" || { echo 'download failed'; exit 1; }
fi
tar --zstd -xf "$pkg" -C "$dest" || { echo 'extract failed'; exit 1; }

if [[ -x "$dest/opt/llvm-mingw/bin/x86_64-w64-mingw32-clang++" ]]; then
    echo "installed $dest/opt/llvm-mingw (build-host.sh finds it automatically)"
    exit 0
fi
echo 'extracted, but x86_64-w64-mingw32-clang++ is not where build-host.sh looks' >&2
find "$dest" -maxdepth 4 -name 'x86_64-w64-mingw32-clang++' | head -4 >&2
exit 1
