#!/usr/bin/env bash
# Builds native/qemu/bin/qemu-aarch64-nb: QEMU 8.2.2 (downloaded from download.qemu.org, checksum verified) + patches/*.patch,
# aarch64-linux-user only, nothing installed system-wide. ~5 min, ~1.5 GB temporary disk.
# The patches make the emulated game ~2x faster with the same game state bit for bit; they are switched on at run time by
# vendor/nbpc/instance.py (QEMU_NB_XPAGE=1 QEMU_NB_INLINE=1 QEMU_CPU=max,pauth-impdef=on):
#   0001 jump cache 16 bits + AArch64-friendly hash      0002 cross-page goto_tb chaining in user mode (QEMU_NB_XPAGE)
#   0003 inline jump-cache probe for indirect branches (QEMU_NB_INLINE) + tcg_gen_goto_ptr
# Without this build the frame generator falls back to the system qemu-aarch64 (apt install qemu-user): same frames, ~2x slower.
# Needs: gcc, ninja, meson, pkg-config, python3, libglib2.0-dev, flex, bison, curl, xz.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"
VER=8.2.2; SHA=847346c1b82c1a54b2c38f6edbd85549edeb17430b7d4d3da12620e2962bc4f3
TAR="src/qemu-$VER.tar.xz"
mkdir -p src bin
if [ ! -f "$TAR" ]; then
  echo "downloading QEMU $VER ..."; curl -fL --retry 3 -o "$TAR.part" "https://download.qemu.org/qemu-$VER.tar.xz" && mv "$TAR.part" "$TAR"
fi
echo "$SHA  $TAR" | sha256sum -c - >/dev/null || { echo "checksum mismatch for $TAR (delete it and retry)"; exit 1; }
rm -rf nb-src nb-build && mkdir -p nb-src nb-build
tar xf "$TAR" -C nb-src --strip-components=1
(cd nb-src && for p in "$HERE"/patches/*.patch; do patch -p1 -s < "$p"; done)
cd nb-build
../nb-src/configure --target-list=aarch64-linux-user --disable-system --enable-linux-user --disable-docs --disable-tools \
  --disable-werror --disable-qom-cast-debug --disable-plugins --enable-trace-backends=nop --enable-lto > configure.log 2>&1 || { tail -30 configure.log; exit 1; }
ninja qemu-aarch64 > build.log 2>&1 || { tail -40 build.log; exit 1; }
cp qemu-aarch64 "$HERE/bin/qemu-aarch64-nb.tmp" && mv -f "$HERE/bin/qemu-aarch64-nb.tmp" "$HERE/bin/qemu-aarch64-nb"
cd "$HERE" && rm -rf nb-src nb-build
echo "built native/qemu/bin/qemu-aarch64-nb"
