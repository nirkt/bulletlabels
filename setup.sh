#!/usr/bin/env bash
# One-time setup on Linux x86_64 (tested on Ubuntu 24.04):
#   1. system packages (asks for sudo), 2. Python packages, 3. the game APK -> game/, 4. the patched qemu (~5 min), 5. a check.
#   ./setup.sh                 everything
#   ./setup.sh --no-apt        skip the system packages (install them yourself: see APT below)
#   APK=/path/to/game.apk ./setup.sh     use an APK you already have instead of downloading it
set -euo pipefail
cd "$(dirname "$0")"
REPO=$(pwd)

# ---- the game: the APK whose libg.so matches native/LIBG_SHA256 (the game host's patches are made for exactly that build)
APK_URL="https://pub-3e0f25080eb949a08641fd278fad96d1.r2.dev/nb_68.279_release_7c5c5152.apk"

APT="python3 python3-pip python3-venv unzip curl xz-utils util-linux gcc ninja-build meson pkg-config flex bison libglib2.0-dev \
libegl-dev libgles-dev libx11-dev zlib1g-dev libzstd-dev libegl1 libgles2 qemu-user"
if [ "${1:-}" != "--no-apt" ] && command -v apt-get >/dev/null; then
  echo "== system packages"; sudo apt-get update -qq && sudo apt-get install -y -qq $APT
fi

echo "== python packages"
python3 -m pip install --user -q -r requirements.txt 2>/dev/null || python3 -m pip install --user -q --break-system-packages -r requirements.txt

echo "== game files"
WANT=$(cut -d' ' -f1 native/LIBG_SHA256)
if [ ! -f game/lib/arm64-v8a/libg.so ]; then
  APKF=${APK:-game.apk}
  if [ ! -f "$APKF" ]; then
    [ -n "$APK_URL" ] || { echo "no APK: set APK_URL in setup.sh or run APK=/path/to/game.apk ./setup.sh"; exit 1; }
    echo "downloading the APK ..."; curl -fL --retry 3 -o game.apk.part "$APK_URL" && mv game.apk.part game.apk; APKF=game.apk
  fi
  rm -rf game && mkdir -p game && unzip -q -o "$APKF" 'lib/arm64-v8a/*' 'assets/*' -d game
fi
HAVE=$(sha256sum game/lib/arm64-v8a/libg.so | cut -d' ' -f1)
[ "$HAVE" = "$WANT" ] || { echo "WRONG GAME VERSION: libg.so sha256 $HAVE, this repo needs $WANT"; exit 1; }
echo "game ok ($REPO/game)"

echo "== patched qemu (native/qemu/build.sh, ~5 min; the system qemu-aarch64 works too, ~2x slower)"
[ -x native/qemu/bin/qemu-aarch64-nb ] || native/qemu/build.sh || echo "qemu build failed: the system qemu-aarch64 will be used"

echo "== GL host: the prebuilt native/prebuilt/nbgl is used; rebuild with native/glhost/build.sh if it does not start"
python3 bl.py doctor
