#!/usr/bin/env bash
# Builds native/glhost/nbgl: the GL host. The game (libg.so under qemu) serialises every GLES 3.0 call into a shared-memory ring;
# nbgl replays it natively (EGL + GLES 3, a hidden X11 window) and returns the frames. idpass.c adds the per-draw ID pass
# (which draw call wrote each pixel + its coverage) that the labeller reads; rec.c can record the GL stream (NBGL_REC=<file.zst>).
# A prebuilt x86_64 binary ships in native/prebuilt/nbgl; a local build here is preferred when it exists (labeller/rc.py).
# Needs: gcc, libegl-dev, libgles-dev, libx11-dev, zlib1g-dev, libzstd-dev.
set -e
cd "$(dirname "$0")"
gcc -O1 -g -rdynamic -Wall -o nbgl.tmp nbgl.c -lEGL -lz -lzstd -lGLESv2 -lX11 -lm
mv -f nbgl.tmp nbgl && echo "built native/glhost/nbgl"
