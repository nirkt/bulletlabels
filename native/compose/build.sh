#!/usr/bin/env bash
# Builds native/compose/replay: re-executes a GL stream recorded by the GL host (NBGL_REC=<file.zst>) on a headless GLES 3.2
# context, writes chosen frames (bit-identical to the captured frames), and the COMPOSITE mode (docs/GPU_COMPOSITING.md):
#   replay REC.zst --frames G --out DIR [--half] --comp G --bg OTHER_FRAME.ppm --keep 105,109,...|all|none
# Based on rendclone's gen/replay.c (+ dump.c: --draws per-draw dumps). Needs: gcc, libegl-dev, libgles-dev, libzstd-dev, zlib1g-dev.
set -e
cd "$(dirname "$0")"
gcc -O2 -g -w -o replay replay.c -lEGL -lGLESv2 -lzstd -lz -lm
echo "built native/compose/replay"
