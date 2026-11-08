#!/bin/sh
# Builds native/loader/nbhost_spawnfix2 from these sources: the game host (loads the Android libg.so, fakes the Android / JNI side,
# runs the battle offline, sends the GL stream to nbgl). spawnfix.c re-creates the child objects the shipped client never creates
# (bullet explosions, chains, wall tiles, items).
# NOTE: the labels in this repo were made and checked with native/prebuilt/nbhost_spawnfix, an earlier build of these sources whose
# spawnfix.c had only the bullet-explosion part (the later version's sources are the ones here). The frame generator always uses the
# prebuilt binary; with this build use NBHOST_BIN=native/loader/nbhost_spawnfix2 NB_SPAWNFIX_CHAIN=0 NB_SPAWNFIX_WALLS=0 to get
# close to it (not verified to be identical).
# Needs: gcc-aarch64-linux-gnu (static aarch64 binary; runs under qemu-aarch64).
set -e
cd "$(dirname "$0")"
aarch64-linux-gnu-gcc -O2 -g -ffp-contract=off -fno-omit-frame-pointer -static -D_GNU_SOURCE -Wall -Wno-unused-function -Wno-incompatible-pointer-types \
  -Wno-int-conversion -fno-strict-aliasing -Iinc -o nbhost_spawnfix2.tmp main.c elfload.c bionic_shim.c android_shim.c jni_fake.c boot.c hook.c offline.c battle_fix.c env.c prof.c encode.c heapcheck.c bar_fix.c gl_client.c gl_client_gen.c spawnfix.c trace.c thunk.S -lm -lpthread
mv -f nbhost_spawnfix2.tmp nbhost_spawnfix2 && echo "built native/loader/nbhost_spawnfix2"
