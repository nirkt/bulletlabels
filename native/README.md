# Native parts

| part | what | binary used | rebuild |
|---|---|---|---|
| `loader/` | the game host: loads the Android `libg.so` (aarch64), fakes the Android / JNI side, runs the battle offline, streams the GL calls to the GL host, answers the Python side over shared memory | `prebuilt/nbhost_spawnfix` (static aarch64) | `loader/build.sh` (needs `gcc-aarch64-linux-gnu`) |
| `glhost/` | the GL host `nbgl`: replays the game's GLES 3 stream with EGL (hidden X11 window), returns frames, writes the ID pass (`idpass.c`) | `glhost/nbgl` if built, else `prebuilt/nbgl` (x86_64) | `glhost/build.sh` |
| `qemu/` | QEMU 8.2.2 user mode + 3 speed patches (same game state, ~2x faster) | `qemu/bin/qemu-aarch64-nb` if built, else the system `qemu-aarch64` | `qemu/build.sh` (downloads QEMU, ~5 min) |
| `payloads/` | two small server messages the offline game boots from | - | - |
| `LIBG_SHA256` | the game build everything here is made for | | |

**Game host version.** The labels were made and reviewed with `prebuilt/nbhost_spawnfix`. Its sources are an earlier state of
`loader/`: the later `spawnfix.c` (here) also re-creates chain bullets, wall tiles and items, which were not reviewed. The frame
generator therefore always uses the prebuilt binary (`labeller/rc.py`). `loader/build.sh` builds `nbhost_spawnfix2` from these
sources; `NBHOST_BIN=native/loader/nbhost_spawnfix2 NB_SPAWNFIX_CHAIN=0 NB_SPAWNFIX_WALLS=0` comes close to the prebuilt one but
is not verified to be identical.

Every game process runs as `unshare -rn qemu-aarch64 nbhost ...` (no network). Logs: `<capture>/instances/`.
