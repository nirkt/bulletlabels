"""Repo paths and defaults. Every module imports this first (`import rc`).

Layout (relative to the repo root):
  game/                 the unpacked game APK (setup.sh): game/lib/arm64-v8a/libg.so, game/assets/...
  labeller/             the labeller (this folder) + labeller/fixes/ (per-type rules, run in name order)
  framegen/             the frame generator (posecap.py drives the game)
  vendor/               small helper modules (game launcher nbpc/, state decoder state2.py, battle reader dumps.py)
  native/               the game host (loader), the GL host with the ID pass (glhost), qemu patches, boot payloads
  data/                 tables the labeller reads (decoded effect emitters, clip kinds, ...)

Environment overrides: BL_GAME_DIR (the unpacked APK), NBHOST_BIN, NBGL_BIN, NBHOST_QEMU (path, or 'system')."""
import os, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GAME = os.path.abspath(os.environ.get('BL_GAME_DIR') or os.path.join(REPO, 'game'))
ASSETS = os.path.join(GAME, 'assets')
LIBDIR = os.path.join(GAME, 'lib', 'arm64-v8a')
LIBG = os.path.join(LIBDIR, 'libg.so')
DATA = os.path.join(REPO, 'data')
NATIVE = os.path.join(REPO, 'native')
VENDOR = os.path.join(REPO, 'vendor')


def _first(*paths):
    for p in paths:
        if p and os.path.exists(p): return p
    return paths[-1]


# the game host / GL host: a local build (native/build.sh) of the GL host wins over the shipped prebuilt binary.
# The game host is always the shipped prebuilt nbhost_spawnfix (the build every label rule was checked on; see native/README.md).
NBHOST_BIN = os.path.join(NATIVE, 'prebuilt', 'nbhost_spawnfix')
NBGL_BIN = _first(os.path.join(NATIVE, 'glhost', 'nbgl'), os.path.join(NATIVE, 'prebuilt', 'nbgl'))
QEMU_NB = os.path.join(NATIVE, 'qemu', 'bin', 'qemu-aarch64-nb')
PAYLOADS = os.path.join(NATIVE, 'payloads')

os.environ.setdefault('NBHOST_APK', GAME)
os.environ.setdefault('NBHOST_BIN', NBHOST_BIN)
os.environ.setdefault('NBGL_BIN', NBGL_BIN)
os.environ.setdefault('NB_PIX_CAP', '10368000')        # frame slot of the game host: 2400 x 1080 x 4

for _p in (VENDOR, os.path.join(REPO, 'labeller'), os.path.join(REPO, 'framegen')):
    if _p not in sys.path: sys.path.append(_p)
