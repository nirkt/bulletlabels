# bulletlabels

Training data for vision models from the **real Null's Brawl client**: game frames plus exact, per-pixel labels of every attack,
super, spawned character and damage zone on screen, with its type, team, brawler and skin.

The game's own `libg.so` runs under a (patched) qemu user-mode emulator inside our game host; a GL host renders its frames and
records, for every pixel, which draw call wrote it (the ID pass). The labeller turns that into labels following a set of
hand-checked rules (`docs/LABEL_RULES.md`): every brawler and skin was reviewed frame by frame.

```
frames  ->  images/<frame>.png      1200 x 540 game frame
labels  ->  masks/<frame>.png       16-bit instance map (0 = background, i = instance i)
            labels/<frame>.json     per instance: class, team (own / enemy), type name, brawler, skin, owner, bbox, area
                                    (+ optional motion: direction, end point, end tick)
```

## Requirements

* Linux x86_64 with a desktop session (the GL host opens a hidden X11 window; on a headless box use `xvfb-run -a`).
* Python 3.10+, ~5 GB free disk (the game ~1.7 GB, captures ~0.3 GB each while they exist).
* Tested on Ubuntu 24.04.

## Setup

```bash
./setup.sh
```

It installs the system and Python packages, downloads the game APK and unpacks it to `game/`, checks it is the exact game
build this repo was made for (`native/LIBG_SHA256`), builds the patched qemu (~5 min) and runs `python3 bl.py doctor`.
Already have the APK? `APK=/path/to/game.apk ./setup.sh`. Skip the apt step: `./setup.sh --no-apt`.

## Make a dataset

```bash
python3 bl.py skins --brawler Knight                     # brawlers / skins (internal names: Knight = Ash, Gunslinger = Colt ...)
python3 bl.py run --skin KnightPinata,RuffsBrawloween datasets/ds1 --overlay
```

`run` captures one battle per skin (~30 s each) and labels it; captures are deleted afterwards (`--keep` keeps them). Options:
`--angles 0,90,180,270` (attack directions), `--motion` (direction / end point / end tick per projectile), `--overlay`
(outlined frames for a quick look), `--every N` (every N-th frame), `--seed`.

Step by step: `bl.py capture --skin X CAP`, then `bl.py label CAP DATASET`. `bl.py pages CAP OUT` makes the contact sheets
(raw | outlined) the rules were reviewed on.

### Capture modes

* **solo** (default, reviewed): the brawler alone in an open part of the map; it idles, attacks in each direction, then uses its
  super. Every label in this mode was checked by hand.
* **ring** (`--ring`, NOT reviewed yet): up to 10 brawlers around the camera's brawler. By default they take turns (one attacker at a
  time, the others stand still); `--mixed` makes everyone attack at once. Crowded scenes can have wrong labels; a review round for
  this mode is planned.

### Class schemes

The labels keep everything about each instance, so the classes can be chosen afterwards:

```bash
python3 bl.py classes datasets/ds1 --mode team                # own / enemy
python3 bl.py classes datasets/ds1 --mode brawler_team --masks
```

Modes: `all`, `team`, `kind`, `kind_team`, `brawler`, `brawler_team`, `brawler_kind_team`, `skin`, `skin_team`,
`skin_kind_team`, `type`, `type_team`. Output in `DATASET/classes/MODE/` (`classes.json`, per-instance class ids, optional
class masks). `--class-list` reuses the ids of an earlier dataset; `--kinds attack,super` keeps only some kinds.

## Found a wrong label?

See `debug/AGENTS.md`: it explains how the labeller works and how to fix a label rule (written so an AI coding agent can follow
it). Tools: `debug/inspect_frames.py` (what the labeller does on some frames) and `debug/compare_fix.py` (try a fix before
installing it).

## Layout

| folder | what |
|---|---|
| `bl.py` | the command line |
| `labeller/` | the labeller: `idlabels.py` (core), `fixes/*.py` (per-type rules, run in name order), `export.py`, `classes.py` |
| `framegen/` | the frame generator: `posecap.py` drives one battle through the game host |
| `vendor/` | small helpers: the game launcher (`nbpc/`), state decoding (`state2.py`), battle reader (`dumps.py`) |
| `native/` | the game host (`loader/`), the GL host with the ID pass (`glhost/`), qemu patches, boot payloads, prebuilt binaries |
| `data/` | tables the labeller reads, the skin list, the label decisions |
| `docs/` | `LABEL_RULES.md` |
| `debug/` | tools and the guide for fixing labels |
| `game/` | the unpacked APK (made by `setup.sh`, not in git) |

## Notes

* The game files are not part of this repo; `setup.sh` downloads them.
* Labels are made for the game build in `native/LIBG_SHA256`; another build will not run (the game host patches fixed addresses).
* Frames come from the real client, so a capture is not bit-repeatable (thread timing): two captures of the same skin differ a
  little. The labels of a given capture are deterministic.
