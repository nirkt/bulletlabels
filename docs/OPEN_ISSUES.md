# Open issues (not started; each needs game-host research)

## 1. Brawlers always face the camera

In every frame of the real client running offline (this repo's game host), brawlers face forward (towards the camera) whatever they
do: attacking at 0 / 90 / 180 degrees or using a super leaves the model's facing unchanged. Seen in this repo's solo captures (Colt
at aim 0, 90 and super 45: identical pose) and noted earlier in rendclone's measurements ("the real frames show every brawler facing
forward, a quirk of the simulator"). Labels are not affected (they come from the projectiles' draws), but the frames are less
realistic and a model trained on them may learn "brawlers face down".

What is known:
* the player actions reach the game SIMULATION directly (`vendor/nbpc/instance.py` `step(actions=...)` -> game host `env.c`), not
  through the client's touch input; `POSECAP_IDLE_LOCAL=1` routes the local player's input through the client instead (wall-clock
  timing, not repeatable) - untested for facing;
* other players (bots) face forward too, so it is not only the local player's client-side prediction;
* the shipped `libg.so` has server-only code compiled out (bare `ret` stubs: charge supers 0xb475cc, gadget activation 0xb27c58,
  area apply whitelist 0xb29494; see native/loader/spawnfix.c for how such missing parts were re-implemented); the character
  rotation / facing update may be one of them or may need a field the offline battle never sets.

Where to start: find the client's character visual (the object that holds the model and its rotation) and what writes its angle
(Ghidra on libg.so v68.279: search the writes of the rotation field around the character render code, compare with what the
simulation's character holds: `state['chars']` row columns), then set it from the simulation in `native/loader/env.c` (or a hook
like spawnfix.c), rebuild with `native/loader/build.sh`, and check: the brawler turns to its aim; labels of a few skins stay
identical (`debug/compare_fix.py` style check on captures before / after).

## 2. Other modes' map layouts (Brawl Ball, Knockout, Bounty, ...)

The game host only accepts locations of the Showdown game-mode variation (`env.c`, search `div.location`: `loc_variation(loc) !=
loc_variation(g_play_loc)` -> rejected, err bit 30). Their themes work on Showdown maps (`bl.py themes`). Supporting the layouts
means letting the offline battle start with another mode's location and checking it runs (spawns, mode objects, no crash).

## 3. Ring mode labels not reviewed

`bl.py --ring` (several brawlers; by default they take turns) has not had the frame-by-frame label review solo mode had; crowded
scenes showed Ash attack hulls stretching between distant pieces. Needs a review round (`bl.py pages` sheets) and fixes.
