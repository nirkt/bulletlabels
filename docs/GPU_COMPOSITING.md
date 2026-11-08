# GPU compositing (next step, in progress)

**Goal.** Frames with many attacks at once, each attack labelled from its own clean single-brawler capture: render one skin's attack,
label it, render the next skin's attack *over* that frame and label only it, and so on. The labels never mix (the labeller only ever
sees one brawler's attack), and the GPU does the blending exactly as the game would (the game's own shaders, textures and blend state).

Status (2026-11-08): steps 1-4 below are built and verified; step 5 (trail / glow draws) is the first thing to do next.
A first composite: `docs/images/composite_first.jpg` (left: Shelly's frame, middle: Colt's frame, right: Colt's super bullets
drawn onto Shelly's frame by the GPU; only the bullet cores, see step 5).

## How it works

```
capture A (solo, GL stream recorded)        capture B (solo, GL stream recorded)
   frame FA  --replay-->  A.ppm (2400x1080)      labeller on B's frame FB -> labels + P['draws'] (per object: its draw ids)
                                                  |
   replay B's recording up to frame FB; just before FB's first on-screen draw, A.ppm is blitted into the framebuffer;
   then only B's attack draws (by ID-pass draw id) are executed  ->  composite.ppm
labels of the composite = A's labels + B's labels on top (both are in screen space; B's pixels win where they overlap)
```

## Verified facts (each measured)

1. **Recording.** The GL host records the full GL stream when `NBGL_REC` is set (`%s` = renderer name):
   `NBGL_REC=/path/rec_%s.nbrec.zst python3 bl.py capture --skin GunSlingerDefault --angles 0,90 CAP` (about 30-40 MB per solo
   capture; works together with the ID pass).
2. **Replay is exact.** `native/compose/replay REC --frames F --out DIR --half` writes frame F; its half-resolution image equals the
   captured frame of the same swap id **bit for bit** (max difference 0, solo Colt capture, frame 2193).
3. **Draw ids line up.** The ID pass numbers the on-screen draws of a frame 1..N in order (`id/<renderer>/f<swap>_draws.jsonl`);
   the replay executes the same on-screen draws in the same order (123 = 123 draws, all vertex / fragment shader hashes equal in
   order). So ID-pass draw `i` = the replay's `i`-th draw into framebuffer 0 of that frame.
4. **Composite mode** (`native/compose/replay.c`, options `--comp G --bg A.ppm --keep ids|all|none`). Self-checks: `--keep all` over a
   black background = the normal replay exactly; `--keep none` = the background exactly (100 % of pixels). Frame convention: the draws
   of image G run while the replay's frame counter is G - 1 (`frame + 1 == G`), like its `--steps` option.
5. **Which draws are an object's.** `labeller/idlabels.py` now also returns `P['draws']`: object id -> the ID-pass draw ids that drew
   that projectile type's own sprites near it (sprite code C > 0 inside its search region). Labels are unchanged by this (verified
   identical on a 75-frame capture).

Tried and dropped: rendclone's executor (`gen/nbx` on a `replay --draws` dump + `dl.py frame_from_dump`) renders these recordings
wrongly (map tiles and textures missing); the replay-based composite above avoids it.

## Next steps (in order)

5. **Trails and glows.** `P['draws']` gives Colt's super bullets their core draws (105, 109, 111, 113, 115 on frame 2249) but not their
   blue trails, so the composite shows thin cores. Find the trail / glow draws of each object: candidates are the draws right next to
   the core draw in the frame (104/106 ...), draws whose sprite code is 2/3/4 (`idlabels.sprite_pixels`) anywhere in the object's
   trail region (`TRAIL_NEAR`), and draws using the same effect texture (`th` in draws.jsonl). Check visually: B's own frame with only
   its kept draws over a black background (`--bg black.ppm`) should show the whole attack and nothing else (no brawler, no map, no
   hit explosion: those are not labelled either).
6. **A tool**: `bl.py composite --base CAP_A --add CAP_B[,CAP_C...] OUT` (or a module in `labeller/`): for each output frame pick a
   base frame and one frame per added capture (same map / theme and solo spot; captures of different skins), label each, collect the
   kept draw ids, replay base and composites, write the dataset files exactly as `labeller/export.py` does (images / masks / labels
   JSON; instances of B over A, with B's `team`, `owner_*` fields).
   Several layers: composite B onto A, then C onto the result (the result is the next `--bg`), or extend `--keep` to two recordings.
7. **Placement / variety.** All solo captures put the shooter on the same spot, so stacked attacks start at the same point. Options:
   different attack angles per layer (`--angles`), the enemy shooter (`POSECAP_SHOOTER=1`, red-tinted shots from another player next
   to the camera's brawler), or a framegen option for the shooter's position. Mind the camera: it drifts by up to ~20 px between
   frames; labels stay correct because each layer's labels come from its own frame in screen space.
8. **Depth.** B's draws are tested against B's own depth buffer, which has only what B's kept draws wrote, so B's bullets are drawn over
   A's walls. Fine for most frames; for exactness also replay B's wall / map draws depth-only (colour mask off), or skip frames where
   B's attack overlaps A's walls.
9. **HUD.** A's HUD (joystick, buttons) is in the background, B's bullets are drawn over it; keep B's attack away from the HUD corners
   or re-draw A's HUD draws last.
10. **Verify** like the rest of the repo: B's labels over a composite must match B's pixels exactly (B's region of the composite =
   B's own frame where B's draws are opaque), and the page review (`bl.py pages`-style sheets) of a sample by a person.

## Files

* `native/compose/replay.c` (+ `dump.c`, `build.sh`): the replay with the composite mode (from rendclone's `gen/replay.c`).
* `labeller/idlabels.py`: `P['draws']` (search for `obj_draws`).
* Scratch scripts used for the checks above are not in the repo; the commands in this file reproduce them.
