# Fixing a wrong label (guide for people and AI coding agents)

Read this whole file before changing anything. The labels of every brawler and skin were reviewed by hand (`docs/LABEL_RULES.md`);
a careless change to shared code silently breaks hundreds of approved labels. The safe way is almost always: **add a small fix file
for the one projectile type that is wrong, and prove that nothing else changes.**

## 1. How a label is made

```
framegen/posecap.py  ->  capture folder               one battle of the real client
    chunk0000.json         per frame: tick, GL swap id, camera matrix ('mat'), phase
    chunk0000.states.npz   per frame: the game state (players, characters, projectiles, items)
    frames.npz             the saved frames: 'swap' ids + 'img' (N, 540, 1200, 3) uint8
    id/<renderer>/f<swap>_id.z, _cv.z, _draws.jsonl     the ID pass: per pixel the draw call that wrote it + its coverage (0-255),
                                                        and per draw call the sprite / clip / program it drew
    script.json            per tick: every player's action; the capture's brawlers / skins ('info')

labeller/track.py        follows every projectile over the frames (also ENDED ones while their effect is visible)
labeller/bulletlabels.py frame_objects(tracker): per projectile its type t, own (team of the camera's brawler), anchor, predicted
                         sprite polygons (where the client SHOULD draw it), radius, ...
labeller/idlabels.py     labels(objs, cam, R, state, ended, img) -> P
```

`P` (one frame): `ids` (540 x 1200 int: index into `lids`, -1 = none), `share` (coverage of the owner, a pixel is labelled when
>= 0.5), `lids` (`(object id, -1, part)`, part = body / ground / spawn / area), `meta` (`{object id: kind, brawler, skin, name}`),
`motion` (with `IDLABELS_MOTION_*`), `zones`. `labeller/export.py` turns it into the dataset files.

Inside `idlabels.labels`, per projectile:
1. every draw call of the frame is given to the object whose predicted sprites it overlaps most;
2. the object's drawn pixels are classified by sprite code `C`: 1 clip body, 2 clip glow, 3 trail body, 4 trail drawn over the
   body, 5 shared; the body = its own body pixels with enough coverage;
3. per-type lists shape it (all defined at the top of `idlabels.py`, extended by the fix files):
   `ONE_LABEL[t]` ('hull' = convex hull, an int = closing radius), `BRIGHT_BODY`, `TRAIL_ATTACH`, `STREAK`, `BRIGHT_RIM`,
   `DARK_DROP`, `TRAIL_SKIP` ...;
4. `POST_BODY[t](body, ctx)` hooks run last for that type. `ctx` has: `o` (the object), `t`, `own`, `C`, `region`, `cv`, `img`,
   `ids_h` (draw id per pixel), `R`, `cam`, `pred_any`, `pred_body`, `mine`, `state`;
5. one owner per pixel (nearest prediction wins); then spawned characters (`spawn_labels`) and zones (`zone_labels`).

## 2. Fix files

`labeller/fixes/*.py` are executed **in name order inside the `idlabels` module namespace** when it is imported (a0..a7, then
`z_<date>_*`, `zz_*`, `zzz_*`). A fix can:
* extend a list: `ONE_LABEL[6000123] = 'hull'`;
* chain a per-type hook (keep the earlier one!):
  ```python
  _myfix_prev = POST_BODY.get(6000123)
  def _myfix(body, ctx):
      if _myfix_prev: body = _myfix_prev(body, ctx)
      ...                       # numpy / scipy on the 540 x 1200 masks
      return body
  POST_BODY[6000123] = _myfix
  ```
* wrap the whole function (for frame-level rules): `_myfix_prev_labels = labels` then `def labels(objs, cam, R, state=None, ended=(), img=None): P = _myfix_prev_labels(...)`.

Conventions: a unique prefix for every global (`_myfix_...`), a header comment saying which skin / type / frame and why, a
switch to turn it off (env var) when it is general. Never edit an older fix to change its behaviour: add a new file that sorts
after it. Enemy (red-tinted) shots go through the same code with `ctx['own'] == False`; their red clips can use other shapes, so
check both colours.

Switches (env): `IDLABELS_NOFIXES=1` (no fix files), `IDLABELS_SKIP=a.py,b.py` (leave some out), `IDLABELS_OVERRIDE=file.py`
(add a file after all of them - how candidates are tested), `IDLABELS_MOTION_DIR/_END/_ENDTIME=1`, `IDLABELS_GLOW=1` (old glow
layer), `IDLABELS_SAME_ATTACK=0`.

Type ids and names: `python3 -c "import sys; sys.path.insert(0,'labeller'); import rc, projtab; T = projtab.table(); print(T[6001046])"`
(name, sprite file, blue / red clip, radius, speed ...). `debug/inspect_frames.py` prints the type name of every label.

## 3. Workflow

1. **Reproduce.** From the dataset JSON take the skin (`owner_skin`) and the tick. Capture that skin and keep the capture:
   `python3 bl.py capture --skin KnightPinata caps/s131` (solo, attacks at 0/90/180/270 + super; `--angles 0,90` is faster).
   Captures are not bit-repeatable, so the exact frame may differ; look for the same situation.
2. **Look.** `python3 debug/inspect_frames.py caps/s131 --ticks 300-330 --crop --scale 2 --out look.png`: raw | labels, and a
   table of every label (object, part, class, type, brawler, team, live / ENDED, pixels). Compare with `docs/LABEL_RULES.md` and
   the skin's decisions there: some shapes that look odd were chosen on purpose.
3. **Write a candidate** fix file outside `labeller/fixes/` (e.g. `/tmp/fix_ash.py`), limited to the wrong type(s).
4. **Compare** on the same capture: `python3 debug/compare_fix.py caps/s131 /tmp/fix_ash.py --out cmp.png` lists every frame
   that changes (+/- pixels) and renders raw | before | after. Iterate until only the wrong frames change, the right way.
5. **Prove nothing else moves.** Capture 3-5 other skins (other skins of the same brawler, and brawlers with similar shots) and
   run compare_fix on each: it must say `0 of N frames change`.
6. **Enemy version.** Capture with an enemy shooter: `POSECAP_SHOOTER=1 python3 bl.py capture --skin KnightPinata caps/s131_red`
   (player 1, red-tinted, fires next to the camera's brawler) and check the red label has the same shape as the blue one.
7. **Install**: copy the file to `labeller/fixes/z_<yyyymmdd>_<what>.py` (it then runs after every existing fix), regenerate the
   dataset, and note the change in the commit message.

Do not edit `labeller/idlabels.py`, `track.py` or `bulletlabels.py` unless a fix file truly cannot express the change; if you do,
run step 5 on many skins (every brawler if possible) - those files shape every label.
