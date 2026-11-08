# sweep 10-27 fix for skin 070 (Haiku fix, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s070/)
# Haiku fix task s070 (sweep 2026-10-27), skin 070 SilencerDefault, finding pair 1256 (extra_label, attack 0 deg).
# Type 6001195 SilencerProjectile (clip effects_brawler.sc, shapes 63 / 28 / 888 / ...).
# Finding: a red outline on a small detached teal puff about 10 px behind the projectile's tail. A trail puff is a
# particle and must stay unlabelled (RULES 4, RULES 'what a correct label is' 4). In the new capture the puff is at
# tick 269 (31 px, 1 px away from the body); at tick 271 it is already gone and the old pair's tick is not in this capture.
# Cause: the puff's pixels (mostly shape 63) touch the projectile body with a 1 px gap. The shared clean-up (z_1099)
# bridges gaps of up to 3 px and keeps parts of 25 px or more, so the puff stayed as a second part of the label.
# Approaches considered:
#   1. drop shape 63 pixels everywhere for this type: rejected, the body itself has shape 63 pixels (4 to 13 px per part);
#   2. drop small parts by size alone: rejected, the 90 deg row of the same skin has detached shape 63 bits of 30 to
#      47 px, and this task is not the 90 deg row; tick 265 also has a 14 px detached part at the nose of the next flame
#      (64% shape 63) that cannot be told apart from a lower tail strand on the drawn frame, so it is left as it is;
#   3. drop detached parts (8-connected, no bridging) that are at most SILENCER_DETACH_MAX px and at least
#      SILENCER_PUFF_SHARE shape 63, only for shots within about 25 deg of horizontal (the attack 0 deg row). Chosen:
#      the puff is a separate piece of the puff's own sprite shape, and the main body and the 90 deg row stay the same.
# The largest part of the body is always kept, so no projectile loses its label.
SILENCER_PROJ = 6001195
SILENCER_DETACH_MAX = 40     # px: a detached part of at most this size is a puff, not body
SILENCER_PUFF_SHAPE = 63     # clip shape of the puff (effects_brawler.sc, the same shape id is used by some body pixels)
SILENCER_PUFF_SHARE = 0.8    # share of the part's pixels that must be the puff shape
SILENCER_HORIZ_COS = 0.9     # |cos| of the shot direction: about 25 deg of horizontal (attack 0 deg); 90 deg shots are not touched


def _silencer_shot_dir(o):
    import numpy as np
    p = o.get('path') or []
    tg = o.get('target')
    if not p or tg is None: return None
    v = np.array([float(tg[0]) - float(p[0][0]), float(tg[1]) - float(p[0][1])])
    n = float(np.hypot(v[0], v[1]))
    return v / n if n > 1.0 else None


def _fix_silencer_puff(body, ctx):
    import numpy as np
    from scipy.ndimage import label as cc
    if not body.any(): return body
    d = _silencer_shot_dir(ctx['o'])
    if d is None or abs(d[0]) < SILENCER_HORIZ_COS: return body
    L, n = cc(body, structure=np.ones((3, 3), int))
    if n < 2: return body
    sz = np.bincount(L.ravel())[1:]
    big = 1 + int(np.argmax(sz))
    T = projtab.table()[SILENCER_PROJ]
    clips = tuple(c for c in (T.get('blue'), T.get('red')) if c)
    SP = _a4_shape_pixels(ctx['R'], T['file'], clips)
    drop = []
    for k in range(n):
        j = k + 1
        if j == big or sz[k] > SILENCER_DETACH_MAX: continue
        m = L == j
        if (SP[m] == SILENCER_PUFF_SHAPE + 1).mean() >= SILENCER_PUFF_SHARE: drop.append(j)
    if not drop: return body
    return body & ~np.isin(L, drop)


POST_BODY[SILENCER_PROJ] = _fix_silencer_puff
