# 2026-11-07 enemy-label fix, group grpB (PowerLeveler / Cocooner / Twins). Prefix _eB_. Installed as py/fixes/z_1116_enemy_grpB.py.
# Own (blue) labels are never touched: every rule below returns early for ctx['own'] / own objects.
import numpy as _eB_np

# ---- 590 PowerLevelerBuzz: Surge008Projectile 6000806 (the flying fist; only this skin) ----
# CAUSE: as 051 (z_1109_enemy_pl): at the first drawn tick of a shot the RED fist is covered by a big white additive muzzle burst (draws that
# belong to no sprite code of the bolt), so the ID-pass body (a4_agent4: bright pixels of its own sprites) keeps only a few exposed chunks
# (attack 0 t265: 527 px in two pieces against 811 px of the blue ellipse, recall 0.48; attack 90 t305: 640 px, recall 0.42). The blue fist
# has no burst. The predicted body of this type is only the ellipse's core (about 1/5 of the label), so z_1109's "hull of body + predicted
# body" would be too big here.
# FIX (enemy only, ctx['own'] False): when less than 85 % of the predicted core is in the body (the fist is hidden), the label is the
# convex hull of the exposed body pixels within 12 px of the core plus the core, cut to the region (hidden-part rule: no holes / gaps).
# Ticks where the fist is visible (core inside the body) and every own label are untouched.
import numpy as _eB_np
from scipy.ndimage import binary_dilation as _eB_dil

_eB_PL = 6000806
_eB_pl_prev = POST_BODY.get(_eB_PL)
_eB_disk12 = (lambda r: (lambda y, x: x * x + y * y <= r * r)(*_eB_np.ogrid[-r:r + 1, -r:r + 1]))(12)


def _eB_pl_post(body, ctx):
    if _eB_pl_prev is not None: body = _eB_pl_prev(body, ctx)
    if ctx.get('own') or not body.any(): return body
    pb = ctx['pred_body']
    if not pb.any() or (body & pb).sum() >= 0.85 * pb.sum(): return body
    near = body & _eB_dil(pb, _eB_disk12)
    return hull_mask(near | pb) & ctx['region']


POST_BODY[_eB_PL] = _eB_pl_post


# ---- Cocooner cocoon spawnable (characters CocoonerCocoon, every Cocooner skin: 085 343 357 438 ...) ----
# CAUSE (085 343 357 438, super ticks 11-17, 'spawn:CocoonerCocoon' labelled in the enemy run only): the cocoon is no 3D model (it is the pink
# glowing egg, drawn from sprites; in the own runs it carries no label), so spawn_labels never finds a draw of it, but it gives every
# runtime-texture draw nearest to a spawnable within CHAR_NEAR = 80 px to it. In the enemy run (shooter 450 units below the camera) the egg
# lands 78 px from the pool's water edge at the screen bottom and that stray model draw (5250 px, ticks 353-359) became its label.
# FIX (enemy cocoons only): the cocoon of a player of another team than the camera's (the hero nearest to the screen centre) gets no spawn
# label. Own cocoons, every other spawnable and all projectiles are untouched.
if '_eB_spawn_base' not in globals(): _eB_spawn_base = spawn_labels


def spawn_labels(state, cam, R, taken):
    out = _eB_spawn_base(state, cam, R, taken)
    if not out: return out
    import owners as _eB_ow
    heroes = []; coc = {}
    for c in state['chars']:
        ch = _eB_ow.character(int(c[1]))
        if int(c[2]) >= 0 and ch['kind'] == 'hero':
            heroes.append((int(c[2]), _eB_np.array(_proj(cam, int(c[3]), int(c[4]), int(c[5])))))
        if ch['name'] == 'CocoonerCocoon': coc[int(c[0])] = int(c[2])
    drop = [cid for cid in out if cid in coc]
    if not drop or not heroes: return out
    team = {int(p[0]): int(p[1]) for p in state['players']}
    viewer = min(heroes, key=lambda h: float(_eB_np.hypot(*(h[1] - _eB_np.array([600.0, 270.0])))))[0]
    for cid in drop:
        if team.get(coc[cid]) != team.get(viewer): del out[cid]
    return out
