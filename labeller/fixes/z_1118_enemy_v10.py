# enemy-label fix 2026-11-08, group v10 (skins 330 Arcade, 354 Buzz, 359 Sandy, 361 Tara, 375 Crow). Global prefix _f10_.
# Installed as py/fixes/z_1118_enemy_v10.py. Every rule acts on ENEMY objects / red-only sprite data (ctx['own'] False, sprites(t, own=False)); the own
# (blue) labels are never touched (proved with pixel diffs of the own captures).
import numpy as _f10_np
import functools as _f10_ft
import re as _f10_re
from scipy.ndimage import binary_dilation as _f10_dil, binary_fill_holes as _f10_fill

# ------------------------------------------------------------------------------------------------------------------------------------------
# 330 ArcadeHauntedHouse, Arcade005Projectile 6000566 (the skull bullets, attack 0 / 90 / super).
# CAUSE: the RED run draws a big pink muzzle-burst puff at the shooter's gun for the first ticks of every shot (the blue run has none). The puff is
# built from the one soft round sprite that is ALSO one of the bullet clip's body sprites (atlas texture 0, region 0.687..0.694 x 0.675..0.683), so every
# puff pixel has sprite code 1 (clip body): the label of the newest bullet became the whole puff (about 800 px instead of the 300 px skull, ticks 265-315,
# 3 pieces per tick) and the skull under it was part of that blob.
# FIX (enemy objects only): the body is cut to the dilated convex hull of the predicted body (the tracker's skull, which lines up with the drawn skull in
# both runs); when the cut leaves less than 60 % of the predicted body (the skull is hidden under the puff) the hull of the exposed body and the
# prediction is the label (the hidden-part rule of z_1109_enemy_pl).
_F10_ARC = 6000566
_F10_ARC_K = 5
_f10_prev_arc = POST_BODY.get(_F10_ARC)


def _f10_arcade(body, ctx):
    res = _f10_prev_arc(body, ctx) if _f10_prev_arc is not None else body
    if ctx.get('own') or not res.any(): return res
    pb = ctx['pred_body']
    if not pb.any(): return res
    keep = _f10_dil(hull_mask(pb), iterations=_F10_ARC_K)
    cut = res & keep
    if cut.sum() < 0.6 * pb.sum():
        cut = (hull_mask(cut | pb) & ctx['region']) | cut
    return cut


POST_BODY[_F10_ARC] = _f10_arcade

# ------------------------------------------------------------------------------------------------------------------------------------------
# 354 RopeDudeFenrir, Buzz006UltiProjectile 6000991 (super: the thrown ice sword).
# CAUSE: the ice crystals around the sword (the rubble behind it, shards beside it) are trail sprites. The blue art draws them with textured trail
# sprites (sprite code 4 = 'trail over', never body). The RED trail clips (buzz_006_ulti_trail_ice_red_NN) draw them with the flat-colour swatch region
# (atlas texture 1, uv 0.615 / 0.707) that is ALSO a body sprite of the red bullet clip, so every crystal pixel has code 1 (clip body) and the crystals
# became separate pieces of the label (2 + 2 extra components, +17 % area at the first ticks; the sword itself is one piece overlapping the prediction).
# FIX (enemy objects only): body pieces that have no pixel within 3 px of the predicted body (the tracker's sword) are dropped; the sword, its chain rings
# and everything that touches the prediction stay. Own labels untouched.
from scipy.ndimage import label as _f10_cc
_F10_BUZZ = 6000991
_f10_prev_buzz = POST_BODY.get(_F10_BUZZ)


def _f10_buzz(body, ctx):
    res = _f10_prev_buzz(body, ctx) if _f10_prev_buzz is not None else body
    if ctx.get('own') or not res.any(): return res
    pb = ctx['pred_body']
    if not pb.any(): return res
    near = _f10_dil(pb, iterations=3)
    L, n = _f10_cc(res, structure=_f10_np.ones((3, 3)))
    if n < 2: return res
    hit = _f10_np.unique(L[near & res]); hit = hit[hit > 0]
    return _f10_np.isin(L, hit) if hit.size else res


POST_BODY[_F10_BUZZ] = _f10_buzz
