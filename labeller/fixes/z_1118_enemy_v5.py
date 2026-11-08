# 2026-11-08 enemy-label fix, group v5 (skins 018 179 028 231). Prefix _f5_. Installed as py/fixes/z_1118_enemy_v5.py.
# Every hook below returns the body unchanged for own (blue) objects (ctx['own'] True).
import numpy as _f5_np
from scipy.ndimage import binary_dilation as _f5_dil, binary_fill_holes as _f5_fill

# ---- Brock attack rockets: 018 RocketGirlDefault (6000053, RocketGirlProjectile) and 179 RocketGirlBrown (6000067, Brock008Projectile; its burst is a cream puff + yellow spikes, enemy body up to 34 px off the hull, own 0).
# CAUSE: the red shot has an enemy-only muzzle burst (big white / yellow spike star + flame, drawn at the shooter on the first ticks, rel 3
# and 5) right on the rocket. Its sprites lie inside the rocket clip's atlas shapes (sprite code 1) so the generic rule (mine = clip body pixels)
# takes the whole burst into the label (body up to 24 px outside the rocket's predicted hull; the own run has no burst).
# FIX (enemy only): the body is cut to the 4 px dilated convex hull of the tracker's predicted body (own bodies lie within 4 px of it, measured on
# all own ticks of 018 except one spray tick at 8 px); holes closed inside it.
_f5_BR_TYPES = (6000053, 6000067)
_f5_BR_K = 4

def _f5_br(prev):
    def post(body, ctx):
        if not ctx.get('own', True) and body.any():
            pb = ctx['pred_body']
            if pb.any():
                keep = _f5_dil(hull_mask(pb), iterations=_f5_BR_K)
                cut = body & keep
                if cut.any(): body = cut | (_f5_fill(cut) & keep & ctx['region'] & ~body & (ctx['C'] > 0))
        return prev(body, ctx) if prev is not None else body
    return post

for _f5_t in _f5_BR_TYPES:
    if not getattr(POST_BODY.get(_f5_t), '_f5', False):
        _f5_f = _f5_br(POST_BODY.get(_f5_t)); _f5_f._f5 = True; POST_BODY[_f5_t] = _f5_f
