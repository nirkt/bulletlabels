# 2026-11-08 enemy-label fix, group v6 (skins 057 427 Ruffs, 068 331 Janet). Prefix _f6_. Installed as py/fixes/z_1118_enemy_v6.py.
# Own (blue) labels are never touched: every hook returns the body unchanged for ctx['own'] objects.
#
# 057 RuffsDefault, RuffsProjectile 6000887 (attack 90, first drawn tick of a shot): the RED run has an enemy muzzle burst (a big additive
# white-pink arrow of spikes drawn at the shooter, which the blue run does not have) right behind the two flames. Parts of the burst show
# the type's sprite shapes in the flames' search region, so they became body and the label of the two flames grew a big arrow joining them
# (a 69x33 px blob where the blue run has two 43x15 flame outlines).
# FIX (enemy objects only): the body is cut to the 4 px dilated convex hull of ctx['pred_body'] (the tracker's predicted flame, which lines
# up with the drawn flame in both runs), the same recipe as z_1117_enemy_cocoon.
import numpy as _f6_np
from scipy.ndimage import binary_dilation as _f6_dil

_f6_BURST_TYPES = (6000887,)
_f6_K = 4


def _f6_burst(prev):
    def post(body, ctx):
        if not ctx.get('own') and body.any():
            pb = ctx['pred_body']
            if pb.any():
                body = body & _f6_dil(hull_mask(pb), iterations=_f6_K)
        return prev(body, ctx) if prev is not None else body
    return post


for _f6_t in _f6_BURST_TYPES:
    if not getattr(POST_BODY.get(_f6_t), '_f6', False):
        _f6_f = _f6_burst(POST_BODY.get(_f6_t)); _f6_f._f6 = True; POST_BODY[_f6_t] = _f6_f
