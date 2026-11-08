# v8 enemy-label fix (2026-11-08), prefix _f8_. Installed as py/fixes/z_1118_enemy_v8.py. Enemy objects only (ctx['own'] False): own labels untouched.
#
# 485 PowerLevelerKnight, Surge004Projectile 6000798 (attack 0, first drawn tick of a shot): the RED run has a big white additive muzzle burst
# sitting on the cone. Its draws use sprite shapes the type's clip also has (sprite code 1, coverage solid), so the burst's spikes / plume
# (1829 px against a 1031 px predicted cone) became body, and z_1027_s485's smoothing then took the whole burst into the label.
# FIX: when more than F8_OUT of the body lies outside the 4 px dilated convex hull of the tracker's predicted body, the body is cut to that
# hull (the bolt lines up with its prediction in both runs), before the installed hook (smoothing, cone pixels).
import numpy as _f8_np
from scipy.ndimage import binary_dilation as _f8_dil

F8_OUT = 0.2
F8_K = 4
_f8_t485 = 6000798
_f8_prev485 = POST_BODY.get(_f8_t485)


def _f8_post485(body, ctx):
    if not ctx.get('own') and body.any():
        pb = ctx['pred_body']
        if pb.any():
            keep = _f8_dil(hull_mask(pb), iterations=F8_K)
            if (body & ~keep).sum() > F8_OUT * body.sum(): body = body & keep
    return _f8_prev485(body, ctx) if _f8_prev485 is not None else body


POST_BODY[_f8_t485] = _f8_post485


# 611 PowerLevelerPorcelain, Surge010Projectile_k2 6000808 (attack 90 / attack 0, first drawn ticks of a shot): the RED bolt is covered by the big white
# additive muzzle burst, so the ID-pass body is only the exposed red rim (301 of 758 predicted px at tick 305) and, as the aligned prediction then slides
# onto the bolt's shadow, the glow-code pixels of the shadow ellipse below the bolt are body too: the enemy label lacked the hidden flame / body
# (own: 150 px, 6 % of the label area) and had a stray ring on the shadow. FIX (enemy only, as the 051 rule of z_1109_enemy_pl): cut the body to the
# 4 px dilated hull of the predicted body (the shadow ring goes), and when less than F8_VIS of the predicted body is still body (the bolt hides under
# another effect) the label is the hull of the exposed body and the predicted body, cut to the region.
F8_VIS = 0.75
_f8_t611 = 6000808
_f8_prev611 = POST_BODY.get(_f8_t611)


def _f8_post611(body, ctx):
    if _f8_prev611 is not None: body = _f8_prev611(body, ctx)
    if ctx.get('own') or not body.any(): return body
    pb = ctx['pred_body']
    if not pb.any(): return body
    body = body & _f8_dil(hull_mask(pb), iterations=F8_K)
    if not body.any(): return body
    if (body & pb).sum() < F8_VIS * pb.sum():
        body = hull_mask(body | pb) & ctx['region'] | body
    return body


POST_BODY[_f8_t611] = _f8_post611
