# 2026-11-07 enemy-label fix, group cocoon (Cocooner skins 085 343 357 438). Prefix _ecc_. Installed as py/fixes/z_1117_enemy_cocoon.py.
# Own (blue) labels are never touched: the rule below returns the body unchanged for ctx['own'] objects.
#
# 357 CocoonerVirus Cocooner003Projectile 6001382, 085 CocoonerDefault CocoonerProjectile 6001376 (attack 0, the first drawn ticks of a shot).
# CAUSE: the swirl ball is the label; the pink / cyan rod behind it (the 16x16 beam) has no sprite code and is never body. In the RED run the
# enemy muzzle burst (a big additive star of spikes drawn at the shooter, which the blue run does not have) sits right behind the ball: some of its
# draws show the type's sprite shapes (code 1) / are blended sprite pixels (code 0 / 4) inside the ball's search region, so a few spikes (3 pieces,
# 10-80 px, 3-25 px left of the ball) became body. The type's ONE_LABEL 'hull' (357) / the beam-notch fill (085, hull of the body) then stretched
# the label from the ball back over the whole rod (357: area ratio 1.47, precision 0.73).
# 085 had a second part: the red run's echo ball (a trailing copy of the ball sprite, drawn right behind it) made the ID-pass body the echo ball and the
# aligned prediction slid onto it, so the real ball's own layer (class 4, coverage 28-80) was not body and the label sat on the echo (ratio 1.42).
# FIX (enemy objects only, ctx['own'] False): the body is cut to the 4 px dilated convex hull of ctx['pred_body'] (the tracker's predicted ball, which
# lines up with the drawn ball in both runs: every own body of these ticks lies inside it); bits of the muzzle burst outside it are dropped. Run before
# the existing hooks / ONE_LABEL, so the hull / notch fill then see the ball only. Then the predicted ball's own sprite pixels (class 1/3/4/5,
# coverage >= half the type's cut) are added back and holes inside the ball closed. Own labels and every other type are untouched.
import numpy as _ecc_np
from scipy.ndimage import binary_dilation as _ecc_dil, binary_fill_holes as _ecc_fill

_ecc_TYPES = (6001382, 6001376)
_ecc_K = 4


def _ecc_make(prev):
    def post(body, ctx):
        if not ctx.get('own') and body.any():
            pb = ctx['pred_body']
            if pb.any():
                keep = _ecc_dil(hull_mask(pb), iterations=_ecc_K)
                body = body & keep
                # the ball's own translucent layer under the echo balls (class 4 / 1 sprite pixels) is not body when the aligned prediction slid
                # onto the echo ball: take the predicted ball's own sprite pixels back (low coverage), then close holes inside the ball
                add = pb & (ctx['C'] > 0) & (ctx['C'] != 2) & (ctx['cv'] >= FADE_CUT.get(ctx['t'], CUT) // 2)
                body = body | add
                body = body | (_ecc_fill(body) & keep & ctx['region'])
        return prev(body, ctx) if prev is not None else body
    return post


for _ecc_t in _ecc_TYPES:
    if not getattr(POST_BODY.get(_ecc_t), '_ecc', False):
        _ecc_f = _ecc_make(POST_BODY.get(_ecc_t)); _ecc_f._ecc = True; POST_BODY[_ecc_t] = _ecc_f
