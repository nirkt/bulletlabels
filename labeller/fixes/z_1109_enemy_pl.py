# 2026-11-04 enemy-label fix, group pl_janet (skin 051 PowerLevelerDefault, type 6000787 PowerLevelerProjectile).
# CAUSE: the RED (enemy) Surge bolt is covered at its first drawn tick(s) by a big white additive muzzle burst (draws that belong
# to no sprite code of the bolt). The ID-pass body (sprite codes 1/3/4 of the bolt) then only keeps the few exposed rim pixels
# (tick 305: 307 px vs 664 px of the blue bolt, only 17% of the predicted core still part of the body): the enemy label lacked
# the hidden part. The blue (own) bolt has no such burst, its label is unchanged.
# FIX (enemy only, ctx['own'] False): when less than half of the predicted body is in the ID-pass body (the bolt is hidden under
# another effect) the label is the convex hull of the exposed body and the predicted body (the hidden-part rule: a hull has no
# holes / hidden gaps), cut to the region. Everything else (own labels, enemy ticks where the bolt is visible) is untouched.
import numpy as _en4_np

_en4_PL_T = 6000787
_en4_pl_prev = POST_BODY.get(_en4_PL_T)


def _en4_pl_post(body, ctx):
    if _en4_pl_prev is not None: body = _en4_pl_prev(body, ctx)
    if ctx.get('own') or not body.any(): return body
    pb = ctx['pred_body']
    if not pb.any() or (body & pb).sum() >= 0.5 * pb.sum(): return body
    return hull_mask(body | pb) & ctx['region'] | body


POST_BODY[_en4_PL_T] = _en4_pl_post
