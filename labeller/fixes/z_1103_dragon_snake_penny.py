# AI-review batch 2026-11-01 fix (Haiku fix, reviewed by the coordinator, installed 11-01 LOCAL ONLY; source out/sweep2/haikutasks/siteai/dragon_snake_penny/fix.py)
# AI-review fixes 2026-11-01, task dragon_snake_penny (skins 371, 434, 424, 463, 430, 440). Exec'd after the installed fixes
# (IDLABELS_OVERRIDE). Explicit type ids only. Existing POST_BODY hooks of the same types are kept and run first (chained).
#
# 371 DragonRider002 (type 6001448, attack 0 and attack 90): the streak's label at attack 0 tick 269 has a loop around the
#     violet blob at its rear (a hole in the outline) and its upper-arm tail is a separate piece. Fix: the label is the object's
#     own outline, closed and filled solid (small gaps closed, enclosed holes filled). The thin upper-arm tail is part of the
#     attack and stays. The dark gap between the arms is not the attack and stays out (it is open, not a hole).
# 434 DragonRider003 (type 6001450, attack 0 and attack 90): the breath is split into several pieces. One convex hull per shot
#     (the task's rule for splintered multi-piece attacks), minus the opaque draws in it: the rider in front of the breath is a
#     brawler and is never labelled.
# 424 Byron005 (type 6000870, attack 90 tick 317): the bottle has sunk into the water (the object is finished, the bottle is no
#     longer drawn, only its splash columns remain). Impact / leftover, not labelled.
# 463 Byron006 (type 6000872, attack 90 tick 307): the bottle is outlined in pieces with a red line (the green lever) through its
#     interior. Fix: one convex hull of the bottle's pieces per shot (a multi-piece attack of one object), inside its region.
#     Tick 317 (not flagged): the finished, sunk bottle leaves a small piece at the water's edge; not labelled (as 424).
# 430 Penny008Ulti (type 6000467, super 45 tick 349): the box outline has notches and interior loops. Fix: closed and filled
#     solid, inside its region. (Penny008 rocket, type 6000465, unchanged: see report.)
# 440 Penny009 (type 6000469, attack 0 and attack 90): the label covers the arrowhead only, the blue flame body beside it stays
#     outside. Fix: the flame is connected to the arrow (its own clip body is body wherever it reaches: EXTEND_BODY).

import numpy as _np
from scipy.ndimage import binary_fill_holes as _bfh, label as _ccl


def _chain_post(t, fn):
    """POST_BODY[t] = fn after the hook already installed for t (if any)"""
    prev = POST_BODY.get(t)
    if prev is None:
        POST_BODY[t] = fn
    else:
        POST_BODY[t] = (lambda p, f: (lambda body, ctx: f(p(body, ctx), ctx)))(prev, fn)


def _solid_fn(r, min_frac=0.0):
    """close the body by r px, fill its enclosed holes, keep it inside its region; parts smaller than min_frac of the largest
    are dropped (specks)"""
    def f(body, ctx):
        if not body.any(): return body
        m = _bfh(_close(body, r)) & _dilate(ctx['region'], r)
        if min_frac > 0 and m.any():
            L, n = _ccl(m); sz = _np.bincount(L.ravel())[1:]
            if n > 1: m = _np.isin(L, 1 + _np.nonzero(sz >= min_frac * sz.max())[0]) & m
        return m
    return f


# 424 Byron005 attack 90 tick 317: finished (sunk) bottle is not labelled
def _f6000870_finished(body, ctx):
    if ctx['o'].get('finished'): return _np.zeros_like(body)
    return body


_chain_post(6000870, _f6000870_finished)

# 463 Byron006 attack 0 / 90: the bottle is several pieces (the green lever and the pinched waist are not drawn with its
# opaque sprites): one convex hull per shot of the bottle's own pieces (the task's rule for multi-piece attacks of one object)
def _f6000872(body, ctx):
    if not body.any(): return body
    return hull_mask(body) & _dilate(ctx['region'], 3)


_chain_post(6000872, _f6000872)
# 463 Byron006 attack 90 tick 317 (not flagged, same defect as 424 tick 317): the finished (sunk) bottle leaves only a small piece
# at the water's edge; ended leftovers are not labelled
_chain_post(6000872, _f6000870_finished)

# 430 Penny008Ulti (box, super 45): notches and interior loops around the box (not the rocket: its tail notches are between flame tongues)
_chain_post(6000467, _solid_fn(5))

# 371 DragonRider002 attack 0 / 90 (streak / V flame): one solid outline, gaps closed, enclosed loops filled
_chain_post(6001448, _solid_fn(4))


# 434 DragonRider003 breath: one hull per shot, minus the opaque draws (the rider in front of it)
def _f6001450(body, ctx):
    if not body.any(): return body
    R = ctx['R']; cv = ctx['cv']; ids_h = ctx['ids_h']
    opaque_draws = [d for d, v in R['draws'].items() if v.get('blend') == 0]
    opaque = _np.isin(ids_h, opaque_draws) & (cv >= CUT)
    return hull_mask(body) & ~opaque & _dilate(ctx['region'], 2)


_chain_post(6001450, _f6001450)

# 440 Penny009 arrow (attack 0 / attack 90): the blue flame body is body wherever its clip body reaches. Its clip body is
# translucent (coverage 32-127, measured on sw2_440 t267): the lower coverage cut, as for other translucent bodies.
FADE_CUT[6000469] = 32
EXTEND_BODY.add(6000469)
