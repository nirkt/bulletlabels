# installed 11-01 from out/sweep2/haikutasks/siteai2/sandy_spawner/fix.py (2nd AI sweep: Sandy clouds 296/359/363 solid, 363 super loop out, 369 hidden star filled)
# AI-review batch 2026-11-01, second sweep, task sandy_spawner (brawlers Sandstorm and SpawnerDude), fixes for the reviewer's
# findings that were real defects. Exec'd in idlabels' namespace after all installed fixes (IDLABELS_OVERRIDE), keyed to explicit
# type ids only. Checked per frame against the owner's rules (RULES.md + the coordinator clarification in task.md).
#
# (1) 363 Sandy006 super (type 6000677, SandstormDeepsea super 45, reviewer t345 "should not be labelled": a small pink loop
#     sitting on the brawler's dome; the streak next to it is fine). At t345 the super's label has three pieces: the streak
#     (90 px, overlaps the predicted body), a 57 px loop on the brawler's dome (no overlap with the prediction) and a 1 px speck.
#     Fix: keep only the pieces of at least 5 px that touch the predicted body. Ticks 347 and 349 are one piece and do not change.
#
# (2) 369 SpawnerDude005 super (type 6000709, SpawnerDude "Make" super 45, reviewer t345 "too small / cut off": the magenta label
#     covers only the left half of the cyan star). At t345 the star is under the cyan burst: 59 % of its predicted body has no
#     sprite in front (hidden). Hidden-parts rule: the label keeps the full shape, so the predicted body (the star's own shape,
#     which equals its visible shape at t347 and t349: 806 predicted vs 807 drawn) is added. Ticks 347 and 349 have no hidden
#     share and do not change.
#
# (3) Sandy clouds 296 Sandy004 attack (6000672, attack 0 and 90) and 363 Sandy006 attack (6000676): ragged zone outlines (bites,
#     notches, interior zigzag lines; reviewer 296 t311, own inspection of 296 t307-t311 and 363 t305). Same owner decision as
#     the installed fix for 414 (Sandy007, z_1103_spawner_sandy.py): one solid cloud outline, bays and holes filled, spurs
#     removed; the same parameters, limited to the predicted region.
#
# (4) 359 Sandy005 attack (6000674, Sandy cloud, reviewer t269-t271 "split into several small outlines"): the cloud's puffs had
#     separate outlines (2 pieces at t265, 4 at t267, 3 large + 1 small at t269). Owner rule for Sandy: one solid zone over the whole cloud.
#     At t271 to t275 the cloud is one piece (or a dispersing leftover): unchanged here (needs_user).
#     Fix: the puffs are joined by a closing of radius X_MERGE (bays and gaps between puffs filled), then holes filled and the
#     same smoothing as (3), limited to the predicted region. Not a hull (the owner wants the cloud shape).
#     Guard: a body with solidity below X_SOLID (scattered burst debris, 296 t271 / t313) is left as it is.
#
# NOT changed (see report): the 291 / 439 spawnable turret footprint (z_1100_s045 clips every skin of SpawnerDudeTurret to its
# 120-unit collision circle; the hex / barbell models overhang it: needs_user); the Sandy burst leftovers 296 t271, 359 t273
# and t275 (needs_user, the open point of the coordinator); 414 t271 (the upper lobe is a fade, needs_user).
#
# Types touched: 6000677 (363 super), 6000709 (369 super), 6000672 (296 attack 0/90), 6000676 (363 attack 0/90),
# 6000674 (359 attack 0). Other skins of the same types: none (each type belongs to one skin).

import numpy as _x_np
from scipy.spatial import ConvexHull as _x_hull
from scipy.ndimage import (binary_closing as _x_close, binary_fill_holes as _x_fill, binary_opening as _x_open,
                           label as _x_cc)

X_CLOSE = 5           # px (frame res), as in z_1103_spawner_sandy.py (414): bays and bites narrower than 2 * X_CLOSE filled
X_OPEN = 3            # px: spurs narrower than 2 * X_OPEN + 1 removed
X_MERGE = 12          # px: gaps between the puffs of the 359 cloud closed (the puffs one solid cloud)
X_SOLID = 0.70        # body area / convex hull area: an intact cloud is 0.74 or more (measured on all ticks of the rows);
                      # scattered leaves / flakes of a burst that ended are 0.31 - 0.62 and stay as they are (needs_user)
X_PIECE = 300         # px: a piece of the 359 cloud counts for the merge when at least this big
X_SPECK = 5           # px: pieces of the 363 super smaller than this are specks


def _x_disc(r):
    yy, xx = _x_np.mgrid[-r:r + 1, -r:r + 1]
    return (xx * xx + yy * yy) <= r * r


def _x_chain(t, fn):
    """runs fn after the hook that was installed before for type t (kept, so earlier fixes stay)"""
    prev = POST_BODY.get(t)

    def hook(body, ctx):
        if prev is not None: body = prev(body, ctx)
        return fn(body, ctx)
    POST_BODY[t] = hook


def _x_solidity(m):
    ys, xs = _x_np.nonzero(m)
    if len(xs) < 10: return 0.0
    try: return len(xs) / float(_x_hull(_x_np.c_[xs, ys]).volume)
    except Exception: return 0.0


def _x_smooth(body, ctx, r_close=X_CLOSE):
    if not body.any(): return body
    if _x_solidity(body) < X_SOLID: return body   # scattered debris of an ended burst: unchanged
    P = r_close + 1
    cl = _x_close(_x_np.pad(body, P), structure=_x_disc(r_close))[P:-P, P:-P] | body
    f = _x_fill(cl) & ctx['region']
    Q = X_OPEN + 1
    op = _x_open(_x_np.pad(f, Q), structure=_x_disc(X_OPEN))[Q:-Q, Q:-Q]
    return op if op.any() else body


def _x_sand_cloud(body, ctx):
    return _x_smooth(body, ctx)


def _x_merge_cloud(body, ctx):
    if not body.any(): return body
    L, n = _x_cc(body); sz = _x_np.bincount(L.ravel())[1:]
    if int((sz >= X_PIECE).sum()) < 2: return body   # one piece (or leftovers): unchanged
    P = X_MERGE + 1
    cl = _x_close(_x_np.pad(body, P), structure=_x_disc(X_MERGE))[P:-P, P:-P] | body
    m = _x_fill(cl) & ctx['region']
    return _x_smooth(m, ctx) if m.any() else body


def _x_hidden_star(body, ctx):
    pb = ctx['pred_body']; C = ctx['C']
    if not pb.any(): return body
    if (C[pb] == 0).mean() < 0.2: return body   # not hidden: unchanged (ticks 347, 349)
    return body | (pb & ctx['region'])


def _x_loop_off(body, ctx):
    pb = ctx['pred_body']
    if not body.any() or not pb.any(): return body
    L, n = _x_cc(body)
    if n <= 1: return body
    keep = [k for k in range(1, n + 1) if int(((L == k) & pb).sum()) > 0 and int((L == k).sum()) >= X_SPECK]
    if not keep: return body
    return _x_np.isin(L, keep) & body


# (1) 363 Sandy006 super: the streak only, the loop on the dome out
_x_chain(6000677, _x_loop_off)

# (2) 369 SpawnerDude005 super: hidden under the burst, the star's full shape
_x_chain(6000709, _x_hidden_star)

# (3) Sandy clouds 296 (attack 0 / 90) and 363 (attack 0 / 90): one solid, smooth cloud outline (as 414)
_x_chain(6000672, _x_sand_cloud)
_x_chain(6000676, _x_sand_cloud)

# (4) 359 Sandy005 cloud: the puffs joined into one solid cloud (only while the cloud is in two or more pieces)
_x_chain(6000674, _x_merge_cloud)
