# sweep 10-27 fix for skin 433 (Sonnet revision, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s433/)
# Beamer Mecha 433 attack: Beamer007Projectile (type 6001310, clip mandy_007_atk_projectile). Skin 433, attack 0 deg and
# attack 90 deg rows. Sweep 2026-10-27, task s433 (second attempt): pair 8124 (tick 269, spur) and pair 8127 (tick 305, ragged),
# plus the reviewer's findings on fix.py (jagged cut over the head at 313, zigzag edges, head split off the chevron).
# Causes (atlas shape ids of effects_brawler_mandy.sc, see the shape breakdown):
#  - shape 267 is the whole chevron silhouette: outer thin arm lines, the bands, and the cyan arrowhead in its mouth;
#  - shape 36 is the soft blue glow tail behind it: a blob behind the arms (the spur on attack 0, tick 269) and a tall
#    column above the head on attack 90 (ticks 309-315); the same shape also shows through the gaps between head and
#    bands; shapes 4 and 79 are the green trail shards flying with the attack (they overlap the head on tick 313).
# Fix: the label is the solid convex silhouette of the attack's own sprite (shape 267), thin outer arms included:
#  1. candidate pixels = the labelled pixels + shape 267 + shapes 36/4/79, but only inside the hull of the shape-267
#     pixels (1.5 px tolerance), and shapes 36/4/79 only within 3 px of a shape-267 pixel: this drops the glow column /
#     spur and the shards flying past, but keeps the shard / glow pixels that lie over the head or between head and bands;
#  2. closing (3 px) and hole filling; parts smaller than 20 % of the largest part are dropped (the lone dot of the head
#     at tick 265, the sparkle above the cone at tick 305);
#  3. the result is filled to its own convex hull (the arms are thin lines: the user wants one solid piece there).
# Not touched: the super (type 6001311, own fix z_1023_beamer_mecha.py), the other types, the shared clean-up.
import numpy as _np

B7_TYPE = 6001310
B7_FILE = 'sc/effects_brawler_mandy.sc'
B7_NAMES = ('mandy_007_atk_projectile', 'mandy_007_atk_projectile_red') + tuple(
    n for f, n in _trail_clips(B7_TYPE) if f == B7_FILE)
B7_EDGE = 267            # the attack's own silhouette (SP below is shape id + 1)
B7_SOFT = (36, 4, 79)    # glow tail, trail shards
B7_TOL = 1.5             # px: tolerance of the silhouette hull
B7_LOOSE = 2.5           # px: tolerance for labelled pixels without a shape (sprite edges)
B7_NEAR = 3              # px: soft shapes count only this close to shape 267
B7_CLOSE = 3             # px: closing radius
B7_MINFRAC = 0.2         # parts below this fraction of the largest are specks


def _hull_in(pts, shape, tol):
    from scipy.spatial import ConvexHull
    H = ConvexHull(pts.astype(float))
    gy, gx = _np.mgrid[0:shape[0], 0:shape[1]]; ok = _np.ones(shape, bool)
    for a_, b_, c_ in H.equations: ok &= (a_ * gx + b_ * gy + c_) <= tol
    return ok


def _beam007_fix(body, ctx):
    from scipy.ndimage import (binary_closing, binary_fill_holes, binary_dilation, generate_binary_structure,
                               label as cclabel)
    if not body.any(): return body
    SP = _a4_shape_pixels(ctx['R'], B7_FILE, B7_NAMES)
    # shape 267 may also show elsewhere in the frame (the same sprite of other effects): only near this label
    edge = (SP == B7_EDGE + 1) & binary_dilation(body, structure=generate_binary_structure(2, 1), iterations=3)
    if edge.sum() < 8: return body
    ys, xs = _np.nonzero(edge)
    # work in a window around the shape so the hull grid stays small
    y0, y1, x0, x1 = max(0, ys.min() - 12), min(body.shape[0], ys.max() + 13), max(0, xs.min() - 12), min(body.shape[1], xs.max() + 13)
    sl = (slice(y0, y1), slice(x0, x1))
    b = body[sl]; sp = SP[sl]; ed = edge[sl]
    pts = _np.stack([_np.nonzero(ed)[1], _np.nonzero(ed)[0]], 1)
    try:
        ins = _hull_in(pts, b.shape, B7_TOL); loose = _hull_in(pts, b.shape, B7_LOOSE)
    except Exception: return body
    soft = _np.isin(sp, [s + 1 for s in B7_SOFT])
    near = binary_dilation(ed, structure=generate_binary_structure(2, 1), iterations=B7_NEAR)
    cand = ((b | ed | soft) & ins) | (b & loose & ~soft)
    cand &= ~(soft & ~near)
    cand &= ~((sp == 1) & ~b)          # the brawler's ground disc (shape 0) is not part of the attack
    pad = 12
    cl = binary_closing(_np.pad(cand, pad), structure=generate_binary_structure(2, 1), iterations=B7_CLOSE)[pad:-pad, pad:-pad]
    res = binary_fill_holes(cand | cl) & loose
    lab, n = cclabel(res, structure=_np.ones((3, 3)))
    if n > 1:
        sz = _np.bincount(lab.ravel())[1:]
        res = _np.isin(lab, _np.nonzero(sz >= B7_MINFRAC * sz.max())[0] + 1)
    ry, rx = _np.nonzero(res)
    if len(rx) >= 3:
        try: res = res | _hull_in(_np.stack([rx, ry], 1), res.shape, 0.5)
        except Exception: pass
    new = _np.zeros_like(body); new[sl] = res
    return new


POST_BODY[B7_TYPE] = _beam007_fix
