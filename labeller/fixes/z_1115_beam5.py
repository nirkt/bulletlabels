# beam5 fix file (round 5 + re-check, 2026-11-06) -> W/py/fixes/z_1115_beam5.py. Every global name has the prefix _bm5_.
#
# 405 BeamerCarnival, Beamer005Projectile (6001306; only skin 405 uses it; its enemy copies share the shape lookup), both attack rows
# (owner round 5: "too small / cut off" on attack 0 t265 / t267 and attack 90 t307 / t309; re-check: "the labels should also be tighter").
# Measured on two fresh captures (hullcap beam5 405): the labels of the earlier steps (z_1104 tail growth, z_1111_cutoff3, z_1114_cut5, and
# the first version of this file) are built from the ID pass shapes / coverage, and the ID pass does not see the beam the way it looks:
#   * the bright yellow-white HEAD of the shot is drawn by additive glow draws / a front draw of another sprite (ID shape 0 / none, coverage
#     27-214), so rules on shape + coverage >= 128 miss half of it: the label narrows to a thread in front of the yellow core;
#   * the label takes the soft teal glow around the tail (shape 133, coverage 26-60: 3 px bulge on each side of the bright band).
# New rule (the beam as it LOOKS: its bright body, not the halo; data-derived from the frame colours, no per-tick shapes): the label of this
# type is rebuilt as the bright body = pixels near the shot (the search region) with
#     G >= 215                                   (yellow core, white-yellow head, light green, cyan-white: the bright part of everything),
#   or G >= 170 and R <= 130 and G >= B - 30      (the teal band of the tail, down to about 3/4 of its peak brightness: the dark teal fade
#                                                  and the blue glow around it are below this and stay out),
#   or amber ring of the head: R >= 235, G >= 150, B <= 75,
#   or the clip's own body pixels (ID-pass code 1, coverage >= 128: the amber ring of the head etc.),
# connected (8-neighbour) to the old label / the beam's own core shapes, then closed (disc 3) on pixels that are not dark, holes filled.
# The old label is only kept where it agrees with the new one (pixels of the old label that are not bright are NOT kept = tighter), plus the
# hidden-parts fill of z_1114_cut5 (old label inside the predicted body). Not included (owner: soft glow / halo stays out): dark teal fade,
# blue glow column, the orange-brown crescent in front of the head, the leaf, the hit star.
import numpy as _bm5_np
from scipy.ndimage import label as _bm5_cc, binary_closing as _bm5_bc, binary_fill_holes as _bm5_fh, binary_opening as _bm5_bo

_bm5_T = 6001306
_bm5_prev = POST_BODY.get(_bm5_T)
_bm5_CORE = (47, 52, 63, 136)       # the beam's own yellow core pieces (shape ids from shapediag on this capture)
_bm5_CLOSE = 3


def _bm5_disc(r):
    y, x = _bm5_np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r + r


def _bm5_beam(body, ctx):
    if _bm5_prev is not None: body = _bm5_prev(body, ctx)
    if not body.any(): return body
    T = projtab.table()[_bm5_T]
    names = tuple(n for n in (T['blue'], T['red']) if n)
    SP = _a4_shape_pixels(ctx['R'], T['file'], names) - 1
    region = ctx['region']; cv = ctx['cv']
    im = ctx['img'].astype(_bm5_np.int16); r, g, b = im[..., 0], im[..., 1], im[..., 2]
    bright = (g >= 215) | ((g >= 170) & (r <= 130) & (g >= b - 30)) | ((r >= 235) & (g >= 150) & (b <= 75))
    core = _bm5_np.isin(SP, _bm5_CORE) & (cv >= 64)
    clipbody = (ctx['C'] == 1) & (cv >= CUT)                       # the clip's own body pixels as the labeller / cutcheck count them
    V = (bright & region) | (core & region) | (clipbody & region)
    # the little leaf sprites flying beside the beam (lime / orange, shape 133 too) are separate draws that carry none of the core shapes:
    # their pixels are out (never the draw that holds most of the bright pixels = the tail)
    ih = ctx['ids_h']; leafpx = _bm5_np.zeros_like(V)
    cores = _bm5_np.isin(SP, _bm5_CORE) | (SP == 135)
    cnt = {int(d): int((V & (ih == d)).sum()) for d in _bm5_np.unique(ih[V]) if d > 0}
    top = max(cnt, key=cnt.get) if cnt else None
    for d in cnt:
        if d == top: continue
        m = (ih == d) & region
        if (m & (SP == 133)).sum() >= 10 and not (m & cores).any(): leafpx |= m
    V &= ~leafpx
    # the brawler / scenery in front of or behind the beam (draws that carry no beam shape at all: the dress highlights at the muzzle)
    # are not beam pixels; interior pixels shown through additive layers are recovered by the closing / hole fill below
    own_draw = _bm5_np.zeros_like(V)
    for d in _bm5_np.unique(ih[region]):
        if d <= 0: continue
        m = (ih == d) & region
        if (m & (SP >= 0)).sum() >= 8: own_draw |= m
    strong = (g >= 215)
    V &= (own_draw | ((ih == 0) & strong))      # soft teal / amber pixels only from the beam's own draws (not the lit water)
    V &= strong | clipbody | ((SP >= 0) & (SP != 0))       # ... and not from its wide glow layer (shape 0), which only counts where it is bright
    seed = body | core
    L, n = _bm5_cc(V | body, structure=_bm5_np.ones((3, 3), bool))
    keep = _bm5_np.unique(L[seed & V]); keep = keep[keep > 0]
    out = _bm5_np.isin(L, keep) & V
    if not out.any(): return body
    ys, xs = _bm5_np.nonzero(out); pad = _bm5_CLOSE + 2
    y0, y1, x0, x1 = max(0, ys.min() - pad), ys.max() + pad + 1, max(0, xs.min() - pad), xs.max() + pad + 1
    sl = (slice(y0, y1), slice(x0, x1))
    sub = _bm5_np.pad(out[sl], _bm5_CLOSE)
    cl = _bm5_bc(sub, structure=_bm5_disc(_bm5_CLOSE))[_bm5_CLOSE:-_bm5_CLOSE, _bm5_CLOSE:-_bm5_CLOSE]
    res = out.copy(); res[sl] |= cl & (g[sl] >= 100) & ~leafpx[sl]      # close notches, never over dark pixels or the leaf
    res[sl] = _bm5_fh(res[sl])
    pb = ctx.get('pred_body')
    if pb is not None and pb.any(): res |= body & pb          # hidden-parts fill (z_1114_cut5) stays
    return res


POST_BODY[_bm5_T] = _bm5_beam
