# Fix (round 3, 2026-10-28, skin 008 RosaRosaBrawloween, pairs 172 / 174 / 176 / 177, attack 0 deg, ticks 265-275; the
# 90 deg row of the same type is improved too). Type 6000585 = Rosa002Projectile (a 3D mesh of three fist-lumps, one object
# and one draw per lump; the lumps are separate labels).
# Cause: each lump's mesh has a translucent trailing part (motion streaks drawn in the same draw, coverage fading 255 -> 0
# toward the tail, saw-tooth geometry). The label took coverage >= 128 and so followed the streaks (spurs, and at t273 a
# reach over the brawler's hand). Two earlier fixes (coverage threshold alone, then threshold + closing/opening of the
# union) removed the spurs but cut a notch into the body at t273.
# Fix: per lump (per object = per draw) the solid core = its pixels with coverage >= min(234, 0.92 * p95); thin spurs of the
# core opened away (disc 4, only when that keeps >= 50% of it), holes filled, largest part (>= 15% of it) kept, and then
# the CONVEX HULL of that core as the lump's label: a fist lump is convex, so the hull is its own outline and cannot have
# notches / holes. Pixels are still given to one owner each by the pipeline (nearest predicted label). The label is never
# emptied (when < 60% of the old label would stay it is left unchanged).
import numpy as _np6000585
_T6000585 = 234
_KEEP6000585 = 0.6
_OP6000585 = 3
_RIM6000585 = 2
_CLU6000585 = 4


def _disc6000585(r):
    y, x = _np6000585.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r + 0.5


def _parts6000585(m, frac=0.15):
    from scipy.ndimage import label as _cc
    L, n = _cc(m)
    if n > 1:
        sz = _np6000585.bincount(L.ravel())[1:]
        m = _np6000585.isin(L, 1 + _np6000585.nonzero(sz >= frac * sz.max())[0])
    return m


def _fix_6000585(body, ctx):
    from scipy.ndimage import binary_fill_holes, binary_opening, binary_dilation
    cv = ctx['cv']
    if not body.any(): return body
    q = float(_np6000585.percentile(cv[body], 95))
    T = min(_T6000585, 0.92 * q)
    keep = body & (cv >= T)
    if keep.sum() < _KEEP6000585 * body.sum(): return body
    keep = binary_fill_holes(_parts6000585(keep))
    pad = _OP6000585 + 2
    op = binary_opening(_np6000585.pad(keep, pad), _disc6000585(_OP6000585))[pad:-pad, pad:-pad]
    if op.sum() >= 0.5 * keep.sum():
        h = hull_mask(_parts6000585(op, 0.5))   # core without thin spurs; solid pixels outside its hull (spurs) dropped
        keep = keep & h
    out = hull_mask(keep)
    # thin pointed tips of the hull (a sliver of the tail) opened away; the rounded body stays
    o2 = _parts6000585(binary_opening(_np6000585.pad(out, pad), _disc6000585(_OP6000585))[pad:-pad, pad:-pad], 0.5)
    if o2.sum() >= 0.85 * out.sum(): out = hull_mask(o2)
    # the antialiased rim (coverage 128..233) just outside the hull belongs to the lump: back in, at most _RIM6000585 px wide
    out = out | (binary_dilation(out, _disc6000585(_RIM6000585)) & body)
    if out.sum() < _KEEP6000585 * body.sum(): return body
    return out


POST_BODY[6000585] = _fix_6000585

# After the pipeline's one-owner-per-pixel step (lumps' hulls overlap each other) a lump label can be left with 1-3 px
# crumbs: those separate bits (< 25 px) of this type's labels are dropped again.
_prev6000585 = labels


def labels(objs, cam, R, state=None, ended=(), img=None):
    from scipy.ndimage import label as _cc
    import bulletlabels as BL
    P = _prev6000585(objs, cam, R, state=state, ended=ended, img=img)
    ids, lids, share = P['ids'], P['lids'], P['share']
    mine = {BL._lid(o) for o in objs if o['t'] == 6000585}
    # seams / small gaps between the lumps of one shot (hulls rounded, a gap left at a corner): the union of this type's
    # labels is closed (radius _CLU6000585), new pixels (never another label's) go to the nearest lump
    ks = [k for k in _np6000585.unique(ids[ids >= 0]) if lids[k][2] == 'body' and lids[k][0] in mine]
    if len(ks) >= 2:
        from scipy.ndimage import binary_closing, distance_transform_edt
        U = _np6000585.isin(ids, ks)
        ys, xs = _np6000585.nonzero(U); pd = _CLU6000585 + 2
        y0, y1, x0, x1 = max(0, ys.min() - pd), ys.max() + pd + 1, max(0, xs.min() - pd), xs.max() + pd + 1
        u = U[y0:y1, x0:x1]
        cl = binary_closing(_np6000585.pad(u, pd), _disc6000585(_CLU6000585))[pd:-pd, pd:-pd]
        new = cl & ~u & (ids[y0:y1, x0:x1] < 0)
        if new.any():
            _, (iy, ix) = distance_transform_edt(~u, return_indices=True)
            sid = ids[y0:y1, x0:x1]; ssh = share[y0:y1, x0:x1]
            sid[new] = sid[iy[new], ix[new]]; ssh[new] = BL.CUT
    for k in _np6000585.unique(ids[ids >= 0]):
        if lids[k][2] != 'body' or lids[k][0] not in mine: continue
        L, n = _cc(ids == k)
        if n < 2: continue
        sz = _np6000585.bincount(L.ravel())
        for j in range(1, n + 1):
            if sz[j] < 25 and sz[j] < 0.5 * sz[1:].max():
                ids[L == j] = -1; share[L == j] = 0
    return P
