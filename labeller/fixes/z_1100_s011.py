# human review 2026-10-28 fix for skin 011 (Haiku fix, reviewed by the coordinator, installed 10-29; source out/sweep2/haikutasks/human1028/s011/fix2.py; tested via IDLABELS_OVERRIDE after all fixes incl. z_1099_cleanup, hence z_1100_)
# Fix 2 (human review 2026-10-28, skin 011 RosaMantis, attack 0 deg ticks 265, 269, 275; coordinator review of fix.py).
# Type 6000588 = Rosa005Projectile: a 3D mesh, one object and one draw per lump, three lumps per shot; its translucent
# motion blur is drawn in the same draws, trailing to the left (against the motion) on attack 0.
# Problem with fix.py: its left edge followed the blur's teeth (a zigzag), which the user rejects as jagged.
# Fix: the solid body of each row is the reference on the trailing side: blur pixels (coverage < 128) that lie left of the
# first solid pixel of their row are cut back, so the left edge is the lump's solid body. Then the outline is smoothed with
# a larger kernel (Gaussian sigma _SIG6000588) so the edge is one clean curve. Growth only into this type's own mesh pixels
# (coverage >= _CV6000588), never background or another object. Top and leading sides (attack 90 tops, the bright rims) keep
# the blur of fix.py. No convex hull. Scope: keyed to type 6000588 only.
import numpy as _np6000588
_CV6000588 = 24     # coverage from which a mesh pixel may join the shape (the blur included)
_SOLID6000588 = 128  # coverage of the solid body: the reference for the trailing side
_NEAR6000588 = 10   # px: candidates lie this close to a label of the type
_R6000588 = 7       # px: closing radius (notches and seams between the lumps)
_O6000588 = 2       # px: opening radius (teeth of the added blur)
_HOLE6000588 = 80   # px: enclosed holes up to this area are filled
_SIG6000588 = 5.0   # px: Gaussian sigma of the trailing-side outline smoothing
_ZONE6000588 = 12   # px: the trailing-side band: from the row's solid left edge inwards
_prev6000588 = labels


def _disc6000588(r):
    y, x = _np6000588.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r + 0.5


def labels(objs, cam, R, state=None, ended=(), img=None):
    from scipy.ndimage import binary_closing, binary_fill_holes, binary_opening, binary_dilation, distance_transform_edt
    from scipy.ndimage import gaussian_filter, label as _cc6000588
    import bulletlabels as BL
    P = _prev6000588(objs, cam, R, state=state, ended=ended, img=img)
    ids, lids, share = P['ids'], P['lids'], P['share']
    mine = {BL._lid(o) for o in objs if o['t'] == 6000588}
    ks = [k for k in _np6000588.unique(ids[ids >= 0]) if lids[k][2] == 'body' and lids[k][0] in mine]
    if not ks: return P
    U0 = _np6000588.isin(ids, ks)
    if not U0.any(): return P
    ids_h = R['id'][::2, ::2]; cv_h = R['cv'][::2, ::2]
    # this type's own draws: the draws of which at least half of the pixels are labelled as this type
    cnt_all = _np6000588.bincount(ids_h.ravel()); cnt_u = _np6000588.bincount(ids_h[U0].ravel(), minlength=len(cnt_all))
    own_draws = _np6000588.nonzero((cnt_u >= 0.5 * cnt_all) & (cnt_u > 0))[0]; own_draws = own_draws[own_draws > 0]
    if not own_draws.size: return P
    ys, xs = _np6000588.nonzero(U0); pd = _R6000588 + _NEAR6000588 + 6
    y0, y1 = max(0, ys.min() - pd), min(ids.shape[0], ys.max() + pd + 1)
    x0, x1 = max(0, xs.min() - pd), min(ids.shape[1], xs.max() + pd + 1)
    u0 = U0[y0:y1, x0:x1]; idc = ids[y0:y1, x0:x1]; sh = share[y0:y1, x0:x1]
    cvc = cv_h[y0:y1, x0:x1]
    mesh = _np6000588.isin(ids_h[y0:y1, x0:x1], own_draws) & (cvc >= _CV6000588)
    other = (idc >= 0) & ~u0
    near = binary_dilation(u0, _disc6000588(_NEAR6000588))
    cand = (mesh & near & ~other) | u0
    # closing and hole fill of the candidates (notches and seams between the lumps), only mesh pixels join
    S = binary_closing(_np6000588.pad(cand, _R6000588 + 2), _disc6000588(_R6000588))[_R6000588 + 2:-_R6000588 - 2, _R6000588 + 2:-_R6000588 - 2]
    S = (S | binary_fill_holes(cand)) & (cand | (S & mesh))
    # the added blur only is opened (its teeth); the labelled body is kept
    O = binary_opening(_np6000588.pad(S, _O6000588 + 2), _disc6000588(_O6000588))[_O6000588 + 2:-_O6000588 - 2, _O6000588 + 2:-_O6000588 - 2]
    S = u0 | (O & ~other)
    # small enclosed holes of this type's own mesh are filled
    hol = binary_fill_holes(S) & ~S
    if hol.any():
        L, n = _cc6000588(hol); sz = _np6000588.bincount(L.ravel())
        mh = _np6000588.isin(ids_h[y0:y1, x0:x1], own_draws)
        mine_frac = _np6000588.bincount(L.ravel(), weights=mh.ravel().astype(float), minlength=len(sz)) / _np6000588.maximum(sz, 1)
        ok = (sz <= _HOLE6000588) & ((mine_frac >= 0.5) | (sz <= 3)) & (_np6000588.arange(len(sz)) > 0)
        S = S | _np6000588.isin(L, _np6000588.nonzero(ok)[0])
    # smoothing as in fix.py (sigma 3) everywhere: the top fur and the leading rims keep their shape
    G3 = gaussian_filter(S.astype(_np6000588.float32), 3.0) > 0.5
    grow3 = G3 & ~S & mesh
    drop3 = S & ~G3 & ~u0
    S = (S & ~drop3) | grow3
    # trailing side: per row, the blur left of the row's first solid pixel (coverage < _SOLID6000588) is cut back to the body
    solid = u0 | (mesh & (cvc >= _SOLID6000588))
    rows = solid.any(axis=1)
    big = _np6000588.iinfo(_np6000588.int64).max // 4
    xl = _np6000588.where(rows, solid.argmax(axis=1), big)
    xr = _np6000588.where(rows, solid.shape[1] - 1 - solid[:, ::-1].argmax(axis=1), -big)
    cols = _np6000588.arange(S.shape[1])[None, :]
    trail = (cols < xl[:, None]) & (cvc < _SOLID6000588) & ~u0
    S = S & ~trail
    # in a band of _ZONE6000588 px from the solid left edge of each row, the outline is smoothed with the larger
    # kernel: one clean curve, not the blur's teeth. Growth only into this type's own mesh pixels.
    Z = rows[:, None] & (cols < (xl + _ZONE6000588)[:, None])
    G5 = gaussian_filter(S.astype(_np6000588.float32), _SIG6000588) > 0.5
    S = (S & ~(Z & S & ~G5)) | (Z & G5 & ~S & mesh)
    if S.sum() < 0.6 * u0.sum(): return P           # never shrink a label away: leave the frame as the pipeline had it
    # pixels that left the shape: unlabelled; new pixels: the nearest lump's id
    orig = idc.copy()
    gone = u0 & ~S
    idc[gone] = -1; sh[gone] = 0
    new = S & ~u0 & ~other
    if new.any():
        _, (iy, ix) = distance_transform_edt(~u0, return_indices=True)
        idc[new] = orig[iy[new], ix[new]]; sh[new] = BL.CUT
    ids[y0:y1, x0:x1] = idc; share[y0:y1, x0:x1] = sh
    return P
