# sweep 10-27 fix for skin 431 (Sonnet revision, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s431/)
# Poco Surfer Overcharge 431 (sweep 2026-10-27, task s431, pair 8119, second attempt): the super disc's thick pale-yellow outer rim
# was left outside the label (sw2_431 super 45 deg, ticks 349 / 351 / 353, the upper-right (front) disc: the outline sat at the
# outer edge of the dark grey ring, the whole pale rim outside; the lower discs of those frames are outlined to their rim).
# This file REPLACES the first attempt (fix.py): it does not use the thresholded prediction window, nor the ID-pass shapes.
#
# Cause: the super disc (type 6000354, Poco010UltiProjectile, skin 431 only) is labelled from its predicted clip polygons
# (z_1023_poco431 _p431_raster, shapes 8 / 21 / 22 / 23 / 24, then z_1026 block cut). The thick pale rim of the front disc
# (a flat pale-yellow ring R = G = 255, drawn around the dark ring, with a pointed ear on it) is not part of any predicted
# clip polygon at ticks 349-353 (at 349 / 351 it is not even in the ID pass: additive), and the predicted disc is also at
# another scale than the drawn one (the rim lies at 1.17-1.42 x the predicted ellipse at 349, 1.31-1.55 x at 351, 1.12-1.37 x at
# 353), so neither a fixed window around the prediction nor the drawn shapes can reach the rim.
#
# Fix (only type 6000354; the attack 6000353 and all other types are untouched): MEASURE the rim in the frame.
#   - the disc's ellipse shape and orientation come from its prediction (covariance of the predicted disc); rays are cast from its
#     centre in that ellipse's own normalised coordinates (so everything below scales with the disc, any zoom);
#   - on each ray the rim is the first pale-yellow run (G >= 150, R >= 150, R - B >= 20: the rim's colour, not the bluish stone
#     pillars, nor the cyan wedge) starting between 1.0 and 1.7 x the predicted ellipse; its outer end is the rim's outer edge
#     (half level between the flat rim and the dark / soft glow outside);
#   - a circle (centre + radius, in normalised coordinates, the ellipse's shape) is fitted to these outer-edge points by trimmed
#     least squares: the pointed ear, notes touching the rim, the cyan wedge and the bright pillar behind are trimmed as outliers;
#   - the fit is used only when it is clean (>= 40 % of the rays give an edge, >= 55 % of those are inliers, residual spread
#     <= 0.04, centre drift <= 0.45, radius between 1.08 and 1.9): a disc whose rim already lies inside its prediction (thin
#     rim: the lower discs, ticks 345-347) keeps its label unchanged;
#   - label = the label as before (z_1023 / z_1026) + the fitted rim ellipse, minus the nearer discs' rim ellipses (the disc in
#     front keeps its rim, as before), minus the solid map blocks in front (the z_1026 cut stays), then the z_1026 clean-up
#     (holes that are not block pixels filled, 1-px spurs opened, parts < P431_SPECK px dropped).
# The outline is therefore a clean ellipse at the rim's outer edge: no notches, no ragged steps (the first attempt's tick 351 / 353
# defects), and it never loses a pixel of the previous label.

S431_TYPE = 6000354
S431_T = 150          # pale-yellow rim: G and R at least this (half level between the flat 255 rim and the dark / glow outside)
S431_RB = 20          # ... and R - B at least this (rim: 53-104; bluish stone and cyan wedge: <= 0)
S431_START = (1.0, 1.7)   # rim starts this far out, in ellipse units
S431_NB = 144         # rays
S431_CACHE = [None, None]   # [R, {object id: fit or None}]


def _s431_fit(img, D):
    """(centre (y, x), A, Ai, cq, s): the rim = {p : |Ai (p - c) - cq| <= s}, or None when there is no clean rim outside D"""
    from scipy.ndimage import binary_fill_holes
    Df = binary_fill_holes(D)
    ys, xs = np.nonzero(Df)
    if len(ys) < 50: return None
    c = np.array([ys.mean(), xs.mean()])
    P = np.stack([ys - c[0], xs - c[1]], 1).astype(float)
    C = np.cov(P.T) * 4
    w, V = np.linalg.eigh(C)
    if w.min() <= 1e-6: return None
    A = V @ np.diag(np.sqrt(w)) @ V.T; Ai = np.linalg.inv(A)
    im = img.astype(np.int32)
    br = (im[..., 0] >= S431_T) & (im[..., 1] >= S431_T) & (im[..., 0] - im[..., 2] >= S431_RB)
    H, W = br.shape
    rr = np.arange(1.0, 2.1, 0.01)
    th = np.linspace(0, 2 * np.pi, S431_NB, endpoint=False)
    Q = []
    for a in th:
        u = np.array([np.sin(a), np.cos(a)])
        pts = c[:, None] + (A @ u)[:, None] * rr[None, :]
        y = np.round(pts[0]).astype(int); x = np.round(pts[1]).astype(int)
        ok = (y >= 0) & (y < H) & (x >= 0) & (x < W)
        b = np.zeros(len(rr), bool); b[ok] = br[y[ok], x[ok]]
        idx = np.nonzero(b & (rr >= S431_START[0]) & (rr <= S431_START[1]))[0]
        if not len(idx): continue
        i = idx[0]
        while i < len(b) - 1 and (b[i + 1] or (i + 2 < len(b) and b[i + 2])): i += 1
        e = rr[i] + 0.005
        Q.append(Ai @ (c + A @ u * e - c))            # outer-edge point in ellipse units (centred on D)
    if len(Q) < 0.4 * S431_NB: return None
    Q = np.array(Q); keep = np.ones(len(Q), bool)
    cq = np.zeros(2); s = float(np.median(np.hypot(Q[:, 0], Q[:, 1])))
    for _ in range(6):
        q = Q[keep]
        if len(q) < 12: return None
        M = np.concatenate([2 * q, np.ones((len(q), 1))], 1); rhs = (q ** 2).sum(1)
        sol = np.linalg.lstsq(M, rhs, rcond=None)[0]
        cq = sol[:2]; s2 = sol[2] + cq @ cq
        if s2 <= 0: return None
        s = float(np.sqrt(s2))
        res = np.hypot(Q[:, 0] - cq[0], Q[:, 1] - cq[1]) - s
        mad = 1.4826 * np.median(np.abs(res[keep] - np.median(res[keep])))
        keep = np.abs(res) <= max(2.5 * mad, 0.02)
    res = np.hypot(Q[:, 0] - cq[0], Q[:, 1] - cq[1]) - s
    if keep.sum() < 0.55 * len(Q) or res[keep].std() > 0.04: return None
    if np.hypot(*cq) > 0.45 or not (1.08 <= s <= 1.9): return None
    return c, A, Ai, cq, s


def _s431_ellipse(fit, shape):
    c, A, Ai, cq, s = fit
    H, W = shape
    # bounding box of the ellipse {c + A (cq + s u)}
    ctr = c + A @ cq
    ext = s * np.sqrt((A ** 2).sum(1)) + 3
    y0, y1 = max(0, int(ctr[0] - ext[0])), min(H, int(ctr[0] + ext[0]) + 1)
    x0, x1 = max(0, int(ctr[1] - ext[1])), min(W, int(ctr[1] + ext[1]) + 1)
    m = np.zeros(shape, bool)
    if y1 <= y0 or x1 <= x0: return m
    yy, xx = np.mgrid[y0:y1, x0:x1]
    q = np.einsum('ij,jhw->ihw', Ai, np.stack([yy - c[0], xx - c[1]]).astype(float)) - cq[:, None, None]
    m[y0:y1, x0:x1] = (q ** 2).sum(0) <= s * s
    return m


def _s431_fits(ctx):
    """{object id: (rim ellipse mask or None)}: the frame's fits, once per frame"""
    R = ctx['R']
    if S431_CACHE[0] is R: return S431_CACHE[1]
    img = ctx.get('img'); out = {}
    for oid, (dz, D) in DISC431.items():
        out[oid] = None
        if img is None or not D.any(): continue
        try:
            f = _s431_fit(img, D)
        except Exception:
            f = None
        if f is not None: out[oid] = _s431_ellipse(f, D.shape)
    S431_CACHE[0] = R; S431_CACHE[1] = out
    return out


def _s431_super_rim(body, ctx):
    m = _S431_PREV[0](body, ctx)                              # the disc as before (z_1023 / z_1026)
    o = ctx['o']
    if not m.any() or o['id'] not in DISC431: return m
    fits = _s431_fits(ctx)
    E = fits.get(o['id'])
    if E is None or not E.any(): return m
    from scipy.ndimage import binary_fill_holes
    dz0 = DISC431[o['id']][0]
    near = np.zeros_like(m)
    for oid, (dz2, D2) in DISC431.items():                    # nearer discs keep their own rim
        if oid != o['id'] and dz2 < dz0:
            e2 = fits.get(oid)
            near |= e2 if e2 is not None else binary_fill_holes(D2)
    cut = _t38_block_front(ctx)
    c = m | (E & ~near & ~cut)
    return _t38_tidy(c, cut) if c.any() else m


_S431_PREV = [POST_BODY[S431_TYPE]]
POST_BODY[S431_TYPE] = _s431_super_rim
