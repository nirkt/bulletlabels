# Buzz SB 376 attack: one fitted disc / ellipse per bubble (user 10-23; Haiku task 12, reviewed)
# task 12 (2026-10-23, v6): RopeDude 376 (Buzz SB), attack type 6000998 (Buzz008Projectile, a wave of bubbles fired one after
# another; trail clip buzz_008_atk_projectile). The user's decision: each flying bubble ONE clean solid disc label, never
# split; no slivers, specks or pieces of popping bubbles (a popping bubble is an end animation: no label); small floating
# bubbles, splashes and pop stars are not labelled. Super untouched (other type). Replaces agent 6's POST_BODY[6000998].
# Coordinator review (2026-10-23): every bubble is a disc: a least-squares circle fitted to the bubble's own rim (shape
# 829), the label is that disc; a bubble partly hidden by an opaque object in front is the disc minus that object's pixels.
#
# Steps (the group label is the union over its members; each member computes the same result):
#  1. body without the liquid splash (shapes 407 / 35339 of liquid_atk_*, where no bubble shape is drawn) and without the
#     small floating bubbles' rims (shape 828 of jessie_008_atk_bubble_*)
#  2. the bubbles are found by a partition: the body closed by _T12_CLOSE px, one marker per bubble (its inner fill, shape
#     47, or a distance-transform peak far from every fill), watershed on the distance transform (touching bubbles split at
#     their necks). A glint (shape 137) on a bubble's centre hides part of its rim: its bubble is found from the glint
#     instead (a glint whose disc holds popping pixels is a pop star: no label)
#  3. each bubble: a least-squares circle (Kasa fit) to its rim pixels (shape 829, frame-wide, within the bubble's own
#     partition zone), inliers within _T12_FIT_TOL px, refitted; the disc is the label
#  4. a disc is cut by an opaque object in front of the bubble: pixels whose front draw is opaque and not a bubble body draw
#     and whose depth is nearer than the bubble's own opaque body (by DEPTH_TOL + 0.1)
#  5. dropped: basins under _T12_MIN_AREA px; basins with fewer than _T12_FIT_RIM rim pixels or fewer than _T12_FIT_SECT
#     of 8 sectors of rim (no reliable fit); fits with a radius outside [5, 22] px or a median residual over _T12_FIT_RES px;
#     any disc or basin holding a pop flash (shapes 577 / 6671 / 92: the end animation)
_T12_BUZZ = dict(file='sc/effects_brawler.sc', bubble=('buzz_008_atk_projectile',),
                 liquid=('liquid_atk_R', 'liquid_atk_L'), small=('jessie_008_atk_bubble_01', 'jessie_008_atk_bubble_02',
                                                              'jessie_008_atk_bubble_03'))
_T12_MIN_AREA = 60     # px (frame): the visible part of a bubble is at least this
_T12_FIT_RIM = 20      # px: a bubble needs at least this many rim pixels (shape 829) in its own zone to be fitted
_T12_FIT_SECT = 4      # of 8 sectors of 45 degrees: the rim must cover at least this many (half the circle)
_T12_FIT_TOL = 2.0     # px: rim pixels this close to the fitted circle are its inliers
_T12_FIT_RES = 1.5     # px: the median residual of the inliers must stay below this
_T12_OCC_MIN = 15    # px: an object in front cuts a disc only as a part at least this large touching its outline
_T12_FIT_FRAC = 0.6    # the share of the zone's rim that must lie on the fitted circle (ellipses fail: not a circle)
_T12_CLOSE = 8         # px: closing radius of the partition (a bubble's broken rim bridged)
_T12_CORE_MIN = 4      # px: a fill core smaller than this is no marker
_T12_STAR_R = 18       # px: rim pixels this far from a glint's centre belong to its bubble
_T12_STAR_DRIFT = 5    # px: the glint bubble's fitted centre stays this close to the glint's centre
_T12_LOG = None        # debugging: a list receives per-bubble decisions (object, basin, px, rim px, fit, kept)


def _t12_ws(M, markers, elev):
    """priority-flood watershed of the mask M from the marker labels (lowest elevation first)"""
    import heapq
    H, W = M.shape; lab = markers.astype(np.int32).copy(); pq = []; cnt = 0
    for y, x in zip(*np.nonzero(lab)):
        pq.append((elev[y, x], cnt, y, x)); cnt += 1
    heapq.heapify(pq)
    while pq:
        _, _, y, x = heapq.heappop(pq); l = lab[y, x]
        for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= yy < H and 0 <= xx < W and M[yy, xx] and lab[yy, xx] == 0:
                lab[yy, xx] = l; heapq.heappush(pq, (elev[yy, xx], cnt, yy, xx)); cnt += 1
    return lab


def _t12_peaks(dt, h=2.0, dmin=3.0):
    """(y, x, value) of the h-maxima of the distance transform dt (one per plateau, at its deepest point)"""
    from scipy.ndimage import grey_dilation, label as cc
    rec = dt - h
    while True:
        nxt = np.minimum(grey_dilation(rec, size=(3, 3)), dt)
        if np.array_equal(nxt, rec): break
        rec = nxt
    mx = (rec >= grey_dilation(rec, size=(3, 3))) & (dt >= dmin)
    L, n = cc(mx); out = []
    for k in range(1, n + 1):
        ys, xs = np.nonzero(L == k); i = int(np.argmax(dt[ys, xs]))
        out.append((int(ys[i]), int(xs[i]), float(dt[ys[i], xs[i]])))
    return out


def _t12_circle(y, x):
    """least-squares circle through the points (Kasa fit): centre (cy, cx), radius"""
    A = np.c_[x, y, np.ones(len(x))]; b = x * x + y * y
    a, bb, c = np.linalg.lstsq(A, b, rcond=None)[0]
    cx, cy = a / 2.0, bb / 2.0
    return cy, cx, float(np.sqrt(max(c + cx * cx + cy * cy, 0.0)))


def _t12_sectors(y, x, cy, cx):
    return len(np.unique(((np.arctan2(y - cy, x - cx) + np.pi) / (np.pi / 4)).astype(int) % 8))


def _t12_ellipse(y, x):
    """least-squares conic (A x^2 + B xy + C y^2 + D x + E y + F = 0) through the points, as an ellipse:
    (x0, y0, Mq, rg) with x^T Mq x = 1 about the centre (x0, y0); rg = geometric mean of the semi-axes. None if no ellipse.
    The points are normalised first (the conic fit is ill-conditioned in raw pixel coordinates)."""
    mx, my = x.mean(), y.mean(); sc = max(float(np.std(np.r_[x - mx, y - my])), 1e-6)
    xn = (x - mx) / sc; yn = (y - my) / sc
    M = np.c_[xn * xn, xn * yn, yn * yn, xn, yn, np.ones(len(xn))]
    _, _, Vt = np.linalg.svd(M, full_matrices=False)
    A, B, C, D, E, F = Vt[-1]
    den = B * B - 4 * A * C
    if den >= 0: return None
    x0 = (2 * C * D - B * E) / den; y0 = (2 * A * E - B * D) / den
    F0 = F + D * x0 / 2 + E * y0 / 2
    s = -F0
    if s == 0: return None
    if s < 0: A, B, C, s = -A, -B, -C, -s
    Mn = np.array([[A, B / 2], [B / 2, C]]) / s
    lam = np.linalg.eigvalsh(Mn)
    if np.any(lam <= 0): return None
    semi = 1 / np.sqrt(lam)
    Mq = Mn / (sc * sc)                                   # back to frame px: x^T Mq x = 1 about the centre (in frame px)
    return x0 * sc + mx, y0 * sc + my, Mq, float(np.sqrt(semi[0] * semi[1]) * sc)


def _t12_ell_dist(E, y, x):
    """approximate distance (px) of points from the ellipse E: (sqrt(x^T Mq x) - 1) * rg"""
    x0, y0, Mq, rg = E; qx = x - x0; qy = y - y0
    v = Mq[0, 0] * qx * qx + 2 * Mq[0, 1] * qx * qy + Mq[1, 1] * qy * qy
    return (np.sqrt(np.maximum(v, 0)) - 1) * rg


def _t12_fit(own_rim_y, own_rim_x, zone_y, zone_x):
    """a bubble's disc from its rim. Circle first: a least-squares circle through the own rim pixels (basin), inliers within
    _T12_FIT_TOL px, refitted; then through the zone's rim pixels on it. A circle is accepted when at least _T12_FIT_FRAC of
    the rim within R + 6 px of its centre lies on it, with radius in [5, 22] and a median residual <= _T12_FIT_RES. Otherwise
    an ellipse (the same steps, a conic fit: a bubble in perspective) is tried, with aspect <= 2.2. Returns (spec, reason):
    spec ('c', cy, cx, R) or ('e', E), reason None when reliable."""
    if len(own_rim_y) < _T12_FIT_RIM: return None, 'rim %d < %d' % (len(own_rim_y), _T12_FIT_RIM)
    py, px = own_rim_y.astype(float), own_rim_x.astype(float)
    zy, zx = zone_y.astype(float), zone_x.astype(float)
    # circle
    cy, cx, R = _t12_circle(py, px); reason = None
    for _ in range(4):
        ok = np.abs(np.hypot(py - cy, px - cx) - R) <= _T12_FIT_TOL
        if ok.sum() < _T12_FIT_RIM: reason = 'inliers %d' % ok.sum(); break
        cy, cx, R = _t12_circle(py[ok], px[ok])
    if reason is None:
        dz_ = np.hypot(zy - cy, zx - cx); ok = np.abs(dz_ - R) <= _T12_FIT_TOL
        if ok.sum() < _T12_FIT_RIM: reason = 'zone inliers %d' % ok.sum()
        else:
            cy, cx, R = _t12_circle(zy[ok], zx[ok])
            dz_ = np.hypot(zy - cy, zx - cx); near_ = dz_ <= R + 6; ok = np.abs(dz_ - R) <= _T12_FIT_TOL
            frac = ok[near_].sum() / max(1, near_.sum())
            res = float(np.median(np.abs(dz_[ok] - R)))
            if not (5 <= R <= 22): reason = 'radius %.1f' % R
            elif frac < _T12_FIT_FRAC: reason = 'circle: %.2f of rim on it' % frac
            elif res > _T12_FIT_RES: reason = 'circle: residual %.2f' % res
            else: return ('c', cy, cx, R), None
    # ellipse
    E = _t12_ellipse(py, px)
    if E is None: return None, 'no ellipse (%s)' % reason
    for _ in range(4):
        ok = np.abs(_t12_ell_dist(E, py, px)) <= _T12_FIT_TOL
        if ok.sum() < _T12_FIT_RIM: return None, 'ellipse inliers %d (%s)' % (ok.sum(), reason)
        E = _t12_ellipse(py[ok], px[ok])
        if E is None: return None, 'ellipse degenerate (%s)' % reason
    dz_ = _t12_ell_dist(E, zy, zx); ok = np.abs(dz_) <= _T12_FIT_TOL
    if ok.sum() < _T12_FIT_RIM: return None, 'ellipse zone inliers %d' % ok.sum()
    E = _t12_ellipse(zy[ok], zx[ok])
    if E is None: return None, 'ellipse degenerate (zone)'
    dz_ = _t12_ell_dist(E, zy, zx); ok = np.abs(dz_) <= _T12_FIT_TOL
    near_ = np.hypot(zx - E[0], zy - E[1]) <= 3 * E[3] + 6     # E[0]: centre x, E[1]: centre y
    frac = ok[near_].sum() / max(1, near_.sum())
    res = float(np.median(np.abs(dz_[ok])))
    lam = np.linalg.eigvalsh(E[2]); semi = 1 / np.sqrt(lam)
    if not (5 <= semi.max() <= 22) or semi.max() / semi.min() > 2.2: return None, 'ellipse axes %.1f/%.1f' % (semi.max(), semi.min())
    if frac < _T12_FIT_FRAC: return None, 'ellipse: %.2f of rim on it (%s)' % (frac, reason)
    if res > _T12_FIT_RES: return None, 'ellipse: residual %.2f' % res
    return ('e', E), None


def _t12_mask(spec, yy, xx):
    """the disc (or ellipse) of a spec as a full-frame mask, with 0.5 px of rim outside its outline"""
    if spec[0] == 'c':
        _, cy, cx, R = spec; return np.hypot(yy - cy, xx - cx) <= R + 0.5
    x0, y0, Mq, rg = spec[1]; qx = xx - x0; qy = yy - y0
    v = Mq[0, 0] * qx * qx + 2 * Mq[0, 1] * qx * qy + Mq[1, 1] * qy * qy
    return np.sqrt(np.maximum(v, 0)) * rg <= rg + 0.5


def _t12_glint_discs(star, rim, shape):
    """one bubble per glint: the glint's centre is the bubble's centre (start point); a circle is fitted to the rim pixels
    near it (outliers beyond _T12_FIT_TOL dropped, refitted); the centre stays within _T12_STAR_DRIFT px of the glint.
    Returns [(cy, cx, R)]."""
    from scipy.ndimage import label as cc
    from scipy.ndimage import binary_dilation
    L, n = cc(binary_dilation(star, iterations=2)); out = []
    ry, rx = np.nonzero(rim)
    for k in range(1, n + 1):
        gy, gx = np.nonzero((L == k) & star)
        if len(gy) < 10: continue
        c0y, c0x = gy.mean(), gx.mean(); cy, cx, R = c0y, c0x, 0.0
        near = np.hypot(ry - c0y, rx - c0x) <= _T12_STAR_R
        for _ in range(4):
            if near.sum() < 8: break
            py, px = ry[near].astype(float), rx[near].astype(float)
            cy, cx, R = _t12_circle(py, px)
            ok = np.abs(np.hypot(py - cy, px - cx) - R) <= _T12_FIT_TOL
            if ok.sum() < 8: break
            cy, cx, R = _t12_circle(py[ok], px[ok])
            near = np.hypot(ry - c0y, rx - c0x) <= _T12_STAR_R
        if np.hypot(cy - c0y, cx - c0x) > _T12_STAR_DRIFT or not (5 <= R <= 22): continue
        near = np.hypot(ry - cy, rx - cx) <= _T12_STAR_R
        ok = np.abs(np.hypot(ry[near] - cy, rx[near] - cx) - R) <= _T12_FIT_TOL
        if ok.sum() < _T12_FIT_RIM or _t12_sectors(ry[near][ok], rx[near][ok], cy, cx) < _T12_FIT_SECT: continue
        out.append((cy, cx, R))
    return out


def _t12_buzz_sb(body, ctx):
    if not body.any(): return body
    from scipy.ndimage import binary_closing, binary_dilation, binary_fill_holes, distance_transform_edt, label as cc
    R, region = ctx['R'], ctx['region']; F = _T12_BUZZ['file']
    SB = _a4_shape_pixels(R, F, _T12_BUZZ['bubble']); SL = _a4_shape_pixels(R, F, _T12_BUZZ['liquid'])
    SJ = _a4_shape_pixels(R, F, _T12_BUZZ['small'])
    def has(S, sids): return np.isin(S, [s + 1 for s in sids])
    liq = has(SL, (407, 35339)) & (SB == 0)          # the liquid's shapes also show inside a bubble: only where no bubble shape
    m = body & region & ~liq & ~has(SJ, (828,))
    pop = binary_dilation(has(SB, (577, 6671, 92)), iterations=3)       # frame-wide: a pop flash anywhere on a bubble counts
    star = has(SB, (137,))                                              # the glints, frame-wide (a glint can lie outside the region)
    m = m & ~has(SB, (137,))
    rim_all = has(SB, (829,))                                           # the whole rim, frame-wide
    H, W = m.shape; yy, xx = np.mgrid[0:H, 0:W]
    # occlusion: the opaque draws (blend 0) of the bubbles' own body vs the opaque objects in front (nearer)
    ids_h = R['id'][::2, ::2]; cv = R['cv'][::2, ::2]; wd = R.get('wd')
    opaque = [d for d, r in R['draws'].items() if r.get('blend') == 0]
    op_front = np.isin(ids_h, opaque) & (cv >= CUT)
    def occlude(disc):
        # the bubble's own body is the opaque draw covering most of the disc; an opaque draw in the disc at least
        # DEPTH_TOL + 0.1 nearer than that body is an object in front (a crate, a wall). Only nearer parts that touch the
        # disc's outline (3 px band) and are >= _T12_OCC_MIN px are cut (a crate enters from outside); then the disc keeps its
        # largest connected part (no fragments left by the cut)
        if wd is None: return disc
        oi = disc & op_front
        if oi.sum() < 10: return disc
        dvals, dcnt = np.unique(ids_h[oi], return_counts=True)
        body_d = dvals[np.argmax(dcnt)]
        dz = float(np.median(wd[oi & (ids_h == body_d)]))
        cand = disc & op_front & (wd < dz - DEPTH_TOL - 0.1)
        if not cand.any(): return disc
        from scipy.ndimage import binary_erosion, label as cc
        edge = disc & ~binary_erosion(disc, iterations=3)
        L, n = cc(cand); cut = np.zeros(disc.shape, bool)
        for k_ in np.unique(L[edge & cand]):
            if k_ == 0: continue
            comp = L == k_
            if comp.sum() >= _T12_OCC_MIN: cut |= comp
        d2 = disc & ~cut
        L2, n2 = cc(d2)
        if n2 > 1:
            sz = np.bincount(L2.ravel()); sz[0] = 0; d2 = L2 == int(np.argmax(sz))
        return d2
    def record(*a):
        if _T12_LOG is not None: _T12_LOG.append((ctx['o']['id'],) + a)
    out = np.zeros(m.shape, bool)
    # 2a. the glint bubbles
    S_disc = np.zeros(m.shape, bool)
    for cy, cx, Rg in _t12_glint_discs(star, rim_all, m.shape):
        disc = np.hypot(yy - cy, xx - cx) <= Rg + 0.5
        S_disc |= disc
        if (disc & pop).any(): record('glint', round(cy), round(cx), 'pop'); continue
        record('glint', round(cy), round(cx), round(Rg, 1), 'kept'); out |= occlude(disc)
    m = m & ~binary_dilation(S_disc, iterations=2)
    # 2b. the other bubbles: partition, then one circle per basin
    if m.any():
        ys, xs = np.nonzero(m); y0, y1 = max(0, ys.min() - 8), min(H, ys.max() + 9)
        x0, x1 = max(0, xs.min() - 8), min(W, xs.max() + 9)
        k = _T12_CLOSE
        def close_fill(a):
            return binary_fill_holes(binary_closing(np.pad(a, k), iterations=k)[k:-k, k:-k] | a)
        Mc = close_fill(m[y0:y1, x0:x1]); P = pop[y0:y1, x0:x1]
        Fl = (m & has(SB, (47,)))[y0:y1, x0:x1]
        Lc, nc = cc(binary_closing(np.pad(Fl, 2), iterations=2)[2:-2, 2:-2] & Mc)
        sz = np.bincount(Lc.ravel()); mk = np.isin(Lc, np.nonzero((sz >= _T12_CORE_MIN) & (np.arange(len(sz)) > 0))[0])
        dt = distance_transform_edt(Mc)
        for py, px, v in _t12_peaks(dt):
            ys_, xs_ = np.nonzero(mk)
            if not len(ys_) or np.hypot(ys_ - py, xs_ - px).min() > 0.6 * v + 3: mk[py, px] = True
        markers, nm = cc(mk)
        if nm:
            lab = _t12_ws(Mc, markers, -dt)
            own = binary_dilation(m[y0:y1, x0:x1], iterations=3)
            RIMc = rim_all[y0:y1, x0:x1]
            for j in range(1, nm + 1):
                b = (lab == j) & own
                if b.sum() < _T12_MIN_AREA: record('basin', j, int(b.sum()), 'area'); continue
                if (b & P).any(): record('basin', j, int(b.sum()), 'pop'); continue
                bf = np.zeros(m.shape, bool); bf[y0:y1, x0:x1] = b
                zone = binary_dilation(bf, iterations=12) & rim_all
                by, bx = np.nonzero(bf & rim_all); zy, zx = np.nonzero(zone)
                spec, why = _t12_fit(by, bx, zy, zx)
                if why is not None: record('basin', j, int(b.sum()), int((b & RIMc).sum()), why); continue
                disc = _t12_mask(spec, yy, xx)
                if (disc & pop).any(): record('basin', j, int(b.sum()), 'disc pop'); continue
                record('basin', j, int(b.sum()), spec[0], 'kept')
                out |= occlude(disc)
    return out


POST_BODY[6000998] = _t12_buzz_sb


# The user (2026-10-24): Buzz SB's attack is ONE convex hull over the whole shot (all its bubbles), not one disc per bubble.
# The wave group's bubbles share one label id; after labels(), that label is replaced by the convex hull of its clean bubble
# discs (pixels already owned by another label stay theirs).
BUZZ376_HULL = {6000998}
_buzz376_prev_labels = labels


def labels(objs, cam, R, state=None, ended=(), img=None):
    import bulletlabels as BL
    P = _buzz376_prev_labels(objs, cam, R, state=state, ended=ended, img=img)
    lids, ids = P['lids'], P['ids']
    done = set()
    for o in objs:
        if o['t'] not in BUZZ376_HULL: continue
        key = (BL._lid(o), -1, 'body')
        if key in done or key not in lids: continue
        done.add(key); k = lids.index(key); cur = ids == k
        if not cur.any(): continue
        h = hull_mask(cur) & (ids < 0)
        ids[h] = k; P['share'][h] = np.maximum(P['share'][h], BL.CUT)
    return P
