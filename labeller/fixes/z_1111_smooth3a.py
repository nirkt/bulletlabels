# smooth3a (round 3, 2026-11-05)
# ---- 407 MinigunDudeAprilFools, Pam008Projectile (6000396; toast / egg / bacon). Owner note: "holes".
# The fried egg (and others) label took half of the soft round glow sprite (clip shape 127, role 'glow' in pam_008_atk_projectile[_red])
# drawn around every food bullet: a ring-shaped, ragged, partly-glow label with the body pixels only patchy inside. Fix: the glow
# shape is cut by shape per pixel (_a4_shape_pixels, shape 127 + 1); the label is the bullet's own body shapes (every other shape of the
# clip) of its draws, sprite-edge pixels without shape kept within 2 px, closed and filled solid, one piece.
# Shape 127 is only used by this type's own clips; applies to every skin whose projectile is type 6000396 (MinigunDude AprilFools 407 and
# any other skin using Pam008Projectile).
import numpy as _s3a_np
_S3A_FILE = 'sc/effects_brawler2.sc'
_S3A_NAMES = ('pam_008_atk_projectile', 'pam_008_atk_projectile_red')
_S3A_GLOW = 127
_S3A_DBG = False


def _s3a_fix_pam008(body, ctx):
    from scipy.ndimage import binary_fill_holes, label as cc
    if not body.any(): return body
    SP = _a4_shape_pixels(ctx['R'], _S3A_FILE, _S3A_NAMES)
    glow = SP == _S3A_GLOW + 1
    core = (SP > 0) & ~glow
    # body shapes of this bullet's own draws (cut by the coverage like everything else) inside its region
    own = _s3a_np.unique(ctx['ids_h'][body & core]); own = own[own > 0]
    cand = core & _s3a_np.isin(ctx['ids_h'], own) & (ctx['cv'] >= 64) & ctx['region']
    m = (body & core) | cand
    if not m.any(): return body
    edge = (body | (_s3a_np.isin(ctx['ids_h'], own) & (ctx['cv'] >= 64) & ctx['region'])) & (SP == 0) & _dilate(m, 2)          # sprite-edge pixels (no shape) at the body's border
    m = m | edge
    m = _close(m, 2)
    m = binary_fill_holes(m)
    # one piece: the largest part, other parts only if big
    L, n = cc(m)
    if n > 1:
        sz = _s3a_np.bincount(L.ravel())[1:]
        m = _s3a_np.isin(L, 1 + _s3a_np.nonzero(sz >= 0.25 * sz.max())[0])
    return m & ctx['region']


POST_BODY[6000396] = _s3a_fix_pam008


# ---- 408 InsectManJapan, Angelo003Projectile (6001434; crown + ball head, orange shaft, fletching tail). Owner note: "holes and ragged".
# The arrow's long orange shaft is drawn under the soft glow column (clip shape 148); the ID pass shows the glow in front over most of the
# shaft, so the label's shaft was 1 px thinner than the visible bar, broke into pieces (shaft / head / tail apart) and had a hollow look
# (two outlines inside the bar). The arrow is one object: one solid silhouette. Fix: along the arrow's own axis (principal axis of its
# label) the gaps between its label pieces are filled with the shaft corridor (width of the thin labelled rows + 1 px, only pixels that
# are not blue glow / background: r >= b - 10), then the label is closed and filled. Head and tail pieces are unchanged.
# The lightning crackle sprites over the shaft (trail clips angelo_003_ulti_area_elec_01/02) are cut out by shape first.
# Type 6001434 only (Angelo003Projectile: InsectMan Japan 408 and the other skins that use this projectile).
def _s3a_fix_angelo003(body, ctx):
    from scipy.ndimage import binary_fill_holes, label as cc
    if body.sum() < 40: return body
    # the crackling lightning sprites drawn over the shaft (trail clips angelo_003_ulti_area_elec_*, shape 138) are not the arrow
    SPo = _a4_shape_pixels(ctx['R'], 'sc/effects_brawler_angelo.sc', ('angelo_003_atk_projectile', 'angelo_003_atk_projectile_red'))
    TPe = _a4_shape_pixels(ctx['R'], 'sc/effects_brawler_angelo.sc', ('angelo_003_ulti_area_elec_01', 'angelo_003_ulti_area_elec_02'))
    body = body & ~((TPe > 0) & (SPo == 0))
    if body.sum() < 40: return body
    ys, xs = _s3a_np.nonzero(body)
    pts = _s3a_np.c_[xs, ys].astype(float); c = pts.mean(0)
    w, V = _s3a_np.linalg.eigh(_s3a_np.cov((pts - c).T)); d = V[:, 1]; nrm = _s3a_np.array([-d[1], d[0]])
    if w[1] < 4 * w[0]: return binary_fill_holes(body)     # not elongated: only fill
    u = (pts - c) @ d; v = (pts - c) @ nrm
    ub = _s3a_np.round(u).astype(int); u0 = ub.min(); n = ub.max() - u0 + 1
    vmin = _s3a_np.full(n, 1e9); vmax = _s3a_np.full(n, -1e9); cnt = _s3a_np.zeros(n)
    for k in range(len(ub)):
        i = ub[k] - u0; vmin[i] = min(vmin[i], v[k]); vmax[i] = max(vmax[i], v[k]); cnt[i] += 1
    occ = cnt > 0; wid = _s3a_np.where(occ, vmax - vmin + 1, 0)
    if not occ.any(): return body
    thin = occ & (wid <= max(6.0, 0.35 * wid.max()))
    if thin.sum() < 3: return binary_fill_holes(_close(body, 2))
    v0 = _s3a_np.median(((vmin + vmax) / 2)[thin]); hw = _s3a_np.percentile(wid[thin], 25) / 2 + 1.0
    print('DBG', v0, hw, int(thin.sum()), n, wid.max()) if _S3A_DBG else None
    # corridor: between the first and the last occupied u slice
    H, W_ = body.shape; yy, xx = _s3a_np.mgrid[0:H, 0:W_]
    uu = (xx - c[0]) * d[0] + (yy - c[1]) * d[1]; vv = (xx - c[0]) * nrm[0] + (yy - c[1]) * nrm[1]
    near = _dilate(body, 14)
    corr = near & (_s3a_np.abs(vv - v0) <= hw) & (uu >= u0 - 0.5) & (uu <= u0 + n - 0.5)
    img = ctx['img'].astype(_s3a_np.int16)
    notblue = img[..., 0] >= img[..., 2] - 10
    bar = corr & notblue & ctx['region']
    b2 = body
    # shaft zone = between the head (shapes 145 / 76 / 102 / 149: crown + ball) and the tail cap (shape 38): only the shaft corridor
    # there; glow-column pixels / lightning remains beside the bar are outside the arrow
    hp = body & _s3a_np.isin(SPo, [146, 77, 103, 150]); cp = body & (SPo == 39)
    if hp.any() and cp.any():
        uh = uu[hp]; uc = uu[cp]
        if uc.mean() > uh.mean(): z0, z1 = uh.max(), uc.min()
        else: z0, z1 = uc.max(), uh.min()
        zone = (uu > z0) & (uu < z1)
        b2 = body & ~(zone & (_s3a_np.abs(vv - v0) > hw + 0.5))
    if not hp.any() and cp.any():
        # head hidden under the launch smoke (hidden parts keep the full shape): the predicted body of the head side of the cap
        pb = ctx['pred_body'] & ctx['region']
        sh = body & (SPo == 1)                                      # the bar (shape 0) lies on the head side of the cap
        sgn = 1.0 if (sh.any() and uu[sh].mean() > uu[cp].mean()) else -1.0
        head_side = (uu - uu[cp].mean()) * sgn > 0
        add = pb & head_side
        L0, k0 = cc(add | b2); kp = _s3a_np.unique(L0[b2]); add = _s3a_np.isin(L0, kp[kp > 0]) & add
        b2 = b2 | add
    m = b2 | bar
    m = binary_fill_holes(_close(m, 2))
    # the shaft pixels only count when they connect the label's pieces: keep the parts that touch the label
    L, k = cc(m); keep = _s3a_np.unique(L[b2]); m = _s3a_np.isin(L, keep[keep > 0])
    return m


POST_BODY[6001434] = _s3a_fix_angelo003


# ---- 409 BlackHoleJapan, Tara007Projectile (6000423; fan of 3 crystal daggers per attack, each dagger = 7 body sprite shapes 351-357/358 of
# clip tara_007_atk_projectile[_red] + glow shape 0). Owner note: "ragged".
# Cause: the three daggers of one shot overlap / touch (the shot starts as a stack) and the logic objects' predicted positions are a few px
# off, so the per-object claim handed pieces of one dagger to the neighbour's label (necks between the labels, one dagger label holding half
# of its neighbour, whole parts of a dagger in no label at all) and the outline followed the leftovers of the glow sprite (frayed).
# Fix (labels() wrapper, this type only): the labels of all daggers of the frame are rebuilt from the clip's own body shapes (cut by shape per
# pixel, _a4_shape_pixels), completed to the whole dagger, split between the daggers by the dagger's own tail spike (shape 355: one per
# dagger) with a cut along the shot direction (perpendicular distance to the spike's axis), each dagger closed + filled solid
# as one piece, and given to the object whose old label overlaps it most; touching daggers keep a 1-px seam.
# Type 6000423 = skin 409 (BlackHole Japan) and every other skin using Tara007Projectile.
_S3A_TD = 6000423
_S3A_TD_FILE = 'sc/effects_brawler_tara.sc'
_S3A_TD_NAMES = ('tara_007_atk_projectile', 'tara_007_atk_projectile_red')
_S3A_TD_BODY = (351, 352, 353, 354, 355, 356, 357, 358)
_S3A_TD_SPIKE = 355
_s3a_prev_labels = labels


def _s3a_td_rebuild(P, objs, R):
    from scipy.ndimage import label as cc, binary_fill_holes
    ts = [o for o in objs if o['t'] == _S3A_TD]
    if not ts: return P
    keys = {BL._lid(o) for o in ts}
    ks = [k for k, l in enumerate(P['lids']) if l[0] in keys and l[2] == 'body']
    if not ks: return P
    ids = P['ids']; U = _s3a_np.isin(ids, ks)
    if not U.any(): return P
    SP = _a4_shape_pixels(R, _S3A_TD_FILE, _S3A_TD_NAMES)
    bodyshape = _s3a_np.isin(SP, [s + 1 for s in _S3A_TD_BODY])
    near = _dilate(U, 8)
    M0 = bodyshape & near
    if M0.sum() < 12: return P
    # one seed per dagger: the connected pieces of its spike shape
    sp = (SP == _S3A_TD_SPIKE + 1) & near
    Ls, ns = cc(sp, structure=_s3a_np.ones((3, 3)))
    seeds = []
    for i in range(1, ns + 1):
        yy, xx = _s3a_np.nonzero(Ls == i)
        if len(yy) >= 3: seeds.append((xx.mean(), yy.mean()))
    if not seeds: return P
    seeds = _s3a_np.array(seeds)
    ys, xs = _s3a_np.nonzero(M0); pts = _s3a_np.c_[xs, ys].astype(float)
    # shot direction: from each spike towards the body mass within reach of it
    dv = _s3a_np.zeros(2)
    for s in seeds:
        dd = pts - s; r = _s3a_np.hypot(dd[:, 0], dd[:, 1]); sel = (r < 30) & (r > 4)
        if sel.any():
            c = (pts[sel] - s).mean(0); dv += c / (_s3a_np.hypot(*c) + 1e-9)
    if _s3a_np.hypot(*dv) < 1e-6: return P
    d = dv / _s3a_np.hypot(*dv); nrm = _s3a_np.array([-d[1], d[0]])
    perp = _s3a_np.stack([_s3a_np.abs((pts - s) @ nrm) for s in seeds], 1)
    along = _s3a_np.stack([(pts - s) @ d for s in seeds], 1)
    perp = _s3a_np.where((along >= -6) & (along <= 45), perp, 1e9)
    own = perp.argmin(1); ok = perp.min(1) < 1e8
    n = len(seeds); masks = []
    for i in range(n):
        m = _s3a_np.zeros(M0.shape, bool); sel = ok & (own == i); m[ys[sel], xs[sel]] = True
        if m.sum() < 8: masks.append(m); continue
        m = binary_fill_holes(_close(m, 2))
        L, k = cc(m)
        if k > 1:
            sz = _s3a_np.bincount(L.ravel())[1:]; m = _s3a_np.isin(L, 1 + _s3a_np.nonzero(sz >= 0.2 * sz.max())[0])
        masks.append(m)
    # closing may have crossed into the neighbour: the pixel goes to the dagger whose axis is nearer
    H, W_ = M0.shape; yy, xx = _s3a_np.mgrid[0:H, 0:W_]
    pf = _s3a_np.stack([_s3a_np.abs((xx - s[0]) * nrm[0] + (yy - s[1]) * nrm[1]) for s in seeds], 0)
    claim = _s3a_np.stack(masks, 0); cnt = claim.sum(0)
    win = pf.argmin(0)
    for i in range(n): masks[i] = masks[i] & ((cnt <= 1) | (win == i))
    # dagger -> object: the object whose old label overlaps it most (greedy, one dagger per object)
    old = {k: (ids == k) for k in ks}
    pairs = sorted(((int((masks[i] & old[k]).sum()), i, k) for i in range(n) for k in ks), reverse=True)
    take_d = {}; take_k = set()
    for ov, i, k in pairs:
        if ov <= 0: break
        if i in take_d or k in take_k or masks[i].sum() < 8: continue
        take_d[i] = k; take_k.add(k)
    # daggers without an overlapping object: the nearest free object (old label centroid)
    free = [k for k in ks if k not in take_k]
    for i in range(n):
        if i in take_d or masks[i].sum() < 8 or not free: continue
        cy, cx = _s3a_np.nonzero(masks[i]); cy, cx = cy.mean(), cx.mean(); best = None
        for k in free:
            oy, ox = _s3a_np.nonzero(old[k])
            if len(oy) == 0: continue
            dist = (oy.mean() - cy) ** 2 + (ox.mean() - cx) ** 2
            if best is None or dist < best[0]: best = (dist, k)
        if best: take_d[i] = best[1]; free.remove(best[1])
    if not take_d: return P
    new = ids.copy(); new[U] = -1
    allm = _s3a_np.zeros(U.shape, bool)
    for i, k in take_d.items(): new[masks[i]] = k; allm |= masks[i]
    from scipy.ndimage import distance_transform_edt as _edt
    hole = binary_fill_holes(allm) & ~allm              # pin-holes between the daggers: given to the nearest dagger
    if hole.any():
        _, (iy, ix) = _edt(~allm, return_indices=True)
        new[hole] = new[iy[hole], ix[hole]]; allm = allm | hole
    for k in ks:                                          # an object that got no dagger keeps its own pixels away from the daggers
        if k not in take_k:
            keep = old[k] & ~_dilate(allm, 2); new[keep & (new < 0)] = k
    # 1-px seam where two daggers touch (separate bullets stay separate outlines)
    tk = [k for k in take_d.values()]; isk = _s3a_np.isin(new, tk); out = new.copy()
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nb = _s3a_np.roll(new, (dy, dx), (0, 1)); nk = _s3a_np.roll(isk, (dy, dx), (0, 1))
        cut = isk & nk & (new > nb) & (nb >= 0)
        if dy == 1: cut[0, :] = False
        if dy == -1: cut[-1, :] = False
        if dx == 1: cut[:, 0] = False
        if dx == -1: cut[:, -1] = False
        out[cut & (out == new)] = -1
    # a seam pixel with all four neighbours labelled would be a pin-hole between three daggers: it stays labelled
    lone = (out < 0) & (new >= 0)
    nbl = _s3a_np.zeros(lone.shape, bool); cntn = _s3a_np.zeros(lone.shape, int)
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)): cntn += (_s3a_np.roll(out, (dy, dx), (0, 1)) >= 0)
    back = lone & (cntn == 4)
    out[back] = new[back]
    cv = R['cv'][::2, ::2]
    sh = P['share'].copy(); ch = (out != ids) & (out >= 0)
    sh[ch] = _s3a_np.maximum(cv[ch] / 255.0, BL.CUT); sh[out < 0] = 0
    P2 = dict(P); P2.update(ids=out, share=sh.astype(_s3a_np.float32))
    return P2


def labels(objs, cam, R, state=None, ended=(), img=None):
    P = _s3a_prev_labels(objs, cam, R, state=state, ended=ended, img=img)
    return _s3a_td_rebuild(P, objs, R)


# ---- 420 KickerDudeDemon, Fang008Projectile (6001117; teal wave = beam stem, clip shape 3, + wave blob, clip shape 500, both body in
# fang_008_atk1_projectile[_red]). Owner: tick 265 (attack 0) and 305 (attack 90) "holes / ragged".
# Cause: (a) the blob's translucent rim (shape 500, coverage 30-120 under the label cut) is labelled only where the coverage happens to pass
# the cut: a frayed, stair-stepped outline; (b) the stem is drawn under the glow so its middle is missing from the ID pass: the stem piece
# beside the brawler floats apart from the blob, one object in two pieces with a hole between them.
# Fix (this type only): (a) while the wave is solid (>= 60 % of its blob shape pixels near the label are already labelled; once it dissolves
# into the faint cloud, the ball core stays the label as before, owner: fades are not labelled) the blob's rim pixels of shape 500 within
# 4 px of the label are added (coverage >= 24); (b) the pieces are joined by closing along the wave's own axis (principal axis, line
# element 28 px) = the stem corridor (width of the stem piece), then filled solid.
_S3A_FANG = 6001117
_S3A_FANG_FILE = 'sc/effects_brawler_fang.sc'
_S3A_FANG_NAMES = ('fang_008_atk1_projectile', 'fang_008_atk1_projectile_red')


def _s3a_line_close(m, d, L):
    from scipy.ndimage import binary_dilation, binary_erosion
    k = int(L) + 2; se = _s3a_np.zeros((2 * k + 1, 2 * k + 1), bool)
    for t in _s3a_np.linspace(-L / 2.0, L / 2.0, int(L * 3) + 1):
        se[int(round(k + t * d[1])), int(round(k + t * d[0]))] = True
    p = k + 2
    mp = _s3a_np.pad(m, p)
    return binary_erosion(binary_dilation(mp, structure=se), structure=se)[p:-p, p:-p]


def _s3a_fix_fang(body, ctx):
    from scipy.ndimage import binary_fill_holes, label as cc
    if body.sum() < 40: return body
    SP = _a4_shape_pixels(ctx['R'], _S3A_FANG_FILE, _S3A_FANG_NAMES)
    blob = SP == 501; cv = ctx['cv']
    tot = blob & _dilate(body, 8)
    if tot.sum() == 0 or (body & blob).sum() < 0.6 * tot.sum(): return body
    m = body | (blob & _dilate(body, 4) & (cv >= 24) & ctx['region'])
    ys, xs = _s3a_np.nonzero(m); pts = _s3a_np.c_[xs, ys].astype(float)
    w, V = _s3a_np.linalg.eigh(_s3a_np.cov((pts - pts.mean(0)).T)); d = V[:, 1]
    m = m | _s3a_line_close(m, d, 28)
    m = binary_fill_holes(m)
    return m & (ctx['region'] | body | _dilate(body, 4))


POST_BODY[_S3A_FANG] = _s3a_fix_fang
