# Skin 521 MechaDudeMonsterTruck, attack 90 deg (pair 9441, tick 325) and attack 0 deg: the Mecha ulti fists (type 6001075).
# Round 3 (the 10-27 rebuild from sprite geometry, fix2, spilled onto puffs / background where a fist is buried in a puff).
# Cause: the fist's clip (mecha_003_atk2_projectile, sc/effects_brawler.sc) also draws the blue-white impact puffs the fist
# throws off where it lands, and the label took them (rule 4). This version uses the client's own per-draw records:
#   1. OWN DRAWS: every sprite draw of one fist (body 24740, fingers 24741, flame pieces) is written at the same eye depth
#      (R['wd'], one value per fist, e.g. -1.405), the puffs / tendrils of the impact sit in draws of other depths (-1.451 ...).
#      Depth is measured on the pixels of the fist body shape (24740); the label keeps only pixels of that depth: puffs out.
#   2. OCCLUDED PART: where a puff is drawn over the fist (the ID pass has only the front-most draw) the pixels are completed
#      from the fist's own sprites: the body-shape similarity map (fitted on its visible body pixels, RANSAC) places the
#      clip's rigid pieces (body 24740 at the identity matrix, the two finger sprites 24741 at their per-frame matrices of the
#      clip 25544, frame chosen by the visible fingers) and the atlas alpha of each piece gives the silhouette. It is added
#      ONLY on pixels whose front-most draw is a blended effect draw (puff) and not another fist's body: never on floor, walls,
#      characters, background; so nothing spills.
#   3. holes of the result filled, specks apart from the main part dropped.
# Type touched: 6001075 only (no other skin uses it in projtab).
F7 = {6001075: dict(file='sc/effects_brawler.sc', clips=('mecha_003_atk2_projectile', 'mecha_003_atk2_projectile_red'),
                    body=24740, fing=24741, flames=(99, 567, 717, 7294, 63, 137, 708, 7384), fit_tol=0.0005, min_fit=25, fl_x=(30, 260), fl_y=120)}
# 10-28: the enemy (red) team's clip mecha_003_atk2_projectile_red has its own body / finger shapes (24736 / 24738, same
# flame shapes and same rig layout); own (blue) objects keep exactly the parameters above.
F7_RED = dict(F7[6001075], geom_clips=('mecha_003_atk2_projectile_red',), body=24736, fing=24738)
F7_BLUE_GEOM = ('mecha_003_atk2_projectile', 'mecha_003_atk2_projectile_red')


def _f7_sim(A, tg):
    n = len(A); M = np.zeros((2 * n, 4)); r = np.zeros(2 * n)
    M[0::2, 0] = A[:, 0]; M[0::2, 1] = -A[:, 1]; M[0::2, 2] = 1; r[0::2] = tg[:, 0]
    M[1::2, 0] = A[:, 1]; M[1::2, 1] = A[:, 0]; M[1::2, 3] = 1; r[1::2] = tg[:, 1]
    a, b, tu, tv = np.linalg.lstsq(M, r, rcond=None)[0]
    return np.array([[a, b], [-b, a], [tu, tv]]), (a, b)


def _f7_groups(m, uv, tol, min_n, scale=(-9.5e-4, -5.0e-4)):
    """screen (x, y, 1) -> atlas uv similarity per rigid body sprite among the visible body pixels m (several fists of one depth
    overlap): RANSAC, one group after the other; [(T, inlier mask)], largest first"""
    ys, xs = np.nonzero(m)
    if len(ys) < min_n: return []
    A = np.c_[xs, ys, np.ones(len(xs))].astype(np.float64); tg = uv[ys, xs].astype(np.float64)
    rng = np.random.RandomState(12345); left = np.ones(len(ys), bool); out = []
    sane = lambda ab: -scale[1] <= np.hypot(*ab) <= -scale[0]
    while left.sum() >= min_n and len(out) < 6:
        idx = np.nonzero(left)[0]; best = None
        for _ in range(300):
            tri = rng.choice(idx, 3, replace=False); T, ab = _f7_sim(A[tri], tg[tri])
            if not sane(ab): continue
            inl = left & (np.abs(A @ T - tg).max(1) < tol)
            if best is None or inl.sum() > best.sum(): best = inl
        if best is None or best.sum() < min_n: break
        ok = True
        for _ in range(2):
            T, ab = _f7_sim(A[best], tg[best])
            if not sane(ab): ok = False; break
            best = left & (np.abs(A @ T - tg).max(1) < tol)
        if not ok or best.sum() < min_n: left[np.nonzero(best)[0]] = False; continue
        mm = np.zeros(m.shape, bool); mm[ys[best], xs[best]] = True
        out.append((T, mm)); left &= ~best
    return out


@functools.lru_cache(maxsize=None)
def _f7_geom(file, clips, body, fing):
    """the fist's rigid pieces: body / finger shape affine xy -> atlas px, the finger matrices of every frame of the clip, atlas alpha"""
    S = PD.sc(file); G = {}
    def find(cid, depth=0):
        c = S.clips[cid]
        for f in range(max(1, c['nframes'])):
            for slot, child, cm, cc, bl in S.frame(cid, f):
                if child == body: return cid
        for slot, child, cm, cc, bl in S.frame(cid, 0):
            if child in S.clips and depth < 4:
                r = find(child, depth + 1)
                if r is not None: return r
    inner = None
    for nm in clips:
        cid = S.exports.get(nm)
        if cid is not None: inner = find(cid); break
    if inner is None: return None
    def aff(sid):
        tex, uv, xy = S.shapes[sid][0]; w, h = tex_size(file, tex)
        P = np.asarray(uv, np.float64) / 65535.0 * [w, h]; X = np.c_[np.asarray(xy, np.float64), np.ones(len(xy))]
        M = np.linalg.lstsq(X, P, rcond=None)[0]   # (3, 2): [x y 1] @ M = atlas px
        return tex, M
    G['body_tex'], G['Mb'] = aff(body); G['fing_tex'], G['Mf'] = aff(fing)
    for key, sid, tex in (('b', body, G['body_tex']), ('f', fing, G['fing_tex'])):
        pl = _a4_polys(file, clips)[(file, tex)]; kk = [i for i, (s_, _) in enumerate(pl) if s_ == sid][0] + 1
        G['rm_' + key] = _a4_rmap(file, clips, tex) == kk   # the shape's own polygon in its atlas
    c = S.clips[inner]; G['nf'] = max(1, c['nframes']); G['bm'] = None; G['fm'] = []
    for f in range(G['nf']):
        fl = []
        for slot, child, cm, cc, bl in S.frame(inner, f):
            if child == body and f == 0: G['bm'] = np.asarray(cm, np.float64)
            if child == fing: fl.append(np.asarray(cm, np.float64))
        G['fm'].append(fl)
    return G


def _f7_apply(cm, xy):
    """SC matrix (a, b, c, d, tx, ty): x' = a x + c y + tx, y' = b x + d y + ty"""
    a, b, c, d, tx, ty = cm
    return np.c_[a * xy[:, 0] + c * xy[:, 1] + tx, b * xy[:, 0] + d * xy[:, 1] + ty]


def _f7_inv(cm, xy):
    a, b, c, d, tx, ty = cm; det = a * d - b * c
    x = xy[:, 0] - tx; y = xy[:, 1] - ty
    return np.c_[(d * x - c * y) / det, (-b * x + a * y) / det]


def _f7_sil(G, T, file, shape_tex_atlas, H, W, window, fing_frame=None):
    """silhouettes (body, finger frames list) predicted in the window (x0, y0, x1, y1) through the body map T"""
    x0, y0, x1, y1 = window
    gy, gx = np.mgrid[y0:y1, x0:x1]; X = gx.ravel(); Y = gy.ravel()
    uvn = np.c_[X, Y, np.ones(len(X))] @ T
    tw, th = tex_size(file, G['body_tex']); px = uvn * [tw, th]
    # body-local xy from atlas px: [x y 1] @ Mb = px  ->  xy = (px - Mb[2]) @ inv(Mb[:2])
    Mb = G['Mb']; xy_b = (px - Mb[2]) @ np.linalg.inv(Mb[:2])
    A_b = tex_alpha(file, G['body_tex']); hb, wb = A_b.shape
    ix = np.round(px[:, 0]).astype(int); iy = np.round(px[:, 1]).astype(int)
    inb = (ix >= 0) & (ix < wb) & (iy >= 0) & (iy < hb)
    return gx, gy, X, Y, xy_b, inb


def _f7_pred(G, T, file, window, frame, which):
    """boolean (h, w) of the window: 'body' or the fingers of a frame, from the body map T"""
    x0, y0, x1, y1 = window; hh, ww = y1 - y0, x1 - x0
    gy, gx = np.mgrid[y0:y1, x0:x1]; X = gx.ravel(); Y = gy.ravel()
    uvn = np.c_[X, Y, np.ones(len(X))] @ T
    tw, th = tex_size(file, G['body_tex']); px = uvn * [tw, th]
    Mb = G['Mb']; xy_b = (px - Mb[2]) @ np.linalg.inv(Mb[:2])
    out = np.zeros(hh * ww, bool)
    if which == 'body':
        A = tex_alpha(file, G['body_tex']) * G['rm_b']; h_, w_ = A.shape; pp = px
        ix = np.round(pp[:, 0]).astype(int); iy = np.round(pp[:, 1]).astype(int)
        ok = (ix >= 0) & (ix < w_) & (iy >= 0) & (iy < h_)
        # inside the body quad (the sprite polygon): xy within the shape's extent
        out[ok] = A[iy[ok], ix[ok]] >= 128
        return out.reshape(hh, ww)
    xy_c = _f7_apply(G['bm'], xy_b)
    A = tex_alpha(file, G['fing_tex']) * G['rm_f']; h_, w_ = A.shape; Mf = G['Mf']
    for cm in G['fm'][frame]:
        xy_f = _f7_inv(cm, xy_c); pf = np.c_[xy_f, np.ones(len(xy_f))] @ Mf
        ix = np.round(pf[:, 0]).astype(int); iy = np.round(pf[:, 1]).astype(int)
        ok = (ix >= 0) & (ix < w_) & (iy >= 0) & (iy < h_)
        out[ok] |= A[iy[ok], ix[ok]] >= 128
    return out.reshape(hh, ww)


F7_DEBUG = {}


def _f7_xy(G, T, file, window):
    """body-local clip xy (the body sprite's own coordinates: x along the fist, flames at x > 45) of every pixel of the window"""
    x0, y0, x1, y1 = window; gy, gx = np.mgrid[y0:y1, x0:x1]
    uvn = np.c_[gx.ravel(), gy.ravel(), np.ones(gx.size)] @ T
    tw, th = tex_size(file, G['body_tex']); px = uvn * [tw, th]
    xy = (px - G['Mb'][2]) @ np.linalg.inv(G['Mb'][:2])
    return xy.reshape(y1 - y0, x1 - x0, 2)


def _f7(body, ctx):
    from scipy.ndimage import binary_fill_holes, label as cc, distance_transform_edt
    if not body.any(): return body
    p = F7[ctx['t']]
    if not ctx['own']: p = F7_RED
    R = ctx['R']; ids_h = ctx['ids_h']; uv = R['uv']; cvv = ctx['cv']
    SP = _a4_shape_pixels(R, p['file'], p['clips'])
    G = _f7_geom(p['file'], p.get('geom_clips', p['clips']), p['body'], p['fing'])
    isb = SP == p['body'] + 1
    core = body & isb
    groups = _f7_groups(core, uv, p['fit_tol'], p['min_fit']) if (G is not None and core.any()) else []
    if not groups: return F7_NOFIT(body, ctx, SP, p)
    H, W = body.shape; ys, xs = np.nonzero(body); pad = 45
    window = (max(0, xs.min() - pad), max(0, ys.min() - pad), min(W, xs.max() + pad + 1), min(H, ys.max() + pad + 1))
    x0, y0, x1, y1 = window; sl = (slice(y0, y1), slice(x0, x1))
    fing_all = (body & (SP == p['fing'] + 1))[sl]
    sils = []; cens = []; zones = []
    for T, gm in groups:
        sb = _f7_pred(G, T, p['file'], window, 0, 'body')
        cand = [_f7_pred(G, T, p['file'], window, f, 'fing') for f in range(G['nf'])]
        any_f = np.any(cand, 0); fv = fing_all & _dilate(any_f, 1); vo = body[sl] & ~fing_all & _dilate(any_f, 1) & ~sb
        if fv.sum() >= 12:
            sc = [(pf & fv).sum() - 0.5 * (fv & ~pf).sum() - 0.5 * (pf & vo).sum() for pf in cand]; sf = cand[int(np.argmax(sc))]
        else: sf = np.mean(cand, 0) >= 0.5
        sils.append(sb | sf)
        xy = _f7_xy(G, T, p['file'], window); zones.append(((xy[..., 0] >= p['fl_x'][0]) & (xy[..., 0] <= p['fl_x'][1]) & (np.abs(xy[..., 1]) <= p['fl_y']), np.abs(xy[..., 1])))
        b_, a_ = np.nonzero(sb); cens.append(np.array([a_.mean() + x0, b_.mean() + y0]))
    # which fist: the group nearest the object's predicted label (not taken by another object of this frame)
    pb = ctx['pred_body']; pa = ctx['pred_any']; ref = pb if pb.sum() >= 300 else pa
    if not ref.any(): return body
    ry, rx = np.nonzero(ref); c = np.array([rx.mean(), ry.mean()])
    dist = [float(np.hypot(*(q - c))) for q in cens]; order = np.argsort(dist)
    if dist[order[0]] > 30: return F7_NOFIT(body, ctx, SP, p)
    used = R.setdefault('_f7_used', {}); oid = ctx['o']['id']
    if oid in used: used.clear()
    pick = None
    for i in order:
        if dist[i] > 30: break
        if not any(np.hypot(*(cens[i] - u)) < 2 for u in used.values()): pick = int(i); break
    if pick is None: pick = int(order[0])
    used[oid] = cens[pick]
    sil = np.zeros_like(body); sil[sl] = sils[pick]
    # a flame pixel inside the zones of two fists (they fly side by side) belongs to the one whose axis is nearer
    ay = np.stack([np.where(z_[0], z_[1], 1e9) for z_ in zones]); zone = np.zeros_like(body); zone[sl] = (ay[pick] < 1e9) & (ay.argmin(0) == pick)
    # the occluded part of the sprites: only where a blended effect draw (a puff) is the front-most and it is not another fist's body
    draws = R['draws']
    eff = [int(x) for x in np.unique(ids_h[ids_h > 0]) if int(x) in draws and draws[int(x)].get('blend') == 1]
    mine_core = groups[pick][1]
    other_body = isb & ~mine_core & ~_dilate(sil, 0)
    allowed = np.isin(ids_h, eff) & ~other_body
    # the visible pixels of its own body / finger sprites at the silhouette (edge pixels of the fit), flames of its flame zone
    vis = body & (isb | (SP == p['fing'] + 1)) & _dilate(sil, 1) & allowed
    fl = body & np.isin(SP, [s_ + 1 for s_ in p['flames']]) & zone & (cvv >= 128)
    m = (sil & allowed) | vis
    m = binary_fill_holes(_close(m, 1))
    L, n = cc(m, structure=np.ones((3, 3)))
    if n > 1:   # specks apart from the fist body
        sz = np.bincount(L.ravel())[1:]; m = np.isin(L, 1 + np.nonzero(sz >= 0.2 * sz.max())[0])
    # its flame: flame-sprite pixels of the flame zone that touch the fist, joined to it (a gap of a few px of translucent flame)
    L, n = cc(_dilate(fl, 1), structure=np.ones((3, 3))); keep = np.unique(L[_dilate(m, 3) & fl]); keep = keep[keep > 0]
    flk = fl & np.isin(L, keep)
    m = binary_fill_holes(m | flk | (_close(m | flk, 3) & _dilate(flk, 3)))
    L, n = cc(m, structure=np.ones((3, 3)))
    if n > 1: sz = np.bincount(L.ravel()); m = np.isin(L, np.nonzero((sz >= 25) & (np.arange(len(sz)) > 0))[0])   # crumbs of the flame apart from it
    if F7_DEBUG.get('on'): print('   final', int(m.sum()), 'fl kept', int(flk.sum()))
    return m


def F7_NOFIT(body, ctx, SP, p):
    """no body sprite of this object is visible (fully buried / not found): only its own flame pixels, if any"""
    fl = body & np.isin(SP, [s_ + 1 for s_ in p['flames']]) & (ctx['cv'] >= 128)
    if F7_DEBUG.get('on'): print('   NOFIT obj', ctx['o']['id'], 'body', int(body.sum()), 'flame px', int(fl.sum()))
    return fl


for _t in F7: POST_BODY[_t] = _f7
