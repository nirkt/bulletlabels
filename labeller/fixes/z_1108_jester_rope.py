# Fix file, owner complaints sweep 4 (2026-11-02), task siteai5 / jester_voodoo_rope. Runs after every installed fix
# (IDLABELS_OVERRIDE). Four skins, keyed to type ids:
#
# 119 JesterLoki, Chester003Projectile (6001275), the crystal hexagons. Owner: 'separate projectiles (hexes) keep separate
#     solid outlines, no interior lines or holes'. Attack 90 tick 305: an interior slit and a spur / loop of the back hex
#     (the outline ran along the front hex and up into the glow); tick 307: two touching hexes drawn as one outline.
#     Rule: each hex = the convex hull of its own visible pixels (a hexagon is convex; the hull drops spurs and loops; the
#     label's pixels within 3 px of its core are the corners), no holes. Known residual: a 2-3 px spur of the beam above the
#     left hex at tick 309. Where two hexes overlap or touch, the pixel goes to the hex whose draw shows it (front-most),
#     else to the nearer centre, and the other hex keeps a 1 px seam, so the two outlines are separate. Other objects'
#     labels are never taken. Only type 6001275.
#
# 121 VoodooSandsOfTimes (zone 17000802): NOT changed here. The partial run's full-circle-from-birth rule for every tick
#     conflicts with the recorded puddle-phase decision (dome disc, then puddle only, 10-20; z_1100_s121) and its claimed
#     '11-02 ruling' is not in RULINGS_1102.md. Open question for the owner, see report.md.
#
# 123 JesterBrawlAcademy, Chester006Projectile (6001290), the flame orbs. Owner: 'holes / ragged' and 'interior hole and
#     a ragged outline between the two flames' (attack 90 tick 305). Rule: the notch between the flames is closed (a slit 1-4 px
#     wide opening into the union), the union's enclosed holes are filled, and where the two orbs of one shot touch, each
#     union pixel goes to the orb whose projected centre is nearer (a straight 1 px split line), so each orb keeps its own
#     outline. A detached speck of a label (under 40 px) is dropped. Only type 6001290.
#
# 182 RopeDudeGodzilla, Buzz007Projectile (6000994), the attack beam (logic RopeDudeProjectile). Owner: 'attack label covers
#     the whole projectile body (cyan body, not only the white core), one hull per shot for thin beams'. Rule: the body is
#     the white core plus the cyan pixels of the object's own clip / trail-over draws that touch it (within the projectile's
#     region, nearest-owner rule as usual), then ONE convex hull per projectile. A finished (impact) projectile stays
#     unlabelled as before. The super (Buzz007UltiProjectile, 6000993) is not touched.
#     Known residual: at attack 90 tick 317 and attack 0 tick 277 the cyan impact burst of a finished shot shares the clip and
#     the flying projectile's footprint, so the hull still reaches it (not separable by draw or footprint).

import numpy as _f_np
from scipy import ndimage as _f_nd

F119_TYPE = 6001275
F123_TYPE = 6001290
F182_ATK = 6000994
F182_NEAR = 10   # px: the cyan body is searched this far from the projectile's predicted footprint
F123_SPECK = 40        # px: a detached speck of a type-123 label, dropped


def _f_disk(r):
    y, x = _f_np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r


def _f_centres(objs, cam, kk):
    out = {}
    for oid, (k, o) in kk.items():
        if o.get('anchor') is not None:
            a = _f_np.asarray(o['anchor'], float)
            out[k] = _proj(cam, a[0], a[1], a[2])
    return out


def _f_hex(P, objs, cam, R, img):
    """type 6001275: one convex, solid outline per hex (see the header)"""
    import bulletlabels as BL
    from scipy.spatial import ConvexHull
    from PIL import Image as _Im, ImageDraw as _Dr
    ids, lids, share = P['ids'], P['lids'], P['share']
    cv = R['cv'][::2, ::2]; ids_h = R['id'][::2, ::2]
    kk = {}
    for o in objs:
        if o['t'] != F119_TYPE: continue
        lb = (BL._lid(o), -1, 'body')
        if lb in lids and (ids == lids.index(lb)).any(): kk[o['id']] = (lids.index(lb), o)
    if not kk: return P
    ks = [k for k, _ in kk.values()]
    H, W = ids.shape
    yy, xx = _f_np.mgrid[0:H, 0:W]
    ctr = _f_centres(objs, cam, kk)
    other = (ids >= 0) & ~_f_np.isin(ids, ks)
    # which hex's draw shows each pixel (the hex whose label holds most of that draw)
    hist = {}
    for k in ks:
        m = ids == k
        dv, cnt = _f_np.unique(ids_h[m & (ids_h > 0)], return_counts=True)
        hist[k] = dict(zip(dv.tolist(), cnt.tolist()))
    downer = {}
    for k, h in hist.items():
        for d, c in h.items():
            if d not in downer or c > downer[d][1]: downer[d] = (k, c)
    fin = {}
    for k in ks:
        m = ids == k
        if m.sum() < 30: continue
        op = _f_nd.binary_opening(_f_np.pad(m, 8), structure=_f_disk(5))[8:-8, 8:-8]
        L, n = _f_nd.label(op)
        if n:
            sz = _f_np.bincount(L.ravel()); sz[0] = 0; core = L == sz.argmax()
        else:
            core = m
        # the label's own pixels within 3 px of that core (the corners the opening rounds off come back; a spur further out goes)
        core = m & _f_nd.binary_dilation(core, iterations=3)
        ys, xs = _f_np.nonzero(core)
        if len(ys) < 3: fin[k] = m; continue
        pts = _f_np.c_[xs, ys].astype(float)
        try: h = ConvexHull(pts)
        except Exception: fin[k] = m; continue
        I = _Im.new('L', (W, H), 0); _Dr.Draw(I).polygon([tuple(pts[v]) for v in h.vertices], fill=1, outline=1)
        hull = _f_np.asarray(I, bool)
        S = hull | _f_nd.binary_dilation(core, iterations=1)
        S = S & ((ids < 0) | (ids == k)) & ~other
        fin[k] = _f_nd.binary_fill_holes(S) & ~other
    # overlaps / touching: the pixel goes to the hex whose draw shows it, else the nearer centre; the other hex keeps a 1 px seam
    names = list(fin)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            O = fin[a] & fin[b]
            if not O.any(): continue
            if a in ctr and b in ctr:
                near_a = _f_np.hypot(xx - ctr[a][0], yy - ctr[a][1]); near_b = _f_np.hypot(xx - ctr[b][0], yy - ctr[b][1])
                tie_a = near_a <= near_b
            else:
                tie_a = _f_np.ones(ids.shape, bool)
            ow = _f_np.where(tie_a, 1, 2).astype(_f_np.int8)
            dsel = _f_np.zeros(ids.shape, _f_np.int8)
            for d, (k, _) in downer.items():
                if k == a: dsel[(ids_h == d) & O] = 1
                elif k == b: dsel[(ids_h == d) & O] = 2
            ow = _f_np.where(dsel > 0, dsel, ow)
            a_wins = O & (ow == 1); b_wins = O & (ow == 2)
            fin[a] = fin[a] & ~(b_wins)
            fin[b] = fin[b] & ~(a_wins)
            # seam: the loser loses the 1 px ring around the winner's pixels in the overlap zone
            Wa = fin[a] & ~O | a_wins; Wb = fin[b] & ~O | b_wins
            ring_a = _f_nd.binary_dilation(Wa, iterations=1) & ~Wa
            ring_b = _f_nd.binary_dilation(Wb, iterations=1) & ~Wb
            fin[b] = fin[b] & ~(ring_a & fin[b])
            fin[a] = fin[a] & ~(ring_b & fin[a])
    for k in fin:
        old = ids == k; ids[old] = -1; share[old] = 0
    for k in fin:
        f = fin[k] & (ids < 0)
        ids[f] = k; share[f] = _f_np.maximum(cv[f] / 255.0, BL.CUT)
    P['ids'] = ids; P['share'] = share
    return P


def _f_flames(P, objs, cam, R, img):
    """type 6001290: holes enclosed by the orb labels filled, specks dropped"""
    import bulletlabels as BL
    ids, lids, share = P['ids'], P['lids'], P['share']
    cv = R['cv'][::2, ::2]
    ks = []
    for o in objs:
        if o['t'] != F123_TYPE: continue
        lb = (BL._lid(o), -1, 'body')
        if lb in lids and (ids == lids.index(lb)).any(): ks.append(lids.index(lb))
    if not ks: return P
    U = _f_np.isin(ids, ks)
    other = (ids >= 0) & ~U
    # notches between the flames (a slit 1-4 px wide that opens into the union) closed: the pixels go to the nearest orb label
    C = _f_nd.binary_closing(_f_np.pad(U, 4), structure=_f_disk(2))[4:-4, 4:-4] & ~U & (ids < 0)
    if C.any():
        _, (iy, ix) = _f_nd.distance_transform_edt(~U, return_indices=True)
        nb = ids[iy, ix]
        ids[C] = nb[C]; share[C] = _f_np.maximum(cv[C] / 255.0, BL.CUT)
        U = _f_np.isin(ids, ks)
    # two orbs of one shot: each pixel of their union goes to the orb whose projected centre is nearer (a straight split line),
    # and the pixels on that line are left out, so the two orbs keep separate solid outlines
    ctr = {}
    for o in objs:
        if o['t'] != F123_TYPE or o.get('anchor') is None: continue
        lb = (BL._lid(o), -1, 'body')
        if lb in lids and lids.index(lb) in ks:
            a_ = _f_np.asarray(o['anchor'], float); ctr[lids.index(lb)] = _proj(cam, a_[0], a_[1], a_[2])
    seam = _f_np.zeros(ids.shape, bool)
    if len(ks) == 2 and len(ctr) == 2 and U.any():
        a, b = ks
        yy_, xx_ = _f_np.mgrid[0:ids.shape[0], 0:ids.shape[1]]
        da = _f_np.hypot(xx_ - ctr[a][0], yy_ - ctr[a][1]); db = _f_np.hypot(xx_ - ctr[b][0], yy_ - ctr[b][1])
        own = _f_np.where(da <= db, a, b)
        ids[U] = own[U]
        # the split line: union pixels whose neighbour belongs to the other orb (1 px wide, left open)
        for dy, dx in ((0, 1), (1, 0)):
            sh = _f_np.roll(_f_np.roll(ids, dy, 0), dx, 1)
            seam |= U & (sh != ids) & (sh >= 0)
        ids[seam] = -1; share[seam] = 0
        U = _f_np.isin(ids, ks)
    # enclosed holes of the union are filled (the split line is not a hole: it is cut from both orbs' outlines only at its ends)
    holes = _f_nd.binary_fill_holes(U) & ~U & ~other & ~seam
    if holes.any():
        # each enclosed hole goes to the label it touches most (the orb behind / in front of it)
        L, n = _f_nd.label(holes)
        for j in range(1, n + 1):
            hm = L == j
            ring = _f_nd.binary_dilation(hm, iterations=1) & ~hm
            vals, cnt = _f_np.unique(ids[ring & U], return_counts=True)
            if vals.size:
                tgt = int(vals[cnt.argmax()])
                ids[hm] = tgt; share[hm] = _f_np.maximum(cv[hm] / 255.0, BL.CUT)
    for k in ks:
        m = ids == k
        if not m.any(): continue
        L, n = _f_nd.label(m, structure=_f_np.ones((3, 3), bool))
        if n < 2: continue
        sz = _f_np.bincount(L.ravel()); sz[0] = 0
        big = int(sz.argmax())
        drop = m & (L != big) & (sz[L] < F123_SPECK)
        if drop.any(): ids[drop] = -1; share[drop] = 0
    P['ids'] = ids; P['share'] = share
    return P


_f_prev_labels = labels


def labels(objs, cam, R, state=None, ended=(), img=None):
    P = _f_prev_labels(objs, cam, R, state=state, ended=ended, img=img)
    if any(o['t'] == F119_TYPE for o in objs): P = _f_hex(P, objs, cam, R, img)
    if any(o['t'] == F123_TYPE for o in objs): P = _f_flames(P, objs, cam, R, img)
    return P


# ---- 182: attack beam (Buzz007Projectile, 6000994): the whole body, one hull
def _f182_body(body, ctx):
    from scipy.ndimage import label as _cc, binary_fill_holes as _fh
    if ctx['o'].get('finished'): return _f_np.zeros_like(body)     # the impact burst of a finished shot: no label
    if not body.any(): return body
    img = ctx['img']; region = ctx['region']; C = ctx['C']
    im = img.astype(_f_np.int16); Rr, Gg, Bb = im[..., 0], im[..., 1], im[..., 2]
    cyan = (Gg >= 150) & (Bb >= 150) & (Rr < 200)                  # the cyan body and the white core (not the violet haze)
    # only near the flying projectile's own predicted footprint: an impact burst of a finished shot (same clip) stays out
    near = _dilate(ctx['pred_any'], F182_NEAR) if ctx.get('pred_any') is not None else region
    ownc = region & near & ((C == 1) | (C == 4)) & cyan
    cand = ownc | body
    L, n = _cc(cand, structure=_f_np.ones((3, 3), bool))
    touch = _f_np.unique(L[body]); touch = touch[touch > 0]
    U = body | (_f_np.isin(L, touch) & cand)
    return _fh(hull_mask(U))                                       # ONE convex hull per projectile


POST_BODY[F182_ATK] = _f182_body
