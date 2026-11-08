# hosted review 2026-10-29 fix (Haiku fix, reviewed by the coordinator, installed 10-29 LOCAL ONLY; source out/sweep2/haikutasks/site1029/g_jester/fix.py; runs after z_1100_s118/s119a/s119b and replaces their rims/hulls for 6001270/6001275; tested via IDLABELS_OVERRIDE, hence z_1101_)
# g_jester task (hosted review 2026-10-29), fix file for the Jester skins 118 / 119 / 123 (one file, keyed to type ids).
# Runs after every installed fix (IDLABELS_OVERRIDE). Section 118 (type 6001270 = Chester002Projectile, skin 118 JesterWoodspirit).
#
# User finding (binding): skin 118 row "attack 90", ticks 305 and 307, "too small / cut off": the ghost's solid green rim (the bottom
# and right flap of the right ghost, its lit rim) lies outside the label. The label stopped at the cyan body plus the rim band
# within 4 px (z_1100_s118 caps the rim at 4 px from the body and needs G - B >= 50). Measured on sw2_118 (tick 307, frame px):
# the solid green rim is R 27-33, G 175-255, B 133-160 (G - B 42-105); the dark teal halo beside it is G 50-120, G - B about 0.
# The rim is clip shapes 63, 884, 407, 7022, 64 (and solid pixels of the glow shape 0); the green spark dots above the ghosts are
# TRAIL shapes (64065, 64068, 64070), which are never taken.
# Rule (type 6001270 only): a pixel joins the ghost when it is a clip shape (not only the glow layer) with cyan or solid-green
# colour, at least half opaque, connected to the ghost's label through such pixels, within G118_NEAR px of the label; the
# union is smoothed (closing, fill, opening, blur threshold, fill) to one solid piece. Other labels are never taken; shared
# pixels go to the ghost whose own draw shows them (the front one), else to the nearer body.
import numpy as np

G118_TYPE = 6001270
G118_NEAR = 12        # px: the rim is searched this far from the ghost's label (the flap at 307 reaches about 10 px)
G118_G = 150          # solid green: G at least this, R below G118_R
G118_GB = 30          # solid green: G - B at least this
G118_R = 120
G118_SMALL = 60       # px: draws smaller than this in the frame are sparks when far from the body (see labels)
_g118_prev_labels = labels


def _g118_disk(r):
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r


def _g118_smooth(V, pad=6):
    from scipy import ndimage as nd
    W = np.pad(V, pad)
    W = nd.binary_closing(W, structure=_g118_disk(2))
    W = nd.binary_fill_holes(W)
    W = nd.binary_opening(W, structure=np.ones((3, 3), bool))
    g = nd.gaussian_filter(W.astype(np.float32), 1.0)
    W = nd.binary_fill_holes(g > 0.5)
    return W[pad:-pad, pad:-pad]


def labels(objs, cam, R, state=None, ended=(), img=None):
    import bulletlabels as BL
    from scipy import ndimage as nd
    P = _g118_prev_labels(objs, cam, R, state=state, ended=ended, img=img)
    mine = [o for o in objs if o['t'] == G118_TYPE]
    if not mine or img is None: return P
    ids, lids, share = P['ids'], P['lids'], P['share']
    cv = R['cv'][::2, ::2]
    kk = {}
    for o in mine:
        lb = (BL._lid(o), -1, 'body')
        if lb in lids and (ids == lids.index(lb)).any(): kk[o['id']] = lids.index(lb)
    if not kk: return P
    T = projtab.table()[G118_TYPE]; names = tuple(sorted({T[k] for k in ('blue', 'red') if T.get(k)}))
    SP = _a4_shape_pixels(R, T['file'], names)          # shape id + 1 per pixel of the clip's shapes (0 = not a clip shape)
    im = img.astype(np.int16); Rr, Gg, Bb = im[..., 0], im[..., 1], im[..., 2]
    solid = (Gg >= G118_G) & (Gg - Bb >= G118_GB) & (Rr < G118_R)
    cyan = (Bb >= 200) & (Gg >= G118_G) & (Rr < G118_R) & (Gg - Bb < G118_GB)
    shape_ok = SP >= 2                                   # any clip shape except the glow layer (shape 0 = SP 1)
    # no coverage gate: a translucent draw (coverage 30-90) shows the same cyan body (measured: sw2_118 tick 305, draw 102)
    cand = (SP >= 1) & (shape_ok & (solid | cyan) | ((SP == 1) & solid))
    ids_h = R['id'][::2, ::2]
    ks = list(kk.values())
    other = (ids >= 0) & ~np.isin(ids, ks)               # labels of other objects: never taken
    # the green spark dots above the ghosts are small draws of their own, lying off the cyan body (measured: sw2_118 tick 305,
    # the dot above the left ghost is a 49 px draw about 6 px from its cyan face). The rim lies within 3 px of the cyan body,
    # or is part of a large draw (the flap at tick 307 is a 745 px draw). Candidates that are neither are not taken.
    cyan_core = cyan & (SP >= 2)
    dcore = nd.distance_transform_edt(~cyan_core) if cyan_core.any() else np.full(ids.shape, 99.0)
    dsz = np.bincount(ids_h.ravel().astype(np.int64) + 1)        # draw id + 1 -> px count of the draw in the frame
    big = dsz[ids_h.astype(np.int64) + 1] >= G118_SMALL
    cand = cand & (big | (dcore <= 3))
    body = {k: ids == k for k in ks}
    near = {k: nd.binary_dilation(body[k], iterations=G118_NEAR) for k in ks}
    # projected centre of each ghost (its anchor): a seam pixel between two ghosts goes to the nearer centre
    ctr = {}
    for o in mine:
        if o['id'] in kk and o.get('anchor') is not None:
            a = np.asarray(o['anchor'], float); ctr[kk[o['id']]] = _proj(cam, a[0], a[1], a[2])
    yy, xx = np.mgrid[0:ids.shape[0], 0:ids.shape[1]]
    cdist = {k: np.hypot(xx - ctr[k][0], yy - ctr[k][1]) for k in ks if k in ctr}
    fin = {}
    for k in ks:
        m = body[k]
        if m.sum() < 30: continue
        C = cand & near[k] & ~other & ((ids < 0) | (ids == k))
        if k in cdist and len(ks) > 1:
            nearest = np.argmin(np.stack([cdist[j] if j in cdist else np.full(ids.shape, 1e9) for j in ks]), 0)
            C = C & (nearest == ks.index(k))
        L, n = nd.label(C | m, structure=np.ones((3, 3), bool))
        keep = np.unique(L[m]); keep = keep[keep > 0]
        add = np.isin(L, keep) & C & ~m
        V = _g118_smooth(m | add)
        V = nd.binary_fill_holes(V) & near[k] & ~other
        L, n = nd.label(V)
        if n > 1:   # one piece: the parts touching the body
            touch = np.unique(L[m & V]); touch = touch[touch > 0]
            V = np.isin(L, touch)
        fin[k] = V | (m & ~V)   # never lose a body pixel
    # shared pixels: the nearer ghost body owns them (one owner per pixel)
    for i, a in enumerate(list(fin)):
        for b in list(fin)[i + 1:]:
            O = fin[a] & fin[b]
            if not O.any(): continue
            if a in cdist and b in cdist: to_b = O & (cdist[b] < cdist[a])
            else: to_b = O & (nd.distance_transform_edt(~body[b]) < nd.distance_transform_edt(~body[a]))
            fin[a] = fin[a] & ~to_b; fin[b] = fin[b] & ~(O & ~to_b)
    for k in fin:
        ids[body[k]] = -1; share[body[k]] = 0
    for k in fin:
        f = fin[k] & ((ids < 0) | (ids == k))
        ids[f] = k; share[f] = np.maximum(cv[f] / 255.0, BL.CUT)
    P['ids'] = ids; P['share'] = share
    return P


# ---- section 119 (type 6001275 = Chester003Projectile, skin 119 JesterLoki): the crystal hexagons (attack 90 / attack 0).
# User findings (binding): row "attack 90", tick 305 "too small / cut off" (the back hex's visible lit facet and rim outside its
# label); ticks 307-315 "holes / ragged" (notches where the hex's lit facets lie: the bright grey-white side facets and the
# dark side). Note: "all labels are a little ragged around the edges".
# Cause (measured, owners on sw2_119 tick 315): the hex's lit facets are drawn translucent (draws 101 and 103: coverage median
# 65 and 31, 108 and 264 px, clip shapes of this type), so the coverage gate (>= 128) leaves them out; the installed s119b hull
# took only the main face draw (its hull was smaller than the hexagon, so its outline had notches).
# Rule (type 6001275 only): the hexagon's own clip-shape pixels, any coverage, assigned by the nearest projected centre (a seam
# between two hexes goes to the nearer), connected to the hex's label, hulled (a hexagon is convex; the hull is clipped to 2 px
# around the pixels it was made from), one owner per pixel. Other labels are never taken.
G119_TYPES = {6001275}
G119_NEAR = 10       # px: candidates within this of the hex's label
G119_KEEP = 2        # px: the hull may reach this far beyond its own pixels (the rim)
G119_GLOW_R = 80     # glow colour: red below this and blue - red at least G119_GLOW_BR (measured: glow (34-60, 102-116, 204-208),
G119_GLOW_BR = 130   # the hex face (78, 144, 210) and facets (160+) fall outside it)


def _g119_glow(img):
    im = img.astype(np.int16)
    return (im[..., 0] < G119_GLOW_R) & (im[..., 2] - im[..., 0] >= G119_GLOW_BR)
_g119_prev_labels = labels


def labels(objs, cam, R, state=None, ended=(), img=None):
    import bulletlabels as BL
    from scipy import ndimage as nd
    from scipy.spatial import ConvexHull
    from PIL import Image as _Im, ImageDraw as _Dr
    P = _g119_prev_labels(objs, cam, R, state=state, ended=ended, img=img)
    mine = [o for o in objs if o['t'] in G119_TYPES]
    if not mine: return P
    ids, lids, share = P['ids'], P['lids'], P['share']
    cv = R['cv'][::2, ::2]
    kk = {}
    for o in mine:
        lb = (BL._lid(o), -1, 'body')
        if lb in lids and (ids == lids.index(lb)).any(): kk[o['id']] = (lids.index(lb), o)
    if not kk: return P
    ks = [k for k, _ in kk.values()]
    ctr = {}
    for oid, (k, o) in kk.items():
        if o.get('anchor') is not None:
            a = np.asarray(o['anchor'], float); ctr[k] = _proj(cam, a[0], a[1], a[2])
    other = (ids >= 0) & ~np.isin(ids, ks)
    fin = {}
    H, W = ids.shape
    yy, xx = np.mgrid[0:H, 0:W]
    cdist = {k: np.hypot(xx - ctr[k][0], yy - ctr[k][1]) for k in ks if k in ctr}
    for oid, (k, o) in kk.items():
        T = projtab.table()[o['t']]; names = tuple(sorted({T[q] for q in ('blue', 'red') if T.get(q)}))
        if not names: continue
        SP = _a4_shape_pixels(R, T['file'], names)
        m = ids == k
        if m.sum() < 30: continue
        near = nd.binary_dilation(m, iterations=G119_NEAR)
        C = (SP >= 2) & near & ~other & ((ids < 0) | (ids == k))
        C &= ~_g119_glow(img)          # the saturated blue glow of the hexes (its shapes are shared with the hex): not the crystal
        if cdist and len(ks) > 1 and k in cdist:
            nearest = np.argmin(np.stack([cdist[j] if j in cdist else np.full(ids.shape, 1e9) for j in ks]), 0)
            C &= nearest == ks.index(k)
        L, n = nd.label(C | m, structure=np.ones((3, 3), bool))
        keep = np.unique(L[m]); keep = keep[keep > 0]
        U = np.isin(L, keep) & (C | m)
        ys, xs = np.nonzero(U)
        if len(ys) < 3: continue
        pts = np.c_[xs, ys].astype(float)
        try: h = ConvexHull(pts)
        except Exception: fin[k] = m; continue
        I = _Im.new('L', (W, H), 0); _Dr.Draw(I).polygon([tuple(pts[v]) for v in h.vertices], fill=1, outline=1)
        hull = np.asarray(I, bool)
        S = hull & nd.binary_dilation(U, iterations=G119_KEEP) & ~other
        S |= U & ~other
        # spurs (glow pixels or sparks touching the crystal's edge): an opening of 3 px radius keeps the crystal (a convex face
        # 20+ px wide) and cuts the thin parts; the piece holding the hex's own pixels is kept
        yy_, xx_ = np.mgrid[-3:4, -3:4]; disk3 = (xx_ * xx_ + yy_ * yy_) <= 9
        O_ = nd.binary_opening(np.pad(S, 4), structure=disk3)[4:-4, 4:-4]
        L2, n2 = nd.label(O_)
        if n2 > 0:
            t2 = np.unique(L2[m & O_]); t2 = t2[t2 > 0]
            if t2.size: O_ = np.isin(L2, t2)
        S = (O_ | (m & ~other)) & S
        fin[k] = nd.binary_fill_holes(S) & ~other
    # one owner per pixel: the hex whose centre is nearer
    for i, a in enumerate(list(fin)):
        for b in list(fin)[i + 1:]:
            O = fin[a] & fin[b]
            if not O.any(): continue
            if a in cdist and b in cdist: to_b = O & (cdist[b] < cdist[a])
            else: to_b = O & (nd.distance_transform_edt(~(ids == b)) < nd.distance_transform_edt(~(ids == a)))
            fin[a] = fin[a] & ~to_b; fin[b] = fin[b] & ~(O & ~to_b)
    for k in fin:
        old_m = ids == k; ids[old_m] = -1; share[old_m] = 0
    for k in fin:
        f = fin[k] & ((ids < 0))
        ids[f] = k; share[f] = np.maximum(cv[f] / 255.0, BL.CUT)
    P['ids'] = ids; P['share'] = share
    return P


# ---- sections 123 and 119 super (types 6001290 = Chester006Projectile, skin 123; 6001276 = Chester003UltiProjectileExploding,
# skin 119 super). Findings (binding): 123 attack 0 ticks 265-269 and attack 90 ticks 305, 315 "holes / ragged"; attack 90 ticks
# 309-313 "too small / cut off". 119 super 345-365 "holes / ragged". Cause: the labels take only the coverage >= 128 pixels of the
# body draws, so translucent parts (the flame, the lit crystal facets) leave holes and notches, and the hull-free silhouette
# has spurs. Rule (these two types only): the object's own clip-shape pixels, any coverage, minus the saturated glow colour and
# other labels, assigned by the nearest projected centre, connected to its label within G_NEAR px; closing (radius 3) fills the
# holes and notches between parts, the hole fill, an opening (radius 2) removes spurs; one piece touching the label. The
# droplets of the flame (s123 rule) stay out: they are not connected through shape pixels of the orb's flame at 3 px.
G_SIL_TYPES = {6001290: dict(near=10, close=3, opn=2), 6001276: dict(near=10, close=3, opn=2)}
_g_sil_prev = labels


def _g_disk(r):
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r


def labels(objs, cam, R, state=None, ended=(), img=None):
    import bulletlabels as BL
    from scipy import ndimage as nd
    P = _g_sil_prev(objs, cam, R, state=state, ended=ended, img=img)
    mine = [o for o in objs if o['t'] in G_SIL_TYPES]
    if not mine or img is None: return P
    ids, lids, share = P['ids'], P['lids'], P['share']
    cv = R['cv'][::2, ::2]
    kk = {}
    for o in mine:
        lb = (BL._lid(o), -1, 'body')
        if lb in lids and (ids == lids.index(lb)).any(): kk[o['id']] = (lids.index(lb), o)
    if not kk: return P
    ks = [k for k, _ in kk.values()]
    H, W = ids.shape
    yy, xx = np.mgrid[0:H, 0:W]
    ctr = {}
    for oid, (k, o) in kk.items():
        if o.get('anchor') is not None:
            a = np.asarray(o['anchor'], float); ctr[k] = _proj(cam, a[0], a[1], a[2])
    cdist = {k: np.hypot(xx - ctr[k][0], yy - ctr[k][1]) for k in ks if k in ctr}
    other = (ids >= 0) & ~np.isin(ids, ks)
    glow = _g119_glow(img)
    fin = {}
    for oid, (k, o) in kk.items():
        prm = G_SIL_TYPES[o['t']]
        T = projtab.table()[o['t']]; names = tuple(sorted({T[q] for q in ('blue', 'red') if T.get(q)}))
        if not names: continue
        SP = _a4_shape_pixels(R, T['file'], names)
        m = ids == k
        if m.sum() < 30: continue
        near = nd.binary_dilation(m, iterations=prm['near'])
        C = (SP >= 2) & near & ~other & ~glow & ((ids < 0) | (ids == k))
        if len(ks) > 1 and k in cdist:
            nearest = np.argmin(np.stack([cdist[j] if j in cdist else np.full(ids.shape, 1e9) for j in ks]), 0)
            C &= nearest == ks.index(k)
        U = (C | m)
        Z = np.pad(U, 6)
        Z = nd.binary_closing(Z, structure=_g_disk(prm['close']))
        Z = nd.binary_fill_holes(Z)
        Z = nd.binary_opening(Z, structure=_g_disk(prm['opn']))[6:-6, 6:-6]
        L, n = nd.label(Z)
        if n > 1:
            t = np.unique(L[m & Z]); t = t[t > 0]
            Z = np.isin(L, t) if t.size else Z
        S = (Z | m) & ~other & near
        fin[k] = nd.binary_fill_holes(S)
    for i, a in enumerate(list(fin)):
        for b in list(fin)[i + 1:]:
            O = fin[a] & fin[b]
            if not O.any(): continue
            if a in cdist and b in cdist: to_b = O & (cdist[b] < cdist[a])
            else: to_b = O & (nd.distance_transform_edt(~(ids == b)) < nd.distance_transform_edt(~(ids == a)))
            fin[a] = fin[a] & ~to_b; fin[b] = fin[b] & ~(O & ~to_b)
    for k in fin:
        old_m = ids == k; ids[old_m] = -1; share[old_m] = 0
    for k in fin:
        f = fin[k] & (ids < 0)
        ids[f] = k; share[f] = np.maximum(cv[f] / 255.0, BL.CUT)
    P['ids'] = ids; P['share'] = share
    return P
