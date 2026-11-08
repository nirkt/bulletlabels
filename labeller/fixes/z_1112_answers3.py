# answers3 (owner answers 2026-11-05 to three open questions; binding). Override file, exec'd in idlabels' namespace, runs after the installed
# fixes (z_1110_merge, z_1110_misc, ...) and before the global smoothing v2. Global names carry the prefix _a3ans_ / A3ANS_.
#
# ---- 223 BanditGirlPrereg SUPER (Shelly002UltiProjectile 6000003; single-skin type; the attack 6000002 is per-star since z_1110_merge) ----
# OWNER: "if the stars overlap, the labels can merge, just make sure that everything gets labeled, since now around 50% of stars at the
#   start get ignored."
# Cause: the super's label is the ID pass's pixels at coverage >= 128 (CUT) of the star sprite. Each star is drawn as a bright core
#   (coverage >= 128) inside a translucent rim (coverage 30-127) and the stars are packed edge to edge, so only the cores were labelled:
#   9 jagged blobs, each star only partly (and the cores of overlapped stars lost). Fix: one label per star (one per clip pellet, 36 per
#   super, 'pellet' of the object = its star) taken from the star sprite pixels at coverage >= A3ANS_CV (the whole star incl. its rim), gaps
#   between neighbouring stars closed, shared out between the stars by the nearest PREDICTED star centre, each filled solid; stars whose
#   centres are closer than A3ANS_MERGE are one label (stars lying on each other at the first frames); a 1-px seam is left where two
#   star labels touch. Stars partly hidden by smoke / a flash / the explosion get the predicted star disc where an effect sprite is in front.
#   Label ids: (object id, pellet, 'body') per star; the object's old (id, -1, 'body') is emptied.
import numpy as _a3ans_np
from scipy.ndimage import (binary_fill_holes as _a3ans_fill, label as _a3ans_cc, binary_closing as _a3ans_closing,
                           binary_opening as _a3ans_opening, binary_dilation as _a3ans_dil, distance_transform_edt as _a3ans_edt)

A3ANS_STAR_TYPES = {6000003}
A3ANS_CVT = {6000003: 56, 6000005: 110}   # 364 super (BanditGirlStar_remake, one star per object): the star is a ring at coverage >= CUT with a translucent white middle
A3ANS_CV = 56            # coverage (0-255) from which a star sprite pixel is the star (core + rim; the soft edge below is not)
A3ANS_SHARED = {6000005}
A3ANS_MERGE = 6.0        # px (half res): star centres closer than this are one label
A3ANS_R = 9.0            # px: a pixel farther than this from every star centre is not a star's (grey trail particle, bridge, glow)
A3ANS_GAP = 2            # px: gaps between neighbouring stars (the dark seam where two sprites meet) are closed
A3ANS_MIN = 8
A3ANS_HIDMIN = 8         # px: a star with at least this much drawn is there; the predicted body fills in the parts an effect hides


def _a3ans_disc(r):
    k = int(_a3ans_np.ceil(r)); y, x = _a3ans_np.ogrid[-k:k + 1, -k:k + 1]
    return (x * x + y * y) <= r * r + 1e-6


def _a3ans_close(m, r):
    p = int(r) + 2
    return _a3ans_closing(_a3ans_np.pad(m, p), structure=_a3ans_disc(r))[p:-p, p:-p]


def _a3ans_stars(P, objs, cam, R):
    import bulletlabels as BL
    sel = [o for o in objs if o['t'] in A3ANS_STAR_TYPES and o.get('drawn')]
    if not sel: return P
    ids = P['ids'].copy(); share = P['share'].copy(); lids = list(P['lids'])
    old = [k for k, l in enumerate(lids) if l[0] in {BL._lid(o) for o in sel} and l[2] == 'body']
    Pp = BL.labels(objs, cam, cut=0.01)
    pids, plids, psh = Pp['ids'], Pp['lids'], Pp['share']
    marks = []                                                   # (object lid, pellet, centre, predicted star body)
    for o in sel:
        lid = BL._lid(o)
        ent = [kk for kk, l in enumerate(plids) if l[0] == lid and l[2] == 'body' and (pids == kk).any()]
        pel = [kk for kk in ent if plids[kk][1] >= 0]
        for kk in (pel or ent):                                  # one star per clip pellet; an object without pellets is one star
            marks.append([lid, plids[kk][1], _a3ans_np.argwhere(pids == kk).mean(0), (pids == kk) & (psh >= BL.CUT)])
    if not marks: return P
    grp = list(range(len(marks)))
    for a in range(len(marks)):                                  # stars lying on each other: one label
        for b in range(a):
            if _a3ans_np.hypot(*(marks[a][2] - marks[b][2])) < A3ANS_MERGE:
                ga, gb = grp[a], grp[b]; grp = [min(ga, gb) if g in (ga, gb) else g for g in grp]
    groups = {}
    for i, g in enumerate(grp): groups.setdefault(g, []).append(i)
    keys = sorted(groups); cen = _a3ans_np.array([_a3ans_np.mean([marks[i][2] for i in groups[g]], 0) for g in keys])
    predb = [_a3ans_np.any([marks[i][3] for i in groups[g]], 0) for g in keys]
    pa = [int(m.sum()) for m in predb if m.sum() > 0]
    req = _a3ans_np.sqrt(float(_a3ans_np.percentile(pa, 75)) / _a3ans_np.pi) if pa else 6.0     # a full star's equivalent radius in this frame
    rmax = max(A3ANS_R, 1.5 * req)
    cv = R['cv'][::2, ::2]; C = sprite_pixels(sel[0]['t'], sel[0]['own'], R)
    t0 = sel[0]['t']
    allpred = _a3ans_np.any([mk[3] for mk in marks], 0)
    cand = (C == 1) & (cv >= A3ANS_CVT.get(t0, A3ANS_CV))
    if t0 in A3ANS_SHARED:                                       # star sprite shared with its trail copies: only on a predicted star
        cand = cand | ((C == 5) & (cv >= A3ANS_CVT[t0]) & _a3ans_dil(allpred, iterations=2))
        cand = _a3ans_fill(cand)
    cand = _a3ans_close(cand, A3ANS_GAP) | cand
    ys, xs = _a3ans_np.nonzero(cand)
    dd = _a3ans_np.hypot(ys[:, None] - cen[None, :, 0], xs[:, None] - cen[None, :, 1])
    nn = dd.argmin(1); keep = dd.min(1) <= rmax
    H, Wd = ids.shape
    ids[_a3ans_np.isin(ids, old)] = -1
    ids_h = R['id'][::2, ::2]
    solid = [d for d, r in R['draws'].items() if r.get('dmask')]
    eff = ~_a3ans_np.isin(ids_h, solid) & (ids_h > 0)                    # an effect sprite (smoke, flash, explosion: no depth write) is drawn here
    taken = _a3ans_np.zeros((H, Wd), bool)
    ms = {}
    for n, g in enumerate(keys):                                  # pass 1: every star's drawn pixels
        sn = keep & (nn == n)
        if not sn.any(): continue
        m = _a3ans_np.zeros((H, Wd), bool); m[ys[sn], xs[sn]] = True
        ms[g] = _a3ans_fill(m); taken |= ms[g]
    new_ks = []
    for n, g in enumerate(keys):
        m = ms.get(g)
        if m is None: continue                                    # no drawn pixel: the star is gone / behind a wall, nothing to label
        op = _a3ans_opening(m, structure=_a3ans_np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool))
        if op.sum() >= 0.6 * m.sum(): m = op
        L, c_ = _a3ans_cc(m)
        if c_ > 1: sz = _a3ans_np.bincount(L.ravel())[1:]; m = L == 1 + sz.argmax()
        if m.sum() >= A3ANS_HIDMIN:                               # hidden parts: the predicted star body where an effect sprite is in front of it
            hid = predb[n] & eff & ~taken & (C != 1)
            if hid.any(): m = _a3ans_fill(_a3ans_close(m | hid, 1))
        if m.sum() < A3ANS_MIN: continue
        i0 = groups[g][0]; lp = (marks[i0][0], marks[i0][1], 'body')
        if lp not in lids: lids.append(lp)
        kp = lids.index(lp); ids[m & (ids < 0)] = kp; new_ks.append(kp)
    ids = _mg_seam(ids, new_ks)
    sh = share.copy(); nz = (ids >= 0) & ((P['ids'] != ids))
    sh[nz] = _a3ans_np.maximum(cv[nz] / 255.0, BL.CUT); sh[ids < 0] = 0
    P2 = dict(P); P2.update(ids=ids, share=sh.astype(_a3ans_np.float32), lids=lids)
    return P2


_a3ans_prev_labels = labels


def labels(objs, cam, R, state=None, ended=(), img=None):
    P = _a3ans_prev_labels(objs, cam, R, state=state, ended=ended, img=img)
    return _a3ans_stars(P, objs, cam, R)


# ---- 326 ControllerAres ATTACK axes (Nani005Projectile 6000778; single-skin type; per-axe hulls in z_1110_merge.py) ----
# OWNER (two axes cross into an "A", row attack 90 t307): "one label, but make it right, now it includes part of the background."
# Rule: axes that touch / overlap (their own opaque pixels, as shared out by z_1110_merge._mg_assign) are ONE label; that label is the
#   axes' own drawn pixels (blades + handles: the union of their opaque parts, closed 1 px, tiny holes filled), NOT a hull, so no background
#   between the blades or inside the "A". Axes that do not touch stay separate bullets: their label is unchanged (z_1110_merge hull of the
#   bullet's own pixels, Voronoi cut, 1-px seam). The see-through blue crescent behind an axe stays out (coverage cut MG_CV of the merge fix).
#   The cluster's label lives on its first bullet (lowest label id); the other members' old hull labels are emptied.
A3ANS_AXE_TYPE = 6000778
A3ANS_AXE_TOUCH = 1       # px: opaque parts of two axes this close (8-neighbour) touch
A3ANS_AXE_CLOSE = 1       # px: closing of the cluster's drawn pixels (joins a blade and its handle)
A3ANS_AXE_SINGLE_OWN = True   # a single axe too is its own drawn pixels (plain bullet rule: 326 is not a hull brawler); False = keep the z_1110_merge hull for single axes
A3ANS_AXE_HOLE = 4        # px: enclosed holes up to this size (specks of the blade's shading) are filled; larger ones are real background


def _a3ans_axes(P, objs, R):
    import bulletlabels as BL
    ht = [o for o in objs if o['t'] == A3ANS_AXE_TYPE and o.get('drawn')]
    if len(ht) < 1: return P
    lids = list(P['lids']); ids = P['ids'].copy(); share = P['share'].copy()
    asg = {}
    for o in ht:
        lid = BL._lid(o); m = _mg_state['asg'].get(lid)
        if m is not None and m.any() and (lid, -1, 'body') in lids: asg[lid] = m
    if len(asg) < 1: return P
    keys = sorted(asg); S8 = _a3ans_np.ones((3, 3), bool)
    par = {k: k for k in keys}
    def find(k):
        while par[k] != k: k = par[k]
        return k
    for i, a in enumerate(keys):
        da = _a3ans_dil(asg[a], structure=S8, iterations=A3ANS_AXE_TOUCH)
        for b in keys[i + 1:]:
            if (da & asg[b]).any(): par[find(b)] = find(a)
    cl = {}
    for k in keys: cl.setdefault(find(k), []).append(k)
    cv = R['cv'][::2, ::2]; ch = False; new_ks = []
    for root, mem in cl.items():
        if len(mem) < 2 and not A3ANS_AXE_SINGLE_OWN: continue
        mem = sorted(mem); u = _a3ans_np.any([asg[k] for k in mem], 0)
        pad = A3ANS_AXE_CLOSE + 3
        u = _a3ans_closing(_a3ans_np.pad(u, pad), structure=_a3ans_disc(A3ANS_AXE_CLOSE))[pad:-pad, pad:-pad] | u
        holes = _a3ans_fill(u) & ~u                                  # tiny enclosed holes only
        Lh, nh = _a3ans_cc(holes)
        if nh:
            sz = _a3ans_np.bincount(Lh.ravel())[1:]
            u = u | _a3ans_np.isin(Lh, 1 + _a3ans_np.nonzero(sz <= A3ANS_AXE_HOLE)[0])
        ks = [lids.index((k, -1, 'body')) for k in mem]
        ids[_a3ans_np.isin(ids, ks)] = -1
        kk = ks[0]; ids[u & (ids < 0)] = kk; new_ks.append(kk); ch = True
    if not ch: return P
    allk = [lids.index((k, -1, 'body')) for k in keys]
    ids = _mg_seam(ids, allk)
    nz = ids != P['ids']
    share[nz & (ids >= 0)] = _a3ans_np.maximum(cv[nz & (ids >= 0)] / 255.0, BL.CUT); share[ids < 0] = 0
    P2 = dict(P); P2.update(ids=ids, share=share.astype(_a3ans_np.float32))
    return P2


_a3ans_prev_labels2 = labels


def labels(objs, cam, R, state=None, ended=(), img=None):
    P = _a3ans_prev_labels2(objs, cam, R, state=state, ended=ended, img=img)
    return _a3ans_axes(P, objs, R)


# ---- 328 GunslingerMonkey banana bullets (Colt007Projectile 6000032 attack, Colt007UltiProjectile 6000033 super; single-skin types) ----
# OWNER: "its good like this for colt [306, untouched], but theres still room for improvement for the banana bullets." (glow tube stays out)
# Installed _ab5m_328 (z_1110_misc) keeps the banana's frame-sprite shapes but closes 2 px and keeps 'no shape' edge pixels (bitten tips /
# bulging notches / spurs). Here the label is exactly the banana sprite: pixels whose atlas shape is one of the clip's banana frame shapes
# (every shape except the soft glow tube 145 and the soft discs / shadow 0, as in _ab5m_328), coverage >= A3ANS_328_CV so the antialiased
# edge and the stem tip are in, connected to what the installed rule found; closed 1 px, holes filled, one piece. Pixels with no shape that
# are not enclosed by the banana are dropped (they were glow fringe).
A3ANS_328_CV = {6000032: 20, 6000033: 40}


def _a3ans_328_hook(t, prev):
    def fn(body, ctx):
        body = prev(body, ctx) if prev else body
        if not body.any(): return body
        names = AB5M_328[t]
        SP = _a4_shape_pixels(ctx['R'], 'sc/effects_brawler.sc', names if ctx['own'] else (names[1],))
        shp = (SP > 0) & ~_a3ans_np.isin(SP, AB5M_328_GLOW)
        cand = shp & ctx['region'] & (ctx['cv'] >= A3ANS_328_CV[t])
        core = body & shp
        if not core.any(): return body
        L, n = _a3ans_cc(cand | core, structure=_a3ans_np.ones((3, 3), bool))
        keep = _a3ans_np.unique(L[core]); keep = keep[keep > 0]
        out = _a3ans_np.isin(L, keep) & (cand | core)
        out = _a3ans_fill(_a3ans_close(out, 1) & (out | shp | ctx['region']))
        L, n = _a3ans_cc(out)
        if n > 1: sz = _a3ans_np.bincount(L.ravel())[1:]; out = _a3ans_np.isin(L, 1 + _a3ans_np.nonzero(sz >= 0.2 * sz.max())[0])
        return out
    return fn


for _t_a3ans in AB5M_328:
    POST_BODY[_t_a3ans] = _a3ans_328_hook(_t_a3ans, POST_BODY.get(_t_a3ans))


# ---- 364 BanditGirlStar_remake SUPER (Shelly004UltiProjectile 6000005; single-skin type): same owner rule as 223 (coordinator, 2026-11-05):
#   "if the stars overlap, the labels can merge, just make sure that everything gets labeled". Every visible star gets a label, in the star's own
#   outline, no glow, no music-note sprites, no background.
# Cause: the predicted star bodies of this type are offset / fewer than the drawn stars (one object each, the prediction sits beside its star), so
#   the installed per-object claim (z_1111_shelly3: Voronoi from the predicted bodies) leaves stars without a seed unlabelled and gives the others
#   neighbours' pixels. Rule here: the stars are read from the drawn pixels themselves: a star = the front-most draw shows the star clip sprite
#   (code 1) at coverage >= CUT, its translucent white middle filled; touching / overlapping stars are one blob (allowed by the owner), separate
#   stars stay separate labels (the blob's seed object = the nearest predicted star; every blob its own label (object id, k, 'body'), k = -1 for the
#   first blob of an object). The old labels of the type's objects are replaced. Music notes / trail glow are other sprites (not code 1).
A3ANS_BLOB_TYPES = {6000005}
A3ANS_BLOB_MIN = 14        # px: smaller blobs are crumbs
A3ANS_BLOB_CLOSE = 1       # px: closing before the fill (joins the ring of a star the notes / flash cut)


def _a3ans_blobs(P, objs, cam, R):
    import bulletlabels as BL
    sel = [o for o in objs if o['t'] in A3ANS_BLOB_TYPES]
    if not sel: return P
    lids = list(P['lids']); ids = P['ids'].copy(); share = P['share'].copy()
    old = [k for k, l in enumerate(lids) if l[0] in {BL._lid(o) for o in sel} and l[2] == 'body']
    cv = R['cv'][::2, ::2]; C = sprite_pixels(sel[0]['t'], sel[0]['own'], R)
    cand = (C == 1) & (cv >= CUT)
    if not cand.any(): return P
    S = _a3ans_fill(_a3ans_close(cand, A3ANS_BLOB_CLOSE) | cand)
    L, n = _a3ans_cc(S, structure=_a3ans_np.ones((3, 3), bool))
    Pp = BL.labels(objs, cam, cut=0.01); pids, plids = Pp['ids'], Pp['lids']
    cen = {}
    for o in sel:
        lid = BL._lid(o)
        ks = [kk for kk, l in enumerate(plids) if l[0] == lid and l[2] == 'body' and (pids == kk).any()]
        if ks: cen[lid] = _a3ans_np.argwhere(_a3ans_np.isin(pids, ks)).mean(0)
    if not cen: return P
    ids[_a3ans_np.isin(ids, old)] = -1
    cnt = {}; new_ks = []
    names = list(cen)
    for i in range(1, n + 1):
        m = L == i
        if m.sum() < A3ANS_BLOB_MIN: continue
        c = _a3ans_np.argwhere(m).mean(0)
        lid = min(names, key=lambda q: float(_a3ans_np.hypot(*(cen[q] - c))))
        j = cnt.get(lid, 0); cnt[lid] = j + 1
        lp = (lid, j - 1, 'body')
        if lp not in lids: lids.append(lp)
        kp = lids.index(lp); ids[m & (ids < 0)] = kp; new_ks.append(kp)
    nz = (ids != P['ids']) & (ids >= 0)
    sh = share.copy(); sh[nz] = _a3ans_np.maximum(cv[nz] / 255.0, BL.CUT); sh[ids < 0] = 0
    P2 = dict(P); P2.update(ids=ids, share=sh.astype(_a3ans_np.float32), lids=lids)
    return P2


_a3ans_prev_labels3 = labels


def labels(objs, cam, R, state=None, ended=(), img=None):
    P = _a3ans_prev_labels3(objs, cam, R, state=state, ended=ended, img=img)
    return _a3ans_blobs(P, objs, cam, R)
