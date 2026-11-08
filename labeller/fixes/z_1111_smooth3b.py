# smooth3b (final review round 3, 2026-11-05; OWNER notes: 425 "ragged", 426 "ragged", 541 "there are some holes in the labels",
# 609 "some are a little ragged and there are holes in some attacks"). Override file, exec'd in idlabels' namespace. Prefix _s3b_.
#
# ===== 425 CrowBrawloween, type 6000327 (Crow009Projectile; skin 425 only) =====
# OWNER: "ragged". Cause: the dagger's clip (effects_brawler_crow.sc crow_009_atk_projectile) is drawn from the dagger sprites (jaw 526, handle 35,
# fang 524, gem 9, highlights 6 / 8 / 34 / 41 / 60) AND two big translucent green puff / glow shapes (0 and 4) that spread 7-12 px behind
# the dagger and, after a hit, form the green clover left on the crate. The label took those puffs, so it had a frayed fringe at the back
# and (after a hit) a loose blob on the crate. Rule: the label is the dagger itself = the pixels of the dagger shapes (all clip shapes but
# 0 and 4), joined over the handle gap and filled; puff pixels (0 / 4) stay only on the 1-px rim of the dagger sprite. Daggers that
# overlap (the cluster hulls of z_1103_crow_barley_gale, _cr_cluster) are left to that fix. Holes enclosed by the labels of the three
# daggers together (a gap between two overlapping daggers) are given to the nearest dagger label.
#
# ===== 426 TrickshotBrawloween, types 6000107 (Rico010Projectile, attack) and 6000108 (Rico010UltiProjectile, super); skin 426 only =====
# OWNER: "ragged". The pufferfish bullet is a translucent bubble dome with fins. (a) The dome's translucent rim fell under the coverage cut,
# so the label bit into the dome and its top edge wobbled: FADE_CUT 48 lets the rim in (16 took the soft glow as well). (b) When a super
# bullet hits, the white hit star (clip shape 15 outside the bullet, shape 20, and the trail shapes 63 / 2598 / 80 / 179) was labelled and
# glued to the bullet. Rule: the label keeps the body but only inside the convex hull (+2 px) of the bullet clip's own shapes (all but 15 / 20:
# the bullet is a convex dome with fins; sparks drawn over it hide some shapes, those pixels stay), holes filled, largest piece.
#
# ===== 541 MechaDudePirate, type 6001079 (Mecha005UltiProjectile, the green shells; skin 541 only) =====
# OWNER: "there are some holes in the labels". The two shells of one swirl overlap; the base ownership hands the pixels of one shell to
# both objects (one label 147 px, the other 484 px: the small one is a torn fragment, the large one has pockets and unlabelled gaps between
# the two). Rule: the labels of the shells of one swirl (labels up to 3 px apart) are merged, holes of the merge filled (the shells of one
# frame are one swirl), the new pixels (holes, gaps between the shells) go to the nearest shell's label; pixels that had a label keep it, thin debris arcs are cut.
#
# ===== 609 LuchadorCarreta, type 6000409 (Primo013Projectile, the fist; skin 609 only) =====
# OWNER: "some are a little ragged and there are holes in some attacks". The glove is a convex mesh with a cyan cuff; the cyan V symbol and
# the yellow spark bar are drawn OVER it, so the label (the installed z_1104_barrel_luchador / a4 fixes only close gaps of 3 px) has deep
# bites where the V / bar cross the glove (305-311, 265, 271) and tiny holes. Hidden-parts rule: the fist label is the whole glove = its
# predicted body (pred_body matches the glove mesh incl. the thumb) with the speed-strip rectangles cut by an opening (r 4), filled;
# fallback (no usable prediction): hull of the old label.
import numpy as _s3b_np
from scipy.ndimage import (binary_fill_holes as _s3b_fill, binary_dilation as _s3b_dil, binary_closing as _s3b_clo, label as _s3b_cc,
                           distance_transform_edt as _s3b_edt)
import bulletlabels as _S3BBL

_S3B_CROW = {6000327: ('sc/effects_brawler_crow.sc', ('crow_009_atk_projectile',), (0, 4))}   # type -> (file, clips, puff shape ids)
_s3b_prev_labels = labels
S3B_NEAR = 2     # px: a dagger pixel up to this far from the dagger's predicted silhouette is its own


def _s3b_dsk(r):
    y, x = _s3b_np.ogrid[-r:r + 1, -r:r + 1]; return (x * x + y * y) <= r * r + 1


def _s3b_crow_body(prev):
    def f(body, ctx):
        if not body.any(): return body
        if _cr_cluster(ctx): return prev(body, ctx) if prev else body
        file, clips, puff = _S3B_CROW[ctx['t']]
        SP = _a4_shape_pixels(ctx['R'], file, clips)
        soft = body & _s3b_np.isin(SP, [s + 1 for s in puff])
        shaped = body & ~soft & (SP > 0)
        core = shaped | (body & (SP == 0) & _s3b_dil(shaped, iterations=1))     # pixels of no shape (sprite edges) only on the rim of the dagger
        pa = ctx['pred_any']
        if pa is not None and int(pa.sum()) >= CR_MIN_PRED:        # the dagger's own pixels: near its predicted silhouette (neighbours' draws are not taken)
            own = core & _s3b_dil(pa, iterations=S3B_NEAR)
            if own.sum() >= 0.3 * pa.sum(): core = own
        if core.sum() < 20: return body
        c = _s3b_clo(_s3b_np.pad(core, 4), structure=_s3b_dsk(3))[4:-4, 4:-4]
        c = _s3b_fill(c | core)
        out = _s3b_fill(c | (soft & _s3b_dil(c, iterations=1)))
        L, n = _s3b_cc(out, structure=_s3b_np.ones((3, 3)))
        if n > 1:
            sz = _s3b_np.bincount(L.ravel())[1:]; out = L == (sz.argmax() + 1)
        return out & ctx['region']
    return f


for _t in _S3B_CROW: POST_BODY[_t] = _s3b_crow_body(POST_BODY.get(_t))


def labels(objs, cam, R, state=None, ended=(), img=None):
    P = _s3b_prev_labels(objs, cam, R, state=state, ended=ended, img=img)
    on = [o for o in objs if o['t'] in _S3B_CROW]
    if not on: return P
    ids = P['ids'].copy(); sh = P['share'].copy(); lids = P['lids']
    mine = [k for k, l in enumerate(lids) if l[2] == 'body' and any(l[0] == _S3BBL._lid(o) for o in on)]
    if mine:
        U = _s3b_np.isin(ids, mine)
        holes = _s3b_fill(U) & ~U & (ids < 0)
        Lh, nh = _s3b_cc(holes)                       # only small holes (a gap between two labels that touch at a corner is not one)
        for i in range(1, nh + 1):
            if (Lh == i).sum() > 0.08 * U.sum(): holes[Lh == i] = False
        if holes.any():
            _, (iy, ix) = _s3b_edt(~U, return_indices=True)
            ys, xs = _s3b_np.nonzero(holes)
            ids[ys, xs] = ids[iy[ys, xs], ix[ys, xs]]; sh[ys, xs] = _S3BBL.CUT
    P2 = dict(P); P2.update(ids=ids, share=sh)
    return P2


# ---- 426 Rico010 (types 6000107 attack, 6000108 super) ----
_S3B_RICO = {6000107: ('sc/effects_brawler_rico.sc', ('rico_010_atk_projectile',)),
             6000108: ('sc/effects_brawler_rico.sc', ('rico_010_ulti_projectile',))}
S3B_RICO_CUT = 48        # coverage cut (the translucent dome rim is in; below ~32 the soft glow comes in)
S3B_RICO_GROW = 2        # px: the hull of the bullet's shapes is grown by this much
S3B_RICO_STAR = (15, 20)  # shapes of the hit star; kept only inside the bullet
for _t in _S3B_RICO: FADE_CUT[_t] = S3B_RICO_CUT


def _s3b_hull(m):
    from scipy.spatial import ConvexHull
    from PIL import Image, ImageDraw
    ys, xs = _s3b_np.nonzero(m)
    if len(ys) < 3: return m
    pts = _s3b_np.c_[xs, ys].astype(float)
    try: h = ConvexHull(pts)
    except Exception: return m
    I = Image.new('L', (m.shape[1], m.shape[0]), 0); ImageDraw.Draw(I).polygon([tuple(pts[q]) for q in h.vertices], fill=1, outline=1)
    return _s3b_np.asarray(I, bool) | m


def _s3b_rico_body(body, ctx):
    if not body.any(): return body
    file, clips = _S3B_RICO[ctx['t']]
    SP = _a4_shape_pixels(ctx['R'], file, clips)
    star = body & _s3b_np.isin(SP, [q + 1 for q in S3B_RICO_STAR])
    main = body & (SP > 0) & ~star
    if main.sum() < 20: return body
    pa = ctx['pred_any']
    if pa is not None and int(pa.sum()) >= 20:
        near = _s3b_dil(pa, iterations=4)
        if (main & near).sum() >= 0.3 * main.sum(): main &= near
    # the bullet's extent = the convex hull of its own shapes (the bullet is a convex dome with fins; sparks / flashes drawn over it hide some
    # of its shapes: those pixels stay inside the hull, the hidden-parts rule); body outside that extent (the hit star and its trail) is dropped
    hull = _s3b_hull(main)
    out = _s3b_fill(body & _s3b_dil(hull, iterations=S3B_RICO_GROW))
    L, n = _s3b_cc(out, structure=_s3b_np.ones((3, 3)))
    if n > 1:
        sz = _s3b_np.bincount(L.ravel())[1:]; out = L == (sz.argmax() + 1)
    return out & ctx['region']


for _t in _S3B_RICO: POST_BODY[_t] = _s3b_rico_body


# ---- 541 Mecha005UltiProjectile (6001079) ----
S3B_MECHA = 6001079
S3B_MECHA_OPEN = 2     # px: disc radius that removes thin debris arcs from a shell label
_s3b_mecha_c = {}
_s3b_mecha_prev = POST_BODY.get(S3B_MECHA)


def _s3b_mecha_body(body, ctx):
    pb = ctx['pred_body']
    if pb is not None and pb.any():
        ys, xs = _s3b_np.nonzero(pb); _s3b_mecha_c[ctx['o']['id']] = (float(ys.mean()), float(xs.mean()), int(pb.sum()))
    return _s3b_mecha_prev(body, ctx) if _s3b_mecha_prev else body


POST_BODY[S3B_MECHA] = _s3b_mecha_body
_s3b_prev_labels2 = labels


def labels(objs, cam, R, state=None, ended=(), img=None):
    _s3b_mecha_c.clear()
    P = _s3b_prev_labels2(objs, cam, R, state=state, ended=ended, img=img)
    on = [o for o in objs if o['t'] == S3B_MECHA]
    if len(on) < 1: return P
    ids = P['ids'].copy(); sh = P['share'].copy(); lids = P['lids']
    ks = {}
    for o in on:
        for k, l in enumerate(lids):
            if l[0] == _S3BBL._lid(o) and l[2] == 'body' and (ids == k).any(): ks[o['id']] = k
    if not ks: return P
    oid = [i for i in ks if i in _s3b_mecha_c]
    # swirls: objects whose labels are within 3 px of each other
    masks = {i: (ids == ks[i]) for i in oid}
    dil = {i: _s3b_dil(masks[i], iterations=3) for i in oid}
    grp = {i: i for i in oid}
    def find(a):
        while grp[a] != a: a = grp[a]
        return a
    for a in oid:
        for b in oid:
            if a < b and (dil[a] & masks[b]).any(): grp[find(b)] = find(a)
    groups = {}
    for i in oid: groups.setdefault(find(i), []).append(i)
    cv = R['cv'][::2, ::2]
    for g in groups.values():
        if len(g) < 2: continue
        U = _s3b_np.zeros(ids.shape, bool)
        for i in g: U |= masks[i]
        Uc = _s3b_fill(_s3b_clo(_s3b_np.pad(U, 4), structure=_s3b_dsk(2))[4:-4, 4:-4] | U)
        free = (ids < 0) | _s3b_np.isin(ids, [ks[i] for i in g])
        Uc &= free
        from scipy.ndimage import binary_opening as _s3b_open0
        Uo = _s3b_open0(_s3b_np.pad(Uc, 4), structure=_s3b_dsk(S3B_MECHA_OPEN))[4:-4, 4:-4]     # thin debris arcs of the swirl
        Uo = (_s3b_dil(Uo, iterations=1) & Uc) | Uo
        if Uo.sum() >= 0.75 * Uc.sum(): Uc = Uo
        # pixels that already had a label keep it (except the thin arcs cut above); pixels that are new (holes, pockets between the shells) go
        # to the nearest of the swirl's original labels
        keep = U & Uc
        for i in g: ids[masks[i]] = -1
        ys, xs = _s3b_np.nonzero(Uc)
        dist = _s3b_np.stack([_s3b_edt(~masks[i]) for i in g], 0)
        own = dist[:, ys, xs].argmin(0)
        for n_, i in enumerate(g):
            sel = own == n_
            was = masks[i][ys, xs] & keep[ys, xs]
            nw = sel & ~(U[ys, xs])
            put = was | nw
            ids[ys[put], xs[put]] = ks[i]; sh[ys[put], xs[put]] = _s3b_np.maximum(cv[ys[put], xs[put]] / 255.0, _S3BBL.CUT)
    # thin curved debris arcs (the green swoosh) hanging on a shell: opening with a 2-px disc, largest part kept (never more than 25 % of the area)
    for i, k_ in ks.items():
        m = ids == k_
        if m.sum() < 40: continue
        o_ = _s3b_np.pad(m, 4)
        from scipy.ndimage import binary_opening as _s3b_open
        o_ = _s3b_open(o_, structure=_s3b_dsk(S3B_MECHA_OPEN))[4:-4, 4:-4]
        o_ = _s3b_dil(o_, iterations=1) & m | o_
        L_, n_ = _s3b_cc(o_, structure=_s3b_np.ones((3, 3)))
        if n_ > 1:
            sz = _s3b_np.bincount(L_.ravel())[1:]; o_ = L_ == (sz.argmax() + 1)
        if o_.sum() >= 0.75 * m.sum() and o_.sum() < m.sum():
            ids[m & ~o_] = -1; sh[m & ~o_] = 0
    P2 = dict(P); P2.update(ids=ids, share=sh)
    return P2


# ---- 609 Primo013 fist (6000409) ----
_s3b_fist_prev = POST_BODY.get(6000409)
S3B_FIST_OPEN = 4      # px: disc radius of the opening that cuts the speed strips
S3B_FIST_MIN = 100     # px: smallest predicted fist trusted


def _s3b_fist(body, ctx):
    if _s3b_fist_prev is not None: body = _s3b_fist_prev(body, ctx)
    if body.sum() < 40: return body
    # the fist's predicted body (pred_body: the glove mesh at the object's own transform; it matches the glove incl. the thumb) is the
    # silhouette; its thin speed-strip rectangles behind the glove are cut by an opening; the label keeps the glove pixels the old rule found
    pb = ctx['pred_body']
    if pb is not None and int(pb.sum()) >= S3B_FIST_MIN:
        from scipy.ndimage import binary_opening as _s3b_open1
        o = _s3b_open1(_s3b_np.pad(pb, 8), structure=_s3b_dsk(S3B_FIST_OPEN))[8:-8, 8:-8]
        L, n = _s3b_cc(o, structure=_s3b_np.ones((3, 3)))
        if n >= 1:
            sz = _s3b_np.bincount(L.ravel())[1:]; o = L == (sz.argmax() + 1)
            if (o & body).sum() >= 0.5 * body.sum(): return _s3b_fill(o)
    L, n = _s3b_cc(body, structure=_s3b_np.ones((3, 3)))
    if n > 1:
        sz = _s3b_np.bincount(L.ravel())[1:]; body = _s3b_np.isin(L, 1 + _s3b_np.nonzero(sz >= 0.12 * sz.max())[0])
    return (_s3b_hull(body) & ctx['region']) | body


POST_BODY[6000409] = _s3b_fist
