# away_1105 group "merge" (owner final review 2026-11-04/05, BAD pages; binding). Override file, exec'd in idlabels' namespace.
# Global names carry the prefix _mg_.
#
# 223 BanditGirlPrereg (type 6000002 Shelly002Projectile attack; 6000003 super checked too). OWNER: "too ragged, some have holes,
#   some are ragged and some get merged even when they shouldnt be". Cause: one logic object draws 4 separate stars (clip pellets
#   0-3, one star sprite each, scattered around a grey disc); the object got ONE label over all four stars (z_1103_shelly closing
#   bridges the gaps; the global fill_smooth joins pieces of one label), so the stars of one object hang together by thin bridge lines
#   and stars of different objects that touch read as one outline. Rule (owner: separate bullets = separate labels, always, even
#   touching): ONE label per star. The object's label pixels are split by the clip's own pellets (BL.labels prediction per pellet,
#   each pellet = one star sprite), every star is filled solid (holes from the grey disc drawn over it) with its own outline, the bridge
#   pixels between stars are dropped, and where two star labels touch a 1-px seam is left between them (the review page outlines
#   the union of the labels, so without it touching stars look like one blob).
#   Label ids: (object id, pellet, 'body') per star; the object's old (id, -1, 'body') is emptied. meta is keyed by object id, so
#   every star keeps its class.
#
# 257 / 326: see the block '257 BowDudeDemonic ... 326 ControllerAres' below (one hull per bullet from the bullet's own pixels, Voronoi cut
# between neighbours, 1-px seam). 326 uses the connected parts of the opaque axe pixels assigned by predicted position.
# Type ids: 6000002 (+6000003 crumbs only) = skin 223 only; 6000363 = skin 257 only; 6000778 = skin 326 only (checked with owners.projectile).

import heapq
import numpy as _mg_np
from scipy.ndimage import (binary_fill_holes as _mg_fill, label as _mg_cc, distance_transform_edt as _mg_edt,
                           binary_closing as _mg_closing, binary_opening as _mg_opening, gaussian_filter as _mg_gauss,
                           maximum_filter as _mg_maxf)
_MG_CROSS = _mg_np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool)

_MG_STAR_TYPES = {6000002}          # 223 attack: one star sprite per clip pellet
MG_NEAR = 3                         # px: a body pixel of the object up to this far from a star's predicted outline belongs to it
MG_MERGE = 7.0                      # px: pellet centres closer than this are one star
MG_SIGMA = 0.7                      # smoothing of the distance transform before the peak search
MG_PEAK_MIN = 1.5                   # px: a bullet's distance-transform peak is at least this deep
MG_PEAK_DIST = 4.5                  # px: peaks closer than this are one bullet
MG_PRED_GROW = 3                    # px: a body pixel up to this far from the bullet's own predicted body is its own
MG_R0 = 4.8                         # px: radius of one star around its predicted centre (half-res)
MG_R = 9.0                          # px: a body pixel farther than this from every star centre is a bridge / crumb: dropped
MG_MIN = 6                          # px: a star label smaller than this is dropped when it has no own pixels


def _mg_seam(ids, ks):
    """remove 1 px along every contact between two different labels of ks (a pixel of the label with the higher index that has a
    4-neighbour of a lower index): touching bullets stay two outlines on the review page"""
    out = ids.copy()
    isk = _mg_np.isin(ids, ks)
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nb = _mg_np.roll(ids, (dy, dx), (0, 1))
        nk = _mg_np.roll(isk, (dy, dx), (0, 1))
        cut = isk & nk & (ids > nb) & (nb >= 0)
        # no wrap around the frame border
        if dy == 1: cut[0, :] = False
        if dy == -1: cut[-1, :] = False
        if dx == 1: cut[:, 0] = False
        if dx == -1: cut[:, -1] = False
        out[cut & (out == ids)] = -1
    return out


def _mg_split_stars(P, objs, cam, types, R):
    """one label per star: the object's label pixels are shared out between the stars (clip pellets) by distance to the pellet's
    predicted centre. A body pixel farther than the star's own radius from every star centre (the middle of a 'flower' of stars, a
    grey trail particle drawn over them) is dropped"""
    import bulletlabels as BL
    sel = [o for o in objs if o['t'] in types and o.get('drawn') and len(o.get('pellets') or []) > 1]
    if not sel: return P
    ids = P['ids'].copy(); share = P['share'].copy(); lids = list(P['lids'])
    Pp = BL.labels(objs, cam, cut=0.01)                       # the prediction per pellet (one star sprite each)
    pids, plids = Pp['ids'], Pp['lids']
    marks = []; old = []
    for o in sel:
        lid = BL._lid(o)
        if (lid, -1, 'body') not in lids: continue
        old.append(lids.index((lid, -1, 'body')))
        pk = [(kk, l[1]) for kk, l in enumerate(plids) if l[0] == lid and l[1] >= 0 and l[2] == 'body' and (pids == kk).any()]
        cen = [_mg_np.argwhere(pids == kk).mean(0) for kk, _ in pk]
        grp = list(range(len(pk)))
        for a in range(len(pk)):                                # pellets of one object closer than MG_MERGE: one blob (first
            for b in range(a):                                  # frames of the shot: the 4 stars sit on top of each other)
                if _mg_np.hypot(*(cen[a] - cen[b])) < MG_MERGE:
                    ga, gb = grp[a], grp[b]
                    grp = [min(ga, gb) if g in (ga, gb) else g for g in grp]
        for g in sorted(set(grp)):
            mem = [i for i in range(len(pk)) if grp[i] == g]
            c = _mg_np.mean([cen[i] for i in mem], 0)
            rad = MG_R0 + max(_mg_np.hypot(*(cen[i] - c)) for i in mem)
            marks.append((lid, pk[mem[0]][1], c, rad))
    if not marks: return P
    U = _mg_np.isin(ids, old)
    if not U.any(): return P
    op = _mg_opening(U, structure=_MG_CROSS)                   # the closing bridges of the old label (thin lines) off
    if op.sum() >= 0.7 * U.sum(): U = op
    ys, xs = _mg_np.nonzero(U)
    C = _mg_np.array([m[2] for m in marks]); Rr = _mg_np.array([m[3] for m in marks])
    dd = _mg_np.hypot(ys[:, None] - C[None, :, 0], xs[:, None] - C[None, :, 1]) - Rr[None, :]
    nn = dd.argmin(1); keep = dd.min(1) <= 0
    H, Wd = ids.shape
    new_ks = []
    ids[_mg_np.isin(ids, old)] = -1
    for n, (lid, pel, c, rad) in enumerate(marks):
        sn = keep & (nn == n)
        if not sn.any(): continue
        m = _mg_np.zeros((H, Wd), bool); m[ys[sn], xs[sn]] = True
        m = _mg_fill(_mg_closing(_mg_np.pad(m, 2), iterations=1)[2:-2, 2:-2] | m)          # solid: holes
        op = _mg_opening(m, structure=_MG_CROSS)                                           # 1-px spurs off
        if op.sum() >= 0.6 * m.sum(): m = op
        sq = _mg_opening(m, structure=_mg_np.ones((3, 3), bool))                           # tails 1-2 px wide off, if the star keeps its body
        if sq.sum() >= 0.85 * m.sum(): m = sq
        L, c_ = _mg_cc(m)
        if c_ > 1:                                                                         # the star's largest part only
            sz = _mg_np.bincount(L.ravel())[1:]; m = L == 1 + sz.argmax()
        if m.sum() < MG_MIN: continue
        lp = (lid, pel, 'body')
        if lp not in lids: lids.append(lp)
        kp = lids.index(lp)
        ids[m & (ids < 0)] = kp
        new_ks.append(kp)
    ids = _mg_seam(ids, new_ks)
    P2 = dict(P); P2.update(ids=ids, share=share, lids=lids)
    return P2


_MG_CRUMB_TYPES = {6000003}          # 223 super: nine objects of interleaved stars
MG_CRUMB_PX = 14                    # px: a part of a label smaller than this (and under 6% of its largest part) is a crumb


def _mg_crumbs(P, objs, types):
    """223 super: tiny separate parts of a label (specks inside / beside another object's label) are dropped; the global step then
    fills the holes they leave in the label around them"""
    import bulletlabels as BL
    keys = {BL._lid(o) for o in objs if o['t'] in types}
    ids = P['ids']; lids = P['lids']; ch = False
    for k, l in enumerate(lids):
        if l[0] not in keys or l[2] != 'body': continue
        m = ids == k
        if not m.any(): continue
        L, n = _mg_cc(m)
        if n < 2: continue
        sz = _mg_np.bincount(L.ravel())[1:]
        bad = [i + 1 for i in range(n) if sz[i] < MG_CRUMB_PX and sz[i] < 0.06 * sz.max()]
        if bad:
            if not ch: ids = ids.copy(); share = P['share'].copy(); ch = True
            d = _mg_np.isin(L, bad); ids[d] = -1; share[d] = 0
    if not ch: return P
    P2 = dict(P); P2.update(ids=ids, share=share)
    return P2


# ---- 257 BowDudeDemonic (6000363 Bo004Projectile) and 326 ControllerAres (6000778 Nani005Projectile): separate bullets stay separate.
# OWNER 257: "they shouldnt be merged, no matter how close they are"; 326: "some are merged even though theres space between them".
# Cause: the type's label is ONE convex hull per object (z_1103_bo_ruffs_shaman / z_1103_misc_b). The hulls of neighbouring bullets
# overlap (a hull of a slanted bullet reaches over the next one, and in 257 it also took the brawler's own model pixels), so their
# labels fuse into one outline on the review page. Rule (owner): one hull per bullet, but a hull never takes pixels nearer to another
# bullet's own body than to its own, never takes the brawler's model, and a 1-px seam is left wherever two bullets touch.
_MG_HULL_TYPES = {6000363, 6000778}
_MG_COMP_TYPES = {6000778}          # hulls from the drawn pixels' connected parts (the prediction is off for the spinning axes)
MG_PART_MIN = 8                     # px: connected parts smaller than this are specks
MG_CV = 110                         # coverage (0-255) of the ID pass: an axe pixel is at least this opaque
MG_JOIN = 2                         # px: closing that joins an axe's handle and blade into one part
_mg_state = dict(collect=False, stash={}, asg={})


def _mg_assign(bodies, preds):
    """{lid: mask}: the pooled opaque pixels of the bullets (axes) shared out by connected part. The chain's own ownership of these
    pixels is unreliable (the client batches the axes of one shot into one draw: one object got both axes, the other a few
    pixels), so each part goes to the bullet(s) whose PREDICTED position is nearest to it (every bullet takes the part nearest its
    prediction, so none is lost; a part nobody is nearest to goes to the nearest bullet). A part taken by several bullets
    (touching axes) is cut between them by the nearest predicted centre."""
    if not bodies: return {}
    pool = _mg_np.any(list(bodies.values()), 0)
    pad = MG_JOIN + 1
    L, n = _mg_cc(_mg_closing(_mg_np.pad(pool, pad), iterations=MG_JOIN)[pad:-pad, pad:-pad] | pool)
    keys = list(bodies)
    cen = {k: _mg_np.argwhere(preds[k]).mean(0) if preds[k].any() else _mg_np.argwhere(bodies[k]).mean(0) for k in keys}
    comps = []
    for c in range(1, n + 1):
        m = (L == c) & pool
        if m.sum() >= MG_PART_MIN: comps.append(m)
    out = {k: _mg_np.zeros(pool.shape, bool) for k in keys}
    if not comps: return {k: bodies[k].copy() for k in keys}
    pts = [_mg_np.argwhere(m) for m in comps]
    def dist(k, i):
        return float(_mg_np.hypot(*(pts[i] - cen[k]).T).min())
    take = {i: [] for i in range(len(comps))}
    for k in keys: take[min(range(len(comps)), key=lambda i: dist(k, i))].append(k)
    for i, m in enumerate(comps):
        ks = take[i] or [min(keys, key=lambda k: dist(k, i))]
        if len(ks) == 1: out[ks[0]] |= m; continue
        ys, xs = _mg_np.nonzero(m)
        d = _mg_np.stack([_mg_np.hypot(ys - cen[k][0], xs - cen[k][1]) for k in ks]).argmin(0)
        for j, k in enumerate(ks): out[k][ys[d == j], xs[d == j]] = True
    for k in keys:
        if not out[k].any(): out[k] = bodies[k].copy()
    return out


def _mg_hull_post(t):
    prev = POST_BODY.get(t)

    def fn(body, ctx):
        if prev is not None: body = prev(body, ctx)
        if not body.any(): return body
        st, cam = ctx.get('state'), ctx.get('cam')
        if st is not None and cam is not None and t == 6000363:
            try: body = body & ~_bo_brawler_draws(ctx['R'], st, cam)       # the brawler's own model is never the arrow
            except Exception: pass
        lid = BL._lid(ctx['o'])
        if t in _MG_COMP_TYPES:
            body = body & (ctx['cv'] >= MG_CV)                  # the axe itself is opaque; the translucent blue crescent behind it is not
            if not body.any(): return body
            if _mg_state['collect']:
                _mg_state['stash'][lid] = (body.copy(), (ctx['pred_body'] if ctx['pred_body'].any() else ctx['pred_any']).copy())
                return body
            mine = _mg_state['asg'].get(lid)
            if mine is None or not mine.any(): mine = body
            h = hull_mask(mine)
            oth = [m for k, m in _mg_state['asg'].items() if k != lid]
            if oth:
                O = _mg_np.any(oth, 0)
                if O.any(): h = h & (_mg_edt(~mine) <= _mg_edt(~O)) & ~_dilate(O & ~mine, 1)
            return h | mine
        pb = ctx['pred_body'] if ctx['pred_body'].any() else ctx['pred_any']
        if _mg_state['collect']:
            _mg_state['stash'][lid] = (body.copy(), pb.copy())
            return body
        others = [(b, p) for k, (b, p) in _mg_state['stash'].items() if k != lid]
        # the batched draws of neighbouring bullets share pixels: only the pixels on the bullet's OWN predicted body (the prediction
        # follows each arrow's own sprites exactly) are its
        own = body & _dilate(pb, MG_PRED_GROW)
        if own.sum() >= 0.5 * body.sum() or not others: body = own if own.any() else body
        h = hull_mask(body)
        if others:
            body_o = _mg_np.any([b & _dilate(p, MG_PRED_GROW) for b, p in others], 0)
            if body_o.any(): h = h & (_mg_edt(~body) <= _mg_edt(~body_o)) & ~_dilate(body_o & ~body, 1)  # nearer to another bullet: its
        return h | body
    POST_BODY[t] = fn
    ONE_LABEL.pop(t, None)                                                  # the hull is done in fn (after the cleaning)


_mg_hull_post(6000363)
_mg_hull_post(6000778)

_mg_prev_labels = labels


def labels(objs, cam, R, state=None, ended=(), img=None):
    ht = [o for o in objs if o['t'] in _MG_HULL_TYPES]
    _mg_state['stash'] = {}
    if ht:                                                    # pass 1: every bullet's own body (no hull), for the Voronoi cut
        _mg_state['collect'] = True
        try: _mg_prev_labels(objs, cam, R, state=state, ended=ended, img=img)
        finally: _mg_state['collect'] = False
        _mg_state['asg'] = _mg_assign(*[{k: v[i] for k, v in _mg_state['stash'].items() if k in {BL._lid(o) for o in ht if o['t'] in _MG_COMP_TYPES}} for i in (0, 1)])
    P = _mg_prev_labels(objs, cam, R, state=state, ended=ended, img=img)
    if ht:
        keys = {BL._lid(o) for o in ht}
        ks = [k for k, l in enumerate(P['lids']) if l[0] in keys and l[2] == 'body']
        if ks:
            ids = _mg_seam(P['ids'].copy(), ks); sh = P['share'].copy(); sh[ids < 0] = 0
            P = dict(P); P.update(ids=ids, share=sh)
    cv = R['cv'][::2, ::2]
    P = _mg_crumbs(P, objs, _MG_CRUMB_TYPES)
    P2 = _mg_split_stars(P, objs, cam, _MG_STAR_TYPES, R)
    if P2 is not P:                                           # new / changed pixels: share = the drawn coverage, at least the cut
        ch = (P2['ids'] != P['ids']) & (P2['ids'] >= 0)
        sh = P2['share'].copy(); sh[ch] = _mg_np.maximum(cv[ch] / 255.0, BL.CUT); sh[P2['ids'] < 0] = 0
        P2['share'] = sh.astype(_mg_np.float32)
    return P2
