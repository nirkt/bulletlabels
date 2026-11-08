# round4 group "shelly4" (owner re-check 2026-11-06, pages marked BAD; binding). Override file, exec'd in idlabels' namespace after all installed
# fixes (z_1110_merge, z_1111_*, z_1112_answers3) and before the global smoothing v2. Global names carry the prefix _sh4_.
#
# ---- 223 BanditGirlPrereg: attack 6000002 (Shelly002Projectile), super 6000003 (Shelly002UltiProjectile); single-skin types (223 only) ----
# OWNER 11-06: "why are there holes, when bullets are close, the labels should be merged" (attack 0 t269; super 45 t345..353).
# Cause: z_1110_merge / z_1112_answers3 give every star its own label and cut a 1-px seam (_mg_seam) wherever two stars touch, so a cluster of
#   touching stars reads as a lattice of holes.
# Rule (owner, overrides the 1-px-seam idea for this case): stars whose labels touch or lie within _sh4_GAP px of each other are ONE label: the
#   union of their pixels, the seams / channels / enclosed gaps between them closed (only pixels no other label owns), outer outline only
#   (holes filled). Stars farther apart stay separate labels.
import numpy as _sh4_np
from scipy.ndimage import (binary_fill_holes as _sh4_fill, label as _sh4_cc, binary_closing as _sh4_closing, binary_opening as _sh4_bopen, binary_dilation as _sh4_dil, distance_transform_edt as _sh4_edt)
import bulletlabels as _sh4_BL

_SH4_STAR_TYPES = {6000002, 6000003}
_sh4_GAP = 2.5          # px (half res): labels of stars whose pixel gap is at most about this (touching, or a 1-2 px seam) are one label
_sh4_CLOSE = 2        # px: disc closing radius of the merged blob (seams / channels between the stars)


def _sh4_disc(r):
    k = int(_sh4_np.ceil(r)); y, x = _sh4_np.ogrid[-k:k + 1, -k:k + 1]
    return (x * x + y * y) <= r * r + 1e-6


def _sh4_merge_stars(P, objs, R):
    keys = {_sh4_BL._lid(o) for o in objs if o['t'] in _SH4_STAR_TYPES}
    if not keys: return P
    ids = P['ids'].copy(); share = P['share'].copy(); lids = P['lids']
    ks = [k for k, l in enumerate(lids) if l[0] in keys and l[2] == 'body' and (ids == k).any()]
    if len(ks) < 1: return P
    masks = {k: ids == k for k in ks}
    par = {k: k for k in ks}
    def find(k):
        while par[k] != k: k = par[k]
        return k
    dist = {k: _sh4_edt(~masks[k]) for k in ks}
    pairs = []
    for i, a in enumerate(ks):
        for b in ks[i + 1:]:
            if dist[a][masks[b]].min() <= _sh4_GAP: par[find(b)] = find(a); pairs.append((a, b))
    cl = {}
    for k in ks: cl.setdefault(find(k), []).append(k)
    cv = R['cv'][::2, ::2]
    ch = False
    for root, mem in cl.items():
        if len(mem) < 2: continue
        mem = sorted(mem); u = _sh4_np.any([masks[k] for k in mem], 0)
        p = _sh4_CLOSE + 3
        c = _sh4_closing(_sh4_np.pad(u, p), structure=_sh4_disc(_sh4_CLOSE))[p:-p, p:-p] | u
        for a, b in pairs:                                     # the convex hull of every touching pair: no thin necks / wires between two stars
            if a in mem: c = c | hull_mask(masks[a] | masks[b])
        c = _sh4_fill(c)
        kk = mem[0]
        for k in mem[1:]: ids[ids == k] = -1
        new = c & ((ids < 0) | (ids == kk))
        ids[new] = kk
        share[new & ~u] = _sh4_np.maximum(cv[new & ~u] / 255.0, _sh4_BL.CUT)
        ch = True
    if not ch: return P
    share[ids < 0] = 0
    P2 = dict(P); P2.update(ids=ids, share=share.astype(_sh4_np.float32))
    return P2


_sh4_prev_labels = labels


def labels(objs, cam, R, state=None, ended=(), img=None):
    P = _sh4_prev_labels(objs, cam, R, state=state, ended=ended, img=img)
    return _sh4_merge_stars(P, objs, R)


# ---- 240 ShotgunGirlWitch: attack 6000006 (Shelly003Projectile), super 6000007 (Shelly003UltiProjectile); single-skin types (240 only) ----
# OWNER 11-06 (flagged: attack 0 t265 / t267 "missing label", t269 "holes / ragged"; super 45 t345 / t347 / t353 "missing label"); owner answers
#   11-06: the thin cyan rims cling to the bullet and DO count ("include them").
# Cause: the installed rule (z_1111_shelly3) builds each pellet from the ID pass: only the front-most draw's pixels at coverage >= CUT plus the
#   predicted body. A translucent / overlapped pellet shows no ID-pass body pixels where another sprite is drawn in front, so whole leaves of the
#   super (the middle band at t345 / t347, upper-left leaves at t353) get no label, the attack ghosts get notches and holes, and the thin
#   cyan rim (coverage 30-50) is left out.
# Rule (data-derived from the game's own draw data, as z_1103_arcade_electro / z_1111_cutoff3 (2) do for Arcade): every pellet's drawn sprite is
#   rasterised from the object's clip pieces (opaque normal-blend pieces: the ghost / leaf body, its rim and the blue bubbles; the big soft
#   alpha-115 / 154 glow disc is NOT a body piece), coverage >= 1/2 of the sprite, so each bullet gets its full silhouette incl. the rim,
#   whatever is drawn in front of it in the ID pass. Pixels where a solid world draw (wall, crate, character) is the front-most draw are
#   left out (the bullet is behind it); effects (smoke, flash) do not cut it. Objects whose sprites overlap or touch (gap <= 1.5 px) are one
#   group and get ONE label (the group's lowest object id, -1, 'body'): union closed 1.5 px, holes filled (owner 11-06 on 223: close bullets = one label).
#   Objects that have no drawn piece keep the installed label.
import masks as _sh4_masks
import projdraw as _sh4_PD

_SH4_T240 = {6000006, 6000007}
_sh4_ALPHA240 = 250       # clip pieces drawn with multiplier alpha >= this are body (the glow disc is 115 / 154)
_sh4_MIN240 = 10
_sh4_GAP240 = 1.5          # px: sprites of two objects this close (touching / overlapping) are one group / one label


def _sh4_raster(o, cam, pieces_ok):      # pieces_ok(q) -> bool: which clip pieces count as body
    S = _sh4_PD.sc(o['file']); m = _sh4_np.zeros((_sh4_masks.H_FULL, _sh4_masks.W_FULL), bool)
    for q in o['parts'].get('clip', []):
        if q.get('blend') != 'normal' or int(q['mul'][3]) < _sh4_ALPHA240 or not pieces_ok(q): continue
        r = _sh4_masks.raster_poly(S, q, cam)
        if r is None: continue
        x0, y0, C_, T = r
        mm = (1.0 - T[..., 0]) >= 0.5
        m[y0:y0 + mm.shape[0], x0:x0 + mm.shape[1]] |= mm
    return m[::2, ::2]


def _sh4_t240(P, objs, cam, R):
    sel = [o for o in objs if o['t'] in _SH4_T240 and o.get('drawn', True)]
    if not sel: return P
    ids = P['ids'].copy(); share = P['share'].copy(); lids = list(P['lids'])
    cv = R['cv'][::2, ::2]; ids_h = R['id'][::2, ::2]
    solid = [d for d, r in R['draws'].items() if r.get('dmask')]
    front = ~_sh4_np.isin(ids_h, solid)
    res = []
    for o in sel:
        m = _sh4_raster(o, cam, lambda q: True) & front
        if m.sum() < _sh4_MIN240: continue
        res.append((o, m))
    if not res: return P
    for o, m in res:                                           # the installed labels of these objects are replaced
        for k, l in enumerate(lids):
            if l[0] == _sh4_BL._lid(o) and l[2] == 'body': ids[ids == k] = -1
    # objects whose sprites overlap / touch are ONE bullet group (owner 11-06, 223: "when bullets are close, the labels should be merged"):
    # their sprites are interleaved, so cutting them apart would leave fragments and seams
    par = list(range(len(res)))
    def find(i):
        while par[i] != i: i = par[i]
        return i
    dist = [_sh4_edt(~m) for _, m in res]
    for i in range(len(res)):
        for j in range(i):
            if dist[i][res[j][1]].min() <= _sh4_GAP240: par[find(i)] = find(j)
    grp = {}
    for i in range(len(res)): grp.setdefault(find(i), []).append(i)
    taken = _sh4_np.zeros(ids.shape, bool)
    for root in sorted(grp):
        mem = grp[root]; o = res[min(mem)][0]
        m = _sh4_np.any([res[i][1] for i in mem], 0)
        p = 3
        g = _sh4_closing(_sh4_np.pad(m, p), structure=_sh4_disc(1.5))[p:-p, p:-p] | m
        g = _sh4_fill(g)
        L, n = _sh4_cc(g, structure=_sh4_np.ones((3, 3)))
        if n > 1:
            sz = _sh4_np.bincount(L.ravel())[1:]
            g = _sh4_np.isin(L, 1 + _sh4_np.nonzero(sz >= 0.1 * sz.max())[0])
        g &= (ids < 0) & ~taken & front
        if g.sum() < _sh4_MIN240: continue
        lp = (_sh4_BL._lid(o), -1, 'body')
        if lp not in lids: lids.append(lp)
        kk = lids.index(lp); ids[g] = kk; taken |= g
        share[g] = _sh4_np.maximum(cv[g] / 255.0, _sh4_BL.CUT)
    share[ids < 0] = 0
    P2 = dict(P); P2.update(ids=ids, share=share.astype(_sh4_np.float32), lids=lids)
    return P2


_sh4_prev_labels2 = labels


def labels(objs, cam, R, state=None, ended=(), img=None):
    P = _sh4_prev_labels2(objs, cam, R, state=state, ended=ended, img=img)
    return _sh4_t240(P, objs, cam, R)


# ---- 364 BanditGirlStar_remake: attack 6000004 (Shelly009Projectile), super 6000005 (Shelly009UltiProjectile); single-skin types (364 only) ----
# OWNER 11-06 (flagged: attack 90 t313 "missing label"; super 45 t345 / t347 "holes / ragged", t349 "missing label"). The big cyan hit star at the
#   crate stays unlabelled (owner answer 11-06); the music notes / glow are not labelled.
# Cause: each star is a thin cyan ring (clip body sprite, ID-pass code 1, coverage >= CUT) round a translucent white middle. The installed
#   rules (z_1111_shelly3 / z_1112_answers3) close the ring by 1 px only, so stars whose ring is covered / cut by a neighbouring star or by
#   the glow stay open: the label is a fragment (holes, ragged bites) or the star gets none (no predicted body near it for the claim).
# Rule (owner: every visible star labelled, in its own outline, overlapping stars may merge): the ring pixels (code 1, coverage >= _sh4_CVT364)
#   are closed with a disc of _sh4_CLOSE364 px (joins ring pieces cut by a neighbour, closes the open rings) and the holes filled: the star
#   silhouette incl. its white middle. Connected silhouettes = one label each (stars lying on / touching each other are one blob), crumbs
#   (< _sh4_MIN364 px) dropped. Each blob is a label (object id, k, 'body') of the nearest predicted star's object (as z_1112 does), so every
#   drawn star has a label, also those with no predicted body beside them. The installed labels of these types are replaced.
_SH4_T364 = {6000004, 6000005}
_sh4_CVT364 = 110
_sh4_CLOSE364 = {6000004: 2.0, 6000005: 2.5}     # the super's stars are packed (rings cut by neighbours): wider closing
_sh4_MIN364 = 14
_sh4_OPEN364 = 1.5
_sh4_NEAR364 = {6000004}
_sh4_NEAR364R = 6


def _sh4_t364(P, objs, cam, R):
    import bulletlabels as BL
    out = P
    for t in sorted(_SH4_T364):
        sel = [o for o in objs if o['t'] == t]
        if not sel: continue
        lids = list(out['lids']); ids = out['ids'].copy(); share = out['share'].copy()
        old = [k for k, l in enumerate(lids) if l[0] in {BL._lid(o) for o in sel} and l[2] == 'body']
        cv = R['cv'][::2, ::2]; C = sprite_pixels(t, sel[0]['own'], R)
        ring = (C == 1) & (cv >= _sh4_CVT364)
        if not ring.any(): continue
        rc = _sh4_CLOSE364[t]; p = int(_sh4_np.ceil(rc)) + 3
        S = _sh4_closing(_sh4_np.pad(ring, p), structure=_sh4_disc(rc))[p:-p, p:-p] | ring
        S = _sh4_fill(S)
        po = int(_sh4_np.ceil(_sh4_OPEN364)) + 2                  # thin necks between two stars (closing bridges) opened away, tips restored
        so = _sh4_bopen(_sh4_np.pad(S, po), structure=_sh4_disc(_sh4_OPEN364))[po:-po, po:-po]
        if so.sum() >= 0.8 * S.sum(): S = so | (S & _sh4_dil(so, iterations=1) & (ring | _sh4_dil(ring, iterations=1)))
        L, n = _sh4_cc(S, structure=_sh4_np.ones((3, 3)))
        Pp = BL.labels(objs, cam, cut=0.01); pids, plids = Pp['ids'], Pp['lids']
        cen = {}
        for o in sel:
            lid = BL._lid(o)
            ks = [kk for kk, l in enumerate(plids) if l[0] == lid and l[2] == 'body' and (pids == kk).any()]
            if ks: cen[lid] = _sh4_np.argwhere(_sh4_np.isin(pids, ks)).mean(0)
        if not cen: continue
        # attack: a silhouette must lie on a drawn pellet of the type (its predicted body, or where the installed label already was): the bright
        # cyan hit star of a pellet that already hit (a finished object) is an effect, not a bullet
        near = None
        if t in _sh4_NEAR364:
            live = {BL._lid(o) for o in sel if o.get('drawn', True)}
            kk = [q for q, l in enumerate(plids) if l[0] in live and l[2] == 'body' and (pids == q).any()]
            near = _sh4_dil(_sh4_np.isin(pids, kk), iterations=_sh4_NEAR364R) | _sh4_dil(_sh4_np.isin(ids, old), iterations=1)
        ids[_sh4_np.isin(ids, old)] = -1
        cnt = {}; names = list(cen)
        for i in range(1, n + 1):
            m = L == i
            if m.sum() < _sh4_MIN364: continue
            if near is not None and not (m & near).any(): continue
            c = _sh4_np.argwhere(m).mean(0)
            lid = min(names, key=lambda q: float(_sh4_np.hypot(*(cen[q] - c))))
            j = cnt.get(lid, 0); cnt[lid] = j + 1
            lp = (lid, j - 1, 'body')
            if lp not in lids: lids.append(lp)
            kp = lids.index(lp); ids[m & (ids < 0)] = kp
        nz = (ids != out['ids']) & (ids >= 0)
        share[nz] = _sh4_np.maximum(cv[nz] / 255.0, _sh4_BL.CUT); share[ids < 0] = 0
        out = dict(out); out.update(ids=ids, share=share.astype(_sh4_np.float32), lids=lids)
    return out


_sh4_prev_labels3 = labels


def labels(objs, cam, R, state=None, ended=(), img=None):
    P = _sh4_prev_labels3(objs, cam, R, state=state, ended=ended, img=img)
    return _sh4_t364(P, objs, cam, R)


# ---- 460 BanditGirlCybernoodles: attack 6000018 (Shelly010Projectile), super 6000019 (Shelly010UltiProjectile); single-skin types (460 only) ----
# OWNER 11-06 (flagged: attack 0 t265 / t267 / t269 and attack 90 t305 "holes / ragged"; super 45 t345-t351 "holes / ragged", t353 "too big").
# Cause: the installed rules (z_1111_cutoff3 (4) etc.) take the bulb / cat from the ID pass (front-most draw, coverage >= CUT, a few shapes) and
#   patch it with closings, so overlapping bulbs / cats leave notches, holes and torn edges, and an effect drawn over the bullet (the hit star of
#   t271 / t273, the explosion of t353) gets pulled into the label.
# Rule (same data-derived method as 240, from the game's own draw data): each bullet is rasterised from its clip pieces (opaque normal-blend pieces,
#   coverage >= 1/2: the bulb's disc, rings, caps, sign; the cat's body, ears, eyes). The small nested spark / rod sub-clips (path depth 3) of the
#   super and the translucent halo / streak pieces (alpha 77-179) are NOT body. The outline is the sprite's own, solid, no halo, no hit effect.
#   One label per object (object id, -1, 'body'); where two bullets' sprites overlap, the pixel goes to the bullet whose centre is nearer (no seam,
#   no hole). Pixels where a solid world draw (wall, crate, character) is the front-most draw are left out.
_SH4_T460 = {6000018: 9, 6000019: 2}       # type -> deepest clip-path length that is body
_sh4_MIN460 = 10
_SH4_GATE460 = {6000018: 0.85}     # attack only: the super's cats are always complete
_sh4_PART460 = 0.6       # a raster smaller than this share of the type's median is an incomplete clip: the installed label is kept as well


def _sh4_t460(P, objs, cam, R):
    sel = [o for o in objs if o['t'] in _SH4_T460 and o.get('drawn', True)]
    if not sel: return P
    ids = P['ids'].copy(); share = P['share'].copy(); lids = list(P['lids'])
    cv = R['cv'][::2, ::2]; ids_h = R['id'][::2, ::2]
    solid = [d for d, r in R['draws'].items() if r.get('dmask')]
    front = ~_sh4_np.isin(ids_h, solid)
    res = []
    for o in sel:
        pl = _SH4_T460[o['t']]
        m = _sh4_raster(o, cam, lambda q: len(q['path']) <= pl) & front
        # the installed label's pixels right next to the raster (<= 2 px) are the same sprite (a piece the clip data lacks): kept
        ks = [k for k, l in enumerate(lids) if l[0] == _sh4_BL._lid(o) and l[2] == 'body']
        if m.any() and ks: m = m | (_sh4_np.isin(ids, ks) & _sh4_dil(m, iterations=2) & front)
        res.append([o, m])
    # a bullet whose clip lists only part of its pieces (the raster is far smaller than its siblings': a bulb that is being created / a clip
    # frame with missing children) keeps what the installed label had, joined with the raster
    for t in _SH4_T460:
        ar = [int(m.sum()) for o, m in res if o['t'] == t and m.sum() >= _sh4_MIN460]
        if len(ar) < 2: continue
        med = float(_sh4_np.median(ar))
        for r_ in res:
            o, m = r_
            if o['t'] != t or m.sum() >= _sh4_PART460 * med: continue
            ks = [k for k, l in enumerate(lids) if l[0] == _sh4_BL._lid(o) and l[2] == 'body']
            oldm = _sh4_np.isin(ids, ks)
            u = m | oldm
            if u.any():
                p = 4
                u = _sh4_closing(_sh4_np.pad(u, p), structure=_sh4_disc(2))[p:-p, p:-p] | u
                r_[1] = _sh4_fill(u) & front
    # gate (per type and frame): a bulb that spins shows only a thin strip of its clip pieces (the drawn bulb is wider than the pieces the
    # clip data gives): when the raster of a type covers clearly less than the installed labels did, those bullets keep the installed label,
    # cleaned (closed 2 px, holes filled)
    for t in _SH4_T460:
        if t not in _SH4_GATE460: continue
        rs = [r_ for r_ in res if r_[0]['t'] == t]
        if not rs: continue
        ras_u = _sh4_np.any([r_[1] for r_ in rs], 0)
        olds = []
        for o, m in rs:
            ks = [k for k, l in enumerate(lids) if l[0] == _sh4_BL._lid(o) and l[2] == 'body']
            olds.append(_sh4_np.isin(ids, ks))
        old_u = _sh4_np.any(olds, 0)
        if old_u.sum() == 0 or ras_u.sum() >= _SH4_GATE460[t] * old_u.sum(): continue
        for r_, oldm in zip(rs, olds):
            if oldm.any():
                p = 4
                u = _sh4_closing(_sh4_np.pad(oldm, p), structure=_sh4_disc(2))[p:-p, p:-p] | oldm
                r_[1] = _sh4_fill(u) & front
            else: r_[1] = _sh4_np.zeros_like(r_[1])
    res2 = []
    for o, m in res:
        if m.sum() < _sh4_MIN460: continue
        p = 3
        m = _sh4_closing(_sh4_np.pad(m, p), structure=_sh4_disc(1))[p:-p, p:-p] | m
        m = _sh4_fill(m) & front
        L, n = _sh4_cc(m, structure=_sh4_np.ones((3, 3)))
        if n > 1:
            sz = _sh4_np.bincount(L.ravel())[1:]
            m = _sh4_np.isin(L, 1 + _sh4_np.nonzero(sz >= 0.1 * sz.max())[0])
        res2.append((o, m, _sh4_np.argwhere(m).mean(0)))
    res = res2
    if not res: return P
    # where two bullets' sprites overlap the one drawn in front is the one whose installed (ID-pass) label is nearer
    H, Wd = ids.shape
    old = []
    for o, m, c in res:
        ks = [k for k, l in enumerate(lids) if l[0] == _sh4_BL._lid(o) and l[2] == 'body']
        old.append(_sh4_np.isin(ids, ks))
    for o, m, c in res:
        for k, l in enumerate(lids):
            if l[0] == _sh4_BL._lid(o) and l[2] == 'body': ids[ids == k] = -1
    best = _sh4_np.full((H, Wd), -1, _sh4_np.int32); bd = _sh4_np.full((H, Wd), 1e9)
    yy, xx = _sh4_np.mgrid[0:H, 0:Wd]
    for i, (o, m, c) in enumerate(res):
        d = _sh4_edt(~old[i]) if old[i].any() else _sh4_np.hypot(yy - c[0], xx - c[1]) + 1000.0
        d = d + 1e-3 * _sh4_np.hypot(yy - c[0], xx - c[1])
        better = m & (d < bd); best[better] = i; bd[better] = d[better]
    for i, (o, m, c) in enumerate(res):
        g = (best == i) & (ids < 0)
        if g.sum() < _sh4_MIN460: continue
        lp = (_sh4_BL._lid(o), -1, 'body')
        if lp not in lids: lids.append(lp)
        kk = lids.index(lp); ids[g] = kk
        share[g] = _sh4_np.maximum(cv[g] / 255.0, _sh4_BL.CUT)
    share[ids < 0] = 0
    P2 = dict(P); P2.update(ids=ids, share=share.astype(_sh4_np.float32), lids=lids)
    return P2


_sh4_prev_labels4 = labels


def labels(objs, cam, R, state=None, ended=(), img=None):
    P = _sh4_prev_labels4(objs, cam, R, state=state, ended=ended, img=img)
    return _sh4_t460(P, objs, cam, R)
