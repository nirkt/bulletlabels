# away_1104 task mortis_dragon, fix.py. Exec'd in idlabels' namespace after the installed fixes (IDLABELS_OVERRIDE).
#
# (1) Skin 195 UndertakerDemon, page 195_UndertakerDemon_01.png, row "super 45", type 6000304 Mortis008UltiProjectile.
#     OWNER NOTE: "the little eyes shouldnt be included in the hull". The small glowing bats with a yellow eye that fly
#     around / behind the ghost (sprites of the clips mortis_008_ulti_trail_demon / _flip, installed in z_1102_s195) stretched the
#     hull far to the left of the ghost (t349-t361). Fix: the hull is built from the ghost's own mesh pixels only (the
#     opaque program-129 draws, the same pixels a3_agent3 / z_1102_s195 take as 'mesh'); the bat sprites are dropped. The
#     ghost's own big eyes are part of its mesh/face and stay inside the hull. ONE_LABEL 'hull' and the "no label once the
#     logic has finished" rule of z_1102_s195 are kept. Only this type is touched (owners.projectile: skin UndertakerDemon only).

_ab4_md_S195 = 6000304
_ab4_md_prev_195 = POST_BODY.get(_ab4_md_S195)


def _ab4_md_s195_mesh_only(body, ctx):
    R, ids, cv, region = ctx['R'], ctx['ids_h'], ctx['cv'], ctx['region']
    uvneg = R['uv'][..., 0] < 0
    mesh = np.zeros(body.shape, bool)
    for d in np.unique(ids[region & (ids > 0)]):
        r = R['draws'].get(int(d), {})
        if r.get('prog') != S195_MESH_PROG: continue
        m = ids == d
        if uvneg[m].mean() < 0.5: continue
        mesh |= m
    mesh &= region & (cv >= CUT)
    return mesh


POST_BODY[_ab4_md_S195] = _ab4_md_s195_mesh_only
ONE_LABEL[_ab4_md_S195] = 'hull'


# (2) Skin 207 AssaultShotgunDragon (Griff 006), pages 207_AssaultShotgunDragon_01/_02.png, rows attack 0 / attack 90 / super 45.
#     OWNER NOTE: "all are either ragged or have holes" (flagged: attack 0 t265-285, attack 90 t305-325, super 45 t345-359).
#     Types: 6001031 Griff006Projectile (the gem / crystal / gold-hook pellets, attack), 6001033 Griff006UltiProjectile and
#     6001034 Griff006UltiProjectile2 (the scrolls, super), 6001032 Griff006ProjectileSp1Buddy (same pellet, not on the pages).
#     Only skin 207 uses these types (Griff006 is its own name). PLAIN BULLET rule: one solid smooth label per bullet.
#     Cause (looked at sw2_207): each bullet is drawn from several sprites (gem, hook, scroll body, glow layers, a bright
#     overexposed centre) and the client's per-pixel draw ids give the pixels of OVERLAPPING bullets to the objects arbitrarily:
#     the labels are interleaved strips with cracks, notches, holes open to the outside (so the global hole fill cannot
#     close them) and spurs. Fix, on the final labels of these types only:
#      1. the union of all their labels in the frame is cut into connected pieces (gaps up to _AB4_MD_JOIN px joined): pieces
#         that touch are ONE silhouette ("touching pieces of one bullet one silhouette"), bullets that are apart stay apart;
#      2. each piece keeps the label id of the object that owns most of it (ids are not shared while objects are left);
#      3. (scrolls, types 6001033/34: one object = one bullet, no re-cut) the overexposed cream / white centre of a scroll
#         (additive, no draw id) is added when it lies within _AB4_MD_BRIGHT_D px of the scroll's label, nearest label wins;
#      4. each piece is closed (_AB4_MD_CLOSE px), its holes filled, the outline smoothed by a blur (spurs and notches go),
#         crumbs dropped. Only unclaimed pixels are added (never into another type's label), growth limited to _AB4_MD_GROW px,
#         a bullet never loses its label.
_AB4_MD_COMP = {6001031, 6001032}      # pellets: the pieces of overlapping bullets are re-cut by connectivity (step 1-2)
_AB4_MD_OBJ = {6001033, 6001034}       # scrolls: one object = one bullet, its own label (step 3 only)
_AB4_MD_TYPES = _AB4_MD_COMP | _AB4_MD_OBJ
_AB4_MD_CLOSE = {6001031: 4, 6001032: 4, 6001033: 9, 6001034: 9}   # px: crack / gap width bridged (the scrolls' bright centre is wider)
_AB4_MD_BRIGHT = 205      # min red of an overexposed (cream / white) scroll pixel
_AB4_MD_BRIGHT_D = 8      # px: how far from a scroll label such a pixel is taken
_AB4_MD_JOIN = 2          # px: pieces this close are one bullet
_AB4_MD_SIGMA = 1.6
_AB4_MD_GROW = 12
_ab4_md_prev_labels = labels


def _ab4_md_smooth(m, r):
    from scipy.ndimage import binary_fill_holes, gaussian_filter, label as cc
    c = binary_fill_holes(_close(m, r))
    L, n = cc(c, structure=np.ones((3, 3), bool))
    if n > 1:
        sz = np.bincount(L.ravel()); sz[0] = 0
        keep = sz >= max(6, 0.12 * sz.max())
        c = keep[L]
    g = gaussian_filter(np.pad(c, 6).astype(np.float32), _AB4_MD_SIGMA)[6:-6, 6:-6] >= 0.5
    return binary_fill_holes(g) & _dilate(c, 2)


def labels(objs, cam, R, state=None, ended=(), img=None):
    import bulletlabels as BL
    from scipy.ndimage import label as cc
    P = _ab4_md_prev_labels(objs, cam, R, state=state, ended=ended, img=img)
    mine = [o for o in objs if o['t'] in _AB4_MD_TYPES]
    if not mine: return P
    ids = P['ids'].copy(); share = P['share'].copy(); lids = P['lids']
    cv = R['cv'][::2, ::2]
    ks = {}
    for o in mine:
        lid = (BL._lid(o), -1, 'body')
        if lid in lids:
            k = lids.index(lid)
            if (ids == k).any(): ks[k] = o['t']
    if not ks: return P
    # scrolls: each object's own label, closed / filled / smoothed (step 3). First the scroll's overexposed cream / white
    # centre (drawn additively, so it has no draw id of the scroll) is added: bright unlabelled pixels within _AB4_MD_BRIGHT_D px
    # of a scroll label go to the nearest scroll label.
    objk = [k for k, t in ks.items() if t in _AB4_MD_OBJ]
    if objk and img is not None:
        from scipy.ndimage import distance_transform_edt
        im = img.astype(np.int16)
        bright = (im[..., 0] >= _AB4_MD_BRIGHT) & (im[..., 1] >= _AB4_MD_BRIGHT - 12) & (im[..., 2] >= 120)       # white / cream (red high), not the cyan glow
        anyl = np.zeros(ids.shape, bool)
        for k in objk: anyl |= ids == k
        d, (iy, ix) = distance_transform_edt(~anyl, return_indices=True)
        cand = bright & (ids < 0) & (d <= _AB4_MD_BRIGHT_D)
        own = ids[iy, ix]
        for k in objk:
            a = cand & (own == k)
            ids[a] = k; share[a] = np.maximum(cv[a] / 255.0, BL.CUT)
    for k, t in ks.items():
        if t not in _AB4_MD_OBJ: continue
        m = ids == k
        s = _ab4_md_smooth(m, _AB4_MD_CLOSE[t]) & _dilate(m, _AB4_MD_GROW)
        if (s & m).sum() < 0.5 * m.sum(): continue
        add = s & ~m & (ids < 0)
        ids[add] = k; share[add] = np.maximum(cv[add] / 255.0, BL.CUT)
        drop = m & ~s; ids[drop] = -1; share[drop] = 0
    ks = {k: t for k, t in ks.items() if t in _AB4_MD_COMP}
    if not ks:
        P['ids'] = ids; P['share'] = share
        return P
    U = np.zeros(ids.shape, bool)
    for k in ks: U |= ids == k
    L, n = cc(_close(U, _AB4_MD_JOIN) & (U | _dilate(U, _AB4_MD_JOIN)), structure=np.ones((3, 3), bool))
    pieces = []
    for c in range(1, n + 1):
        pc = (L == c) & U
        if pc.sum() >= 8: pieces.append(pc)
    pieces.sort(key=lambda p: -p.sum())
    used = set(); assign = []
    for pc in pieces:
        cnt = {k: int((pc & (ids == k)).sum()) for k in ks}
        free = {k: v for k, v in cnt.items() if k not in used and v > 0}
        k = max(free, key=free.get) if free else max(cnt, key=cnt.get)
        used.add(k); assign.append((pc, k))
    old = {k: ids == k for k in ks}
    for k in ks: ids[old[k]] = -1; share[old[k]] = 0
    for pc, k in assign:
        t = ks[k]
        s = _ab4_md_smooth(pc, _AB4_MD_CLOSE[t]) & _dilate(pc, _AB4_MD_GROW)
        if (s & pc).sum() < 0.5 * pc.sum(): s = pc
        add = s & (ids < 0) & ~(U & ~pc)
        ids[add] = k; share[add] = np.maximum(cv[add] / 255.0, BL.CUT)
        ids[pc & (ids < 0)] = k
    P['ids'] = ids; P['share'] = share
    return P
