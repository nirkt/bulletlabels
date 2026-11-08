# sweep 10-27 fix for skin 168 (Sonnet revision, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s168/)
# fix2 (second attempt, after the reviewer rejected fix.py: the ring at the right-centre star of tick 345, the notch and
# loop of the lower-left shard and of the upper-right star at tick 355, and the detached rings beside the white comet at
# tick 351 were still there). Same cause and key as fix.py (type 6001017, Griff002UltiProjectile, the super's stars, each
# its own label, rule 5): the stars' own outline is drawn in front of by effect sprites (the white sparkle at a star's
# centre with its soft halo, the white comet at ticks 351-355), which cut pockets, channels and notches out of the label.
# Done on each star's own label, after the per-type rules and the shared clean-up:
#  1. parts of a star label detached from its main part (bridging S168_BRIDGE px) and smaller than S168_FRAC of it are
#     dropped (the comet's rings / specks that the star draw picked up at tick 351);
#  2. closing of each star's label (disc S168_R) adding only unlabelled pixels whose front-most draw is an effect sprite
#     (blended, no depth write: sparkle, halo, comet in front of the star), never the floor or other labels, so pockets
#     and notches cut by an occluder are closed and an occluded star's visible part is one solid piece, while the open
#     notches between star arms (floor) keep the star's outline;
#  2b. the same closing (disc S168_RU) on the union of the star labels: narrow slits between two stars;
#  3. enclosed gaps of the union of the star labels (<= S168_HOLE px) are filled with the nearest star's label;
#  1b. a star label whose largest part is under S168_MIN px has no star body left (the star is behind the comet, only
#     comet rings picked up by its draw): removed;
#  3b. pits / slits (unlabelled pixel with 3+ labelled 4-neighbours, two passes) are filled;
#  4. step 1 again (parts that the closing left detached).
# Other types, shots and skins are not changed.

S168_STAR = 6001017
S168_HOLE = 300
S168_R = 7
S168_RU = 4
S168_BRIDGE = 2
S168_FRAC = 0.25
S168_MIN = 150


def _s168_disc(r):
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r + 1


def _s168_fix(P, objs, R):
    import bulletlabels as BL
    from scipy.ndimage import binary_fill_holes, binary_closing, label as cc, binary_dilation, distance_transform_edt
    star = {BL._lid(o) for o in objs if o['t'] == S168_STAR}
    ks = [k for k, l in enumerate(P['lids']) if l[0] in star and l[2] == 'body']
    ids = P['ids'].copy(); share = P['share'].copy()
    if not ks or not np.isin(ids, ks).any(): return P
    cv = R['cv'][::2, ::2]; rid = R['id'][::2, ::2]
    eff = np.zeros(ids.shape, bool)
    for d in np.unique(rid):
        dd = R['draws'].get(int(d)) if d > 0 else None
        if dd and dd.get('blend') and not dd.get('dmask'): eff |= rid == d

    def drop_parts():
        for k in ks:
            m = ids == k
            if not m.any(): continue
            L, n = cc(binary_dilation(m, _s168_disc(S168_BRIDGE)))
            if n < 2: continue
            sz = np.bincount(L[m], minlength=n + 1)[1:]
            d = np.isin(L, 1 + np.nonzero(sz < S168_FRAC * sz.max())[0]) & m
            ids[d] = -1; share[d] = 0

    drop_parts()
    for k in ks:      # a star hidden behind the comet: only comet fragments (rings) are left of its label, no star body
        m = ids == k
        if m.any() and max(np.bincount(cc(m)[0].ravel())[1:]) < S168_MIN: ids[m] = -1; share[m] = 0
    r = S168_R; H, W_ = ids.shape
    for k in ks:
        m = ids == k
        if not m.any(): continue
        ys, xs = np.nonzero(m); p = r + 2
        y0, y1, x0, x1 = max(0, ys.min() - p), min(H, ys.max() + p + 1), max(0, xs.min() - p), min(W_, xs.max() + p + 1)
        sl = (slice(y0, y1), slice(x0, x1)); sm = m[sl]
        cl = binary_closing(np.pad(sm, r + 1), _s168_disc(r))[r + 1:-(r + 1), r + 1:-(r + 1)]
        add = cl & ~sm & eff[sl] & (ids[sl] < 0)
        ids[sl][add] = k; share[sl][add] = np.maximum(cv[sl][add] / 255.0, BL.CUT)
    U = np.isin(ids, ks); r2 = S168_RU
    cu = binary_closing(np.pad(U, r2 + 1), _s168_disc(r2))[r2 + 1:-(r2 + 1), r2 + 1:-(r2 + 1)]
    m = cu & ~U & eff & (ids < 0)       # narrow slits between two stars (tick 345), same effect-sprite restriction
    if m.any():
        _, (iy, ix) = distance_transform_edt(~U, return_indices=True)
        near = ids[iy, ix]; ids[m] = near[m]; share[m] = np.maximum(cv[m] / 255.0, BL.CUT)
    U = np.isin(ids, ks)
    fh = binary_fill_holes(U) & ~U & (ids < 0)
    if fh.any():
        L, n = cc(fh); sz = np.bincount(L.ravel())
        ok = [j for j in range(1, n + 1) if sz[j] <= S168_HOLE]
        if ok:
            m = np.isin(L, ok)
            _, (iy, ix) = distance_transform_edt(~U, return_indices=True)
            near = ids[iy, ix]; ids[m] = near[m]; share[m] = np.maximum(cv[m] / 255.0, BL.CUT)
    for _ in range(2):    # one- to two-pixel slits / pits left in the outline: unlabelled pixel with 3+ star-labelled 4-neighbours
        U = np.isin(ids, ks); pad = np.pad(U, 1)
        nb = pad[:-2, 1:-1].astype(np.int8) + pad[2:, 1:-1] + pad[1:-1, :-2] + pad[1:-1, 2:]
        m = (~U) & (ids < 0) & (nb >= 3)
        if not m.any(): break
        _, (iy, ix) = distance_transform_edt(~U, return_indices=True)
        near = ids[iy, ix]; ids[m] = near[m]; share[m] = np.maximum(cv[m] / 255.0, BL.CUT)
    drop_parts()
    P = dict(P); P['ids'] = ids; P['share'] = share
    return P


_s168_prev_labels = labels


def labels(objs, cam, R, state=None, ended=(), img=None):
    P = _s168_prev_labels(objs, cam, R, state=state, ended=ended, img=img)
    if not any(o['t'] == S168_STAR for o in objs): return P
    return _s168_fix(P, objs, R)
