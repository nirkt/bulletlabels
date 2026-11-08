# sweep 10-27 fix for skin 084 (Haiku fix, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s084/)
# Skin 084 ConductorDefault, attack 0 deg and attack 90 deg (sweep 2026-10-27, task s084: pairs 1459, 1465 spur_or_speck)
# Type 6001369 ConductorProjectile (attack, sc/effects_brawler.sc chuck_def_atk_projectile): each cloud is one solid outline.
# The clouds shed small puffs of the same sprite shape (63) at their edges. They sit in the same draw call as the cloud, so
# the shape cannot tell them from it. Some of them are split off only AFTER the one-owner-per-pixel step (where overlapping
# clouds share pixels), so the cut has to run on the final label map, as the clean-up does.
# Measured on the capture: the puffs are 1-77 px; the main part of each cloud is 328 px or more. The user's rule 1: a
# separate bit is a speck, not part of the label. Rule: a part of a 6001369 label that is under 100 px and under a tenth of
# that label's largest part is dropped. A puff that sat in a notch of the neighbouring cloud leaves a pocket in that cloud's
# outline: the pocket (100 px or less, no other label in it) is filled, as the clean-up fills small holes.
# Keyed to this one type id only (no other skin of the sweep uses it). Runs in a labels() wrapper after the clean-up.

_T20_TYPE = 6001369
_T20_SPECK = 100    # px
_T20_FRAC = 0.1
_T20_HOLE = 100     # px
_t20_prev = labels


def labels(objs, cam, R, state=None, ended=(), img=None):
    from scipy.ndimage import label as cc, binary_fill_holes
    import bulletlabels as BL
    P = _t20_prev(objs, cam, R, state=state, ended=ended, img=img)
    ids, lids, share = P['ids'], P['lids'], P['share']
    ltype = {}
    for o in objs: ltype.setdefault(BL._lid(o), o['t'])
    ks = [k for k in np.unique(ids[ids >= 0]) if lids[k][2] == 'body' and ltype.get(lids[k][0]) == _T20_TYPE]
    if not ks: return P
    # pass 1: the specks go (all objects first: a speck may sit in a neighbour's notch)
    for k in ks:
        m = ids == k
        ys, xs = np.nonzero(m); y0, y1, x0, x1 = max(0, ys.min() - 2), ys.max() + 3, max(0, xs.min() - 2), xs.max() + 3
        sm = m[y0:y1, x0:x1]; sid = ids[y0:y1, x0:x1]; ssh = share[y0:y1, x0:x1]
        L, n = cc(sm)
        if n <= 1: continue
        sz = np.bincount(L.ravel())[1:]
        drop = 1 + np.nonzero((sz < _T20_SPECK) & (sz < _T20_FRAC * sz.max()))[0]
        if drop.size:
            d = np.isin(L, drop) & sm; sid[d] = -1; ssh[d] = 0
    # pass 2: pockets of 100 px or less enclosed by one cloud's outline, with no other label in them, join that cloud
    for k in ks:
        m = ids == k
        if not m.any(): continue
        ys, xs = np.nonzero(m); y0, y1, x0, x1 = max(0, ys.min() - 2), ys.max() + 3, max(0, xs.min() - 2), xs.max() + 3
        sm = m[y0:y1, x0:x1]; sid = ids[y0:y1, x0:x1]; ssh = share[y0:y1, x0:x1]
        hole = binary_fill_holes(sm) & ~sm
        if not hole.any(): continue
        HL, hn = cc(hole)
        for j in range(1, hn + 1):
            hj = HL == j
            if hj.sum() <= _T20_HOLE and (sid[hj] < 0).all():
                sid[hj] = k; ssh[hj] = BL.CUT
    return P
