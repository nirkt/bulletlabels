# (installed 10-27 as z_1099_cleanup.py: runs after every other fix file)
# Clean-up of every projectile / spawnable label (the user, 2026-10-27, after the agent sweep sweep_1027: "go ahead with the
# cleanup"): a label is its object's outline filled solid, one piece (10-24 rule). Applied last, to the final label map:
#  - small holes inside a label (closed loops of at most CLEAN_HOLE px, half resolution, not other labels' pixels) are filled
#    (Rosa 007's splash diamonds, Shaman blob diamonds, shard-fan gaps). Larger openings stay: Lola's (Duplicator) ring
#    interiors and the gaps between her trail rings are ground showing through (user-good page 000, kept as they are:
#    every Lola type, logic DuplicatorProjectile / DuplicatorUltiProjectile, is left out of the hole filling);
#  - specks: parts of a label smaller than CLEAN_SPECK px and smaller than CLEAN_FRAC of its largest part are dropped
#    (single-pixel bits on sparkles beside the attack, crumbs at a cut edge); parts are counted after bridging gaps of
#    CLEAN_BRIDGE px, so a thin ring drawn in pieces (377's bubble) is one part, not specks.
# Zones, landing markers, glow and hit labels are untouched (zones are logic discs, cut by walls on purpose).
import os as _os
CLEAN_HOLE = int(_os.environ.get('CLEAN_HOLE', '40'))     # px
CLEAN_SPECK = int(_os.environ.get('CLEAN_SPECK', '25'))   # px
CLEAN_FRAC = float(_os.environ.get('CLEAN_FRAC', '0.1'))
CLEAN_PARTS = ('body', 'spawn')
CLEAN_BRIDGE = 3   # px: parts closer than 2 x this are one part
CLEAN_KEEP_HOLES = {t for t, d in projtab.table().items() if d['logic'] in ('DuplicatorProjectile', 'DuplicatorUltiProjectile')}
CLEAN_OFF = bool(_os.environ.get('CLEAN_OFF'))   # read at load (regress2 baselines)
_clean_prev = labels


def labels(objs, cam, R, state=None, ended=(), img=None):
    from scipy.ndimage import binary_fill_holes, label as cc
    import bulletlabels as BL
    P = _clean_prev(objs, cam, R, state=state, ended=ended, img=img)
    if CLEAN_OFF: return P
    ids, lids, share = P['ids'], P['lids'], P['share']
    ltype = {}
    for o in objs: ltype.setdefault(BL._lid(o), o['t'])
    for k in np.unique(ids[ids >= 0]):
        if lids[k][2] not in CLEAN_PARTS: continue
        m = ids == k
        ys, xs = np.nonzero(m); y0, y1, x0, x1 = max(0, ys.min() - 2), ys.max() + 3, max(0, xs.min() - 2), xs.max() + 3
        sm = m[y0:y1, x0:x1]; sid = ids[y0:y1, x0:x1]; ssh = share[y0:y1, x0:x1]
        L, n = cc(_dilate(sm, CLEAN_BRIDGE))
        if n > 1:
            sz = np.bincount(L[sm], minlength=n + 1)[1:]   # the label's own pixels per bridged part
            drop = 1 + np.nonzero((sz < CLEAN_SPECK) & (sz < CLEAN_FRAC * sz.max()))[0]
            if drop.size:
                d = np.isin(L, drop) & sm; sid[d] = -1; ssh[d] = 0; sm = sm & ~d
        if ltype.get(lids[k][0]) in CLEAN_KEEP_HOLES: continue
        hole = binary_fill_holes(sm) & ~sm
        if hole.any():
            L, n = cc(hole); sz = np.bincount(L.ravel())
            for j in range(1, n + 1):
                hj = L == j
                if sz[j] <= CLEAN_HOLE and (sid[hj] < 0).all():
                    sid[hj] = k; ssh[hj] = BL.CUT
    return P
