# sweep 10-27 fix for skin 088 (Haiku fix, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s088/)
# s088 (Twins Default 088, sweep 2026-10-27, super 45 deg row, pair 1510, tick 411): the Twins pet's shotgun volley
# (TwinsShotgunProjectile 6001402, pellets 2000014-2000018 spawned on one tick) was labelled as five separate outlines.
# Each pellet's label takes the sprite pixels nearest its own prediction, so one cluster of blades and yellow sparks
# came out as several colours and ragged fragments, with specks at the edge (verifier: extra_label / spur_or_speck).
# Fix (labels() wrapper, keyed to type 6001402 only): the pellets of a volley that touch or overlap in a frame (same
# type, same owner) form ONE label, the user's one-label rule for a multi-part attack (as the Twins Angel's fan, 193),
# filled solid: closed by 2 px into the unlabelled pixels only, holes filled, pieces under 25 px dropped (and pieces
# under 10 % of the largest). Pellets that are apart stay separate labels. No pixel of another object is taken.
# MegaBossTwinsShotgunProjectile (6001707) shares the clip but is another type: unchanged.
import numpy as _s88np
import bulletlabels as BL
from scipy.ndimage import binary_fill_holes as _s88fill, binary_closing as _s88close, label as _s88cc, binary_dilation as _s88dil

S88_TYPES = {6001402}
S88_CLOSE = 2        # px (full-resolution frame): gaps between the pellets' blades
S88_SPECK = 25       # px
S88_FRAC = 0.1
S88_TOUCH = 2        # px: labels closer than this are one volley


def _s88_merge(P, objs):
    ids, lids = P['ids'], P['lids']
    ltype, lown = {}, {}
    for o in objs:
        ltype.setdefault(BL._lid(o), o['t']); lown.setdefault(BL._lid(o), o.get('src'))
    ks = [k for k, l in enumerate(lids) if l[2] == 'body' and ltype.get(l[0]) in S88_TYPES and (ids == k).any()]
    if len(ks) < 2: return P
    isk = _s88np.isin(ids, ks)
    G, n = _s88cc(_s88dil(isk, iterations=S88_TOUCH))              # groups of touching pellets
    share = P['share']
    for g in range(1, n + 1):
        grp = [k for k in ks if (G[ids == k] == g).any()]
        if len(grp) < 2: continue
        owners = {lown.get(lids[k][0]) for k in grp}
        if len(owners) > 1: continue                                  # different shooters: separate labels
        k0 = grp[0]
        m = _s88np.isin(ids, grp)
        m2 = _s88fill(_s88close(_s88np.pad(m, S88_CLOSE), iterations=S88_CLOSE)[S88_CLOSE:-S88_CLOSE, S88_CLOSE:-S88_CLOSE]) | m
        m2 &= m | (ids < 0)                                           # closing only fills unlabelled pixels, never another label
        L, c = _s88cc(m2)
        if c > 1:                                                     # specks of the volley: dropped
            sz = _s88np.bincount(L.ravel())[1:]
            drop = [i + 1 for i in range(c) if sz[i] < S88_SPECK and sz[i] < S88_FRAC * sz.max()]
            if drop: m2 &= ~_s88np.isin(L, drop)
        ids[m & ~m2] = -1
        add = m2 & ~m
        ids[m2] = k0
        share[add] = BL.CUT
    return P


_s88_prev = labels


def labels(objs, cam, R, state=None, ended=(), img=None):
    P = _s88_prev(objs, cam, R, state=state, ended=ended, img=img)
    if not any(o['t'] in S88_TYPES for o in objs): return P
    return _s88_merge(P, objs)
