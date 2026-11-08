# sweep 10-27 fix for skin 423 (Haiku fix, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s423/)
# sweep 10-27 fix for skin 423 JetpackGirlSurfer (Haiku task out/sweep2/haikutasks/sweep1027/s423/), finding pair 7952:
# attack 0 deg, tick 267, "hole": the yellow shard fan has a small enclosed hole (a dark diamond inside the fan, drawn with
# its own red loop). The fan is four separate shard objects of ONE attack type, JetpackGirl008Projectile (6001172, Janet's
# attack 008, janet_008_atk_projectile). Where the shards meet, a few background pixels (1-2 px gaps at ticks 265-269, the
# dark diamond at tick 267) are enclosed by the union of the shards but by no single shard, so the shared clean-up
# (z_1099_cleanup, which fills holes of each object's own mask) leaves them: each becomes a small hole in the outline.
# Rule 1 of the sweep (a label is one solid piece, no holes): holes enclosed by the union of this type's own shards, at most
# 40 px each (the clean-up's CLEAN_HOLE), containing no other label's pixels, are filled and given to the neighbouring shard
# (its most frequent 4-neighbour), with the share of a filled hole set to BL.CUT as the clean-up does. Only pixels that were
# unlabelled are touched, so no projectile loses a pixel; only type 6001172 is affected, nothing else in the map.

import numpy as _np_s423

S423_UNION_TYPES = {6001172}     # JetpackGirl008Projectile: the shards of Janet's attack 008 (skin 423 attack 0 deg)
S423_HOLE_MAX = 40               # px, same bound as the shared clean-up (CLEAN_HOLE)
_labels_s423_prev = labels


def labels(objs, cam, R, state=None, ended=(), img=None):
    from scipy.ndimage import binary_fill_holes, label as _cc
    import bulletlabels as BL
    P = _labels_s423_prev(objs, cam, R, state=state, ended=ended, img=img)
    ids, lids, share = P['ids'], P['lids'], P['share']
    tmap = {o['id']: o['t'] for o in objs}
    ks = [int(k) for k in _np_s423.unique(ids[ids >= 0]) if tmap.get(lids[k][0]) in S423_UNION_TYPES]
    if len(ks) < 2: return P
    m = _np_s423.isin(ids, ks)
    fill = binary_fill_holes(m) & ~m
    if not fill.any(): return P
    L, n = _cc(fill)
    sz = _np_s423.bincount(L.ravel())
    ids = ids.copy(); share = share.copy(); done = 0
    for j in range(1, n + 1):
        if sz[j] > S423_HOLE_MAX: continue
        hj = L == j
        if (ids[hj] >= 0).any(): continue          # the gap holds another label: not ours to fill
        ys, xs = _np_s423.nonzero(hj); nb = []
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            yy = _np_s423.clip(ys + dy, 0, ids.shape[0] - 1); xx = _np_s423.clip(xs + dx, 0, ids.shape[1] - 1)
            v = ids[yy, xx]; nb.append(v[v >= 0])
        nb = _np_s423.concatenate(nb)
        if not nb.size: continue
        k = int(_np_s423.bincount(nb).argmax())
        ids[hj] = k; share[hj] = BL.CUT; done += 1
    if not done: return P
    P = dict(P); P['ids'] = ids; P['share'] = share
    return P
