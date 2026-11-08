# sweep 10-27 fix for skin 582 (Haiku fix, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s582/)
# skin 582 MummyDemon (sweep 2026-10-27, task s582, pair 10118 tick 319 'hole'): Emz006Projectile (type 6000688)
# clouds of attack 90 deg (and 0 deg). The cat-head units are drawn in front of the cloud's lower edge; their pixels are
# not cloud body, so the solid cloud label has notches (open at the bottom, not closed holes: binary_fill_holes and the
# clean-up's hole filling do not reach them). The cloud continues behind the cat heads, so the label is closed with a
# small disc (S582_CLOSE_R px) over the attack's own union, taking only free pixels (no other label, no solid map block):
# the outer silhouette and the cat heads that stick out of the cloud stay as they are (a closing only adds pixels).
# Keyed to type 6000688 only (the Emz types of the other skins are not changed).
# Only the attack's large parts (at least S582_MAIN of the largest part) are closed: a small separate piece of the label
# (a green trail streak beside the cloud at tick 269) is not bridged to the cloud, so it does not become a spur.
import numpy as np
from scipy.ndimage import binary_closing as _s582_closing, label as _s582_cc

S582_TYPES = {6000688}
S582_CLOSE_R = 8                     # px: notches narrower than about 2 x this are closed
S582_MAIN = 0.15                     # share of the largest part a part needs to be closed
_s582_prev_labels = labels


def labels(objs, cam, R, state=None, ended=(), img=None):
    import bulletlabels as BL
    P = _s582_prev_labels(objs, cam, R, state=state, ended=ended, img=img)
    if not any(o['t'] in S582_TYPES for o in objs): return P
    ids, lids, share = P['ids'], P['lids'], P['share']
    ks = set()
    for o in objs:   # the clouds of one attack share one body label (the Emz stack merge of z_1024)
        if o['t'] not in S582_TYPES: continue
        lid = (BL._lid(o), -1, 'body')
        if lid in lids: ks.add(lids.index(lid))
    if not ks: return P
    try: world = _zs_solid(cam, R)            # solid map blocks in front: not the cloud
    except Exception: world = np.zeros(ids.shape, bool)
    r = S582_CLOSE_R
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]; disc = (yy ** 2 + xx ** 2) <= r * r
    for k in sorted(ks):
        U = ids == k
        if not U.any(): continue
        Lc, nc = _s582_cc(U, structure=np.ones((3, 3), bool))
        src = U
        if nc > 1:   # the large parts only (a small separate piece is not bridged to the cloud)
            sz = np.bincount(Lc.ravel())[1:]
            src = np.isin(Lc, 1 + np.nonzero(sz >= S582_MAIN * sz.max())[0])
        cl = _s582_closing(np.pad(src, r + 2), structure=disc)[r + 2:-r - 2, r + 2:-r - 2]
        take = cl & ~U & (ids < 0) & ~world
        if take.any():
            ids[take] = k; share[take] = BL.CUT
    return P
