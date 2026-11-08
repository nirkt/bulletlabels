# human review 2026-10-28 fix for skin 119 (Haiku fix, reviewed by the coordinator, installed 10-29; source out/sweep2/haikutasks/human1028/s119/fix.py; tested via IDLABELS_OVERRIDE after all fixes incl. z_1099_cleanup, hence z_1100_)
# Fix for skin 119 (JesterLoki, human review 2026-10-28, task s119): Chester003Projectile (type 6001275), the Jester's
# attack: a blue crystal hexagon, usually two in one shot, one in front of the other. Findings: row "attack 90", tick 305 and
# tick 307, "too small / cut off": the label of the hex BEHIND stopped at the front hex's edge and left its visible lit
# facet and rim outside (a 2-4 px strip along its left side, a 1 px rim on its top and right edges).
# Cause (checked on the draw ids, ticks 305 and 307): each hex is its own draw call (the back one's face is one draw, the
# front one's another). Both hexes' predicted regions overlap, so a pixel of the front hex was claimed by the back hex's label
# when the back hex came first in the object list (first object wins ties), and the back hex's facet pixels next to the front
# hex were then missing. The per-type rules also keep out the facet pixels that are sprite shape 0 shared with the trail.
# Fix (this type only, labels() wrapper, runs after every other fix):
#  1. draw ownership: every draw is owned by the hex whose label holds most of its pixels; labelled pixels of another hex's
#     draw are removed from that label (the front hex's pixels leave the back hex's label);
#  2. silhouette: per hex, the pixels its OWN draw shows at full coverage within 6 px of its label, where nothing else is labelled
#     (the lit facet and rim, the side the front hex hid). The glow around the hexes is not drawn by the hex's draw with
#     coverage, so it stays out. Other draws (the front hex, the glyph inside it, trail sparks) are never added.
#  3. separate pieces under 40 px that do not touch the hex's own draw (1-px lines of a spark draw above the front hex,
#     found at tick 305) are dropped.
# Also applies to a single hex (attack 0 row, ticks 265-269): its own draw's dark side and rim are added the same way; the
# attack 0 row was not in the user's marks but its outline is now the full hexagon.
# Affects only type 6001275 (Chester003Projectile = skin JesterLoki, skin 119). Other types untouched.
import numpy as np

S119_TYPES = {6001275}
_s119_prev_labels = labels


def labels(objs, cam, R, state=None, ended=(), img=None):
    import bulletlabels as BL
    P = _s119_prev_labels(objs, cam, R, state=state, ended=ended, img=img)
    mine = [o for o in objs if o['t'] in S119_TYPES]
    if not mine: return P
    ids, lids, share = P['ids'], P['lids'], P['share']
    ids_h = R['id'][::2, ::2]; cv = R['cv'][::2, ::2]
    kk = {}
    for o in mine:
        lb = (BL._lid(o), -1, 'body')
        if lb in lids: kk[o['id']] = lids.index(lb)
    if not kk: return P
    # 1. draw ownership: the hex whose label holds most pixels of a draw owns that draw
    hist = {}
    for oid, k in kk.items():
        m = ids == k
        if not m.any(): continue
        dv, cnt = np.unique(ids_h[m & (ids_h > 0)], return_counts=True)
        hist[k] = dict(zip(dv.tolist(), cnt.tolist()))
    owner = {}
    for k, h in hist.items():
        for d, c in h.items():
            if d not in owner or c > owner[d][1]: owner[d] = (k, c)
    for k in hist:
        wrong = np.zeros(ids.shape, bool)
        for d, (ok, _) in owner.items():
            if ok != k and d in hist[k]: wrong |= (ids_h == d)
        rm = (ids == k) & wrong
        if rm.any():
            ids[rm] = -1; share[rm] = 0
    # 2. silhouette: the pixels its own main draw shows at full coverage, near its label, that nothing else labels
    for oid, k in kk.items():
        m = ids == k
        if m.sum() < 30: continue
        own = {d: c for d, (ok, c) in owner.items() if ok == k}
        if not own: continue
        main = max(own, key=own.get)
        # separate small pieces of the label that do not touch the hex's own draw (a draw of the lightning / sparks above it
        # that holds more of this hex's pixels than of the other hex's): dropped
        from scipy.ndimage import label as cc
        L, n = cc(m, structure=np.ones((3, 3)))
        if n > 1:
            touch = np.unique(L[m & (ids_h == main)]); touch = touch[touch > 0]
            sz = np.bincount(L.ravel())
            drop = m & ~np.isin(L, touch) & np.isin(L, np.nonzero(sz < 40)[0])
            if drop.any():
                ids[drop] = -1; share[drop] = 0; m = ids == k
        near = _s119_near(m)
        add = near & (ids_h == main) & (cv >= 128) & ((ids < 0) | (ids == k))
        if add.any():
            ids[add] = k; share[add] = np.maximum(cv[add] / 255.0, BL.CUT)
    P['ids'] = ids; P['share'] = share
    return P


def _s119_near(m, r=6):
    from scipy.ndimage import binary_dilation
    return binary_dilation(m, iterations=r)
