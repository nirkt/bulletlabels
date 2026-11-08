# installed 11-01 from out/sweep2/haikutasks/siteai2/bull/fix.py (2nd AI sweep: 483 orbs split per pellet + hidden fill, 568 orbs split per orb; types 6000043/6000050 are single-skin)
# AI-review sweep 2026-11-01, task bull (skins 483, 516, 517, 531, 568, 466), fix file.
# Exec'd in idlabels' namespace (IDLABELS_OVERRIDE) after all installed fixes; keyed to explicit type ids.
#
# 483 BullDudeBarbking, Bull002Projectile (6000043): the orb shots (attack 0 t267-t273, attack 90 t307-t313).
#   Each object is a cluster of 4 orb pellets (pellet slots 0-3 of its clip, each one orb). idlabels gave the whole
#   object one body label, so 2-4 touching orbs got one blob outline (AI: 'orbs at t305-t313 labelled as blobs covering
#   several orbs'; t311 has 3 objects = 12 orbs in 3 outlines). Owner rule: separate pellets / orbs = separate outlines.
#   Fix: each object's label is split per pellet. A pixel goes to the pellet whose predicted disc is nearest (distance
#   to the predicted disc, centre distance as tie-break). Only pellets whose predicted body is at least _ORB_MIN px are
#   orbs: the small ones (14-33 px) are the orbs still growing out of the same spot (t305, t265) and stay with the
#   full orb, so a single orb is not cut. Hidden parts: a pellet whose drawn part is under _ORB_FILL of its predicted
#   disc (covered by dust / the hit cloud: t307 2000018, t313) is filled to its full disc (own or unlabelled pixels
#   only). A pellet with no drawn pixel gets no label (fully hidden, nothing to outline).
#   Attack 0 of the same skin uses the same type and the same clusters (t267-t271), so it gets the same split.
#
# 568 BullDudeGreek, Bull012? yellow orb pellets of type 6000050 (attack 0 t265-t269, attack 90 t305-t307): each
#   object is a cluster of 2-4 touching orbs and has ONE label, so the orbs of one object were one merged scalloped
#   outline (AI: 'bull pellets merged into one scalloped outline over the orb cluster on t265, t267, t305'). The clip
#   has no pellet slots, so the split is geometric: the seeds are the strong maxima of the label's distance transform
#   (an orb is a disc of about 7 px radius; a seed must reach 4.5 px and lie 9 px from the next seed), and each pixel goes
#   to its nearest seed. One seed = one orb = unchanged. Pieces keep the object's id (pellet index 0..n-1).
#
# Other skins using 6000043 (the same brawler's projectile) change too (re-checked by the coordinator).

import numpy as np
from scipy import ndimage as _orb_ndi

_ORB_TYPES = {6000043}
_ORB_MIN = 70      # px: a pellet whose predicted body is this large has its orb core (smaller ones are blue smear of the growing pellets)
_ORB_FILL = 0.6    # a pellet drawn below this share of its predicted disc is partly hidden: fill to the disc

_orb_prev_labels = labels

_SEED_TYPES = {6000050}
_SEED_E = 4.5      # px: a seed is a distance-transform maximum at least this far from the label edge
_SEED_SEP = 9      # px: seeds closer than this are one orb


def _seed_split_object(o, ids, lids):
    import bulletlabels as BL
    from scipy import ndimage as ndi
    lk = (BL._lid(o), -1, 'body')
    if lk not in lids:
        return
    kb = lids.index(lk)
    M = ids == kb
    if not M.any():
        return
    E = ndi.distance_transform_edt(M)
    mx = ndi.maximum_filter(E, size=9)
    cand = np.argwhere((E == mx) & (E >= _SEED_E) & M)
    if len(cand) < 2:
        return
    cand = cand[np.argsort(-E[cand[:, 0], cand[:, 1]])]
    seeds = []
    for y, x in cand:
        if all((y - yy) ** 2 + (x - xx) ** 2 >= _SEED_SEP ** 2 for yy, xx in seeds):
            seeds.append((int(y), int(x)))
    if len(seeds) < 2:
        return
    yy, xx = np.nonzero(M)
    D = np.stack([(yy - sy) ** 2 + (xx - sx) ** 2 for sy, sx in seeds], 1)
    pick = np.argmin(D, 1)
    for j in range(len(seeds)):
        sel = pick == j
        if not sel.any():
            continue
        if j == 0:
            idx = kb
        else:
            lids.append((BL._lid(o), j, 'body'))
            idx = len(lids) - 1
        ids[yy[sel], xx[sel]] = idx


def _orb_split_object(o, P, ids, lids, cam):
    import bulletlabels as BL
    P0 = BL.labels([o], cam, cut=0.01)
    orbs = []
    for k, l in enumerate(P0['lids']):
        if l[2] != 'body' or l[1] < 0:
            continue
        m = P0['ids'] == k
        if m.sum() >= _ORB_MIN:
            ys, xs = np.nonzero(m)
            orbs.append((l[1], m, xs.mean(), ys.mean(), int(m.sum())))
    lk = (BL._lid(o), -1, 'body')
    if not orbs or lk not in lids:
        return
    kb = lids.index(lk)
    M = ids == kb
    if not M.any():
        return
    yy, xx = np.nonzero(M)
    D = []
    for pel, m, cx, cy, a in orbs:
        D.append(_orb_ndi.distance_transform_edt(~m)[yy, xx] + 0.01 * np.hypot(xx - cx, yy - cy))
    pick = np.argmin(np.stack(D, 1), 1)
    own = []
    for j, (pel, m, cx, cy, area) in enumerate(orbs):
        sel = pick == j
        drawn = int(sel.sum())
        if drawn == 0:
            continue
        if j == 0:
            idx = kb
        else:
            lids.append((BL._lid(o), pel, 'body'))
            idx = len(lids) - 1
        ids[yy[sel], xx[sel]] = idx
        own.append((idx, m, drawn, area))
    # hidden parts: a partly covered orb takes its full predicted disc (where no other object's label is)
    for idx, m, drawn, area in own:
        if drawn < _ORB_FILL * area:
            mine = (ids == idx)
            add = m & ((ids < 0) | mine)
            ids[add] = idx


def labels(objs, cam, R, state=None, ended=(), img=None):
    P = _orb_prev_labels(objs, cam, R, state=state, ended=ended, img=img)
    todo = [o for o in objs if o['t'] in _ORB_TYPES and o.get('drawn')]
    seed = [o for o in objs if o['t'] in _SEED_TYPES and o.get('drawn')]
    if not todo and not seed:
        return P
    ids = P['ids']; lids = list(P['lids'])
    for o in todo:
        _orb_split_object(o, P, ids, lids, cam)
    for o in seed:
        _seed_split_object(o, ids, lids)
    P['lids'] = lids
    return P
