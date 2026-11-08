# human review 2026-10-28 fix for skin 033 (Haiku fix, reviewed by the coordinator, installed 10-29; source out/sweep2/haikutasks/human1028/s033/fix2.py; tested via IDLABELS_OVERRIDE after all fixes incl. z_1099_cleanup, hence z_1100_)
# Human review 2026-10-28, skin 033 (BarrelbotDefault, brawler Barrelbot), task out/sweep2/haikutasks/human1028/s033/.
# Replaces fix.py (its convex hull per logic object was rejected by the coordinator: it takes in empty floor between
# spread-apart pellets, and the user prefers the object's own outline filled solid).
#
# The user's findings on the 'attack 90 deg' row, ticks 305, 309, 315: "holes / ragged". The same type is the
# 'attack 0 deg' row's shot (ticks 265-277), which changes in the same way.
#
# Type 6000427 BarrelBotProjectile (clip bullet_shotgun_def_blue_short, sc/effects_brawler.sc): the blue pellets. The pellets
# are drawn additively, so the ID pass shows no body and the label falls back to the prediction. One logic object can be a
# single pellet or a cluster of pellets (tick 309: the lower groups are clusters; the top row is single pellets).
#
# Fix, per pellet (the user's rule: one label per pellet, several only while they touch or overlap, the Skater rule):
# - pellet centres: the distance-transform maxima of the object's predicted body (one per pellet disc), refined to the
#   centroid of the drawn saturated core (max channel >= 240) within 3 px when that is within 2 px of the prediction;
#   a hidden pellet keeps its predicted centre (the hidden-parts rule: the full disc is labelled);
# - each pellet is a filled disc of radius 5 px at its centre;
# - pellets of one object whose centres are closer than 9 px (discs overlap by at least 1 px) form one group: the union of
#   their discs, closed by 2 px, holes filled. No convex hull;
# - groups that are apart are separate labels (their own label id, the same object's meta), even though the game holds
#   them as one logic object;
# - the depth test of labels() still applies (only nearer world geometry hides a pellet, never flashes or smoke).
# Pellets of different objects (the top row at tick 309, one pellet per object) stay separate, as before (the coordinator's
# review of 2026-10-28).
#
# Other types, the other objects' labels, the glow and the ground stay as they are.

from scipy.ndimage import distance_transform_edt as _s033_edt, maximum_filter as _s033_maxf
from scipy.ndimage import binary_closing as _s033_close, binary_fill_holes as _s033_fill

S033_PELLET = {6000427}   # BarrelBotProjectile, the Barrelbot shot (every skin with this type)
S033_R = 5.0              # pellet disc radius (px): the drawn pellet's bright body
S033_MERGE = 9.0          # centre distance under which two pellets of one object overlap (one label)
S033_DEDUP = 3.0          # distance-transform maxima closer than this are one pellet (plateau)
S033_CORE = 240           # saturated core pixels (max of R, G, B) of a drawn pellet
S033_REFINE_R = 3         # px: window around a centre for the drawn core centroid
S033_REFINE_MAX = 2.0     # px: a refined centre must lie this close to the prediction
S033_GROUPS = {}          # object id -> [mask] per pellet group of the current labels() call (set by the hook)


def _s033_disc(cx, cy, r, shape):
    m = np.zeros(shape, bool)
    y0, y1 = max(0, int(np.floor(cy - r)) - 1), min(shape[0], int(np.ceil(cy + r)) + 2)
    x0, x1 = max(0, int(np.floor(cx - r)) - 1), min(shape[1], int(np.ceil(cx + r)) + 2)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    m[y0:y1, x0:x1] = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
    return m


def _s033_disk(r):
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r + 0.5


def _s033_centres(pb, vis):
    """pellet centres (x, y) of an object's predicted body: distance-transform maxima, plateaus averaged, hidden dropped"""
    dt = _s033_edt(np.pad(pb, 1))[1:-1, 1:-1]
    pk = (dt == _s033_maxf(dt, size=7)) & (dt >= 3.0) & pb
    ys, xs = np.nonzero(pk)
    pts = list(zip(xs.astype(float).tolist(), ys.astype(float).tolist()))
    out, used = [], [False] * len(pts)
    for i, (x, y) in enumerate(pts):
        if used[i]: continue
        grp = [j for j, (a, b) in enumerate(pts) if not used[j] and (a - x) ** 2 + (b - y) ** 2 <= S033_DEDUP ** 2]
        for j in grp: used[j] = True
        cx = float(np.mean([pts[j][0] for j in grp])); cy = float(np.mean([pts[j][1] for j in grp]))
        if vis[int(round(cy)), int(round(cx))]: out.append((cx, cy))
    return out


def _s033_refine(cx, cy, core):
    """centroid of the drawn saturated core near a centre, when it is close to the centre (else the prediction's centre)"""
    H, W = core.shape
    x0, x1 = max(0, int(round(cx)) - S033_REFINE_R), min(W, int(round(cx)) + S033_REFINE_R + 1)
    y0, y1 = max(0, int(round(cy)) - S033_REFINE_R), min(H, int(round(cy)) + S033_REFINE_R + 1)
    ys, xs = np.nonzero(core[y0:y1, x0:x1])
    if len(xs) < 4: return cx, cy
    rx, ry = x0 + xs.mean(), y0 + ys.mean()
    if (rx - cx) ** 2 + (ry - cy) ** 2 > S033_REFINE_MAX ** 2: return cx, cy
    return rx, ry


def _s033_pellet(body, ctx):
    pb = ctx['pred_body']
    if not pb.any(): return body   # no prediction (hidden or not on screen): as before
    o, R, cam, img = ctx['o'], ctx['R'], ctx['cam'], ctx['img']
    H, W = pb.shape
    vis = np.ones((H, W), bool)
    if R.get('wd') is not None and o.get('anchor') is not None:
        dz = _eye_depth(cam, o['anchor'], R['meta'].get('wref', 0.0))[0]
        vis = R['wd'] >= dz - DEPTH_TOL
    cs = _s033_centres(pb, vis)
    if not cs:
        S033_GROUPS[o['id']] = []
        return body
    core = (img.max(-1) >= S033_CORE) if img is not None else None
    pts = [_s033_refine(x, y, core) if core is not None else (x, y) for x, y in cs]
    n = len(pts); par = list(range(n))
    def find(i):
        while par[i] != i: par[i] = par[par[i]]; i = par[i]
        return i
    for i in range(n):
        for j in range(i + 1, n):
            if (pts[i][0] - pts[j][0]) ** 2 + (pts[i][1] - pts[j][1]) ** 2 < S033_MERGE ** 2:
                par[find(j)] = find(i)
    groups = {}
    for i in range(n): groups.setdefault(find(i), []).append(i)
    masks = []
    for g in sorted(groups.values(), key=lambda g: min(g)):
        m = np.zeros((H, W), bool)
        for i in g: m |= _s033_disc(pts[i][0], pts[i][1], S033_R, (H, W))
        mp = np.pad(m, 3)
        mp = _s033_fill(_s033_close(mp, structure=_s033_disk(2)))
        m = mp[3:-3, 3:-3] & vis
        if m.any(): masks.append(m)
    S033_GROUPS[o['id']] = masks
    if not masks: return body
    out = np.zeros((H, W), bool)
    for m in masks: out |= m
    return out


for _s033_t in S033_PELLET:
    POST_BODY[_s033_t] = _s033_pellet


_s033_labels_prev = labels


def labels(objs, cam, R, state=None, ended=(), img=None):
    """labels() as before; the pellet groups of one object that are apart get their own label ids (the object's meta)"""
    import bulletlabels as BL
    S033_GROUPS.clear()
    P2 = _s033_labels_prev(objs, cam, R, state=state, ended=ended, img=img)
    ids = P2['ids']; lids = list(P2['lids'])
    for o in objs:
        gs = S033_GROUPS.get(o['id'])
        if not gs or len(gs) < 2 or o['t'] not in S033_PELLET: continue
        lid = (BL._lid(o), -1, 'body')
        if lid not in lids: continue
        k = lids.index(lid); own = ids == k
        for gi, gm in enumerate(gs[1:], start=1):
            pix = own & gm
            if not pix.any(): continue
            lids.append((BL._lid(o), gi, 'body')); ids[pix] = len(lids) - 1
            own &= ~pix
    P2['lids'] = lids
    return P2
