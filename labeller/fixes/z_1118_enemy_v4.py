# v4 enemy fix (2026-11-08): to be installed as py/fixes/z_1118_enemy_v4.py. Global prefix _f4_. Enemy (red) shots only (o['own'] False).
#
# 048 BlowerDefault super (type 6000732 BlowerUltiProjectile): z_1108_gale_super labels the white snow slab by colour. The RED super is
# launched under a big pink-magenta muzzle burst (no blue counterpart) that washes the middle of the slab at its first tick(s): the
# snow test (weakest channel >= 150, red >= blue + 10) fails there and the label falls into two pieces (the two ends), the middle
# hidden under the burst is not labelled (enemycmp2 048 super tick 3: own 4872 px, enemy 2552 px). The hidden-part rule: when an
# enemy shot's label of this type consists of several pieces (>= 15 % of the largest), the pieces are joined along the slab's axis by
# the hull of their two facing end cross-sections (the slab keeps its width; a hull of the whole pieces took the pink rim), minus world
# props in front. Own (blue) shots and one-piece enemy labels are untouched.
import numpy as _f4_np
from scipy.ndimage import label as _f4_cc

_F4_SNOW = (6000732, 6001468)       # BlowerUltiProjectile (+ overcharged); only the sweep's 048 uses 6000732
_F4_END = 4.0                       # px: depth of the end cross-section of a piece used for the bridge
_F4_PROP_PX = 15000
_F4_prev_labels = labels


def labels(objs, cam, R, state=None, ended=(), img=None):
    P = _F4_prev_labels(objs, cam, R, state=state, ended=ended, img=img)
    mine = [o for o in objs if o['t'] in _F4_SNOW and not o.get('own', True)]
    if not mine: return P
    import bulletlabels as BL
    oid = {BL._lid(o) for o in mine} | {o['id'] for o in mine}
    lids = P['lids']; ids = P['ids']; share = P['share']; out = None
    ids_h = R['id'][::2, ::2]; cnt = None
    for k, l in enumerate(lids):
        if l[0] not in oid or l[2] != 'body': continue
        m = ids == k
        if not m.any(): continue
        L, n = _f4_cc(m, structure=_f4_np.ones((3, 3)))
        sz = _f4_np.bincount(L.ravel()); sz[0] = 0
        keep = [i for i in range(1, n + 1) if sz[i] >= 0.15 * sz.max()]      # crumbs are not slab pieces
        if len(keep) < 2: continue
        # bridge the pieces along the slab's axis: the hull of the two facing end cross-sections of consecutive pieces (a hull of the whole
        # pieces would also take the pink rim / rounded ends and be wider than the slab)
        ys, xs = _f4_np.nonzero(m); pts = _f4_np.c_[xs, ys].astype(float); c0 = pts.mean(0)
        ev, evec = _f4_np.linalg.eigh(_f4_np.cov((pts - c0).T)); ax = evec[:, -1]
        Lb = L[ys, xs]; k_ = _f4_np.isin(Lb, keep); U = (pts - c0) @ ax
        order = sorted(keep, key=lambda i: U[Lb == i].mean())
        hl = _f4_np.zeros_like(m)
        for i, j in zip(order[:-1], order[1:]):
            ui, uj = U[Lb == i], U[Lb == j]
            si = pts[(Lb == i) & (U >= ui.max() - _F4_END)]; sj = pts[(Lb == j) & (U <= uj.min() + _F4_END)]
            if ui.max() >= uj.min(): continue
            q = _f4_np.vstack([si, sj]); bm = _f4_np.zeros_like(m); bm[q[:, 1].astype(int), q[:, 0].astype(int)] = True
            hl |= hull_mask(bm)
        if cnt is None: cnt = _f4_np.bincount(ids_h.ravel())
        occ = [d for d, dr in R['draws'].items() if dr.get('dmask') and not dr.get('blend') and d < len(cnt) and 0 < cnt[d] <= _F4_PROP_PX]
        add = hl & (ids < 0)
        if occ: add &= ~_f4_np.isin(ids_h, occ)
        if not add.any(): continue
        if out is None: out = (ids.copy(), share.copy())
        cv = R['cv'][::2, ::2]
        out[0][add] = k; out[1][add] = _f4_np.maximum(cv[add] / 255.0, BL.CUT)
    if out is None: return P
    P2 = dict(P); P2.update(ids=out[0], share=out[1])
    return P2


# ---- Mr P (SpawnerDude) attack 2 'IndirectProjectile' (045 6000696, 291 6000699, 302 6000705) ---------------------------------------
# The clip of these types is one glow sprite (the launch ball); the suitcase is a 3D model. The labeller takes one of two branches
# depending on whether that glow sprite shows up in the ID pass (sprite code 2): with it ('clip drawn') the label is the suitcase's mesh
# pixels (+ glow where the prediction agrees); without it the projectile's own draw / the prediction (suitcase AND launch ball).
# Own (blue) shots: the blue launch ball is additive, it is never in the ID pass at the shot's first tick, so the first label is
# suitcase + ball. Enemy (red): the red launch ball IS a blended draw in the ID pass (code 2, ~330 px) at that first tick, which flips the
# labeller into the 'clip drawn' branch: ball and halo left out, suitcase cut (enemycmp2 045 / 302 attack 0, first tick: 1129 px against
# 1985 own). Fix: at an enemy shot's FIRST tick the glow code is not reported, so the first label is suitcase + ball as for the own shot.
# Later ticks keep the installed behaviour (own and enemy flip between the branches by what the ID pass shows, they agree there).
_F4_MRP = (6000696, 6000699, 6000705)
_F4_state = {'last': -1, 'seen': set(), 'first': set()}
_f4_prev_sprite_pixels = sprite_pixels
_f4_prev_labels_mrp = labels


def labels(objs, cam, R, state=None, ended=(), img=None):
    tick = int(state['tick']) if state is not None else -1
    if tick < _F4_state['last']: _F4_state['seen'] = set()           # frames go back: a new capture
    _F4_state['last'] = tick
    fs = set()
    for o in objs:
        if o['t'] in _F4_MRP and not o.get('own', True):
            if o['id'] not in _F4_state['seen']: fs.add(o['t'])
            _F4_state['seen'].add(o['id'])
    _F4_state['first'] = fs
    return _f4_prev_labels_mrp(objs, cam, R, state=state, ended=ended, img=img)


def sprite_pixels(t, own, R):
    C = _f4_prev_sprite_pixels(t, own, R)
    if not own and t in _F4_state['first'] and C is not None and (C == 2).any():
        C = C.copy(); C[C == 2] = 0
    return C
