# sweep 10-27 fix for skin 054 (Haiku fix, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s054/)
# Ice Dude attack, IceDudeProjectile (type 6000847, lou_def_atk_projectile): skin 054 IceDudeDefault, sweep 2026-10-27,
# findings 980 (tick 269, attack 0 deg), 985 (305), 986 (307), 988 (313): specks. The label of each blob took small cyan
# bubbles that float between the blobs of the stream (ticks 265-273, 305-313) and the sparkle bubbles above a blob's cone
# apex (307-313) as separate red rings. The user's rule: the attack body only, detached dots stay out.
#
# Final-label wrapper for this type only (after the clean-up, z_1099). Per label of a projectile of this type:
#  - the main body is the parts of its label that stay blob-sized (>= ICE_MAIN px) after an opening with a radius-2 disk;
#    a projectile without such a part (fresh or clipped) keeps its label unchanged, so no projectile loses its label;
#  (a) bubbles: the small opened parts (8 .. ICE_MAIN px) apart from the main body (joined to it only by a neck) and their
#      2-px neighbourhood, minus the main body itself;
#  (b) specks: pieces of the label outside the main body's neighbourhood with at most ICE_SPECK px and an inscribed radius
#      of at least 2 px (a 3-4 px dot; a thin tip or the 34-px cyan disc on a cone apex, radius below 2, stays).
# Checked on the affected rows (sw2_054 attack 0 / attack 90), see s054/report.md.
import numpy as _ice_np
from scipy.ndimage import binary_opening as _ice_open, binary_dilation as _ice_dil, label as _ice_cc, distance_transform_edt as _ice_edt

ICE_DUDE_ATK = 6000847
ICE_MAIN, ICE_CORE_MIN, ICE_SPECK = 150, 8, 25
_ICE_DISK = (_ice_np.mgrid[-2:3, -2:3][0] ** 2 + _ice_np.mgrid[-2:3, -2:3][1] ** 2) <= 4
_ice_dude_prev_labels = labels


def _ice_dude_cut(P, objs):
    import bulletlabels as BL
    ice = {BL._lid(o) for o in objs if o['t'] == ICE_DUDE_ATK}
    if not ice: return P
    ids = P['ids'].copy(); lids = P['lids']; changed = False
    for k, lid in enumerate(lids):
        if lid[2] != 'body' or lid[0] not in ice: continue
        m = ids == k
        if not m.any(): continue
        L, n = _ice_cc(_ice_open(m, structure=_ICE_DISK))
        if n == 0: continue
        sz = _ice_np.bincount(L.ravel())[1:]
        main_ids = _ice_np.nonzero(sz >= ICE_MAIN)[0] + 1
        if main_ids.size == 0: continue                      # no body to keep: the label stays as it is
        main = _ice_np.isin(L, main_ids)
        near = _ice_dil(main, structure=_ICE_DISK)
        drop = _ice_np.zeros_like(m)
        for j in _ice_np.nonzero((sz >= ICE_CORE_MIN) & (sz < ICE_MAIN))[0] + 1:      # (a) bubbles on a neck
            drop |= _ice_dil(L == j, structure=_ICE_DISK) & m
        Lr, nr = _ice_cc(m & ~near)                                                   # (b) specks
        for j in range(1, nr + 1):
            pm = Lr == j
            if pm.sum() <= ICE_SPECK and _ice_edt(pm).max() >= 2.0: drop |= pm
        drop &= ~main
        if drop.any():
            ids[drop] = -1; changed = True
    if not changed: return P
    P2 = dict(P); P2['ids'] = ids
    return P2


def labels(objs, cam, R, state=None, ended=(), img=None):
    P = _ice_dude_prev_labels(objs, cam, R, state=state, ended=ended, img=img)
    return _ice_dude_cut(P, objs)
