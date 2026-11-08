# sweep 10-27 fix for skin 341 (Haiku fix, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s341/)
# sweep 10-27 fix for skin 341 (ShotgunGirlBrawloween, Shelly 006 attack, row "sw2_341 attack 90 deg", pair 6385, tick 317,
# kind too_small; task out/sweep2/haikutasks/sweep1027/s341/).
# At tick 317 the pod 2000022 (type 6000012, Shelly006Projectile) has reached its range (finished 1) and is hidden in the
# prediction (drawn False, no label of its own). What is drawn in its place is its max-range end animation
# (shelly_006_atk_reached: the flat ring-eyed pod between the two live pods; at ticks 321-323 the same rings expand and
# fade). The live pod 2000023 next to it took the left ring of that leftover (19 px at x 587-594, y 482-490): the
# outline was only partly on the flat pod. Ended attacks and their end animations are not labelled (the user, 2026-10-23
# global rule; rule 2). Fix (labels() wrapper, keyed to type 6000012 only): a live pod's body pixels within 28 px of the
# projected centre of an ended pod of the same type (finished by range, hidden) and more than 2 px outside its own
# predicted label, and whose draw (the ID pass's draw id) is not one the pod's own body pixels use in this frame, are dropped.
# The draw test keeps a live pod's own parts that lie beside its prediction (the teal spur of the pod at tick 277, attack 0
# row: its draws are the pod's own), and drops the ring (its draws 131/133/135 are not the pod's). The ended pod keeps no
# label. Nothing changes without such an ended pod (checked on every tick of the three rows of this skin).
# Shared type: 6000012 only (its clip is shelly_006_atk_projectile; other skins use other ids).
import numpy as _s341np
from scipy.ndimage import binary_dilation as _s341dil

S341_TYPES = {6000012}
S341_RANGE = 28    # px: radius around an ended pod's projected centre
S341_OWN = 2       # px: the live pod's own predicted label, grown by this, is kept

_s341_prev = labels


def _s341_proj(cam, p):
    v = cam @ _s341np.array([p[0], p[1], p[2], 1.0])
    return (v[0] / v[3] * 0.5 + 0.5) * 1200, (1 - (v[1] / v[3] * 0.5 + 0.5)) * 540


def labels(objs, cam, R, state=None, ended=(), img=None):
    P = _s341_prev(objs, cam, R, state=state, ended=ended, img=img)
    ends = [o for o in objs if o['t'] in S341_TYPES and not o.get('drawn', True) and o.get('finished') == 1]
    if not ends or not any(o['t'] in S341_TYPES and o.get('drawn', True) for o in objs): return P
    ids, lids, share = P['ids'], P['lids'], P['share']
    PB = BL.labels(objs, cam, cut=0.01)                       # the prediction, as labels() computes it
    H, W = ids.shape; yy, xx = _s341np.mgrid[0:H, 0:W]
    for E in ends:
        cx, cy = _s341_proj(cam, E['anchor'])
        F = (xx - cx) ** 2 + (yy - cy) ** 2 <= S341_RANGE ** 2
        for A in objs:
            if A['t'] not in S341_TYPES or not A.get('drawn', True) or A['id'] == E['id']: continue
            lid = (BL._lid(A), -1, 'body')
            if lid not in lids: continue
            k = lids.index(lid)
            ks = [j for j, l in enumerate(PB['lids']) if l[0] == BL._lid(A)]
            pred = _s341np.isin(PB['ids'], ks) if ks else _s341np.zeros((H, W), bool)
            near = _s341dil(pred, iterations=S341_OWN)
            M = ids == k
            own_draws = _s341np.unique(R['id'][::2, ::2][M & near])        # draws of the pod's own body pixels
            d = M & F & ~near & ~_s341np.isin(R['id'][::2, ::2], own_draws)
            if d.any(): ids[d] = -1; share[d] = 0
    return P
