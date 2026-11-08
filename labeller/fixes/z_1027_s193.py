# sweep 10-27 fix for skin 193 (Haiku fix, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s193/)
# sweep 10-27 fix for skin 193 TwinsAngel, attack 0 deg (sweep 2026-10-27, task s193: pair 3400, tick 265, extra_label)
# Type 6001408 Twins003ThrowerProjectile (attack, sc/effects_brawler.sc twin_003_atk_projectile): the arrow's label also took in
# a small blue pip beside the top-left brawler's ammo bar. That pip is UI (the HUD ammo indicator), which is never labelled.
# It is drawn with the arrow's own atlas texture and clip shape 63, so the shape id cannot tell it from the arrow's wing.
# Measured on the capture (frames 676-698, ticks 263-285): at tick 265 the pip is a separate 33 px part, 14.3 px from the
# arrow body. The arrow's own detached cyan tips (ticks 267, 271, 283) touch the body (gap 1.4-2.0 px) and are kept.
# Rule, keyed to this one type id: a part of its label that is under 40 px and more than 6 px away from the label's largest
# part is dropped. The largest part is never dropped, so the projectile keeps its label. Runs as a labels() wrapper after the
# clean-up (z_1099), so it sees the final label map.

_T193_TYPE = 6001408
_T193_SIZE = 40     # px: a part this small ...
_T193_GAP = 6       # px: ... and this far from the main part is not the attack
_t193_prev = labels


def labels(objs, cam, R, state=None, ended=(), img=None):
    from scipy.ndimage import label as cc, distance_transform_edt
    import bulletlabels as BL
    P = _t193_prev(objs, cam, R, state=state, ended=ended, img=img)
    ids, lids, share = P['ids'], P['lids'], P['share']
    ltype = {}
    for o in objs: ltype.setdefault(BL._lid(o), o['t'])
    ks = [k for k in np.unique(ids[ids >= 0]) if lids[k][2] == 'body' and ltype.get(lids[k][0]) == _T193_TYPE]
    for k in ks:
        m = ids == k
        ys, xs = np.nonzero(m); y0, y1, x0, x1 = max(0, ys.min() - 20), ys.max() + 21, max(0, xs.min() - 20), xs.max() + 21
        sm = m[y0:y1, x0:x1]; sid = ids[y0:y1, x0:x1]; ssh = share[y0:y1, x0:x1]
        L, n = cc(sm)
        if n <= 1: continue
        sz = np.bincount(L.ravel())[1:]
        main = 1 + int(np.argmax(sz))
        dt = distance_transform_edt(L != main)
        drop = [i for i in range(1, n + 1) if i != main and sz[i - 1] < _T193_SIZE and dt[L == i].min() > _T193_GAP]
        if drop:
            d = np.isin(L, drop) & sm; sid[d] = -1; ssh[d] = 0
    return P
