# Skin 481 BarrelBotFootball (round 3, 2026-10-28; unfixed task, attempts 1 and 2 of 10-27 rejected). Darryl003Projectile
# (6000429, paint; both teams since 2026-10-28: enemy red shots were skipped by an own-only filter): the zigzag streak wave was outlined streak by streak (one object per streak row, 5 per
# shot) with gaps between the streaks. The user's rule: an attack drawn as many thin lines = ONE label, the convex hull of the
# whole attack, built from the VISIBLE streak draws only (never from logic points not drawn yet / any more).
# Design (what changed against the rejected fix2, which gave a big hull over empty floor at the fade-in tick and cut visible
# parts at the next ones):
#  * a wave = the objects of one shot: the same o['first'] (the frame the shot's objects appeared). No pixel clustering by
#    distance: two shots at once (ticks 269, 309) = two hulls, however close they are.
#  * the hull is built from the base pipeline's labelled streak pixels of the shot's objects (the front-most streak sprite
#    draws, rows that ended / are not drawn have none: the ended rule is the base's), and only the pixels that are really
#    drawn visibly: the frame's max(R, G, B) >= S481_VIS (a streak is 100-255, the floor 74-80), so a fading-in or faded wave
#    (coverage there, colour = floor) adds nothing. A shot with fewer than S481_MIN visible pixels is not visible yet / any
#    more: NO label (the rejected fix2 hulled the invisible fade-in tick).
#  * parts of the visible pixels under S481_SPECK px (edge crumbs, 8-connected) are dropped before the hull; smoke or
#    anything else is never added (the hull is the hull of the streak pixels).
#  * the hull is NOT cut by crates / smoke / sparks inside it (tested: cutting the crate at tick 311, where the leftmost streak
#    is drawn over the crate, gave a ragged left edge with notches); it is the plain whole-attack hull.
# Other skins: type 6000429 only.

S481_TYPES = {6000429}   # Darryl003Projectile (BarrelBot, skin 481)
S481_VIS = 120           # frame max(R, G, B) from which a pixel of the streak sprite is visible (floor 74-80, streaks 100-255)
S481_MIN = 120           # visible streak pixels (half resolution) a shot needs to be labelled at all (visible waves: 400-2000 / shot)
S481_SPECK = 10          # px: parts of the visible streak pixels smaller than this are crumbs
_s481_prev = labels


def labels(objs, cam, R, state=None, ended=(), img=None):
    import bulletlabels as BL
    from scipy.ndimage import label as cc
    P = _s481_prev(objs, cam, R, state=state, ended=ended, img=img)
    mine = [o for o in objs if o['t'] in S481_TYPES]
    if not mine or img is None: return P
    ids, lids = P['ids'], P['lids']
    byl = {}
    for k, l in enumerate(lids):
        if l[2] == 'body': byl.setdefault(l[0], []).append(k)
    waves = {}
    for o in mine: waves.setdefault(o.get('first'), []).append(BL._lid(o))
    new_ids = ids.copy(); share = P['share'].copy(); cv = R['cv'][::2, ::2]
    vis = img.max(-1) >= S481_VIS
    for w, lid_list in waves.items():
        ks = [k for lid in lid_list for k in byl.get(lid, [])]
        if not ks: continue
        own = np.isin(ids, ks)
        if not own.any(): continue
        v = own & vis
        L, n = cc(v, structure=np.ones((3, 3), bool))
        if n:
            sz = np.bincount(L.ravel()); keep = sz >= S481_SPECK; keep[0] = False; v = keep[L]
        new_ids[own] = -1; share[own] = 0
        if v.sum() < S481_MIN: continue                  # not visible (fade-in / faded): no label
        hull = hull_mask(v)
        free = hull & ((new_ids < 0))
        new_ids[free] = ks[0]; share[free] = np.maximum(cv[free] / 255.0, BL.CUT)
    P['ids'] = new_ids; P['share'] = share
    return P
