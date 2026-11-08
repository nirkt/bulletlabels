# hosted review 2026-10-29 fix (Haiku fix, reviewed by the coordinator, installed 10-29 LOCAL ONLY; source out/sweep2/haikutasks/site1029/s073/fix.py; tested via IDLABELS_OVERRIDE after all fixes incl. z_1100_*, hence z_1101_)
# Skin 073 (ShieldTankDefault), ShieldTankProjectile (type 6001254), review 2026-10-29. The user's finding (binding, page
# 073_ShieldTankDefault_01): "attack should consist of a hull around the bright blue rectangles". Flagged: attack 0 deg
# t265/267/269 and attack 90 deg t305/307/309. The attack is a volley of 3 projectiles per shot, each drawn on a bright blue
# rectangular plate (atlas shape 25032 of the clip buster_def_atk_projectile) with spheres (shape 829) and a thin crescent
# (shape 25025) on it. The labels were the spheres and crescents only, with spurs and frayed rings, and the plate left out.
#
# Fix: ONE label per shot. A shot = the projectiles of this type with the same owner, first shown on the same frame (the Emz
# rule, z_1024_emz_attack). Its plate pieces are the shape-25032 pixels of the type's clip that touch the shot's labelled pixels
# or lie within S073_ANCHOR_R px (half res) of a member's projected anchor (a member can have no label of its own on a tick,
# e.g. obj 2000011 at t307 / t309). The label is the convex hull of those plate pieces: the spheres and crescents lie inside the
# plates, so the hull covers them; the soft blue glow beyond the plate edges is not shape 25032 and stays out. Unlabelled
# pixels inside the hull are taken when their front-most draw is one of this type's draws (or a thin line of the attack, closed
# over a 2 px gap) and not an opaque world draw (crates, walls: a crate in front of the plate is not the attack, as Emz / the
# solid map blocks). The shot's labelled pixels outside the hull are dropped (no spurs). The label goes to the lowest object id
# of the shot that has a label; the other ids of the shot are empty (the Emz / ShieldTank 004 convention). A shot with no label
# on a tick (the ended attack at the wall: the white hit flash on the plates, t271 / t311) stays unlabelled, as before.
#
# Shapes of the clip (sc/effects_brawler.sc): 25032 = the plate (bright blue rectangle), 829 = the sphere rings, 25025 = the
# thin crescent. Only type 6001254 is changed; no other type or skin is touched by this file.

import numpy as _np

S073_TYPE = 6001254
S073_PLATE = 25032 + 1        # shape id + 1 (the _a4_shape_pixels convention)
S073_NEAR = 4                 # half-res px: plate pieces this close to the shot's labelled pixels are the shot's plates
S073_ANCHOR_R = 40            # half-res px: plate pieces this close to a member's projected anchor are the shot's plates
S073_STACKS = []              # per frame: (carrier id, member ids, hull px, px of the shot's label)

_s073_prev_labels = labels


def _fill(m):
    from scipy.ndimage import binary_fill_holes
    return binary_fill_holes(m)


def labels(objs, cam, R, state=None, ended=(), img=None):
    import bulletlabels as BL, projtab, masks
    from scipy.ndimage import label as cc
    S073_STACKS.clear()
    P = _s073_prev_labels(objs, cam, R, state=state, ended=ended, img=img)
    if not any(o['t'] == S073_TYPE for o in objs): return P
    ids, lids, share = P['ids'], P['lids'], P['share']
    shots = {}
    for o in objs:   # one shot: same type, same owner, first shown on the same frame
        if o['t'] == S073_TYPE: shots.setdefault((o.get('owner'), o.get('first')), []).append(o)
    if not shots: return P
    T = projtab.table()[S073_TYPE]
    names = tuple(n for n in (T.get('blue'), T.get('red')) if n)
    SP = _a4_shape_pixels(R, T['file'], names)          # shape id + 1 per pixel, at the labels' resolution
    plate = SP == S073_PLATE
    Lp, nL = cc(plate, structure=_np.ones((3, 3), bool))
    # the draws that show this type's own shapes, and the opaque world draws (blend 0, depth write: crates, walls, floor)
    ids_h = R['id'][::2, ::2]; draws = R['draws']
    own = _np.isin(ids_h, _np.unique(ids_h[SP > 0]))
    wdraw = _np.isin(ids_h, [int(d) for d, r in draws.items() if r.get('blend') == 0 and r.get('dmask') == 1])
    try: world = _zs_solid(cam, R)
    except Exception: world = _np.zeros(ids.shape, bool)
    for key, grp in sorted(shots.items(), key=lambda kv: min(o['id'] for o in kv[1])):
        mem = []
        for o in grp:
            lid = (BL._lid(o), -1, 'body')
            if lid not in lids: continue
            k = lids.index(lid); m = ids == k
            if m.any(): mem.append((o['id'], k, m))
        if not mem: continue                                 # no label on this tick (ended attack): stays unlabelled
        mem = sorted(mem); ks = [k for _, k, _ in mem]; carrier = ks[0]
        U = _np.zeros(ids.shape, bool)
        for _, _, m in mem: U |= m
        near = _dilate(U, S073_NEAR)
        comps = set(int(c) for c in _np.unique(Lp[near & plate])) - {0}
        anchors = []
        for o in grp:
            if o.get('anchor') is None: continue
            p, _w = masks.project(cam, _np.asarray(o['anchor'], float).reshape(1, 3))
            anchors.append((float(p[0][0]) / 2.0, float(p[0][1]) / 2.0))   # full-res px -> half-res (x, y)
        for c in range(1, nL + 1):
            if c in comps: continue
            ys, xs = _np.nonzero(Lp == c)
            if any(_np.hypot(xs - ax, ys - ay).min() <= S073_ANCHOR_R for ax, ay in anchors): comps.add(c)
        if not comps: continue
        H = hull_mask(_np.isin(Lp, sorted(comps)))          # ONE convex hull around the shot's bright blue plates
        for c in range(1, nL + 1):                          # small plate pieces inside that hull are plate too
            if c not in comps and (H & (Lp == c)).any(): comps.add(c)
        H = hull_mask(_np.isin(Lp, sorted(comps)))
        mine = _np.isin(ids, ks)
        core = H & (mine | ((ids < 0) & own & ~wdraw & (plate | ~world)))   # the plate itself is never cut by a map tile
        # thin lines of the attack inside the hull that no shape of this type covers (the crescent's strokes) would cut the
        # label: closed and filled, except opaque world draws and other labels
        filled = _fill(_close(core, 2)) & H & ~wdraw & ((ids < 0) | mine)
        take = core | filled
        ids[mine & ~take] = -1                               # the shot's pixels outside the hull are dropped
        new = take & ~mine
        ids[take] = carrier; share[new] = BL.CUT
        S073_STACKS.append((mem[0][0], tuple(i for i, _, _ in mem), int(H.sum()), int(take.sum())))
    return P
