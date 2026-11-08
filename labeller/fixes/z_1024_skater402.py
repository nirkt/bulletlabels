# Skater Sands Of Times 402 (Skater002Projectile 6001569): touching balls of a shot = ONE label, their merged outline filled (user 10-24; Haiku task 24b, reviewed; coordinator: filled outline instead of convex hull)
# Task 24b (2026-10-24), skin 402 SkaterSandsOfTimes: Skater002Projectile (type 6001569, the balls of attack 0 deg and 90 deg).
# User decision (binding, 2026-10-24, the same rule as for the overlapping Emz clouds): balls whose bodies touch or overlap
# form one STACK, labelled as ONE convex hull over the stack (each ball's whole visible body incl. its bright rim; no holes,
# no dents from the trail or effects drawn over it). A ball that touches no other ball is its own hull. The trail, the soft
# glow and the max-range end bloom (ticks 317-321, 275) stay unlabelled.
#
# Cause: labels() gives each ball its own label id (no wave grouping for this type), so the two balls of a shot are two
# labels; a ball partly hidden behind the other is a crescent, and the cyan trail drawn over a ball leaves a dent in its body.
#
# Method: after the per-object labels are made (labels() below wraps the previous labels()), the balls of this skin are
# grouped by contact of their labelled bodies (1 px dilation touches another body; union-find). Each stack's pixels are
# hulled (hull_mask over the union of its bodies). All pixels of the members lie inside the hull and go to the stack's
# carrier id (the lowest object id of the stack); the other members of the stack keep no pixels. Unlabelled pixels inside the
# hull are taken too, except opaque world geometry nearer the camera than the stack's nearest ball (walls, crates: blend 0 with
# depth writes; the depth test) and pixels of other labels.
# The hull covers the dent of the trail and the gap between two balls (the same as the Emz clouds: glow inside a hull is
# covered). Only type 6001569 of skin SkaterSandsOfTimes; other skins and other types are untouched.

import numpy as np

SK402_TYPE = 6001569
SK402_SKIN = 'SkaterSandsOfTimes'   # skin 402
SK402_CLOSE = 6
# the user (10-24): the balls are separate labels again once they are far enough apart: merged only while they overlap,
# i.e. centre distance (label centroids) < SK402_MERGE_DR x ball radius (sqrt(area / pi) of the larger). Measured on sw2_402:
# overlapping 0.7-1.55 (t265-271, t305-309), touching side by side 1.75-2.3 (t273, t311-315)
SK402_MERGE_DR = 1.7


def _sk402_dr(ma, mb):
    ya, xa = np.nonzero(ma); yb, xb = np.nonzero(mb)
    r = (max(ma.sum(), mb.sum()) / np.pi) ** 0.5
    return float(np.hypot(ya.mean() - yb.mean(), xa.mean() - xb.mean()) / max(r, 1.0))

SK402_TOUCH = 1                     # px (half resolution): bodies within this distance touch
SK402_STACKS = []                   # per frame: (carrier id, member ids, hull px, px taken, px of other labels inside the hull)


def _sk402_on(objs):
    import owners
    out = set()
    for o in objs:
        if o['t'] != SK402_TYPE: continue
        if owners.runtime(o['t'], None, None).get('skin') == SK402_SKIN: out.add(o['id'])
    return out


_sk402_prev_labels = labels


def labels(objs, cam, R, state=None, ended=(), img=None):
    import bulletlabels as BL
    SK402_STACKS.clear()
    on = _sk402_on(objs)
    P = _sk402_prev_labels(objs, cam, R, state=state, ended=ended, img=img)
    if not on: return P
    ids, lids, share = P['ids'], P['lids'], P['share']
    mem = {}
    for o in objs:
        if o['t'] != SK402_TYPE or o['id'] not in on: continue
        lid = (BL._lid(o), -1, 'body')
        if lid not in lids: continue
        k = lids.index(lid); m = ids == k
        if m.any(): mem[o['id']] = (k, m)
    if not mem: return P
    # stacks: balls whose bodies touch or overlap; union-find over the contact pairs
    from scipy.ndimage import binary_dilation
    oids = sorted(mem); par = {i: i for i in oids}
    def find(i):
        while par[i] != i:
            par[i] = par[par[i]]; i = par[i]
        return i
    for a in oids:
        da = binary_dilation(mem[a][1], iterations=SK402_TOUCH) if SK402_TOUCH else mem[a][1]
        for b in oids:
            if b > a and (da & mem[b][1]).any() and _sk402_dr(mem[a][1], mem[b][1]) < SK402_MERGE_DR: par[find(b)] = find(a)
    stacks = {}
    for i in oids: stacks.setdefault(find(i), []).append(i)
    # opaque world geometry (walls, crates, the ground: blend 0 with depth writes) in front of the stack is not the ball: the hull
    # does not take it. Ground seen through the gap between two balls is behind them (the depth test), so it stays in the hull
    ids_h = R['id'][::2, ::2]; draws = R['draws']; world = np.zeros(ids.shape, bool)
    for d in np.unique(ids_h[ids_h > 0]):
        dr = draws.get(int(d), {})
        if dr.get('blend') == 0 and dr.get('dmask') == 1: world |= ids_h == d
    wd = R.get('wd'); wref = R['meta'].get('wref', 0.0)
    anc = {o['id']: o['anchor'] for o in objs if o['id'] in on and o.get('anchor') is not None}
    for root, mm in sorted(stacks.items(), key=lambda kv: min(kv[1])):
        mm = sorted(mm); ks = [mem[i][0] for i in mm]; carrier = ks[0]
        U = np.zeros(ids.shape, bool)
        for i in mm: U |= mem[i][1]
        # the stack in the balls' own merged outline, filled (the user prefers the own shape over a convex hull, 10-24):
        # closing SK402_CLOSE px fills the trail dents and the seam between touching balls, holes filled
        from scipy.ndimage import binary_fill_holes as _bfh
        H = _bfh(_close(U, SK402_CLOSE)) & (_dilate(U, SK402_CLOSE) | U)
        mine = np.isin(ids, ks)
        other = H & (ids >= 0) & ~mine                      # pixels of other labels inside the hull: they keep them
        occ = world
        if wd is not None and any(i in anc for i in mm):
            dz = min(_eye_depth(cam, anc[i], wref)[0] for i in mm if i in anc)   # the stack's nearest ball
            occ = world & (wd < dz - DEPTH_TOL)              # world nearer than the ball: in front of it
        take = H & (((ids < 0) & ~occ) | mine)
        new = take & ~mine
        ids[take] = carrier
        share[new] = BL.CUT
        SK402_STACKS.append((mm[0], tuple(mm), int(H.sum()), int(take.sum()), int(other.sum())))
    return P

