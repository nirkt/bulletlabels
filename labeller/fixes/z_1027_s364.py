# sweep 10-27 fix for skin 364 (Haiku fix, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s364/)
# Skin 364 BanditGirlStar_remake (Shelly's skin; sweep 2026-10-27, task s364): the shotgun pellets of one shot that touch or
# overlap are one label, their merged outline filled solid.
#
# Defect (pairs 6844, 6845, 6846: attack 0 deg, ticks 265, 267, 269 (captured: 267, 270); the same type on attack 90 deg, ticks 305-315):
# the 5 pellets of one shot (Shelly009Projectile, type 6000004, one object each, drawn as crystal stars) overlap in a cluster.
# Each pellet keeps its own label, so the cluster is many small loops with gaps and holes between the stars, not one outline.
#
# User decision applied (binding, the same rule as Skater 402 (z_1024_skater402.py) and the overlapping Emz clouds): projectiles
# of one shot whose bodies touch or overlap form ONE stack and get ONE label, the stack's own outline filled solid (gaps between
# the stars closed, holes filled); a projectile that touches no other keeps its own label (separate again once apart).
#
# Method: after the per-object labels are made (labels() below wraps the previous labels()), the body labels of this type on
# this skin are grouped by contact (4-neighbour touch of their labelled bodies, union-find). Each stack's union is closed by
# SH364_CLOSE px and every enclosed hole is filled; the unlabelled pixels inside that outline go to the stack's carrier id (the
# lowest object id of the stack), together with the members' pixels, except opaque world geometry nearer the camera than the
# stack's nearest pellet (depth test). Other labels' pixels inside the outline keep their labels. Glow is not touched (glow is
# ignored entirely: round-2 decision). No projectile loses its pixels: the stack's pixels contain every member.
#
# Why contact and not the centre distance of Skater 402: the labeller hands some stars to two pellets (sw2_364 tick 273: the top
# and left stars are split between pellets 15 and 16), so label centroids are not the stars' centres; contact of the labelled
# bodies is measured on the drawn pixels. At tick 273 the top and left stars touch (one stack), the right star is apart and stays
# its own label.
#
# Only type 6000004 (Shelly009Projectile) of skin BanditGirlStar_remake: the only skin whose projectiles are this type (owners
# skin table). The super (Shelly009UltiProjectile, 6000005) and every other type are untouched.

import numpy as np

SH364_TYPE = 6000004
SH364_SKIN = 'BanditGirlStar_remake'   # skin 364
SH364_CLOSE = 3                        # px (frame resolution): gaps between the stars closed before the outline is filled
SH364_TOUCH = 1                        # px: bodies within this 4-neighbour distance touch
SH364_STACKS = []                      # per frame: (carrier id, member ids, outline px, px taken, px of other labels inside)


def _sh364_on(objs):
    import owners
    out = set()
    for o in objs:
        if o['t'] != SH364_TYPE: continue
        if owners.runtime(o['t'], None, None).get('skin') == SH364_SKIN: out.add(o['id'])
    return out


_sh364_prev_labels = labels


def labels(objs, cam, R, state=None, ended=(), img=None):
    import bulletlabels as BL
    from scipy.ndimage import binary_dilation, binary_fill_holes
    SH364_STACKS.clear()
    P = _sh364_prev_labels(objs, cam, R, state=state, ended=ended, img=img)
    on = _sh364_on(objs)
    if not on: return P
    ids, lids, share = P['ids'], P['lids'], P['share']
    mem = {}
    for o in objs:
        if o['t'] != SH364_TYPE or o['id'] not in on: continue
        lid = (BL._lid(o), -1, 'body')
        if lid not in lids: continue
        k = lids.index(lid); m = ids == k
        if m.any(): mem[o['id']] = (k, m)
    if len(mem) < 2: return P
    # stacks: bodies that touch; union-find over the contact pairs
    oids = sorted(mem); par = {i: i for i in oids}
    def find(i):
        while par[i] != i:
            par[i] = par[par[i]]; i = par[i]
        return i
    for a in oids:
        da = binary_dilation(mem[a][1], iterations=SH364_TOUCH)
        for b in oids:
            if b > a and (da & mem[b][1]).any(): par[find(b)] = find(a)
    stacks = {}
    for i in oids: stacks.setdefault(find(i), []).append(i)
    # opaque world geometry (walls, crates: blend 0 with depth writes) nearer the camera than the stack's nearest pellet is not
    # the pellet: the outline does not take it; ground or walls seen through the gaps between the stars are behind them (kept)
    ids_h = R['id'][::2, ::2]; draws = R['draws']; world = np.zeros(ids.shape, bool)
    if any(len(mm) > 1 for mm in stacks.values()):
        for d in np.unique(ids_h[ids_h > 0]):
            dr = draws.get(int(d), {})
            if dr.get('blend') == 0 and dr.get('dmask') == 1: world |= ids_h == d
    wd = R.get('wd'); wref = R['meta'].get('wref', 0.0)
    anc = {o['id']: o['anchor'] for o in objs if o['id'] in on and o.get('anchor') is not None}
    for root, mm in sorted(stacks.items(), key=lambda kv: min(kv[1])):
        if len(mm) < 2: continue
        mm = sorted(mm); ks = [mem[i][0] for i in mm]; carrier = ks[0]
        U = np.zeros(ids.shape, bool)
        for i in mm: U |= mem[i][1]
        # the stack's own outline: gaps between the stars closed, every enclosed hole filled (solid, the user's 10-24 rule);
        # it contains every member's pixels
        H = binary_fill_holes(_close(U, SH364_CLOSE) | U)
        mine = np.isin(ids, ks)
        other = H & (ids >= 0) & ~mine                      # pixels of other labels inside the outline: they keep them
        occ = world
        if wd is not None and any(i in anc for i in mm):
            dz = min(_eye_depth(cam, anc[i], wref)[0] for i in mm if i in anc)   # the stack's nearest pellet
            occ = world & (wd < dz - DEPTH_TOL)              # world nearer than the pellets: in front of them
        take = H & (((ids < 0) & ~occ) | mine)
        new = take & ~mine
        ids[take] = carrier
        share[new] = BL.CUT
        SH364_STACKS.append((mm[0], tuple(mm), int(H.sum()), int(take.sum()), int(other.sum())))
    return P
