# sweep 10-27 fix for skin 240 (Haiku fix, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s240/)
# Shelly 003 attack (Shelly003Projectile, type 6000006; skin 240 ShotgunGirlWitch, task s240, sweep 2026-10-27): the
# pellets of one shot are one solid label while they overlap. Pairs 4335 (attack 0 deg, tick 265) and 4342 (attack 90 deg,
# tick 305), both 'hole': the outline of the ball cluster was several loops with inner red seams.
# Cause: one shot spawns five pellet objects on the same tick (2000015-19 at tick 265, 2000020-24 at tick 305). Each has its
# own label id, and the claim step splits the touching cluster between them, so the outline shows one loop per piece and a
# seam between neighbours. The gaps between the balls are the clip's glow (not body): they are open to the outside, so the
# hole fill of idlabels does not close them.
# Rule (keyed to type 6000006 only; the same stack rule as Skater 402, z_1024_skater402, 10-24, and the Emz clouds):
#  1. pellets whose claimed bodies touch form one stack (union-find over the pieces, as the Emz clouds, 10-24: 'touch or
#     overlap'); pieces of one cluster always touch, and a pellet with no contact with the others stays its own label.
#     The centre-distance test of Skater 402 is not used here: the claim splits one cluster between the pellets, so the
#     centroids of the pieces are not the balls' centres (tick 305 kept a 121 px piece apart with it);
#  2. the stack's union is closed by SH003_CLOSE px (half resolution) and its holes filled, inside the union's dilation:
#     one solid outline in the cluster's own shape, not a convex hull (the user, 10-24);
#  3. the stack's pixels go to the lowest object id of the stack (the other members keep none); unlabelled pixels in the
#     outline are taken, except opaque world geometry in front of the stack (the depth test) and other labels' pixels.
# Approaches considered:
#   1. convex hull per stack (ONE_LABEL 'hull'): rejected, the user prefers the object's own outline filled (10-24).
#   2. fill holes of each pellet alone: rejected, the gaps open to the outside, so no hole is enclosed.
#   3. merge every pellet of the shot: rejected, pellets that have separated (tick 273) would be merged.
#   4. stack by contact of the claimed pieces, then close + fill (chosen): the Emz rule (10-24); no threshold, no colour.
# Checked on sw2_240 ticks 263-279 and 303-319 (measured per tick: parts, holes and ids of the pellets' label).

SH003_TYPE = 6000006
SH003_TOUCH = 1         # px (half resolution): bodies within this distance touch
SH003_CLOSE = 3         # px (half resolution): gaps between the balls of a stack closed before filling


_sh003_prev_labels = labels


def labels(objs, cam, R, state=None, ended=(), img=None):
    import bulletlabels as BL
    from scipy.ndimage import binary_fill_holes, binary_dilation
    P = _sh003_prev_labels(objs, cam, R, state=state, ended=ended, img=img)
    if not any(o['t'] == SH003_TYPE for o in objs): return P
    ids, lids, share = P['ids'], P['lids'], P['share']
    mem = {}
    for o in objs:
        if o['t'] != SH003_TYPE: continue
        lid = (BL._lid(o), -1, 'body')
        if lid not in lids: continue
        k = lids.index(lid); m = ids == k
        if m.any(): mem[o['id']] = (k, m)
    if len(mem) < 2: return P
    oids = sorted(mem); par = {i: i for i in oids}

    def find(i):
        while par[i] != i:
            par[i] = par[par[i]]; i = par[i]
        return i
    for a in oids:
        da = binary_dilation(mem[a][1], iterations=SH003_TOUCH)
        for b in oids:
            if b > a and (da & mem[b][1]).any(): par[find(b)] = find(a)
    stacks = {}
    for i in oids: stacks.setdefault(find(i), []).append(i)
    # opaque world geometry (blend 0, depth writes: walls, crates) in front of the stack is not the pellets' outline
    ids_h = R['id'][::2, ::2]; draws = R['draws']; world = np.zeros(ids.shape, bool)
    for d in np.unique(ids_h[ids_h > 0]):
        dr = draws.get(int(d), {})
        if dr.get('blend') == 0 and dr.get('dmask') == 1: world |= ids_h == d
    wd = R.get('wd'); wref = R['meta'].get('wref', 0.0)
    anc = {o['id']: o['anchor'] for o in objs if o['id'] in mem and o.get('anchor') is not None}
    for root, mm in sorted(stacks.items(), key=lambda kv: min(kv[1])):
        mm = sorted(mm); ks = [mem[i][0] for i in mm]; carrier = ks[0]
        U = np.zeros(ids.shape, bool)
        for i in mm: U |= mem[i][1]
        H = binary_fill_holes(_close(U, SH003_CLOSE)) & (_dilate(U, SH003_CLOSE) | U)
        mine = np.isin(ids, ks)
        occ = world
        if wd is not None and any(i in anc for i in mm):
            dz = min(_eye_depth(cam, anc[i], wref)[0] for i in mm if i in anc)
            occ = world & (wd < dz - DEPTH_TOL)
        take = H & (((ids < 0) & ~occ) | mine)
        new = take & ~mine
        ids[take] = carrier
        share[new] = BL.CUT
    return P
