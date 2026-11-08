# ShieldTank 004 Supercell Make (skin 400 ShieldTankSupercellMake), Buster004Projectile (type 6001261): the volley of 3 atoms
# gets ONE label while the atoms touch or overlap, separate labels once they are apart (the user, 10-28, Q7 "one label while
# they touch, or one label per atom?": answer "one label while they touch" = the Skater 402 rule, z_1024_skater402).
#
# Before this fix: every atom had its own label (its own hull, rule 5: the atom is drawn as thin orbit rings), and the three
# hulls of a touching volley drew as one outline on the page, i.e. the labels were not one object per label.
# Merge (as skater402, same constants): atoms whose labelled bodies touch (1 px dilation) AND whose label centroids are closer
# than 1.7 x radius (sqrt(area / pi) of the larger) form one stack (union-find). A stack is ONE label: the union of its members'
# labelled pixels, closed (T261_CLOSE frame px) and filled (holes filled), taken over unlabelled pixels inside it except
# opaque world geometry in front of the stack. The carrier is the lowest object id of the stack. A lone atom keeps its own
# label (its hull), unchanged. Only type 6001261 of skin ShieldTankSupercellMake; other types and skins are untouched.
# The shared type: owners.runtime gives skin ShieldTankSupercellMake for 6001261 only (the other Buster skins use 6001259 /
# 6001263 and their own super types).

import numpy as np

T261_TYPE = 6001261
T261_SKIN = 'ShieldTankSupercellMake'
T261_CLOSE = 6            # frame px: closing before the fill (as skater402)
T261_MERGE_DR = 1.7       # centroid distance / radius below which two touching atoms are one label (as skater402)
T261_STACKS = []          # per frame: (carrier id, member ids, px of the merged label)

# Orb at tick 307 (attack 90): when atom 2000011 stops at the wall its rings vanish and a big bright white-yellow orb (its hit
# flash, not labelled) is drawn at its position. The orb is shape 63 of the atom's own clip (effects_brawler.sc,
# buster_004_atk_projectile), the same shape as each atom's core glow (50-120 px, centred on the projected anchor, measured
# on every tick). The neighbour atom's label took 403 px of the orb. Rule: shape-63 pixels farther than T261_CORE_R frame px
# from the atom's own projected anchor are not its body (the orb); the core glow stays. Only type 6001261.
T261_SHAPE_CORE = 64      # shape 63 + 1 (the shape id + 1 of _a4_shape_pixels)
T261_CORE_R = 12          # frame px: the core glow lies within ~11 px of the anchor (measured 305, 309; max 10.7-11.4)
T261_FILE = 'sc/effects_brawler.sc'
T261_CLIPS = ('buster_004_atk_projectile', 'buster_004_atk_projectile_red')

_t261_prev_post = POST_BODY.get(T261_TYPE)


def _t261_post(body, ctx):
    import masks
    if _t261_prev_post is not None: body = _t261_prev_post(body, ctx)
    if not body.any() or ctx['o'].get('anchor') is None: return body
    SP = _a4_shape_pixels(ctx['R'], T261_FILE, T261_CLIPS)
    p, _w = masks.project(ctx['cam'], np.asarray(ctx['o']['anchor'], float).reshape(1, 3))
    px, py = p[0] / 2.0                                  # full-resolution px -> frame px
    yy, xx = np.mgrid[0:body.shape[0], 0:body.shape[1]]
    near = np.hypot(xx - px, yy - py) <= T261_CORE_R
    orb = body & (SP == T261_SHAPE_CORE) & ~near
    return body & ~orb


POST_BODY[T261_TYPE] = _t261_post

_t261_prev_labels = labels


def labels(objs, cam, R, state=None, ended=(), img=None):
    import owners, bulletlabels as BL
    from scipy.ndimage import binary_dilation
    T261_STACKS.clear()
    P = _t261_prev_labels(objs, cam, R, state=state, ended=ended, img=img)
    on = {o['id'] for o in objs if o['t'] == T261_TYPE and owners.runtime(o['t'], None, None).get('skin') == T261_SKIN}
    if len(on) < 2: return P
    ids, lids, share = P['ids'], P['lids'], P['share']
    anc = {o['id']: o['anchor'] for o in objs if o['id'] in on and o.get('anchor') is not None}
    mem = {}
    for o in objs:
        if o['id'] not in on: continue
        lid = (BL._lid(o), -1, 'body')
        if lid not in lids: continue
        k = lids.index(lid); m = ids == k
        if m.any(): mem[o['id']] = (k, m)
    if len(mem) < 2: return P
    # stacks: atoms whose labelled bodies touch and whose centres are closer than T261_MERGE_DR x radius (union-find)
    oids = sorted(mem); par = {i: i for i in oids}
    def find(i):
        while par[i] != i:
            par[i] = par[par[i]]; i = par[i]
        return i
    for a in oids:
        da = binary_dilation(mem[a][1], iterations=1)
        for b in oids:
            if b <= a or not (da & mem[b][1]).any(): continue
            ya, xa = np.nonzero(mem[a][1]); yb, xb = np.nonzero(mem[b][1])
            r = (max(mem[a][1].sum(), mem[b][1].sum()) / np.pi) ** 0.5
            dr = float(np.hypot(ya.mean() - yb.mean(), xa.mean() - xb.mean()) / max(r, 1.0))
            if dr < T261_MERGE_DR: par[find(b)] = find(a)
    stacks = {}
    for i in oids: stacks.setdefault(find(i), []).append(i)
    # opaque world geometry (blend 0 with depth writes: walls, crates) nearer the camera than the stack's nearest atom is not
    # the atom: the merged label does not take it (the same test as skater402)
    ids_h = R['id'][::2, ::2]; draws = R['draws']; world = np.zeros(ids.shape, bool)
    for d in np.unique(ids_h[ids_h > 0]):
        dr_ = draws.get(int(d), {})
        if dr_.get('blend') == 0 and dr_.get('dmask') == 1: world |= ids_h == d
    wd = R.get('wd'); wref = R['meta'].get('wref', 0.0)
    for root, mm in sorted(stacks.items(), key=lambda kv: min(kv[1])):
        if len(mm) < 2: continue
        mm = sorted(mm); ks = [mem[i][0] for i in mm]; carrier = ks[0]
        U = np.zeros(ids.shape, bool)
        for i in mm: U |= mem[i][1]
        H = _bfh_t261(_close(U, T261_CLOSE)) & (_dilate(U, T261_CLOSE) | U)
        mine = np.isin(ids, ks)
        occ = world
        if wd is not None and any(i in anc for i in mm):
            dz = min(_eye_depth(cam, anc[i], wref)[0] for i in mm if i in anc)
            occ = world & (wd < dz - DEPTH_TOL)
        take = H & (((ids < 0) & ~occ) | mine)
        new = take & ~mine
        ids[take] = carrier
        share[new] = BL.CUT
        T261_STACKS.append((mm[0], tuple(mm), int(take.sum())))
    return P


def _bfh_t261(m):
    from scipy.ndimage import binary_fill_holes
    return binary_fill_holes(m)
