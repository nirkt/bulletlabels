# away_1104 / misc  (final-review round, 2026-11-04). One file, one section per owner complaint. Exec'd after all installed fixes.
# Prefix of every global: _aw4m_
#
# ===== skin 015 BanditGirlDefault, super 45 deg, ticks 345-353: owner "some super bullets are missing" =====
# Type 6000001 (ShotgunGirlUltiProjectile: Shelly's default super; the skins that use the default super projectile take it too).
# Cause (measured on sw2_015): one super shot = 9 objects, each object carries FOUR pellet slots (slot ids -1, 1, 2, 3 of the
# prediction; the installed fix skipped slot -1 as the 'whole object' label, but it is a real 4th ball) and every slot is a visible ball (36 balls at tick 349: each object is a diamond of 4 balls). The installed
# z_1100_s015 kept only slot 3 plus the slots whose core is dark blue (red channel < 88) or hidden, so the balls with a paler core
# (the 2nd..4th ball of every object, which is what the owner calls missing bullets) got no disc, and balls closer than 1.5
# radii to a kept ball were merged away (their extent lost).
# Rule: EVERY slot of a live object is a ball: one solid disc (radius S017_R, the measured outer edge of core + rim) at its slot
# centroid; the object's discs are one label (touching = one label, coordinator rule kept). Slots too small to be a ball
# (< S017_MIN px of prediction, e.g. cut by the screen edge) are dropped like before; a slot hidden by smoke keeps its disc
# (live pellets keep their discs under smoke); the depth test against nearer world geometry is kept; ended objects (logic
# 'finished') stay unlabelled (global owner rule 10-23, unchanged).
import numpy as _aw4m_np
import bulletlabels as _aw4m_BL

_AW4M_S015 = {6000001}


def _aw4m_s015_body(body, ctx):
    o = ctx['o']; R = ctx['R']; cam = ctx['cam']
    if o.get('finished', 0):
        return _aw4m_np.zeros(body.shape, bool)
    if not ctx['pred_body'].any(): return body
    Q = _aw4m_BL.labels([o], cam, cut=0.5)               # this object's own prediction, one label per pellet slot
    qids, qlids = Q['ids'], Q['lids']
    H, W = body.shape
    yy, xx = _aw4m_np.mgrid[0:H, 0:W]
    out = _aw4m_np.zeros((H, W), bool); n = 0
    for k, l in enumerate(qlids):
        if l[0] != o['id'] or l[2] != 'body': continue      # slot -1 is the 4th ball (62-66 px like the others)
        M = qids == k
        if int(M.sum()) < S017_MIN: continue
        ys, xs = _aw4m_np.nonzero(M)
        out |= (xx - xs.mean()) ** 2 + (yy - ys.mean()) ** 2 <= S017_R ** 2; n += 1
    if not n: return body
    wd = R.get('wd')
    if wd is not None and o.get('anchor') is not None:
        dz = _eye_depth(cam, o['anchor'], R['meta'].get('wref', 0.0))[0]
        out &= wd >= dz - S017_DEPTH
    return out


for _t in _AW4M_S015:
    POST_BODY[_t] = _aw4m_s015_body


# Seams: the discs of neighbouring objects of one super (all 36 balls now labelled) leave 1-2 px channels / notches where two
# label shapes meet. The union of the type's labels in a frame is bridged (closing, radius _AW4M_BR px) and its small enclosed holes
# filled; only pixels no label holds are added, each taking the label of its nearest label pixel. (Same idea as z_1107 for Shelly.)
_AW4M_BR = 2
_aw4m_prev_labels = labels


def labels(objs, cam, R, state=None, ended=(), img=None):
    P = _aw4m_prev_labels(objs, cam, R, state=state, ended=ended, img=img)
    from scipy.ndimage import binary_fill_holes as _fh, distance_transform_edt as _edt
    cl = {BL._lid(o) for o in objs if o['t'] in _AW4M_S015}
    ids, lids = P['ids'], P['lids']
    ks = [k for k, l in enumerate(lids) if l[0] in cl and l[2] == 'body']
    if not ks: return P
    U = np.isin(ids, ks)
    if not U.any(): return P
    C = _fh(_close(U, _AW4M_BR))
    add = C & ~U & (ids < 0)
    if not add.any(): return P
    _, (iy, ix) = _edt(~U, return_indices=True)
    ids = ids.copy(); share = P['share'].copy(); cv = R['cv'][::2, ::2]
    ids[add] = ids[iy[add], ix[add]]; share[add] = np.maximum(cv[add] / 255.0, BL.CUT)
    P2 = dict(P); P2['ids'] = ids; P2['share'] = share
    return P2


# ===== skin 118 JesterWoodspirit, attack 90 deg, ticks 305 / 307: owner "too small / cut off" =====
# (the old coordinator note "FIXED 10-29" did not hold: the owner marked the page bad again.)
# Type 6001270 (Chester002Projectile = skin 118's own attack; 119 / 120 fire 6001275 / 6001280, not touched).
# Cause (measured on sw2_118): the two ghosts of one shot overlap at 305 and 307. Between the front ghost's label and the back
# ghost's label there is a vertical strip (305: 10 x 32 px, 206 px; 307: 227 px) of UNLABELLED pixels: the bright cyan face /
# rim of the back ghost seen through the front ghost's translucent edge (colour R 15-90, G 235-250, B 220-245, a translucent
# draw of the glow layer shape 0 of the clip). z_1101_s118jester only takes glow-layer pixels when they are solid green, not
# cyan, so the strip stays out and the page shows a hole between the two outlines.
# Rule: pixels of the type's clip (any shape), bright cyan or solid green (the same colour tests as z_1101), lying inside the
# closing (radius _AW4M_G118_CL) of the union of the ghosts' labels (so only gaps between / inside the two labels), unlabelled, are
# added; each takes the label of its nearest label pixel (the nearer ghost). The dark teal halo between ghosts that have separated
# (309: G 137) fails the colour test and stays out; at 303 / 311 / 313.. nothing changes (no gap).
_AW4M_G118_T = 6001270
_AW4M_G118_CL = 6        # px: closing iterations (3x3): gaps up to ~12 px wide between the two ghost labels
_aw4m_prev_labels2 = labels


def labels(objs, cam, R, state=None, ended=(), img=None):
    P = _aw4m_prev_labels2(objs, cam, R, state=state, ended=ended, img=img)
    mine = [o for o in objs if o['t'] == _AW4M_G118_T]
    if not mine or img is None: return P
    from scipy import ndimage as nd
    ids, lids = P['ids'], P['lids']
    ks = [k for k, l in enumerate(lids) if l[2] == 'body' and l[0] in {BL._lid(o) for o in mine}]
    if not ks: return P
    U = np.isin(ids, ks)
    if not U.any(): return P
    Cl = nd.binary_fill_holes(nd.binary_closing(np.pad(U, 10), structure=np.ones((3, 3), bool), iterations=_AW4M_G118_CL)[10:-10, 10:-10])
    gap = Cl & ~U & (ids < 0)
    if not gap.any(): return P
    T = projtab.table()[_AW4M_G118_T]; names = tuple(sorted({T[k] for k in ('blue', 'red') if T.get(k)}))
    SP = _a4_shape_pixels(R, T['file'], names)
    im = img.astype(np.int16); Rr, Gg, Bb = im[..., 0], im[..., 1], im[..., 2]
    ok = (Gg >= 200) & (Rr < 120) & (Bb >= 100)          # bright cyan / solid green body colour (the halo has G 70-140)
    add = gap & (SP >= 1) & ok
    if not add.any(): return P
    add |= nd.binary_fill_holes(U | add) & ~(U | add) & (ids < 0)       # what is then enclosed by the two ghosts' labels is body too
    _, (iy, ix) = nd.distance_transform_edt(~U, return_indices=True)
    ids = ids.copy(); share = P['share'].copy(); cv = R['cv'][::2, ::2]
    ids[add] = ids[iy[add], ix[add]]; share[add] = np.maximum(cv[add] / 255.0, BL.CUT)
    P2 = dict(P); P2['ids'] = ids; P2['share'] = share
    return P2


# ===== skin 201 TntGuyProgressionSkinLevel9, attack 90 deg, tick 313: owner "should be split / merged" =====
# (the old coordinator note "FIXED 11-01 (hull per projectile)" did not hold at this tick.)
# Types 6000289 (Mike012Projectile, the attack) and 6000290 (Mike012UltiProjectile, the super): the same two types are used by the
# other TntGuyProgressionSkinLevel3..11 skins (z_1102_s201 header); same behaviour for them.
# Cause (measured on sw2_201): z_1102_s201 fuses the two projectiles of a shot into ONE hull whenever their individual hulls touch at
# all. At tick 313 the two goblets (each a yellow stem with a cyan swirl head) only touch with their heads, and at 317 they are 15 px
# apart (their slanted hulls still overlap by 4 % of the smaller hull); both were drawn as one big hull with empty background in it.
# Rule (decided from the frames): two projectiles are ONE visible attack only while they really overlap, i.e. the intersection of
# their hulls is at least _AW4M_S201_MERGE (0.5) of the smaller hull (measured: stacked goblets 0.61-0.91 at attack 0 t265-273 and
# attack 90 t305-309; 313: 0.35; 317: 0.04; separate ones 0). Merged: one hull over the group (as before). Otherwise: each
# projectile keeps its OWN hull (its parts, no sparks); where two hulls overlap, a contested pixel goes to the member whose own
# body is nearer. Everything else of z_1102_s201 is kept (own body per projectile without sparks, props in front, one label id per
# group = the lowest object id). z_1102's own wrapper is switched off here (its type set is emptied) and re-done below.
_AW4M_S201 = set(S201_TYPES)
_AW4M_S201_MERGE = 0.5
S201_TYPES.clear()                    # z_1102's wrapper is now a no-op (it reads this set at call time)
_aw4m_prev_labels3 = labels


def labels(objs, cam, R, state=None, ended=(), img=None):
    from scipy.ndimage import label as _cc, distance_transform_edt as _edt
    P = _aw4m_prev_labels3(objs, cam, R, state=state, ended=ended, img=img)
    if not any(o['t'] in _AW4M_S201 for o in objs): return P
    ids, lids, share = P['ids'], P['lids'], P['share']
    wd = R.get('wd'); cv = R['cv'][::2, ::2]
    ids_h = R['id'][::2, ::2]
    wopaque = [int(d) for d, r in R['draws'].items() if r.get('blend') == 0 and r.get('dmask') == 1]
    wworld = np.isin(ids_h, wopaque)
    mem = []; codes = {}
    for o in objs:   # pass 1 (as z_1102): each projectile's own body
        if o['t'] not in _AW4M_S201 or o.get('ended'): continue
        lid = (BL._lid(o), -1, 'body')
        if lid not in lids: continue
        k = lids.index(lid); m = ids == k
        if not m.any(): continue
        L, n = _cc(m, structure=np.ones((3, 3), bool))
        sz = np.bincount(L.ravel()); sz[0] = 0; big = int(sz.argmax())
        keep = L == big
        ck = (o['t'], o['own'])
        if ck not in codes: codes[ck] = sprite_pixels(o['t'], o['own'], R)
        Cc = codes[ck]
        for j in range(1, n + 1):
            if j == big: continue
            cj = L == j
            if sz[j] >= S201_SPECK or (Cc[cj] == 1).any(): keep |= cj
        ids[m & ~keep] = -1
        mem.append(dict(o=o, k=k, keep=keep, hull=hull_mask(keep)))
    # a stray part (a small piece of one projectile's label) that lies nearer the MAIN part of another projectile than its own main
    # part belongs to that projectile (317: a piece of the left goblet's label lay on the right goblet's tip and stretched its hull)
    if len(mem) > 1:
        mains = []
        for m_ in mem:
            L, n = _cc(m_['keep'], structure=np.ones((3, 3), bool))
            sz = np.bincount(L.ravel()); sz[0] = 0
            mains.append((L, n, int(sz.argmax())))
        dmain = [_edt(~(L == b)) for (L, n, b) in mains]
        for i, m_ in enumerate(mem):
            L, n, b = mains[i]
            for j in range(1, n + 1):
                if j == b: continue
                part = L == j
                d_own = float(dmain[i][part].min())
                dj = [(float(dmain[q][part].min()), q) for q in range(len(mem)) if q != i and mem[q]['o']['t'] == m_['o']['t']]
                if not dj: continue
                d_oth, q = min(dj)
                if d_oth < d_own:
                    m_['keep'] = m_['keep'] & ~part; mem[q]['keep'] = mem[q]['keep'] | part; ids[part] = mem[q]['k']
        for m_ in mem: m_['hull'] = hull_mask(m_['keep'])
    par = list(range(len(mem)))
    def find(i):
        while par[i] != i: par[i] = par[par[i]]; i = par[i]
        return i
    for i in range(len(mem)):
        for j in range(i + 1, len(mem)):
            if mem[i]['o']['t'] != mem[j]['o']['t']: continue
            inter = int((mem[i]['hull'] & mem[j]['hull']).sum())
            if inter >= _AW4M_S201_MERGE * min(int(mem[i]['hull'].sum()), int(mem[j]['hull'].sum())): par[find(j)] = find(i)
    groups = {}
    for i in range(len(mem)): groups.setdefault(find(i), []).append(i)
    # the final labels of the units (a unit = a merged group or a single projectile)
    units = []
    for g in groups.values():
        G = sorted(g, key=lambda i: mem[i]['o']['id'])
        K = np.zeros(ids.shape, bool)
        for i in G: K |= mem[i]['keep']
        hm = hull_mask(K)
        if wd is not None and K.any():
            hm = hm & ~(wworld & (wd < float(wd[K].min()) - DEPTH_TOL) & ~K)
        units.append(dict(car=mem[G[0]]['k'], G=G, K=K, hm=hm))
    for u in units:
        for i in u['G']: ids[mem[i]['keep']] = u['car']
    # unlabelled pixels inside a unit's hull: a contested pixel (inside two hulls) goes to the unit whose body is nearer
    if units:
        dist = [_edt(~u['K']) for u in units]
        for ui, u in enumerate(units):
            new = u['hm'] & ~u['K'] & (ids < 0)
            for uj in range(len(units)):
                if uj != ui and units[uj]['hm'].any():
                    new &= ~(units[uj]['hm'] & (dist[uj] < dist[ui]))
            ids[new] = u['car']
            share[new | u['K']] = np.maximum(cv[new | u['K']] / 255.0, BL.CUT)
    return P
