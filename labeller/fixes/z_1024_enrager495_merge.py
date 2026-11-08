# Enrager Duel 495: the shot's two projectiles = ONE label while they overlap (user 10-26, Q10; Haiku task 35, reviewed)
# Skin 495 EnragerDuel (type Enrager002Projectile 6000876), task 35 (Haiku, 2026-10-26). Loads AFTER py/fixes/z_1024_enrager495_624.py.
# User's decision (binding, 2026-10-26): the two projectiles of one shot are ONE label while they overlap (their labels merged
# into one, outline filled), and separate labels again once they are apart. Same rule as py/fixes/z_1024_skater402.py: merge
# only while the centre distance < E495_MERGE_DR x radius. The rest of the 495 fix (hull of the band per projectile) is kept.
#
# Centre distance: the projectiles' LOGIC centres (the game's own x, y) over the type's radius (projtab, 300). The label
# centroids used by skater402 are not the projectiles' centres here: the band is split between the two projectiles by
# ownership (z_1024_enrager495_624), so each projectile's pixels are one half of the band and their centroids lie apart even
# while the projectiles overlap (t265: 2.35 on label centroids, 1.22 on logic centres; both are one band there).
# Measured on the capture sw2_495: logic distance 366 (t265, t305), 184 (t267, t307), 0 (t269-271, t309-311), so the pair is
# merged on every tick where both are labelled and alive. Separation never happens in these ticks.
#
# The merged label: the union of the two labels, closed and filled (the outline of the stack, not a hull), taken over unlabelled
# pixels inside it except opaque world geometry in front of the stack (same as skater402). The carrier is the lower object id,
# so the label keeps its id from one tick to the next. Other skins and other types are untouched.

E495_TYPE = 6000876
E495_SKIN = 'EnragerDuel'      # skin 495
E495_MERGE_DR = 1.7            # same factor as SK402_MERGE_DR (skater402)
E495_CLOSE = 3                 # px (half resolution) closed before the fill
E495_STACKS = []               # per frame: (carrier id, member ids, px of the merged label)
E495_G, E495_B, E495_R = 180, 235, 200   # the band's colour (z_1024_enrager495_624: bright cyan core and rims)
E495_GROW = 40                 # px (half resolution): the growth of the merged label stays this close to its labelled part

_e495_prev_labels = labels


def labels(objs, cam, R, state=None, ended=(), img=None):
    import owners, bulletlabels as BL
    from scipy.ndimage import binary_fill_holes as _bfh
    E495_STACKS.clear()
    P = _e495_prev_labels(objs, cam, R, state=state, ended=ended, img=img)
    on = [o for o in objs if o['t'] == E495_TYPE and not o.get('ended') and o.get('logic') is not None
          and owners.runtime(o['t'], None, None).get('skin') == E495_SKIN]
    if len(on) < 2: return P
    ids, lids, share = P['ids'], P['lids'], P['share']
    mem = []                                       # (object, label index) of every projectile of this skin on screen
    for o in on:
        lid = (BL._lid(o), -1, 'body')
        if lid in lids: mem.append((o, lids.index(lid)))
    if len(mem) < 2: return P
    R_logic = projtab.table()[E495_TYPE]['radius']
    # stacks: projectiles whose logic centres are closer than E495_MERGE_DR x radius (union-find)
    par = list(range(len(mem)))
    def find(i):
        while par[i] != i:
            par[i] = par[par[i]]; i = par[i]
        return i
    for i in range(len(mem)):
        for j in range(i + 1, len(mem)):
            a, b = mem[i][0]['logic'], mem[j][0]['logic']
            d = float(np.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1])))
            if d < E495_MERGE_DR * R_logic: par[find(j)] = find(i)
    stacks = {}
    for i in range(len(mem)): stacks.setdefault(find(i), []).append(i)
    # opaque world geometry (blend 0 with depth writes) nearer the camera than the stack hides its gap pixels
    ids_h = R['id'][::2, ::2]; draws = R['draws']; world = np.zeros(ids.shape, bool)
    for d in np.unique(ids_h[ids_h > 0]):
        dr = draws.get(int(d), {})
        if dr.get('blend') == 0 and dr.get('dmask') == 1: world |= ids_h == d
    wd = R.get('wd'); wref = R['meta'].get('wref', 0.0)
    for root, mm in sorted(stacks.items(), key=lambda kv: min(mem[i][0]['id'] for i in kv[1])):
        if len(mm) < 2: continue
        mm = sorted(mm, key=lambda i: mem[i][0]['id'])
        ks = [mem[i][1] for i in mm]; carrier = ks[0]
        U = np.zeros(ids.shape, bool)
        for k in ks: U |= ids == k
        if not U.any(): continue
        mine = np.isin(ids, ks)
        occ = world
        anc = [mem[i][0].get('anchor') for i in mm if mem[i][0].get('anchor') is not None]
        if wd is not None and anc:
            dz = min(_eye_depth(cam, a, wref)[0] for a in anc)
            occ = world & (wd < dz - DEPTH_TOL)
        # the stack's band beyond its labelled pixels: unlabelled band-coloured pixels (the band's own bright core and rims, as
        # POST_BODY of z_1024_enrager495_624) connected to the labelled part and within E495_GROW px of it. Found on the check of
        # t307: the front projectile's top diamond (338 px of its clip body) had no label, while the band itself was there
        G = np.zeros(ids.shape, bool)
        if img is not None and E495_GROW:
            from scipy.ndimage import label as _cc
            im = img.astype(np.int16)
            bright = (im[..., 1] >= E495_G) & (im[..., 2] >= E495_B) & (im[..., 0] <= E495_R) & (R['cv'][::2, ::2] >= CUT) & (ids < 0) & ~occ
            L, n = _cc(bright | U, structure=np.ones((3, 3)))
            hit = np.unique(L[U & (L > 0)]); hit = hit[hit > 0]
            if hit.size: G = np.isin(L, hit) & bright & _dilate(U, E495_GROW)
        U2 = U | G
        H = _bfh(_close(U2, E495_CLOSE)) & (_dilate(U2, E495_CLOSE) | U2)
        take = H & (((ids < 0) & ~occ) | mine)
        new = take & ~mine
        ids[take] = carrier
        share[new] = BL.CUT
        E495_STACKS.append((mem[mm[0]][0]['id'], tuple(mem[i][0]['id'] for i in mm), int(take.sum())))
    return P
