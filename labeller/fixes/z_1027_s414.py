# sweep 10-27 fix for skin 414 (Sonnet revision, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s414/)
# sweep 10-27 fix (2nd attempt) for skin 414 SandstormGreekGod (Sandy 007). Task: out/sweep2/haikutasks/sweep1027/s414/ (task.md).
# Capture W/runs/gc/sw2_414. Three confirmed findings:
#  pair 7798 (attack 90 deg, tick 313, ragged): the right cloud's label was a ragged smear. The cause: the label of a Sandy
#     cloud (type 6000678, ZONE) is made of pixels the ID pass shows at >= 50 % coverage plus the predicted body under the
#     cloud's glow sprite. At tick 313 the clouds are already translucent (rim coverage 32..160), so the rim is mostly under
#     the cut, and the predicted body drags in the glow disc around it (zigzag edge, bites, a detached piece).
#  pair 7799 (tick 317, extra_label): the clouds have faded to dim purple smears and still carried labels: a fade animation
#     after the logic end (rule 2), not labelled.
#  pair 7807 (super 45 deg, tick 411, spur_or_speck): the super is the lingering zone Sandy_007_ulti (area effect data id
#     17000223, logic Radius 2000); its label (zone sprites) has detached loops outside the dome.
# Fix (attack, type 6000678 only = skin 414's Sandy007Projectile): the label of each cloud is the cloud's OWN outline filled
# solid: its clip-body sprite is the cyan rim ring (sprite code 1: the rim; the soft purple glow disc is code 2 and is NOT
# body), so the rim pixels (coverage >= S414_RIM_CV, any translucency) of the cloud are taken, closed and filled (idlabels.zone:
# gaps closed, holes filled, small detached parts dropped). The rim pixels are split between the clouds of the shot by the
# nearest predicted body (Voronoi), so two clouds that touch stay two labels. A cloud whose rim is faded (no rim pixel at least
# half opaque: coverage 90th percentile < S414_FADE_P90) is a fade animation and gets no label (tick 317 and also 315 where only
# dim purple remains; at 313 the p90 is 160..250). Brightness, not a tick or an age, decides.
# Super (zone Sandy_007_ulti, did 17000223): the same rule as z_1027_s042.py (skin 042's Sandy super, reviewed): the label is
# the exact projected logic circle minus the solid map blocks and minus other labels, pinholes filled, specks dropped.
# Touched: type 6000678 (attack) and the zone Sandy_007_ulti (did 17000223).
import os as _os
S414_ATK = 6000678
S414_RIM_CV = int(_os.environ.get("S414_RIMCV", 1))         # coverage from which a rim pixel counts (the rim is translucent from about tick 313 on)
S414_FADE_P90 = 128      # a cloud whose rim never reaches this coverage (90th percentile) is fading: unlabelled
S414_STAR_MAX = 150      # px (half resolution): largest star sprite blob taken with the cloud
S414_TRANSL = 0.25       # share of rim pixels at >= 50 % coverage under which the clouds count as translucent
S414_MIN_RIM = 40        # px (half resolution): fewer rim pixels = nothing of a cloud drawn
_labels_s414_prev = labels


import os as _os
S414_CLOSE = int(_os.environ.get('S414_CLOSE', 4)); S414_OPEN = int(_os.environ.get('S414_OPEN', 3)); S414_STARS = int(_os.environ.get('S414_STARS', 1)); S414_PK = int(_os.environ.get('S414_PK', 9))


def _s414_solid(m):
    from scipy.ndimage import binary_closing, binary_fill_holes, binary_opening, label as cc
    c = S414_CLOSE
    z = binary_closing(np.pad(m, c), iterations=c)[c:-c, c:-c] | m
    z = binary_fill_holes(z)
    if S414_OPEN: z = binary_opening(np.pad(z, S414_OPEN), iterations=S414_OPEN)[S414_OPEN:-S414_OPEN, S414_OPEN:-S414_OPEN]
    L, n = cc(z)
    if n > 1:
        sz = np.bincount(L.ravel())[1:]; z = np.isin(L, 1 + np.nonzero(sz >= ZONE_MIN * sz.max())[0])
    if S414_PK:
        # a part of the cloud hidden behind the neighbouring cloud (its rim is not drawn there): the wide pockets of each
        # part's convex hull (narrower than 2 * S414_PK px = the bays between the puffs stay open) are body
        from scipy.ndimage import binary_dilation
        L, n = cc(z); out = z.copy()
        for j in range(1, n + 1):
            part = L == j; hm = hull_mask(part); miss = hm & ~z
            if not miss.any(): continue
            op = binary_opening(np.pad(miss, S414_PK), iterations=S414_PK)[S414_PK:-S414_PK, S414_PK:-S414_PK]
            if op.any(): out |= binary_dilation(op, iterations=S414_PK) & miss
        z = binary_fill_holes(out)
    return z


def labels(objs, cam, R, state=None, ended=(), img=None):
    import bulletlabels as BL
    P = _labels_s414_prev(objs, cam, R, state=state, ended=ended, img=img)
    so = [o for o in objs if o['t'] == S414_ATK]
    ids, lids = P.get('ids'), P.get('lids')
    if not so or ids is None: return P
    # only once the shot's logic has ended (its clouds are in the end phase: fading out); before that the earlier rules stay
    if not any(int(e.get('t', -1)) == S414_ATK for e in ended): return P
    share = P['share']; H, W = ids.shape
    cv = R['cv'][::2, ::2]
    C = np.zeros((H, W), np.int8)
    for o in so:
        c = sprite_pixels(o['t'], o['own'], R); C = np.where((C == 0) & (c > 0), c, C)
    rim = (C == 1) & (cv >= S414_RIM_CV)
    v = cv[rim]
    mylid = {BL._lid(o) for o in so}
    old = [k for k, l in enumerate(lids) if l[2] == 'body' and l[0] in mylid]
    keep_old = np.isin(ids, old) if old else np.zeros((H, W), bool)
    faded = len(v) < S414_MIN_RIM or np.percentile(v, 90) < S414_FADE_P90
    if not faded and (v >= CUT).mean() >= S414_TRANSL: return P     # still opaque: the cut at 50 % coverage sees the body
    if faded:
        # fade animation after the logic end (rule 2): no rim pixel half opaque any more
        if keep_old.any():
            ids[keep_old] = -1; share[keep_old] = 0
        return P
    # translucent clouds (rim coverage mostly under the 50 % cut): the cloud's own outline, filled solid, from its rim
    star = rim
    if S414_STARS:
        # the cloud's own stars (code 3, small sprites touching the rim) close the rim where they sit in front of it; the big
        # launch flash (code 3, 200+ px) and floating sparkles are not part of the cloud
        from scipy.ndimage import label as _cc, binary_dilation as _bd
        c3 = (C == 3) & (cv >= CUT); L3, n3 = _cc(c3, np.ones((3, 3)))
        if n3:
            sz3 = np.bincount(L3.ravel()); near_rim = _bd(rim, iterations=2)
            ok = [j for j in range(1, n3 + 1) if sz3[j] <= S414_STAR_MAX and near_rim[L3 == j].any()]
            star = rim | np.isin(L3, ok)
    z = _s414_solid(star)
    if not z.any(): return P
    ids[keep_old] = -1; share[keep_old] = 0
    lk = (BL._lid(so[0]), -1, 'body')
    if lk not in lids: lids = lids + [lk]
    k = lids.index(lk)
    put = z & (ids < 0)
    ids[put] = k; share[put] = np.maximum(cv[put] / 255.0, BL.CUT)
    P['lids'] = lids; P['ids'] = ids; P['share'] = share
    if state is not None: P['meta'] = meta(objs, state, P)
    return P


# ---- the super: Sandy_007_ulti (did 17000223), as z_1027_s042.py
S414_ZONE_DID = 17000223
S414_ZONE_NAME = 'Sandy_007_ulti'
S414_ZONE_SCALE = 1.0
S414_RING = 1440
S414_SPECK = 0.01


def _s414_circle(cam, x, y, rz, H, W):
    th = np.linspace(0.0, 2.0 * np.pi, S414_RING, endpoint=False)
    pts = [tuple(_proj(cam, x + rz * np.cos(t), y + rz * np.sin(t))) for t in th]
    im = Image.new('L', (W, H), 0); ImageDraw.Draw(im).polygon(pts, fill=1)
    return np.asarray(im, bool)


def _s414_specks(part):
    from scipy.ndimage import label as cc
    L, n = cc(part)
    if n <= 1: return part
    sz = np.bincount(L.ravel())[1:]
    return np.isin(L, 1 + np.nonzero(sz >= S414_SPECK * sz.max())[0])


_s414_zone_labels_prev = zone_labels


def zone_labels(state, cam, R, taken, own_team=None):
    import owners
    out = _s414_zone_labels_prev(state, cam, R, taken, own_team)
    if R.get('wd') is None: return out
    H, W = R['id'][::2, ::2].shape
    solid = None
    for it in state.get('items', []):
        iid, did = int(it[0]), int(it[1])
        if did != S414_ZONE_DID: continue
        Z = owners.zone(did)
        if not Z or Z.get('Name') != S414_ZONE_NAME or owners.area(did)['kind'] != 'zone': continue
        if solid is None: solid = _zs_solid(cam, R)
        x, y = int(it[3]), int(it[4]); rz = float(Z['logic_row'].get('Radius') or 300) * S414_ZONE_SCALE
        disc = _s414_circle(cam, x, y, rz, H, W)
        part = disc & ~solid & ~taken
        if part.any():
            part = _zs_fill_small(part, True, solid, taken)
            part = _s414_specks(part)
        if part.any(): out[iid] = (part, did, int(it[2]))
        else: out.pop(iid, None)
    return out
