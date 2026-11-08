# sweep 10-27 fix for skin 523 (Haiku fix, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s523/)
# s523 (skin 523 MummyDj, sweep 2026-10-27, pair 9470 'hole', attack 90 deg, tick 325; the same defect at ticks 319 and 323).
# Type 6000685 = Emz003Projectile (sc/effects_brawler.sc, emz_003_atk_projectile): an Emz attack is ONE label per attack
# (z_1024_emz_attack). That merge fills holes only where they lie within EMZ_CLOSE px of the clouds' union
# (`H = fill(close(U)) & (dilate(U, EMZ_CLOSE) | U)`), so an enclosed gap deeper than a few px stays open. Here three
# enclosed gaps between the star tips stay unlabelled: 41 px (tick 319), 56 px (tick 323), 122 px (tick 325). Their raw
# colour is the attack's saturated blue (mean about 50, 90, 255), not the navy background, and no solid map block is there.
# Fix (this type only, the user's rule: the label is one solid piece, no holes): after the attack's label is final, every
# gap enclosed by that label is filled, unless it is a solid map block or holds another label's pixels. The label keeps its
# carrier id and the pixels are counted as the attack's own (share = CUT), as the Emz merge does.
# Scope: only 6000685 (EMZ_HOLE_FILL). The same deep-gap rule exists in the Emz merge for every EMZ_ATTACK type; those types
# were not reported with this defect and are not changed here.
EMZ_HOLE_FILL = {6000685}
_s523_prev_labels = labels


def labels(objs, cam, R, state=None, ended=(), img=None):
    from scipy.ndimage import binary_fill_holes
    import bulletlabels as BL
    P = _s523_prev_labels(objs, cam, R, state=state, ended=ended, img=img)
    att = [o for o in objs if o['t'] in EMZ_HOLE_FILL]
    if not att: return P
    ids, lids, share = P['ids'], P['lids'], P['share']
    ks = []
    for o in att:
        lid = (BL._lid(o), -1, 'body')
        if lid in lids: ks.append(lids.index(lid))
    if not ks: return P
    M = np.isin(ids, ks) & (ids >= 0)
    if not M.any(): return P
    new = binary_fill_holes(M) & ~M & (ids < 0)
    if not new.any(): return P
    try: world = _zs_solid(cam, R)
    except Exception: world = np.zeros(ids.shape, bool)
    new &= ~world
    if not new.any(): return P
    vals, cnt = np.unique(ids[M], return_counts=True)
    carrier = int(vals[cnt.argmax()])
    ids[new] = carrier; share[new] = BL.CUT
    return P
