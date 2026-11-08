# sweep 10-27 fix for skin 310 (Haiku fix, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s310/)
# Task s310 (sweep 2026-10-27, skin 310 CrowVulture = brawler Crow): pair 5798, sw2_310 attack 90 deg, tick 305, 'hole'.
# The 90-deg blade attack is Crow005Projectile (type 6000321): three blue blades with a green flame, one object per blade.
# At tick 305 two small unlabelled pockets sit between the blades' tops (1 px and 5 px, around x 602, y 290..293), so the
# page outline shows two small enclosed diamonds. Tick 307 has the same kind of pocket (5 px at x 606, y 320..323, and
# 17 px at x 590..592, y 331..339). The clean-up (z_1099_cleanup.py) fills holes only inside ONE label, and these pockets
# are enclosed by two or three different blade objects, so the clean-up does not see them.
# Fix: after the clean-up, an unlabelled pocket of at most CROW310_POCKET px (the clean-up's own limit) that is enclosed
# only by body labels of the Crow005Projectile type is given to the neighbouring body label with the most shared border.
# A pocket touching any other object's label is left alone, and so are larger pockets (tick 317's 52 px pocket in the
# blade head is above the limit and is not part of this finding). Only pixels inside the attack are filled; no label grows
# outward. Type 6000321 is also the 0-deg attack of this skin: the same kind of 1-3 px pockets at ticks 267 and 269 of the
# 0-deg row are filled too (checked on the 0-deg row, the outline gets solid and nothing else changes). Other Crow skins
# (027, 117, 216, 222, 237, 285, 375, 425) were not checked per skin; if they use this type, the same rule applies there.
CROW310_TYPES = (6000321,)   # Crow005Projectile (sc/effects_brawler.sc)
CROW310_POCKET = 40          # px: largest pocket filled (the clean-up's CLEAN_HOLE)
_crow310_prev = labels


def labels(objs, cam, R, state=None, ended=(), img=None):
    from scipy.ndimage import binary_fill_holes, binary_dilation, label as cc
    import bulletlabels as BL
    P = _crow310_prev(objs, cam, R, state=state, ended=ended, img=img)
    ids, lids, share = P['ids'], P['lids'], P['share']
    ltype = {}
    for o in objs: ltype.setdefault(BL._lid(o), o['t'])
    ks = [int(k) for k in np.unique(ids[ids >= 0]) if ltype.get(lids[k][0]) in CROW310_TYPES and lids[k][2] == 'body']
    if not ks: return P
    M = np.isin(ids, ks)
    H = binary_fill_holes(M) & ~M          # pockets enclosed by the attack's own body labels
    if not H.any(): return P
    L, n = cc(H)
    for j in range(1, n + 1):
        hj = L == j
        if hj.sum() > CROW310_POCKET or (ids[hj] >= 0).any(): continue   # too big, or other labels inside
        ring = binary_dilation(hj) & ~hj
        nb = ids[ring]
        if nb.size == 0 or (nb < 0).any() or not np.isin(nb, ks).all(): continue   # touches an unlabelled gap or another object
        cnt = np.bincount(nb, minlength=int(ids.max()) + 1)
        k = int(cnt.argmax())
        if k not in ks: continue
        ids[hj] = k; share[hj] = BL.CUT
    return P
