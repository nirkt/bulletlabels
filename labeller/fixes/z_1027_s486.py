# sweep 10-27 fix for skin 486 (Haiku fix, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s486/)
# sweep 10-27 fix for skin 486 MummyGiftShop (task out/sweep2/haikutasks/sweep1027/s486/, pair 8969, sw2_486 attack 0 deg,
# tick 279 'hole'; the same defect at tick 285 on the same row, found by the self-check).
# Type 6000684 (Emz002Projectile, the Emz attack of skin 486; listed in EMZ_ATTACK of z_1024_emz_attack.py).
# Cause: a purple debris bit lies on the cloud's lower edge. It is not labelled and it is enclosed by the cloud's own label.
# At tick 279 cleanup (z_1099, holes up to 40 px) already fills it. At tick 285 the enclosed gap is 47 px, above that limit, so
# a closed loop of background stays in the label.
# Rule (keyed to type 6000684 only): after all other label rules, each unlabelled hole that is fully enclosed by the attack's
# own body label(s) (the union of its clouds), at most S486_HOLE px, not on solid world geometry and not claimed by another
# label, is filled as part of the attack (BL.CUT). Open gaps (the background notch behind the cloud) are not enclosed and stay out.
# Approaches considered:
#   1. raise CLEAN_HOLE for every skin: changes the holes of every other label, rejected.
#   2. drop the debris by its colour / shape: the debris has no shape of this type, a colour test is the least robust, rejected.
#   3. fill holes enclosed by this type's own label, keyed to 6000684 (chosen): uses only the label's own geometry, does not
#      touch other types, skins or labels, and only adds pixels.
S486_TYPE = 6000684
S486_HOLE = 80      # px: enclosed unlabelled holes up to this size are filled (the debris is 47 px at tick 285)

_s486_prev = labels


def labels(objs, cam, R, state=None, ended=(), img=None):
    from scipy.ndimage import binary_fill_holes, label as cc
    import bulletlabels as BL
    P = _s486_prev(objs, cam, R, state=state, ended=ended, img=img)
    mine = [o for o in objs if o['t'] == S486_TYPE]
    if not mine: return P
    ids, lids, share = P['ids'], P['lids'], P['share']
    lidset = {BL._lid(o) for o in mine}
    ks = [k for k in np.unique(ids[ids >= 0]) if lids[k][0] in lidset and lids[k][2] == 'body']
    if not ks: return P
    U = np.isin(ids, ks)
    hole = binary_fill_holes(U) & ~U & (ids < 0)
    if not hole.any(): return P
    try: world = _zs_solid(cam, R)
    except Exception: world = np.zeros(ids.shape, bool)
    L, n = cc(hole)
    sz = np.bincount(L.ravel(), minlength=n + 1)
    ok = [j for j in range(1, n + 1) if sz[j] <= S486_HOLE and not (world & (L == j)).any()]
    if not ok: return P
    fill = np.isin(L, ok)
    ids[fill] = min(ks); share[fill] = BL.CUT
    return P
