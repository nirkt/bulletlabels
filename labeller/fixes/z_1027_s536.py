# sweep 10-27 fix for skin 536 (Sonnet revision, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s536/)
# sweep 10-27 fix (2nd attempt) for skin 536 MummyHarajuku (Haiku task s536, pair 9624, attack 90 deg, tick 305, 'merge_split').
# Emz004Projectile (type 6000686, skin 536 only): the clouds of one attack are ONE label (z_1024_emz_attack), but the label is the
# union of the clouds' opaque body pixels closed by only EMZ_CLOSE = 2 px. The clouds' soft dark-blue edges (low coverage, not
# in the ID pass body) leave channels of 5-20 px between the puffs, so the label showed separate arcs / slivers with unlabelled
# gaps between them instead of one solid cloud (ticks 305-325 of the 90 deg row; the 0 deg row has the same cloud).
# Fix: for the attack's label of type 6000686 only, close the label with a disc of EMZ536_CLOSE px, fill holes, and give the label
# the added pixels, but only free pixels (no other label, no solid map block, as in z_1024); the added part is then opened with a
# small disc so no thin spurs / jagged notches stick out, and only the part of it connected to the label is kept.
# The label's own pixels are never removed. Other types, other skins: unchanged (z_1024 result).
EMZ536_TYPE = 6000686
EMZ536_CLOSE = 14       # px: closing disc radius (gaps between the puffs are up to ~25 px wide)
EMZ536_TRIM = 3         # px: opening radius that trims spurs and zig-zag notches off the final label
EMZ536_OPEN = 3         # px: opening radius that trims spurs from the added pixels
_emz536_prev_labels = labels


def _emz536_disc(r):
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    return (yy * yy + xx * xx) <= r * r + r


def labels(objs, cam, R, state=None, ended=(), img=None):
    import bulletlabels as BL
    from scipy.ndimage import binary_fill_holes, binary_closing, binary_opening, label as cc
    P = _emz536_prev_labels(objs, cam, R, state=state, ended=ended, img=img)
    if not any(o['t'] == EMZ536_TYPE for o in objs): return P
    ids, lids, share = P['ids'], P['lids'], P['share']
    typ = {o['id']: o['t'] for o in objs}
    try: world = _zs_solid(cam, R)
    except Exception: world = np.zeros(ids.shape, bool)
    r = EMZ536_CLOSE
    for carrier_id, members, _, _, _ in list(EMZ_STACKS):
        if typ.get(carrier_id) != EMZ536_TYPE: continue
        kc = None
        for k, l in enumerate(lids):
            if l[1:] == (-1, 'body') and l[0] == _emz536_lid_of(objs, carrier_id, BL): kc = k; break
        if kc is None: continue
        C = ids == kc
        if not C.any(): continue
        H = binary_fill_holes(binary_closing(np.pad(C, r), structure=_emz536_disc(r))[r:-r, r:-r])
        free = (ids < 0) & ~world
        add = H & free & ~C
        if EMZ536_OPEN > 0 and add.any():
            # trim thin spurs of the added part (it is judged together with the label so the seam fills stay)
            U = C | add
            Uo = binary_opening(np.pad(U, EMZ536_OPEN), structure=_emz536_disc(EMZ536_OPEN))[EMZ536_OPEN:-EMZ536_OPEN, EMZ536_OPEN:-EMZ536_OPEN]
            add = add & Uo
        U = C | add
        U = binary_fill_holes(U) & (U | free)
        L, n = cc(U, structure=np.ones((3, 3), bool))
        if n > 1:
            keep = np.isin(L, np.unique(L[C & (L > 0)]))
            U = U & keep
        # thin spurs / zig-zag notches of the whole label (its own ragged edge at the walls included): opened with a small disc, the
        # label is never emptied and loses only parts narrower than the disc
        if EMZ536_TRIM > 0:
            t = EMZ536_TRIM
            Uo = binary_opening(np.pad(U, t), structure=_emz536_disc(t))[t:-t, t:-t]
            if Uo.sum() > 0.9 * U.sum(): U = Uo
        new = U & ~C
        gone = C & ~U
        ids[gone] = -1
        ids[new] = kc; share[new] = BL.CUT
    return P


def _emz536_lid_of(objs, oid, BL):
    for o in objs:
        if o['id'] == oid: return BL._lid(o)
