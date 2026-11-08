# AI-review batch 2026-11-01 fix (Haiku fix, reviewed by the coordinator, installed 11-01 LOCAL ONLY; source out/sweep2/haikutasks/siteai/enrager_mummy/fix.py)
# AI-review fixes 2026-11-01, enrager_mummy task (skins 536, 556, 596; 509, 582, 605 checked: no change in this file, see report).
# Exec'd in idlabels' namespace (IDLABELS_OVERRIDE); wraps the labels() chain that is installed at this point.
#
# (1) Skin 536 MummyHarajuku, Emz004Projectile (type 6000686), attack 90 deg, tick 325 (AI finding 'holes / ragged': a pocket
#     cut into the right cloud, a loop inside the outline). Cause: the stack label of the attack (one label per stack of
#     clouds, z_1024_emz_attack / z_1027_s536) leaves out the bright cyan grin of the right cloud at about x 653-672, y 453-472.
#     The ID-pass cut and the world-solid mask (z_1023_zone_solid) do not see those pixels as the cloud, and the label has a slit
#     there that is closed on every side by the label itself. The rule is one filled cloud outline with no holes, so every
#     unlabelled pixel that lies fully enclosed by the stack label is filled in (binary_fill_holes), for this type only.
#     Gaps open to the outside (between clouds, dark background) are not holes and are not changed.
#
# (2) Skin 556 EnragerThief, Enrager007Projectile (type 6000881), attack 90 deg tick 305 and attack 0 deg tick 265 (AI finding
#     'one beam drawn as two separate outlines, split where the character covers it': 'should get a hull'). The beam of one
#     shot is two objects of this type, split by the Thief's body in between. Owner rule 5 / Thin lines: ONE convex hull
#     of the whole beam. Fix: for this type, all labelled pixels of the shot's objects form one label (the carrier =
#     the lowest object id), and its convex hull is taken; the hull adds only free pixels (no other label, no solid
#     block), so the covered gap of the beam is filled (hidden parts keep their full shape).
#
# (3) Skin 596 EnragerKaiju, Enrager009Projectile (type 6000885), attack 0 deg tick 265 (AI finding 'the cyan beam at t265 is
#     split into two separate outlines instead of one hull'). Same rule and same mechanism as (2).

E536_TYPE = 6000686          # Emz004Projectile, skin 536 (see z_1027_s536 for the stack rule)
ENR_BEAMS = (6000881, 6000885)   # 556 Enrager007Projectile, 596 Enrager009Projectile: one convex hull per shot
_enr_prev_labels = labels


def _enr_fill_enclosed_536(P, objs, img):
    """(1) skin 536: fill the holes of the Emz stack label (enclosed unlabelled pixels, not open to the outside). A hole is
    filled only when it is cloud-blue (mean blue >= 150): the dark background enclosed between puffs is not filled."""
    from scipy.ndimage import binary_fill_holes, label as _cc
    import bulletlabels as BL
    ids, lids, share = P['ids'], P['lids'], P['share']
    typ = {o['id']: o['t'] for o in objs}
    for k, l in enumerate(lids):
        if l[1:] != (-1, 'body') or typ.get(l[0]) != E536_TYPE: continue
        C = ids == k
        if not C.any(): continue
        holes = binary_fill_holes(C) & ~C & (ids < 0)
        if not holes.any(): continue
        L, n = _cc(holes)
        for i in range(1, n + 1):
            m = L == i
            if img is not None and img[m, 2].mean() < 150: continue
            ids[m] = k; share[m] = BL.CUT
    return P


def _enr_beam_hull(P, objs, cam, R):
    """(2)(3): all labelled pixels of one beam type's objects = ONE label (carrier: the lowest object id), then its hull"""
    import bulletlabels as BL
    ids, lids, share = P['ids'], P['lids'], P['share']
    try: world = _zs_solid(cam, R)
    except Exception: world = np.zeros(ids.shape, bool)
    for T in ENR_BEAMS:
        mem = {(BL._lid(o), -1, 'body') for o in objs if o['t'] == T}
        ks = [k for k, l in enumerate(lids) if l in mem]
        if not ks: continue
        U = np.isin(ids, ks)
        if U.sum() < 3: continue
        # only a shot whose labelled pieces are split (two or more separate outlines) becomes one hull; a shot that is
        # already one connected outline (e.g. the two touching bulbs of 596 at tick 305, 556 at 307) is left as it is
        from scipy.ndimage import label as _cc
        if _cc(U, structure=np.ones((3, 3), bool))[1] < 2: continue
        k0 = min(ks, key=lambda k: lids[k][0])
        H = hull_mask(U)
        add = H & (ids < 0) & ~world
        for k in ks: ids[ids == k] = k0
        ids[add] = k0; share[add] = BL.CUT
    return P


def labels(objs, cam, R, state=None, ended=(), img=None):
    P = _enr_prev_labels(objs, cam, R, state=state, ended=ended, img=img)
    if any(o['t'] == E536_TYPE for o in objs): P = _enr_fill_enclosed_536(P, objs, img)
    if any(o['t'] in ENR_BEAMS for o in objs): P = _enr_beam_hull(P, objs, cam, R)
    return P
