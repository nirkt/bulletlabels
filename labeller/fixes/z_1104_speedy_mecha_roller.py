# installed 11-01 from out/sweep2/haikutasks/siteai2/speedy_mecha_roller/fix.py (2nd AI sweep: 508 blade hull, 538 streak hull, 515 flame drip legs out; single-skin types)
# AI-review sweep 2026-11-01, task speedy_mecha_roller (skins 489, 508, 515, 538, 541, 545, 616). Exec'd in idlabels' namespace
# (IDLABELS_OVERRIDE), after the installed fixes. Keyed to explicit type ids. Findings:
#
# 508 RollerHeadless, attack 90 (t305, t307, t309) and attack 0 (t265-t269): "one shot of thin green blades is drawn as several
#   small separate outlines". Type 6000907 = Roller002Projectile, the leaf blade: two objects per tick, each one blade made of
#   1-3 leaf pieces. Owner rule: a splintered object = one hull; separate blades keep separate outlines.
# 538 SpeedyLeopard, attack 0 (t267-t279): "the thin streak is outlined as a tangle of small separate pieces instead of one
#   convex hull". Type 6000718 = Max004Projectile, the blue streak; one object per streak (separate streaks keep separate hulls).
# 616 RollerSteamPunk, attack 0 (t267, t271): "gear-spray labels are ragged zigzag outlines, not one convex hull per shot".
#   Type 6000911 = Roller006Projectile, one object per gear. NOT changed (needs_user): adjacent gears overlap, see the hull set below.
# 541 MechaDudePirate, attack 0 (t265): "green swirl with torn outlines, interior lines and holes". Type 6001079 =
#   Mecha005UltiProjectile, the green shells (two overlapping shells per swirl). Each shell gets one clean outline (the
#   installed z_1027_s541 debris cut runs first, unchanged).
# The hull is built per object from the object's OWN assigned pixels (the ID pass's per-pixel ownership), so:
#   - no pixel of one object is taken by another object's hull, and no pixel of an object is lost (checked per tick);
#   - unlabelled hull pixels join only when their ID draw is one of the object's own draws (floor and brawler draws at
#     the hull edge stay out: a plain hull over 508 tick 305 reached the brawler's chin); enclosed gaps are filled.
#
# 515 MechaDudeRhino, attack 90 (t325, flagged "holes / ragged"): the flame label includes the thin blue drip legs under each
#   flame body (effects, rule 3). Type 6001073 = Mecha002UltiProjectile. Fix (POST_BODY): morphological opening with a
#   radius-4 disc removes the legs (3-6 px wide), the largest part is kept and its holes filled; the body is kept inside
#   the original body. Flames at t323-t327 keep one clean outline each.
#
# NOT changed here (see the report): 489 (t275: the lower disc is a hit burst, not the blob); 545 (crest above the crystal:
#   the crest pixels share their shape id with the vertical beam below it, not separable by shape); 541 pair (kept as two
#   shell objects; their hulls overlap and read as one outline per swirl).

# 541 (6001079, green shells) is NOT in the hull set: the hulls of the shell pairs cross at attack 90 (bow-ties); the owner
#   decides one hull per pair or per shell (needs_user).
# 616 (6000911, gears) is NOT in the hull set: neighbouring gears overlap, so each gear's hull takes in the next gear's label
#   (mixed result on the same frame); the owner decides (needs_user).
_HULL_TYPES_T = {6000907, 6000718}
_HULL_CROSS_PX = 20   # a hull that takes in more than this many px of another object's label is not built (crossing pair)
_SKIPPED_T = []
_hull_prev_labels_T = labels


def labels(objs, cam, R, state=None, ended=(), img=None):
    import bulletlabels as BL
    from scipy.ndimage import binary_fill_holes
    P = _hull_prev_labels_T(objs, cam, R, state=state, ended=ended, img=img)
    tl = {BL._lid(o) for o in objs if o['t'] in _HULL_TYPES_T}
    ids, lids = P['ids'], P['lids']
    ids_h = R['id'][::2, ::2]
    ks = [k for k, l in enumerate(lids) if l[0] in tl and l[2] == 'body']
    cv = R['cv'][::2, ::2]
    for k in ks:
        own = ids == k
        if own.sum() < 3: continue
        hull = hull_mask(own)
        cross = int((hull & (ids >= 0) & ~own).sum())
        if cross > _HULL_CROSS_PX:   # the hull would take in another object's label (crossing pair): keep as is
            _SKIPPED_T.append((k, cross))
            continue
        od = np.unique(ids_h[own]); od = od[od > 0]
        cand = hull & (ids < 0) & np.isin(ids_h, od)
        new = own | cand
        fill = binary_fill_holes(new) & hull & (ids < 0)
        take = (cand | fill) & (ids < 0)
        ids[take] = k
        P['share'][take] = np.maximum(cv[take] / 255.0, BL.CUT)
    return P


from scipy.ndimage import binary_opening as _bo6001073
_yy6001073, _xx6001073 = np.ogrid[-4:5, -4:5]
_DISK6001073 = (_xx6001073 ** 2 + _yy6001073 ** 2) <= 16


def _fix_6001073(body, ctx):
    from scipy.ndimage import label as _cc6001073, binary_fill_holes as _fh6001073
    if not body.any(): return body
    op = _bo6001073(body, structure=_DISK6001073)
    L, n = _cc6001073(op)
    if n == 0: return body
    sz = np.bincount(L.ravel())[1:]
    keep = L == (1 + int(np.argmax(sz)))
    return _fh6001073(keep) & body


POST_BODY[6001073] = _fix_6001073
