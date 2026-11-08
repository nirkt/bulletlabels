# Leon 557 / 550 attack: body + bright rims, clean solid (user Leon rule 10-20; Haiku task 13, reviewed)
# task 13 (2026-10-23): Leon's NinjaMecha 557 (Leon008Projectile, type 6000580: the cyan shuriken, clip
# leon_008_atk_projectile) and NinjaBrawloween 550 (Leon007Projectile, type 6000579: the blue orbs, clip leon_007_atk_projectile).
# The user's decision: each projectile labelled as its visible body including the bright additive rims, like Leon's default
# skin (BRIGHT_RIM 6000572); soft halos, trails and hit / end animations out; one clean solid shape per projectile.
# Causes, measured on the kept captures runs/gc/sw2_557 and sw2_550:
#  - the body shapes (46847-46851 for 550; 666, 42652/42653 for 557) cover the solid parts only; the bright white-cyan
#    rims at the blade tips / orb edges are additive and missing from the body (BRIGHT_RIM fills them, at most RIM_R px out);
#  - the bright cores of the orbs and the centre of the shuriken are holes in the body (closing + hole fill makes one shape);
#  - small detached fragments (sprite pieces) are dropped before the closing (ONE_LABEL with a closing radius, not a hull:
#    a hull would cover the notches between the blades).
# Shuriken / orbs that have ended (the hit flash at about tick 273 onwards) keep no label: their objects have no body pixels.
LEON_SKIN_TYPES = (6000580, 6000579)   # 557 NinjaMecha shuriken (6000580), 550 NinjaBrawloween orbs (6000579) only
BRIGHT_RIM.update({t: 120 for t in LEON_SKIN_TYPES})
ONE_LABEL.update({t: 3 for t in LEON_SKIN_TYPES})   # closing radius 3 px: gaps between the parts closed, holes filled


def _t13_leon_solid(body, ctx):
    """drop specks and detached pieces smaller than a fifth of the main part (before the closing above)"""
    from scipy.ndimage import label as cc
    if not body.any(): return body
    L, n = cc(body)
    if n <= 1: return body
    sz = np.bincount(L.ravel())[1:]
    return np.isin(L, 1 + np.nonzero(sz >= 0.2 * sz.max())[0])


for _t in LEON_SKIN_TYPES: POST_BODY[_t] = _t13_leon_solid


# The user (2026-10-24): these projectiles get their outline only, FULLY labelled inside: after labels(), each one's label is
# its largest piece with every hole filled (crumbs split off by neighbours' claims dropped; pixels owned by another label stay)
_leon_prev_labels = labels


def labels(objs, cam, R, state=None, ended=(), img=None):
    import bulletlabels as BL
    from scipy.ndimage import binary_fill_holes, label as cc
    P = _leon_prev_labels(objs, cam, R, state=state, ended=ended, img=img)
    lids, ids = P['lids'], P['ids']
    for o in objs:
        if o['t'] not in LEON_SKIN_TYPES: continue
        key = (BL._lid(o), -1, 'body')
        if key not in lids: continue
        k = lids.index(key); m = ids == k
        if not m.any(): continue
        L, n = cc(m)
        if n > 1:
            sz = np.bincount(L.ravel()); sz[0] = 0; keep = L == sz.argmax()
            ids[m & ~keep] = -1; P['share'][m & ~keep] = 0; m = keep
        h = binary_fill_holes(m) & ~m & (ids < 0)
        ids[h] = k; P['share'][h] = np.maximum(P['share'][h], BL.CUT)
    return P
