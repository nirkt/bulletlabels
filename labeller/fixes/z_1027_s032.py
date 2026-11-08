# sweep 10-27 fix for skin 032 (Sonnet revision fix2.py; the card and eye parts, which the review accepted. The coordinator REMOVED its
# ZONE_DISC.add('BlackHoleUltiExplosion') line: shared by other Tara skins, unverified, and it labels the eye-ball end frames; question for the user)
# Skin 032 BlackholeDefault (BlackHole / Tara, sweep 2026-10-27), task s032, SECOND version (the first, fix.py, was rejected:
# hooks at the eye's ends, closed-lid ticks unchanged, a notch on a card, and a global ZONE_DISC edit).
# Types: 6000411 BlackHoleProjectile (Tara's default attack: playing cards, clip playing_card), 6000412 BlackHoleUltiProjectile
# (Tara's default super, the blue eye, clip tara_def_ulti_projectile). Both type ids belong to this skin only (the
# other BlackHole skins have their own projectile types, Tara002Projectile ...).
#
# 1. pair 623 (attack 90 deg, tick 305): the card at the cluster's upper left (object 2000014) had a thin spur at its top-left
#    corner (the white star flare touching the corner, 2-3 rows above the card edge) and, in the same corner, a notch (the flare
#    is additive, so the ID pass has no pixels under it). Cards are convex tilted rectangles. The pre-label body of a card (POST_BODY)
#    still holds its neighbours' pixels (the claim between cards comes later), so the clean-up is done on the final label map by a
#    labels() wrapper, per card label: opening with a disk of radius 2 (spurs narrower than 5 px go; radius 4 cut too much of the thin
#    visible strips of cards half hidden behind another, tick 265 / 305), a 3x3 opening for the tip,
#    small parts dropped, closing with a disk of radius 5 per part (fills notches narrower than 10 px, never onto another label's
#    pixels) and holes filled. Never lose the projectile: if the opening keeps less than half of the card, it stays as it is.
#
# 2. pair 634 (super 45 deg, ticks 345-359): the eye's label took in the soft blue glow (shape 24919, the halo sprite) and was bumpy.
#    The eye proper = shape 140 (141 in the pixel map: the lid / iris pieces, and the closed lid's slit) plus the shape-829 pixels
#    (830) touching it (the dark rim), both from the ID pass. The label is the convex hull of these pixels grown by 2 px: a clean
#    lens in open frames, a flat lozenge on the closed lid (ticks 345-351, same rule, so no jump between closed and open ticks).
#    The hull replaces the first version's dilate + fill, which gave hooks at the lens ends (stray rim pixels); here the rim pixels
#    count only when they touch the lid pieces. The glow is out. If a frame has fewer than 6 lid pixels the body stays as before.
#
# 3. pairs 635 / 636 (ticks 367, 377): the yellow label there is the grey smoke puff / a speck of the zone's own clip (the zone's
#    drawn ground disc is additive and invisible to the ID pass). Rule 7 / 10: the lingering zone is its full logic circle, cut only
#    by solid map blocks (z_1023_zone_solid), the same mechanism as Penny's burn and Barley's puddles: ZONE_DISC. The name is the
#    game's area-effect row 'BlackHoleUltiExplosion' (logic zone 17000157, radius 450, the super's DelayedDamage zone). That row is
#    one shared row: the BlackHole skins with no own explosion row use it, with the same effect (blackhole_ulti_damage) and the same
#    puff: pages 246 (Tara002) and 267 (Tara003) show the identical yellow puff at tick 367 and speck at 377, so they get the same
#    fix and no frame there gets worse (a puff / speck becomes the circle). Skins with an own row (Tara004UltiExplosion, the
#    overcharged BlackHoleOverchargedUltiExplosion) are not touched. The big dark vortex is BlackHoleUltiSuck (a pull, class
#    super), left unlabelled: question to the user.

_s032_prev_labels = labels


def labels(objs, cam, R, state=None, ended=(), img=None):
    """6000411 (BlackHole cards), on the final label map (after the claim between neighbouring cards and the clean-up): per card
    label an opening with a disk of radius 2 (spurs and slivers narrower than 5 px go; radius 4 cut too much of the thin visible parts of
    cards half hidden behind another), a 3x3 opening for the 1-2 px tip left above a corner
    (kept only if it keeps 90 % of the pixels), parts under 10 % of the largest dropped, then each part closed with a disk of
    radius 5 and holes filled, only onto pixels no label owns. A card whose opening keeps less than half of it stays as it is."""
    from scipy.ndimage import binary_opening, binary_closing, binary_fill_holes, label as cc
    import bulletlabels as BL
    P = _s032_prev_labels(objs, cam, R, state=state, ended=ended, img=img)
    ids, lids, share = P['ids'], P['lids'], P['share']
    ltype = {}
    for o in objs: ltype.setdefault(BL._lid(o), o['t'])
    def disk(r):
        y, x = np.mgrid[-r:r + 1, -r:r + 1]; return (x * x + y * y) <= r * r + 0.5
    for k in np.unique(ids[ids >= 0]):
        if lids[k][2] != 'body' or ltype.get(lids[k][0]) != 6000411: continue
        m = ids == k
        ys, xs = np.nonzero(m); y0, y1, x0, x1 = max(0, ys.min() - 8), ys.max() + 9, max(0, xs.min() - 8), xs.max() + 9
        sm = m[y0:y1, x0:x1]; sid = ids[y0:y1, x0:x1]; ssh = share[y0:y1, x0:x1]
        op = binary_opening(sm, structure=disk(2))
        if op.sum() < 0.5 * sm.sum(): continue                 # never lose the projectile
        op2 = binary_opening(op, structure=np.ones((3, 3), bool))
        if op2.sum() >= 0.9 * op.sum(): op = op2
        L, n = cc(op)
        if n > 1:
            sz = np.bincount(L.ravel())[1:]
            op = np.isin(L, 1 + np.nonzero(sz >= 0.1 * sz.max())[0])
        L, n = cc(op); new = np.zeros_like(op)
        for j in range(1, n + 1):
            pj = np.pad(L == j, 7)
            new |= binary_closing(pj, structure=disk(5))[7:-7, 7:-7] | (L == j)
        new = binary_fill_holes(new) | op
        free = (sid < 0) | (sid == k)
        new &= free
        sid[sm & ~new] = -1; ssh[sm & ~new] = 0
        add = new & ~sm; sid[add] = k; ssh[add] = BL.CUT
    return P


def _s032_eye(body, ctx):
    """6000412 (BlackHole super eye): hull of the lid pieces (shape 140) and the rim pixels touching them (shape 829), grown 2 px"""
    from scipy.ndimage import binary_dilation, binary_opening, label as cc
    if not body.any(): return body
    T = projtab.table()[6000412]
    names = [n for n in (T.get('blue'), T.get('red')) if n] + [n for f, n in _trail_clips(6000412)]
    SP = _a4_shape_pixels(ctx['R'], T['file'], tuple(sorted(set(names))))   # shape id + 1 per pixel, 0: none
    A = body & (SP == 141)
    if A.sum() < 6: return body                           # no lid pieces in the body: as before
    core = A | (body & (SP == 830) & binary_dilation(A, np.ones((3, 3), bool)))
    L, n = cc(binary_dilation(core, np.ones((3, 3), bool)))
    if n > 1:
        sz = np.bincount(L.ravel())[1:]
        core = core & np.isin(L, 1 + np.nonzero(sz >= 0.1 * sz.max())[0])
    m = _dilate(hull_mask(core), 2)
    m2 = binary_opening(m, structure=np.ones((3, 3), bool))   # a 1-px hull vertex
    if m2.sum() >= 0.95 * m.sum(): m = m2
    return m & ctx['region']


POST_BODY[6000412] = _s032_eye

# the BlackHole super's lingering damage zone (row BlackHoleUltiExplosion): its full logic circle (rule 7), not the puff / smoke sprites
# ZONE_DISC.add('BlackHoleUltiExplosion')   # removed by the coordinator (see header)
