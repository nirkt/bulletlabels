# hosted review 2026-11-01 fix (Haiku fix, reviewed by the coordinator, installed 11-01 LOCAL ONLY; source out/sweep2/haikutasks/site1101/g_bull/fix.py; runs after z_1101_*)
# Hosted review 2026-11-01, task g_bull (skins 173 AssaultShotgunArcade and 176 AssaultShotgunTreasure). Fix file, exec'd in
# idlabels' namespace after the installed fixes (IDLABELS_OVERRIDE).
#
# What the user saw:
#  - skin 173 (attack 0 deg rows 263-287, attack 90 deg rows 303-327): 'attack's bad'. Type 6001019 = Griff003Projectile
#    (AssaultShotgun pellets; one shot = a group of 3 pellets spawned on one tick). Each pellet's label was ragged: notches on the
#    outline, inner holes, and pieces of one pellet cut apart from its main part.
#  - skin 176 (super 45 deg rows 345-355, page 02 also 353-355): 'holes / ragged'. Type 6001025 = Griff004UltiProjectile (the
#    bottle-shaped super projectile, one per object): holes inside the outline (tick 351, 353), notches at the neck (357, 359).
#
# Cause (measured on sw2_173 ticks 267-311 and sw2_176 ticks 345-359, ASCII maps of the sprite codes and the labels): the
# particles drawn over the projectile (teal squares, sparkles) take the front-most draw of some of its pixels, so those pixels
# show no sprite of the type. The label has holes and notches exactly there, and the outline of each object is the union of
# whatever body pixels are left. The hidden parts rule (2026-10-28): a projectile partly covered by smoke or particles keeps its
# full shape.
#
# Fix, per object of the two types, on its final body label (after all the per-type rules and the claims):
#  1. the object's own outline is closed by a disc of FIX_CLOSE[t] full-res px (notches and seams between its parts close) and
#     its holes are filled: one solid piece in its own outline (not a convex hull: the user's preference for the object's own shape);
#  2. pixels added by 1 are only unlabelled pixels of no other label, and never opaque world geometry (depth-writing, blend 0:
#     walls, crates) in front of it; the closing never adds the soft halo because it only fills concavities of the outline;
#  3. specks: parts of the object's label smaller than FIX_SPECK of its largest part are dropped (the largest part is always kept,
#     so an object never loses its label).
# Keyed to the two type ids only (both skins' rows and any other skin that uses these types get the same rule). Nothing else
# is changed: other types, the other attack rows (176 attack, type 6001023) and the 173 super (type 6001021) are not touched.

FIX_CLOSE = {6001019: 4, 6001025: 4}   # type -> closing disc radius, full-res px
FIX_SPECK = 0.15                       # parts smaller than this share of the object's largest part are dropped

_fix_prev_labels = labels


def _fix_disc(r):
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r + 1


def _fix_silhouette(P, objs, R):
    import bulletlabels as BL
    from scipy.ndimage import binary_closing, binary_fill_holes, label as cc
    ids = P['ids'].copy(); share = P['share'].copy(); lids = P['lids']
    rid = R['id'][::2, ::2]; cv = R['cv'][::2, ::2]; draws = R['draws']   # R buffers are 2x the label frame
    world = np.zeros(ids.shape, bool)   # opaque world geometry (blend 0, depth writes): never added
    for d in np.unique(rid):
        dd = draws.get(int(d)) if d > 0 else None
        if dd is not None and not dd.get('blend') and dd.get('dmask'): world |= rid == d
    for o in objs:
        r = FIX_CLOSE.get(o['t'])
        if r is None: continue
        lid = (BL._lid(o), -1, 'body')
        if lid not in lids: continue
        k = lids.index(lid); m = ids == k
        if not m.any(): continue
        ys, xs = np.nonzero(m); p = r + 3
        y0, y1 = max(0, ys.min() - p), min(ids.shape[0], ys.max() + p + 1)
        x0, x1 = max(0, xs.min() - p), min(ids.shape[1], xs.max() + p + 1)
        sl = (slice(y0, y1), slice(x0, x1)); sm = m[sl]
        cl = binary_closing(np.pad(sm, r + 1), _fix_disc(r))[r + 1:-(r + 1), r + 1:-(r + 1)]
        filled = binary_fill_holes(cl | sm)
        add = filled & ~sm & (ids[sl] < 0) & ~world[sl]
        if add.any():
            ids[sl][add] = k; share[sl][add] = np.maximum(cv[sl][add] / 255.0, BL.CUT)
        m2 = ids[sl] == k
        hole2 = binary_fill_holes(m2) & ~m2 & (ids[sl] < 0) & ~world[sl]   # enclosed leftovers after the claims
        if hole2.any(): ids[sl][hole2] = k; share[sl][hole2] = np.maximum(cv[sl][hole2] / 255.0, BL.CUT); m2 = ids[sl] == k
        L, n = cc(m2)
        if n > 1:
            sz = np.bincount(L.ravel(), minlength=n + 1); sz[0] = 0; main = int(np.argmax(sz))
            drop = [j for j in range(1, n + 1) if j != main and sz[j] < FIX_SPECK * sz[main]]
            if drop:
                d = np.isin(L, drop) & m2
                ids[sl][d] = -1; share[sl][d] = 0
    P['ids'] = ids; P['share'] = share
    return P


def labels(objs, cam, R, state=None, ended=(), img=None):
    P = _fix_prev_labels(objs, cam, R, state=state, ended=ended, img=img)
    if not any(o['t'] in FIX_CLOSE for o in objs): return P
    return _fix_silhouette(P, objs, R)
