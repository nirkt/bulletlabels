# sweep 10-27 fix for skin 098 (Haiku fix, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s098/)
# s098 (VoodooDefault, sweep 2026-10-27): the Voodoo attack zones' label is one clean circle (rule 7), with no specks.
# Findings: pair 1605 (tick 289), 1606 (tick 301, debris rings; gone in this capture, the same debris specks recur at ticks
# 293-299), 1611 (tick 329: loops around splash bits outside the zone circle). The zone label is the dome disc (while the
# zone is young, ZONE_DOME) plus the zone's sprite pixels (cloud puffs, the puddle, splash bits, debris, the earthworm clip).
# Cause: the zone's effect clips (selena_def_earth_atk: cloudExplode_*, groundFrag_* debris, juju ground) put small
# blue splash rings and crescents outside the logic circle, and brown debris / a yellow fleck on the crate faces inside it.
# zone_labels() keeps every sprite pixel, so these became small detached specks. The clip sets are shared with the real
# puddle (the same atlas shapes, e.g. shape 63 is the puddle's puffs too; speck shapes also occur in the puddle at ticks
# 325-331), so a shape-based cut would remove the puddle; a rule on the label's own parts does not.
# Approaches considered:
#   1. drop atlas shapes of the debris / sparks (763-765, 565): rejected, those shapes occur inside the big puddle (1100+ px at 325-331).
#   2. drop the zone clips by name (like Barley's A5_ZONE_CLIP_SKIP): rejected, the splash clouds and the puddle share the
#      cloudExplode / juju clips (the same shapes), so no clip is only speck.
#   3. clip the label to the logic circle: rejected, the dome's raised silhouette lies a few px outside the ground ellipse
#      (cut 11 px at 289), and the zone is the sprite label after its dome phase.
#   4. CHOSEN, on the label's parts: (a) outside the logic circle (dilated DISC_GROW px) any piece under Z98_OUT_MIN px is
#      dropped (the rings, hooks and crescents of 289-299, the bumps of 325-331; the 698-px splash cluster at 289 stays);
#      (b) inside the circle, a detached part that is not the largest is dropped when under Z98_IN_MIN px (1-34 px specks at
#      293-341), or when brown / yellow (mean R-B above Z98_BROWN, under Z98_BROWN_MAX px: the crate debris and the fleck).
#      The blue detached pieces of 84-147 px inside the circle stay: the puddle's pieces at 333-341 are that size too.
# Only the Voodoo attack zones (ZONE_DOME names, every Voodoo skin with them) and only the zone labels; no projectile type.
Z98_NAMES = set(ZONE_DOME)     # VoodooAttack{Earth,Forest,Water,All} and the Voodoo002 / Voodoo003 skins of them
Z98_OUT_MIN = 200              # px: a piece of the label outside the logic circle smaller than this is a speck
Z98_IN_MIN = 40                # px: a detached piece inside the circle smaller than this is a speck
Z98_BROWN = 15                 # mean R - B of a detached piece above this: debris / a yellow fleck, not the blue puddle
Z98_BROWN_MAX = 150            # px: such a piece is dropped only when this small
Z98_DISC_GROW = 3              # px: the logic circle grown by this (rim antialiasing, the dome's edge)


def _z98_zone_cut(P, state, cam, img):
    from scipy.ndimage import label as cc, binary_dilation
    import owners
    if not state or 'items' not in state: return P
    ids = P['ids'].copy(); share = P['share'].copy(); lids = P['lids']
    H, W = ids.shape; yy, xx = np.mgrid[0:H, 0:W]
    im = img.astype(np.int16) if img is not None else None
    changed = False
    for it in state.get('items', []):
        did = int(it[1]); Zd = owners.zone(did)
        if not Zd or Zd.get('Name') not in Z98_NAMES: continue
        ks = [k for k, l in enumerate(lids) if l[0] == int(it[0]) and l[2] == 'area']
        if not ks: continue
        k = ks[0]; m = ids == k
        if not m.any(): continue
        x, y = int(it[3]), int(it[4]); rz = float(Zd['logic_row'].get('Radius') or 300)
        cx, cy = _proj(cam, x, y); dx = abs(_proj(cam, x + rz, y)[0] - cx); dy = abs(_proj(cam, x, y + rz)[1] - cy)
        if dx < 1 or dy < 1: continue
        disc = ((xx - cx) / dx) ** 2 + ((yy - cy) / dy) ** 2 <= 1
        grown = binary_dilation(disc, iterations=Z98_DISC_GROW)
        drop = np.zeros((H, W), bool)
        # (a) pieces outside the logic circle
        Lo, no = cc(m & ~grown)
        if no:
            szo = np.bincount(Lo.ravel())
            small = np.nonzero(szo < Z98_OUT_MIN)[0]
            small = small[small > 0]
            if small.size: drop |= np.isin(Lo, small)
        m1 = m & ~drop
        # (b) detached specks inside the circle (not the largest part of the label)
        L, n = cc(m1)
        if n > 1:
            sz = np.bincount(L.ravel()); big = int(np.argmax(sz[1:])) + 1
            for j in range(1, n + 1):
                if j == big: continue
                mm = L == j; s_ = int(sz[j])
                if s_ < Z98_IN_MIN:
                    drop |= mm; continue
                if s_ < Z98_BROWN_MAX and im is not None:
                    rb = float((im[..., 0] - im[..., 2])[mm].mean())
                    if rb > Z98_BROWN: drop |= mm
        if drop.any():
            ids[drop] = -1; share[drop] = 0.0; changed = True
    if changed:
        P = dict(P); P['ids'] = ids; P['share'] = share
    return P


_z98_labels_prev = labels


def labels(objs, cam, R, state=None, ended=(), img=None):
    P = _z98_labels_prev(objs, cam, R, state=state, ended=ended, img=img)
    return _z98_zone_cut(P, state, cam, img)
