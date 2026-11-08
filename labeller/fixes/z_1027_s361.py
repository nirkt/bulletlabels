# sweep 10-27 fix for skin 361 (Haiku fix, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s361/)
# sweep 10-27 fix for skin 361 BlackHoleStreetNinja_remake (Tara, super 45 deg), task s361 (Haiku, sweep 2026-10-27).
# Confirmed defect (pair 6801, sw2_361 super 45 deg, tick 363, 'hole'): the lingering zone BlackHoleUltiExplosion (area-effect
# row 17000157, logic radius 450; the zone of the super Tara006UltiSuck) had small closed holes where the white shards of the
# super projectile lie inside it. The projectile (2000015, type 6000422) is 'finished' from tick 361, so its shards are its end
# animation: not labelled (rule 2 / 4), and they cut the zone. The same kind of hole (cyan cards and sparks lying in the zone)
# recurs at ticks 365-371. The user's rule 7: the zone is cut ONLY by solid map objects (walls, crates, barrels); characters,
# bushes, UI and effects never cut it.
#
# Fix: holes enclosed by the zone label are filled when they are not solid map blocks (idlabels' _zs_solid: the capture's own
# tile grid, z_1023_zone_solid) and not another label's pixels (taken). A hole bigger than S361_HOLE_MAX px (whole hole, before
# the solid / taken cut) stays open. Only enclosed holes change: the outline and the notches at the zone's edge (walls and
# crates standing on the zone edge) stay as they are.
#
# Shared row: area-effect row 17000157 (BlackHoleUltiExplosion) is also used by the BlackHole skins that have no own explosion
# row (see py/fixes/z_1027_s032.py), so the same hole fill applies to them; it is the same effect and the same rule. Nothing else
# (the circle, ZONE_DISC, projectiles, the other zones) is changed.

S361_ZONE_DID = 17000157                  # area-effect row of BlackHoleUltiExplosion (see header)
S361_ZONE_NAME = 'BlackHoleUltiExplosion'
S361_HOLE_MAX = 3000                      # px (frame 1200 x 540): enclosed holes larger than this stay open

_s361_zone_labels_prev = zone_labels


def zone_labels(state, cam, R, taken, own_team=None):
    import owners
    from scipy.ndimage import binary_fill_holes, label as cc
    out = _s361_zone_labels_prev(state, cam, R, taken, own_team)
    if R.get('wd') is None or not out: return out
    solid = None
    for it in state.get('items', []):
        iid, did = int(it[0]), int(it[1])
        if did != S361_ZONE_DID or iid not in out: continue
        Z = owners.zone(did)
        if not Z or Z.get('Name') != S361_ZONE_NAME or owners.area(did)['kind'] != 'zone': continue
        m, d_, own_ = out[iid]
        f0 = binary_fill_holes(m) & ~m                      # every enclosed hole of the zone label
        if f0.any():
            L, n = cc(f0)
            if n:
                sz = np.bincount(L.ravel()); ok = sz <= S361_HOLE_MAX; ok[0] = False
                f = ok[L]                                   # holes small enough: effects, characters, bushes, cards
                if f.any():
                    if solid is None: solid = _zs_solid(cam, R)
                    f &= ~solid & ~taken                    # solid map blocks and other labels keep their pixels
                    m = m | f
        out[iid] = (m, d_, own_)
    return out
