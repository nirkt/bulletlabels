# sweep 10-27 fix for skin 020 (Haiku fix, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s020/)
# Skin 020 CactusDefault, sweep 2026-10-27 (pairs 405-407, sw2_020 super 45 deg, ticks 361-411): the bush cuts the Cactus
# super zone (CactusUltiExplosion, area effect 17000030, spawned by CactusUltiProjectile 6000112).
#
# Cause: a2_agent2's zone_labels (ZONE_DISC_UNION) adds the zone's logic ground disc, but only where the ground is the
# nearest surface (depth test against the ground plane). A bush (opaque world geometry standing on a Forest tile, height
# 310-560 logic units) drawn in front of the ground removes the disc pixels under it, and the sprite part of the label
# is cut there too, so the yellow outline follows the bush's top edge instead of the zone's circle (about 240-280 px per
# tick, the notch at the right of the crate). The user's rule 7: bushes never cut a zone.
#
# Fix, for this zone name only: the logic disc pixels whose surface lies on a Forest tile (tile value 7, the bush tile;
# measured on this capture: every bush pixel of the defect is tile 7, none of the crate or wall pixels is) and whose
# front-most draw is opaque world geometry (blend 0, dmask 1: a bush, not a blended effect) are added to the zone label,
# unless another label owns them. Solid blocks still cut (their tiles are Wall / Crate, not Forest; the flash-over-block
# exception of z_1023 is kept as it is). Holes of up to 60 px in the result are filled. Everything else stays as it was.
# Solid map blocks are not changed; the other zone types are not touched.
# Shared types: the zone name CactusUltiExplosion is the base Cactus super zone (area effect 17000030); the Spike_00N
# super zones (ZONE_DISC_UNION in a2_agent2) use their own names and are not changed here.

S020_FOREST_ZONES = {'CactusUltiExplosion'}
S020_FOREST_TILE = 7       # the map's Forest tile (csv tiles: bushes), see z_1023_zone_solid.py
_s020_cache = [None, None]
_s020_zone_prev = zone_labels


def _s020_forest_front(cam, R):
    """(H, W) bool at frame resolution: pixels whose surface point lies on a Forest tile, above the ground, and whose
    front-most draw is opaque world geometry (a bush), cached per frame"""
    if _s020_cache[0] is R: return _s020_cache[1]
    X, Y, Z = _zs_world(cam, R); g = _zs_grid(R)
    cxw = np.floor(X / ZS_TILE).astype(int); cyw = np.floor(Y / ZS_TILE).astype(int)
    ok = (cxw >= 0) & (cxw < 60) & (cyw >= 0) & (cyw < 60)
    t = np.full(X.shape, -1); t[ok] = g[cyw[ok], cxw[ok]]
    ids = R['id'][::2, ::2]
    opaque = [d for d, v in R['draws'].items() if v.get('blend', 0) == 0 and v.get('dmask', 1) == 1]
    m = (t == S020_FOREST_TILE) & (ids > 0) & np.isin(ids, opaque) & (Z > ZS_Z[0]) & (Z < ZS_Z[1])
    _s020_cache[0] = R; _s020_cache[1] = m
    return m


def zone_labels(state, cam, R, taken, own_team=None):
    import owners
    out = _s020_zone_prev(state, cam, R, taken, own_team)
    if R.get('wd') is None: return out
    H, W = R['wd'].shape; yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    for it in state.get('items', []):
        iid, did = int(it[0]), int(it[1]); Z = owners.zone(did)
        if not Z or owners.area(did)['kind'] != 'zone' or Z.get('Name') not in S020_FOREST_ZONES: continue
        if iid not in out: continue
        x, y = int(it[3]), int(it[4]); rz = float(Z['logic_row'].get('Radius') or 300)
        cx, cy = _proj(cam, x, y); dx = abs(_proj(cam, x + rz, y)[0] - cx); dy = abs(_proj(cam, x, y + rz)[1] - cy)
        if dx < 1 or dy < 1: continue
        disc = ((xx - cx) / dx) ** 2 + ((yy - cy) / dy) ** 2 <= 1
        old, odid, own = out[iid]
        add = disc & _s020_forest_front(cam, R) & ~taken & ~old
        if not add.any(): continue
        new = old | add
        new = new | (small_holes(new, 60) & ~taken)
        out[iid] = (new, odid, own)
    return out
