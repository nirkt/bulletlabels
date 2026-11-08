# sweep 10-27 fix for skin 156 (Haiku fix, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s156/)
# Skin 156 Cactusmecha, sweep 2026-10-27 (Haiku task s156; pairs 2612-2614, sw2_156 super 45 deg, ticks 371, 389, 411):
# a bush cuts the yellow Cactus super zone at its lower-right rim (about x 1110-1150 of the pair crop), so the outline
# follows the bush's leafy top edge with notches instead of the zone's full circle. The user's rule 7: bushes never cut
# a zone.
#
# Zone: Spike_003_ulti_explosion (zone did 17000036, item 3000010, logic radius 800), the Cactusmecha super. Measured on
# this capture at tick 371: the zone's label has no pixels in the bush notch, while the logic ground disc there is the
# bush's surface (the bush is opaque world geometry above the ground, on a Forest tile).
#
# Cause: the zone labels of a2_agent2 (ZONE_DISC_UNION, which includes Spike_003) add the logic ground disc only where the
# ground is the nearest surface (depth test against the ground plane). A bush drawn in front of the ground removes the
# disc pixels under it, and the sprite part of the label is cut there too. This is the same mechanism as z_1027_s020
# (CactusUltiExplosion, skin 020), which is keyed to that zone name only, so Spike_003 needed its own entry.
#
# Fix, for this zone name only: the logic disc pixels whose surface lies on a Forest tile (tile value 7, the bush tile)
# and whose front-most draw is opaque world geometry (blend 0, dmask 1: a bush, not a blended effect) are added to the
# zone label, unless another label owns them. Solid blocks still cut (crates, walls, barrels are not on Forest tiles).
# Holes of up to 60 px in the result are filled. Everything else stays as it was. Solid map blocks are not changed,
# and the other zone types are not touched.
# Shared types: zone name Spike_003_ulti_explosion (did 17000036), the Cactusmecha super zone. Other skins that use this
# zone name are expected to show the same bush cut; they were not re-captured here.

S156_FOREST_ZONES = {'Spike_003_ulti_explosion'}
S156_FOREST_TILE = 7       # the map's Forest tile (csv tiles: bushes), as in z_1027_s020
_s156_cache = [None, None]
_s156_zone_prev = zone_labels


def _s156_forest_front(cam, R):
    """(H, W) bool at frame resolution: pixels whose surface point lies on a Forest tile, above the ground, and whose
    front-most draw is opaque world geometry (a bush), cached per frame"""
    if _s156_cache[0] is R: return _s156_cache[1]
    X, Y, Z = _zs_world(cam, R); g = _zs_grid(R)
    cxw = np.floor(X / ZS_TILE).astype(int); cyw = np.floor(Y / ZS_TILE).astype(int)
    ok = (cxw >= 0) & (cxw < 60) & (cyw >= 0) & (cyw < 60)
    t = np.full(X.shape, -1); t[ok] = g[cyw[ok], cxw[ok]]
    ids = R['id'][::2, ::2]
    opaque = [d for d, v in R['draws'].items() if v.get('blend', 0) == 0 and v.get('dmask', 1) == 1]
    m = (t == S156_FOREST_TILE) & (ids > 0) & np.isin(ids, opaque) & (Z > ZS_Z[0]) & (Z < ZS_Z[1])
    _s156_cache[0] = R; _s156_cache[1] = m
    return m


def zone_labels(state, cam, R, taken, own_team=None):
    import owners
    out = _s156_zone_prev(state, cam, R, taken, own_team)
    if R.get('wd') is None: return out
    H, W = R['wd'].shape; yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    for it in state.get('items', []):
        iid, did = int(it[0]), int(it[1]); Z = owners.zone(did)
        if not Z or owners.area(did)['kind'] != 'zone' or Z.get('Name') not in S156_FOREST_ZONES: continue
        if iid not in out: continue
        x, y = int(it[3]), int(it[4]); rz = float(Z['logic_row'].get('Radius') or 300)
        cx, cy = _proj(cam, x, y); dx = abs(_proj(cam, x + rz, y)[0] - cx); dy = abs(_proj(cam, x, y + rz)[1] - cy)
        if dx < 1 or dy < 1: continue
        disc = ((xx - cx) / dx) ** 2 + ((yy - cy) / dy) ** 2 <= 1
        old, odid, own = out[iid]
        add = disc & _s156_forest_front(cam, R) & ~taken & ~old
        if not add.any(): continue
        new = old | add
        new = new | (small_holes(new, 60) & ~taken)
        out[iid] = (new, odid, own)
    return out
