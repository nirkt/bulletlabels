# sweep 10-27 fix for skin 124 (Haiku fix, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s124/)
# sweep 10-27 fix for skin 124 (VoodooStarrGirls, task out/sweep2/haikutasks/sweep1027/s124/): pair 1980 (attack 90 deg,
# tick 341, 'hole'): the yellow zone is a ring with an open middle instead of one filled disc.
# Cause (measured on the capture, zone item 3000003, Voodoo003AttackEarth, radius 600, age 18 at tick 341): the zone's dome is
# gone (ZONE_DOME: 8 ticks), so its label is its sprite pixels only. They cover the middle of the logic circle (r < 0.65: 100 %)
# and about half of the bright ring (r 0.85-0.95), but not the teal puddle between them (r 0.65-0.85: 17 %, 5 %, 10 % labelled)
# nor the dark ground at the ring's outer edge (r 0.95-1.0). That leaves an unlabelled annulus on either side of the ring.
# Fix: once the dome is gone, a zone whose sprite label already covers its middle and at least half of its logic disc also takes
# the logic disc cut by solid objects standing in front of the ground (the disc of ZONE_DISC zones, rule 7: the full circle from
# the game data, cut only by solid map objects). Pixels already labelled by another object are not taken (~taken). The sprite
# pixels of the zone stay. At 333-339 the change is 60-65 px; at 341 it adds 5609 px (the disc: 16600 px labelled).
# Guard: the crate-area zone of this capture (item 3000002, Voodoo003AttackEarth at 7150 1350, ticks 293-301) has its label
# outside its middle (centre not labelled, 28-32 % of the disc labelled), so it is not extended. The zones at ticks 303-321 and
# 343-361 have no sprite label at all (no puddle drawn: crates, walls, the super's ring) and stay unlabelled.
# Only the Voodoo003AttackEarth name: the other Voodoo earth zones (VoodooAttackEarth, Voodoo002AttackEarth) keep their sprite
# labels (s098's puddle is sprite-coded and reviewed as such). Projectile types are not touched.
Z124_NAMES = {'Voodoo003AttackEarth'}
Z124_MIN_FRAC = 0.5      # the sprite label covers at least this fraction of the logic disc (before the cut by front objects)
_z124_zone_prev = zone_labels


def zone_labels(state, cam, R, taken, own_team=None):
    import owners
    out = _z124_zone_prev(state, cam, R, taken, own_team=own_team)
    if not state or R.get('wd') is None: return out
    tick = int(state.get('tick', 0))
    H, W = R['wd'].shape; yy, xx = np.mgrid[0:H, 0:W]
    wref = R['meta'].get('wref', 0.0)
    for it in state.get('items', []):
        did = int(it[1]); Z = owners.zone(did)
        if not Z or Z.get('Name') not in Z124_NAMES: continue
        iid = int(it[0]); age = tick - _ZONE_SEEN.get(iid, tick)
        if age <= ZONE_DOME.get(Z.get('Name'), -1): continue   # the dome phase: the disc is already in the label
        if iid not in out: continue                            # no sprite label: no visible puddle, nothing to complete
        x, y = int(it[3]), int(it[4]); rz = float(Z['logic_row'].get('Radius') or 300)
        cx, cy = _proj(cam, x, y); dx = abs(_proj(cam, x + rz, y)[0] - cx); dy = abs(_proj(cam, x, y + rz)[1] - cy)
        if dx < 1 or dy < 1: continue
        disc = ((xx - cx) / dx) ** 2 + ((yy - cy) / dy) ** 2 <= 1
        m = out[iid][0]
        ci, cj = int(round(cy)), int(round(cx))
        if not (0 <= ci < H and 0 <= cj < W) or not m[ci, cj]: continue       # the middle of the zone is not labelled
        if (m & disc).sum() < Z124_MIN_FRAC * disc.sum(): continue            # the label is not the puddle of this zone
        dm = disc & (R['wd'] >= ground_depth(cam, x, y, rz, wref) - DEPTH_TOL) & ~taken
        if not dm.any(): continue
        out[iid] = (m | dm, did, int(it[2]))
    return out
