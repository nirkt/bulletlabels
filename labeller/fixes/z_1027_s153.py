# sweep 10-27 fix for skin 153 (Sonnet revision, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s153/)
# sweep 10-27 fix (2nd attempt) for skin 153 CactusPink (Haiku fix task out/sweep2/haikutasks/sweep1027/s153/, pairs 2549-2551,
# sw2_153 super 45 deg, ticks 371 / 389 / 411; bushes cutting the zone, loose speck at tick 371).
#
# Zone: Spike_002_ulti_explosion (area effect 17000033, the Pink skin's super zone, owner Cactus; projectiles_skin
# Spike002UltiProjectile). Keyed to this zone name only (same convention as z_1027_s020 for CactusUltiExplosion).
#
# Defect 1 (too_small, pairs 2549-2551): the zone's outline lower right followed the top edge of the green bush. Causes:
#  (a) a2_agent2's ZONE_DISC_UNION adds the logic ground disc only where the ground is the nearest surface; the bush
#      (opaque world geometry on Forest tiles, drawn in front of the ground) removes the disc under it (as s020);
#  (b) the DRAWN ring (the zone's sprite rim) is 6-9 px outside the logic disc all round (left edge 637 vs disc 644,
#      right 855 vs 849 at t371), so even with the disc under the bush filled, the outline stepped in from the ring's outer
#      edge (where the sprite is still visible above the bush) to the smaller disc and followed the bush top edge there.
# Fix: logic disc pixels on a Forest tile whose front-most draw is opaque world geometry (bush) are added to the label (not
# other labels' pixels), using the disc scaled by the measured ring factor k: per frame k = median of the label's max
# radius (in disc radii) over 5-degree bins in the unobstructed sector -35..15 deg (right of the disc, above the bush),
# clipped to 1.0..1.10, 1.0 if fewer than 4 bins. A bush pixel standing against a crate lies on the crate's tile, so a draw
# whose pixels are mostly on Forest tiles (>= half) counts as the bush in full. Crates / walls still cut. Holes <= 60 px filled.
#
# Defect 2 (spur_or_speck, pair 2549 tick 371): an 88 px loose part of the zone label lower left, outside the circle, around
# a small pink particle. Fix: a part of the zone label smaller than S153_SPECK px and entirely outside the logic disc is
# dropped. The cloud (26000+ px) and the detached parts of ticks 361-369 larger than the threshold are kept.
#
# Shared types: the zone name Spike_002_ulti_explosion only; no other zone name is touched, the zone's solid cut is unchanged.

S153_ZONES = {'Spike_002_ulti_explosion'}
S153_FOREST_TILE = 7     # the map's Forest tile (bushes), see z_1023_zone_solid.py
S153_SPECK = 150         # px (full resolution): loose zone parts smaller than this, outside the ground disc, are dropped
_s153_cache = [None, None]
_s153_zone_prev = zone_labels


def _s153_forest_front(cam, R):
    """(H, W) bool at frame resolution: pixels whose surface point lies on a Forest tile, above the ground, and whose
    front-most draw is opaque world geometry (a bush), cached per frame"""
    if _s153_cache[0] is R: return _s153_cache[1]
    X, Y, Z = _zs_world(cam, R); g = _zs_grid(R)
    cxw = np.floor(X / ZS_TILE).astype(int); cyw = np.floor(Y / ZS_TILE).astype(int)
    ok = (cxw >= 0) & (cxw < 60) & (cyw >= 0) & (cyw < 60)
    t = np.full(X.shape, -1); t[ok] = g[cyw[ok], cxw[ok]]
    ids = R['id'][::2, ::2]
    opaque = [d for d, v in R['draws'].items() if v.get('blend', 0) == 0 and v.get('dmask', 1) == 1]
    base = np.isin(ids, opaque) & (ids > 0) & (Z > ZS_Z[0]) & (Z < ZS_Z[1])
    f7 = base & (t == S153_FOREST_TILE)
    # a bush's pixels that stand against a crate or wall lie on that block's tile (the surface point of a bush pixel at
    # height 500 projects onto the neighbouring tile): a draw whose pixels are mostly on Forest tiles is the bush, all of it
    dd, inv = np.unique(ids[base], return_inverse=True)
    tot = np.bincount(inv, minlength=len(dd)); fo = np.bincount(inv, weights=f7[base].astype(np.float64), minlength=len(dd))
    bush = dd[fo >= 0.5 * tot]
    m = f7 | (base & np.isin(ids, bush))
    _s153_cache[0] = R; _s153_cache[1] = m
    return m


def _s153_ring_k(old, xx, yy, cx, cy, dx, dy):
    """how much bigger the drawn ring is than the logic disc, measured on the label in the open sector right of the disc"""
    ys, xs = np.nonzero(old)
    if not ys.size: return 1.0
    r = np.sqrt(((xs - cx) / dx) ** 2 + ((ys - cy) / dy) ** 2)
    ang = np.degrees(np.arctan2((ys - cy) / dy, (xs - cx) / dx))
    v = []
    for lo in range(-35, 15, 5):
        s = (ang >= lo) & (ang < lo + 5) & (r < 1.3)
        if s.sum() >= 3: v.append(r[s].max())
    if len(v) < 4: return 1.0
    return float(np.clip(np.median(v), 1.0, 1.10))


def zone_labels(state, cam, R, taken, own_team=None):
    import owners
    from scipy.ndimage import label as _s153_cc
    out = _s153_zone_prev(state, cam, R, taken, own_team)
    if R.get('wd') is None: return out
    H, W = R['wd'].shape; yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    for it in state.get('items', []):
        iid, did = int(it[0]), int(it[1]); Z = owners.zone(did)
        if not Z or owners.area(did)['kind'] != 'zone' or Z.get('Name') not in S153_ZONES: continue
        if iid not in out: continue
        x, y = int(it[3]), int(it[4]); rz = float(Z['logic_row'].get('Radius') or 300)
        cx, cy = _proj(cam, x, y); dx = abs(_proj(cam, x + rz, y)[0] - cx); dy = abs(_proj(cam, x, y + rz)[1] - cy)
        if dx < 1 or dy < 1: continue
        disc = ((xx - cx) / dx) ** 2 + ((yy - cy) / dy) ** 2 <= 1
        old, odid, own = out[iid]
        # 1. bushes never cut the zone: the logic disc under a bush is the zone
        k = _s153_ring_k(old, xx, yy, cx, cy, dx, dy)
        disck = ((xx - cx) / (dx * k)) ** 2 + ((yy - cy) / (dy * k)) ** 2 <= 1
        add = disck & _s153_forest_front(cam, R) & ~taken & ~old
        new = old | add
        new = new | (small_holes(new, 60) & ~taken)
        # 2. loose specks: small parts of the zone label that lie outside the ground disc are dropped
        L, n = _s153_cc(new)
        if n > 1:
            sz = np.bincount(L.ravel()); drop = np.zeros(n + 1, bool)
            for j in range(1, n + 1):
                if sz[j] < S153_SPECK and not (disc & (L == j)).any(): drop[j] = True
            if drop.any(): new = new & ~drop[L]
        if not new.any(): out.pop(iid, None); continue
        out[iid] = (new, odid, own)
    return out
