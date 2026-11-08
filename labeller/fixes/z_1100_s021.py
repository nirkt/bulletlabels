# human review 2026-10-28 fix for skin 021 (Haiku fix, reviewed by the coordinator, installed 10-29; source out/sweep2/haikutasks/human1028/s021/fix2.py; runs after z_1099_cleanup like the tests)
# Skin 021 (BarkeepDefault, Barley default attack), human review 2026-10-28, task s021, finding: "attack 0, tick 289: it
# includes some extra stuff" (the bits around the pink crate and under the blocks). fix2 replaces the first fix.py (not used).
# Rule (the user's zone rule): the zone = the full logic circle from its birth, cut ONLY by the solid blocks standing on it
# (crates, barrels, walls), CLEANLY: the cut follows each block's own outline, no thin slivers between or under blocks, no loops
# or islands around a crate, no holes; effects (the impact flash, bottle shards, the star) never cut the zone.
# Zone: Barley_def_atk (zone data id 17000058; the only zone row with this name, so every skin that shares it). Only the zone
# labels of this name change: other zones and every projectile label (type 6000248 and the rest) are untouched.
# Causes found at attack 0 tick 289 (sw2_021 capture, probe of the solid cut and of the draws under the pink crate):
#  (a) the white impact flash over the pink crate is taken as flash by z_1023's ZS_SAT exemption: the crate under it is not
#      cut, and the zone runs into the crate in a V-shaped notch with loops around it. For this zone the flash exemption is off
#      (the crate under a flash is still a crate), as z_1027_s429 does for Barley_009_atk.
#  (b) the blended effect draws over a block (flash, cyan star, shards: blend 1, no depth write) carry the depth of the ground,
#      so the block is punched there and the zone runs in as islands and loops around the crate. Those pixels take the block's
#      own silhouette (the tile box, projected, convex hull, shrunk by F2_SHRINK px), the same silhouette method as z_1027_s429.
#      The cut is then closed at effect pixels, opened 3x3 and its holes filled (as z_1023), so each block is one solid shape.
#  (c) after the cut the zone is cleaned: a 3x3 opening removes slivers thinner than 3 px (the strips between a crate row and
#      the circle edge, the tips at the crate corners), and islands (pieces disconnected from the main zone body) smaller than
#      max(F2_PIECE, F2_ISLAND x main body) are dropped. The wedge under the left crate row at tick 289 (the bottom-left sliver the
#      user meant under the blocks) is such an island: 168-201 px, 5-6 % of the main body, the 150 px cut alone would keep it
#      (measured on every zone tick 287-345 of sw2_021). Holes smaller than ZS_HOLE px that are not solid and not another label
#      are filled (z_1023's rule).
# Only the Barley zone's own mask changes; the other zones' masks are the ones the previous chain gives.

from scipy.ndimage import (binary_closing as _f2_close, binary_opening as _f2_open, binary_fill_holes as _f2_fill,
                           binary_erosion as _f2_erode, label as _f2_label)
from scipy.spatial import ConvexHull as _f2_hull

F2_NAMES = ('Barley_def_atk',)
F2_SHRINK = 4        # px: a block's box hull is a little smaller than its drawn silhouette (z_1027_s429)
F2_MINPX = 30        # opaque solid pixels a tile needs to count as a block
F2_PIECE = 150       # px: zone islands (pieces disconnected from the main body) smaller than this are dropped ...
F2_ISLAND = 1.0 / 12  # ... and so are islands smaller than this share of the main body (the wedge under the left crates is
                     # 168-201 px = 5-6 % of the main body: just over F2_PIECE, so the share rule decides; measured sw2_021 t287-325)
_f2_prev_zone_labels = zone_labels


def _f2_boxes(cam, R, S):
    """(H, W) bool: the projected silhouettes of the solid tiles that the opaque solid pixels S belong to (tile box hull,
    shrunk by F2_SHRINK px)"""
    H, W = R['wd'].shape
    X, Y, Z = _zs_world(cam, R); g = _zs_grid(R)
    tx = np.floor(X[S] / ZS_TILE).astype(int); ty = np.floor(Y[S] / ZS_TILE).astype(int); zz = Z[S]
    good = (tx >= 0) & (tx < 60) & (ty >= 0) & (ty < 60)
    tx, ty, zz = tx[good], ty[good], zz[good]
    key = ty * 60 + tx
    out = Image.new('L', (W, H), 0); dr = ImageDraw.Draw(out)
    cam = np.asarray(cam, np.float64)

    def pr(x, y, z):
        v = cam @ np.array([x, -y, 0.6 * z, 1.0])
        return ((v[0] / v[3] + 1) * 0.5 * W, (1 - v[1] / v[3]) * 0.5 * H)

    for k in np.unique(key):
        c = key == k; cx, cy = int(k % 60), int(k // 60)
        if int(g[cy, cx]) not in ZS_SOLID or c.sum() < F2_MINPX: continue
        h = float(np.percentile(zz[c], 90))
        if h < 1.0: continue
        x0, x1, y0, y1 = cx * ZS_TILE, (cx + 1) * ZS_TILE, cy * ZS_TILE, (cy + 1) * ZS_TILE
        pts = np.array([pr(x, y, z) for x in (x0, x1) for y in (y0, y1) for z in (0.0, h)])
        try: hull = _f2_hull(pts)
        except Exception: continue
        dr.polygon([tuple(pts[i]) for i in hull.vertices], fill=1)
    return _f2_erode(np.asarray(out, bool), iterations=F2_SHRINK)


def _f2_solid(cam, R):
    """the block cut of the Barley zone: opaque solid pixels with no flash exemption; blended effect pixels over a block take
    the block's silhouette; then one clean cut (closed at effect pixels, opened 3x3, holes of up to ZS_CLOSE px filled)"""
    old = globals()['ZS_SATPX']; globals()['ZS_SATPX'] = 10 ** 9
    try:
        S = _zs_solid(cam, R)
    finally:
        globals()['ZS_SATPX'] = old
    ids = R['id'][::2, ::2]
    eff = [int(d) for d, rec in R['draws'].items() if rec.get('blend') == 1 and rec.get('dmask') == 0]
    E = np.isin(ids, eff) if eff else np.zeros(ids.shape, bool)
    if not E.any(): return S
    M = _f2_boxes(cam, R, S & ~E)
    S2 = np.where(E, M, S)
    S2 = np.where(E, _f2_close(np.pad(S2, 8), structure=np.ones((7, 7), bool))[8:-8, 8:-8], S2)   # notches at a block's edge
    S2 = _f2_open(S2, structure=np.ones((3, 3), bool), iterations=1)
    fh = _f2_fill(S2) & ~S2
    if fh.any():
        L, n = _f2_label(fh); sz = np.bincount(L.ravel()); okk = sz <= ZS_CLOSE; okk[0] = False
        S2 = S2 | okk[L]
    return S2


def _f2_clean(m):
    """the zone part without slivers (3x3 opening: strips and tips under 3 px go) and without islands: pieces disconnected
    from the main body that are smaller than max(F2_PIECE px, F2_ISLAND of the main body)"""
    if not m.any(): return m
    op = _f2_open(m, structure=np.ones((3, 3), bool), iterations=1)
    if not op.any(): return m
    L, n = _f2_label(op)
    if n > 1:
        sz = np.bincount(L.ravel()); sz[0] = 0
        main = int(np.argmax(sz))
        keep = sz >= max(F2_PIECE, F2_ISLAND * sz[main]); keep[main] = True; keep[0] = False
        op = keep[L]
    return op


def zone_labels(state, cam, R, taken, own_team=None):
    import owners
    out = _f2_prev_zone_labels(state, cam, R, taken, own_team)
    if R.get('wd') is None or not R.get('draws') or len(state.get('items', [])) == 0: return out
    tick = int(state.get('tick', 0)); H, W = R['wd'].shape
    scales = globals().get('A3_DISC_SCALE', {})
    items = []   # the circle phase of every lingering zone, as z_1023 builds it (the same owner rule for every circle)
    for it in state.get('items', []):
        iid, did = int(it[0]), int(it[1]); Z = owners.zone(did)
        if not Z or owners.area(did)['kind'] != 'zone': continue
        nm = Z.get('Name'); age = tick - _ZONE_SEEN.get(iid, tick)
        if not (nm in ZONE_DISC or age <= ZONE_DOME.get(nm, -1)): continue
        rz = float(Z['logic_row'].get('Radius') or 300) * float(scales.get(nm, 1.0))
        x, y = int(it[3]), int(it[4])
        cx, cy = _proj(cam, x, y); dx = abs(_proj(cam, x + rz, y)[0] - cx); dy = abs(_proj(cam, x, y + rz)[1] - cy)
        if dx < 1 or dy < 1: continue
        items.append((iid, did, int(it[2]), cx, cy, dx, dy, nm))
    mine = [k for k, t in enumerate(items) if t[7] in F2_NAMES]
    if not mine: return out
    solid = _f2_solid(cam, R)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    D = np.stack([np.where(((xx - cx) / dx) ** 2 + ((yy - cy) / dy) ** 2 <= 1,
                           ((xx - cx) / dx) ** 2 + ((yy - cy) / dy) ** 2, np.inf)
                  for (_, _, _, cx, cy, dx, dy, _) in items])
    best = np.argmin(D, axis=0); anyin = np.isfinite(D.min(axis=0))
    for k in mine:
        iid, did, own, cx, cy, dx, dy, nm = items[k]
        owned = anyin & (best == k)
        part = owned & ~solid & ~taken
        part = _zs_fill_small(part, owned, solid, taken)
        part = _f2_clean(part)
        if part.any(): out[iid] = (part, did, own)
        else: out.pop(iid, None)
    return out
