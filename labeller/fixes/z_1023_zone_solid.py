# Zone circles: full logic circle, cut only by solid map blocks (user, 2026-10-23); Haiku zone agent (bugtrial3), reviewed and wired (map from the capture via R[dir]). Replaces the first version (bugtrial3/z_1023_zone_solid_v1_opus.py).
# Zone circles (haiku, 2026-10-23, the coordinator's decision of the same day): a zone drawn as a circle (ZONE_DISC, and
# ZONE_DOME while its dome is visible) is labelled as the FULL logic circle: its logic position and radius (times the
# per-zone scale of A3_DISC_SCALE). The circle is cut ONLY by solid map blocks standing on it (walls, crates, barrels and
# similar map blocks). Characters, bushes, the UI, effects (flashes, sparks, the zone's own glows, swirls and meshes) and
# projectiles never cut it.
#
# Solid blocks come from the capture's own map (map.npy: the 60 x 60 tile grid, 300 logic units per tile, values
# 27000000 + the row of csv_logic/tiles.csv): Wall1 (1), Wall2 (2), Crate (3), Barrel (4), Indestructible (6), Themed (10),
# Fragile (12), Fence (13). Open (0), Forest (7, bushes), Water (9), DecoDestructible (5) and InvisibleWater (11) are not solid.
# The capture's map is read from the environment (IDLABELS_MAP=<capture>/map.npy); the run raises without it.
#
# Per pixel the ID pass's eye depth gives the logic point the pixel shows: the exact inverse of idlabels._proj (see
# world below). A pixel is a block when that point lies in a solid tile (ZS_MARGIN, so the block's visible side faces, which
# lie on the tile border, count), stands above the ground (ZS_Z) and is drawn (id > 0). That is the object's own silhouette
# at the pixel level, not a colour or shader test.
#
# Label per zone: the full circle minus the solid blocks and minus the labels of projectiles / spawnables (their own
# labels keep their pixels: one owner per pixel). Where two zone circles overlap, a pixel belongs to the circle whose centre
# is nearer (normalised by the circle's radii; the same rule as the projectiles' nearest predicted label). Holes of less
# than ZS_HOLE px that are not solid or taken are filled; solid pieces of less than ZS_SPECK px are dropped.
# The solid cut is smoothed (3x3 opening, enclosed holes up to ZS_CLOSE px filled) and white flashes over a block do not cut
# (ZS_SAT). The frame is needed for the flash test: labels() is wrapped here to keep it (idlabels.labels passes img).
import os as _zs_os
from scipy.ndimage import label as _zs_label, binary_fill_holes as _zs_fill

ZS_ENV = 'IDLABELS_MAP'
ZS_TILE = 300.0
ZS_SOLID = (1, 2, 3, 4, 6, 10, 12, 13)
ZS_MARGIN = 8.0          # logic units: a point this close to a tile border is taken in both tiles (side faces)
ZS_Z = (30.0, 1000.0)    # logic height of a block surface (the ground shows at z ~ 5.5; walls top out near 600-800)
ZS_SPECK = 6             # px: solid pieces smaller than this are dropped
ZS_OPEN = 2              # 3x3 opening iterations on the solid cut: strips thinner than about 2 * ZS_OPEN + 1 px go
ZS_CLOSE = 40            # px: holes enclosed by the solid cut up to this size are filled
ZS_HOLE = 20             # px: holes of the zone label smaller than this are filled (not solid, not another label)
# a block drawn under an additive flash: the flash is not in the ID pass, so the block's pixels show as the block; the frame
# shows them clipped white. Pure-white pieces (all channels >= ZS_SAT) of at least ZS_SATPX px are the flash over the block:
# they do not cut (the flash is an effect on top), grown by ZS_SATGROW px so the block's rim under the flash edge does not fray.
# Small white highlights stay solid (measured on sw2_226: attack ticks 301-341 have no white piece above 35 px)
ZS_SAT = 230
ZS_SATPX = 40
ZS_SATGROW = 3
_ZS_MAPS = {}
_ZS_WORLD = [None, None]
_ZS_IMG = [None]


_zs_labels_base = labels


def labels(objs, cam, R, state=None, ended=(), img=None):
    """idlabels.labels with the frame kept for the flash test of zone_labels"""
    _ZS_IMG[0] = img
    return _zs_labels_base(objs, cam, R, state=state, ended=ended, img=img)


def _zs_grid(R=None):
    """the capture's 60 x 60 tile grid: IDLABELS_MAP if set, else map.npy of the capture the ID pass frame R comes from
    (R['dir'] = capture/id/<run>/, idbuf.load)"""
    p = _zs_os.environ.get(ZS_ENV, '')
    if not p and R is not None and R.get('dir'):
        d = _zs_os.path.abspath(R['dir'])
        for _ in range(4):
            if _zs_os.path.exists(_zs_os.path.join(d, 'map.npy')): p = _zs_os.path.join(d, 'map.npy'); break
            d = _zs_os.path.dirname(d)
    if not p:
        raise RuntimeError('zone labels need the capture map: set %s=<capture dir>/map.npy' % ZS_ENV)
    if p not in _ZS_MAPS:
        _ZS_MAPS[p] = np.load(p).reshape(60, 60).astype(np.int64) - 27000000
    return _ZS_MAPS[p]


def _zs_world(cam, R):
    """(H, W) logic x, y and height z of the surface each pixel shows, from the ID pass depth wd (w = wd * 1000 + wref):
    solves the camera equations of idlabels._proj for the render point (x, -y, 0.6 z)"""
    if _ZS_WORLD[0] is R: return _ZS_WORLD[1]
    cam = np.asarray(cam, np.float64); wd = np.asarray(R['wd'], np.float64); H, W = wd.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    ndx = 2 * xx / W - 1; ndy = 1 - 2 * yy / H
    w = wd * 1000.0 + R['meta'].get('wref', 0.0)
    A = np.empty((H, W, 3, 3)); b = np.empty((H, W, 3))
    A[..., 0, :] = cam[0, :3] - ndx[..., None] * cam[3, :3]
    A[..., 1, :] = cam[1, :3] - ndy[..., None] * cam[3, :3]
    A[..., 2, :] = cam[3, :3]
    b[..., 0] = -(cam[0, 3] - ndx * cam[3, 3]); b[..., 1] = -(cam[1, 3] - ndy * cam[3, 3]); b[..., 2] = w - cam[3, 3]
    v = np.linalg.solve(A, b[..., None])[..., 0]
    out = (v[..., 0], -v[..., 1], v[..., 2] / 0.6)
    _ZS_WORLD[0] = R; _ZS_WORLD[1] = out
    return out


def _zs_solid(cam, R):
    """(H, W) bool: pixels (drawn, above the ground) whose surface point lies in a solid map tile (with ZS_MARGIN)"""
    X, Y, Z = _zs_world(cam, R); g = _zs_grid(R)
    ids = R['id'][::2, ::2]; solid = np.zeros(ids.shape, bool)
    for dx in (-ZS_MARGIN, 0.0, ZS_MARGIN):
        for dy in (-ZS_MARGIN, 0.0, ZS_MARGIN):
            cx = np.floor((X + dx) / ZS_TILE).astype(int); cy = np.floor((Y + dy) / ZS_TILE).astype(int)
            ok = (cx >= 0) & (cx < 60) & (cy >= 0) & (cy < 60)
            t = np.full(ids.shape, -1); t[ok] = g[cy[ok], cx[ok]]
            solid |= np.isin(t, ZS_SOLID)
    solid &= (ids > 0) & (Z > ZS_Z[0]) & (Z < ZS_Z[1])
    img = _ZS_IMG[0]
    if img is not None:   # the flash over a block: pure-white pieces (see ZS_SAT)
        from scipy.ndimage import binary_dilation as _zs_dil
        white = img.min(-1) >= ZS_SAT
        L, n = _zs_label(white)
        if n:
            sz = np.bincount(L.ravel()); big = sz >= ZS_SATPX; big[0] = False
            flash = _zs_dil(big[L], iterations=ZS_SATGROW)
            solid &= ~flash
    # the block's own outline: thin strips (under about 5 px) that the swirls / flashes leave along a block are cut off,
    # holes of up to ZS_CLOSE px enclosed by the block (its pixels under a translucent effect) are filled
    from scipy.ndimage import binary_opening as _zs_open
    solid = _zs_open(solid, structure=np.ones((3, 3), bool), iterations=ZS_OPEN)
    fh = _zs_fill(solid) & ~solid
    if fh.any():
        L, n = _zs_label(fh); sz = np.bincount(L.ravel()); ok = sz <= ZS_CLOSE; ok[0] = False
        solid |= ok[L]
    L, n = _zs_label(solid)
    if n:
        sz = np.bincount(L.ravel()); small = sz < ZS_SPECK; small[0] = False
        solid &= ~small[L]
    return solid


def _zs_fill_small(part, own, solid, taken):
    """fill the holes of less than ZS_HOLE px in a zone part that are neither solid nor another label, nor another circle's"""
    f = _zs_fill(part) & ~part
    if not f.any(): return part
    L, n = _zs_label(f); sz = np.bincount(L.ravel()); small = sz < ZS_HOLE; small[0] = False
    return part | (small[L] & own & ~solid & ~taken)


_zs_prev_zone_labels = zone_labels


def zone_labels(state, cam, R, taken, own_team=None):
    import owners
    out = _zs_prev_zone_labels(state, cam, R, taken, own_team)
    if R.get('wd') is None: return out
    tick = int(state.get('tick', 0)); H, W = R['wd'].shape
    scales = globals().get('A3_DISC_SCALE', {})
    items = []
    for it in state.get('items', []):
        iid, did = int(it[0]), int(it[1]); Z = owners.zone(did)
        if not Z or owners.area(did)['kind'] != 'zone': continue
        nm = Z.get('Name'); age = tick - _ZONE_SEEN.get(iid, tick)
        if not (nm in ZONE_DISC or age <= ZONE_DOME.get(nm, -1)): continue   # the circle phase only (a dome's puddle is sprites)
        rz = float(Z['logic_row'].get('Radius') or 300) * float(scales.get(nm, 1.0))
        x, y = int(it[3]), int(it[4])
        cx, cy = _proj(cam, x, y); dx = abs(_proj(cam, x + rz, y)[0] - cx); dy = abs(_proj(cam, x, y + rz)[1] - cy)
        if dx < 1 or dy < 1: continue
        items.append((iid, did, int(it[2]), cx, cy, dx, dy))
    if not items: return out
    solid = _zs_solid(cam, R)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    D = np.stack([np.where(((xx - cx) / dx) ** 2 + ((yy - cy) / dy) ** 2 <= 1,
                           ((xx - cx) / dx) ** 2 + ((yy - cy) / dy) ** 2, np.inf)
                  for (_, _, _, cx, cy, dx, dy) in items])
    best = np.argmin(D, axis=0); anyin = np.isfinite(D.min(axis=0))
    for k, (iid, did, own, cx, cy, dx, dy) in enumerate(items):
        disc = np.isfinite(D[k]); owned = anyin & (best == k)
        part = owned & ~solid & ~taken
        part = _zs_fill_small(part, owned, solid, taken)
        old = out[iid][0] if iid in out else np.zeros((H, W), bool)
        new = (old & ~disc) | part
        if new.any(): out[iid] = (new, did, own)
        else: out.pop(iid, None)
    return out
