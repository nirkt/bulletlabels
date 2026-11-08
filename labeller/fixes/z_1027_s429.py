# sweep 10-27 fix for skin 429 (Sonnet revision, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s429/)
# s429 (BarkeepBrawlAcademy, sweep 2026-10-27, pair 8066 ticks 289 / 301; reviewer round 2).
# Barley_009_atk puddle (zone 17000072): the zone is the logic circle minus the solid blocks (crates), ONE clean piece.
# z_1023_zone_solid decides "solid" per pixel from the ID pass depth. Effect sprites that are drawn over a crate (blended, no
# depth write, ID pass: the zone's own cyan blobs / crescents / flash sprites and the flags) carry the depth of the ground behind
# the crate, not of the crate, so the crate is punched there, and pure-white pieces >= 40 px are exempt from the cut (ZS_SAT):
# yellow islands and notches hug the effects on the crates (a piece-size / effect-share threshold only catches some).
# For this zone only: (1) the white-flash exemption is off (the crate under a flash is still a crate); (2) the solid state of a
# pixel whose front-most draw is an effect draw (blend 1, dmask 0) is taken from the map instead of from its depth: the
# silhouette of the solid tile's block (the 8 corners of its box, tile footprint x the block's height measured on its own
# opaque pixels, projected with the frame's camera, convex hull). Pixels of the opaque world draws keep the exact per-pixel
# depth test of z_1023. So an effect over a crate is crate, an effect over the ground stays zone, and the zone is the circle
# minus the solid blocks in one piece.
# Type ids: zone 17000072 (Barley_009_atk) only; the projectile (6000264) is not touched.
import numpy as _s429_np
from scipy.ndimage import (binary_closing as _s429_close, binary_opening as _s429_open, binary_fill_holes as _s429_fill,
                           label as _s429_label)

_S429_NAMES = ('Barley_009_atk',)
_S429_MINPX = 30
_S429_SHRINK = 4   # px: the box hull is a little larger than the block's rounded silhouette
_s429_prev_zone_labels = zone_labels
_s429_prev_solid = _zs_solid


def _s429_boxes(cam, R, S, E):
    """(H, W) bool: union of the projected block silhouettes of the solid tiles that the opaque solid pixels S & ~E belong to"""
    H, W = R['wd'].shape
    X, Y, Z = _zs_world(cam, R); g = _zs_grid(R)
    ok = S & ~E
    tx = _s429_np.floor(X[ok] / ZS_TILE).astype(int); ty = _s429_np.floor(Y[ok] / ZS_TILE).astype(int); zz = Z[ok]
    good = (tx >= 0) & (tx < 60) & (ty >= 0) & (ty < 60)
    tx, ty, zz = tx[good], ty[good], zz[good]
    key = ty * 60 + tx
    out = Image.new('L', (W, H), 0); dr = ImageDraw.Draw(out)
    cam = _s429_np.asarray(cam, _s429_np.float64)
    def pr(x, y, z):
        v = cam @ _s429_np.array([x, -y, 0.6 * z, 1.0])
        return ((v[0] / v[3] + 1) * 0.5 * W, (1 - v[1] / v[3]) * 0.5 * H)
    from scipy.spatial import ConvexHull
    for k in _s429_np.unique(key):
        c = key == k; cx, cy = int(k % 60), int(k // 60)
        if int(g[cy, cx]) not in ZS_SOLID or c.sum() < _S429_MINPX: continue
        h = float(_s429_np.percentile(zz[c], 90))
        x0, x1, y0, y1 = cx * ZS_TILE, (cx + 1) * ZS_TILE, cy * ZS_TILE, (cy + 1) * ZS_TILE
        pts = _s429_np.array([pr(x, y, z) for x in (x0, x1) for y in (y0, y1) for z in (0.0, h)])
        hull = ConvexHull(pts)
        dr.polygon([tuple(pts[i]) for i in hull.vertices], fill=1)
    from scipy.ndimage import binary_erosion as _er
    return _er(_s429_np.asarray(out, bool), iterations=_S429_SHRINK)


def _s429_solid_fn(cam, R):
    old = globals().get('ZS_SATPX'); globals()['ZS_SATPX'] = 10 ** 9
    try:
        S = _s429_prev_solid(cam, R)
    finally:
        globals()['ZS_SATPX'] = old
    ids = R['id'][::2, ::2]
    eff = [int(d) for d, rec in R['draws'].items() if rec.get('blend') == 1 and rec.get('dmask') == 0]
    E = _s429_np.isin(ids, eff) if eff else _s429_np.zeros(ids.shape, bool)
    if not E.any(): return S
    M = _s429_boxes(cam, R, S, E)
    S2 = _s429_np.where(E, M, S)
    S2 = _s429_np.where(E, _s429_close(_s429_np.pad(S2, 8), structure=_s429_np.ones((7, 7), bool))[8:-8, 8:-8], S2)   # notches of an effect at a block's edge
    S2 = _s429_open(S2, structure=_s429_np.ones((3, 3), bool), iterations=1)
    fh = _s429_fill(S2) & ~S2
    if fh.any():
        L, n = _s429_label(fh); sz = _s429_np.bincount(L.ravel()); okk = sz <= ZS_CLOSE; okk[0] = False
        S2 |= okk[L]
    return S2


def zone_labels(state, cam, R, taken, own_team=None):
    import owners
    hit = any((owners.zone(int(it[1])) or {}).get('Name') in _S429_NAMES for it in state.get('items', []))
    if not hit or not R.get('draws') or R.get('wd') is None:
        return _s429_prev_zone_labels(state, cam, R, taken, own_team)
    g = globals(); old = g['_zs_solid']; g['_zs_solid'] = _s429_solid_fn
    try:
        return _s429_prev_zone_labels(state, cam, R, taken, own_team)
    finally:
        g['_zs_solid'] = old
