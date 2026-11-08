# zone184 fix2 (task q1028/zone184, revision 2026-10-28): the cut runs exactly along the wall's drawn top edge.
# Circle rule (user, 10-23): a lingering zone is its full logic circle, cut only by solid map blocks (walls, crates, barrels).
# Characters never cut it, and a wall hidden behind a character still cuts it.
# Cause of fix.py's gap (the user, 10-28: "theres a gap between the label and the wall"): fix.py tested every block as a
# box of the tile's full footprint, up to 606 logic units high for every type. The drawn blocks are smaller than that:
# the visible top of a crate or a wall is inset from its tile, and a barrel is round. The box's far top edge then projects
# above the drawn top edge, so zone ground that is visible just above the drawn top was cut (a strip about 2-8 px high).
# Calibration, on the kept capture sw2_184 (zone crop 700..900 x 180..330, every 12th ID frame; the visible opaque
# ground pixels are the only truth for "not behind a block"; the drawn top edge per column is the truth for the cut line):
#   - visible ground pixels whose ray meets a block (false cuts): fitted per type over top H and footprint inset d.
#     crate (3): inset 10, top 510 -> 0 false cuts (top 506..510: best).
#     walls (1): the cut line matches the drawn top edge (tick 289, columns 773-788, 16 columns) with inset 20, top 590
#     (inset 0 with top 606 leaves a 2 px gap above the drawn edge; top 570..590 with inset 10..20 leave none).
#     barrel (4): a cylinder (round) of radius 120 about the tile centre, top 634 -> 0 false cuts (radius 140: 34..98).
#   - walls (2), indestructible (6), themed (10), fragile (12), fence (13): not measured on this capture; they take the wall
#     calibration (inset 20, top 590). Fence is listed in the report as unverified.
# Cut rule (as fix.py): a zone-label pixel covered by a sprite that does not write depth (blend 1, dmask 0) is removed when
# the ground point it shows lies behind a block, found from the ray through the pixel (the ray at some height z in [0, top]
# is inside the block). Pixels not covered by such a sprite are untouched; the crumb rule of fix.py is kept.
# Brawler vs zone sprites: they share one texture and blend/dmask in this capture, so a character draw cannot be told apart
# from the zone's own sprite draw by its id; the cut uses the geometry for every non-depth sprite pixel (as the coordinator
# allowed). Installed as z_1028_zone184.py: it runs after every z_1027 file, and wraps zone_labels last.
Z184_BOX = {1: (590.0, 20.0), 2: (590.0, 20.0), 6: (590.0, 20.0), 10: (590.0, 20.0), 12: (590.0, 20.0),
            3: (510.0, 10.0), 13: (590.0, 20.0)}            # tile code -> (top, footprint inset) of a box
Z184_BARREL = (4, 120.0, 634.0)                              # tile code, radius (cylinder about the tile centre), top
Z184_STEP = 4.0    # logic units between the samples of a ray (the boxes are 500-634 tall)
Z184_SPECK = 12    # px: a part of the zone label left by the cut that touches the cut and is smaller than this is dropped


def _z184_behind(cam, grid, sx, sy):
    """(n,) bool: the ground point each half-res pixel (sx, sy) shows lies behind a solid block: the ray from the camera through
    the pixel is inside a box (tile footprint inset by d, ground .. top) or a barrel's cylinder at some height z >= 0"""
    out = np.zeros(len(sx), bool)
    zmax = max(max(t for t, _ in Z184_BOX.values()), Z184_BARREL[2])
    for z in np.arange(0.0, zmax + 1e-6, Z184_STEP):
        X, Y = _ft42_unproj(cam, sx + 0.5, sy + 0.5, z)
        gx = np.floor(X / ZS_TILE).astype(int); gy = np.floor(Y / ZS_TILE).astype(int)
        ok = (gx >= 0) & (gx < grid.shape[1]) & (gy >= 0) & (gy < grid.shape[0])
        t = np.full(len(sx), -1); t[ok] = grid[gy[ok], gx[ok]]
        fx = X - gx * ZS_TILE; fy = Y - gy * ZS_TILE
        for code, (top, d) in Z184_BOX.items():
            if z <= top:
                out |= (t == code) & (fx >= d) & (fx <= ZS_TILE - d) & (fy >= d) & (fy <= ZS_TILE - d)
        code, r, top = Z184_BARREL
        if z <= top:
            dist = np.hypot(fx - ZS_TILE / 2, fy - ZS_TILE / 2)
            out |= (t == code) & (dist <= r)
    return out


_z184_prev_zone_labels = zone_labels


def zone_labels(state, cam, R, taken, own_team=None):
    import owners
    from scipy.ndimage import label as _z184_label, binary_dilation as _z184_dil
    out = _z184_prev_zone_labels(state, cam, R, taken, own_team)
    if R.get('wd') is None or not out: return out
    ids_h = R['id'][::2, ::2]; H, W = ids_h.shape
    cov = np.array([d for d, dr in R['draws'].items() if dr.get('blend') == 1 and dr.get('dmask') == 0], np.int64)
    if cov.size == 0: return out
    covered = np.isin(ids_h, cov)                     # front-most draw is a sprite that does not write depth
    tick = int(state.get('tick', 0)); scales = globals().get('A3_DISC_SCALE', {})
    grid = None; yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    for it in state.get('items', []):
        iid, did = int(it[0]), int(it[1])
        if iid not in out: continue
        Z = owners.zone(did)
        if not Z or owners.area(did)['kind'] != 'zone': continue
        nm = Z.get('Name'); age = tick - _ZONE_SEEN.get(iid, tick)
        if not (nm in ZONE_DISC or age <= ZONE_DOME.get(nm, -1)): continue   # the circle phase only, as z_1023
        rz = float(Z['logic_row'].get('Radius') or 300) * float(scales.get(nm, 1.0))
        x, y = int(it[3]), int(it[4])
        cx, cy = _proj(cam, x, y); dx = abs(_proj(cam, x + rz, y)[0] - cx); dy = abs(_proj(cam, x, y + rz)[1] - cy)
        if dx < 1 or dy < 1: continue
        disc = ((xx - cx) / dx) ** 2 + ((yy - cy) / dy) ** 2 <= 1
        m, d_, o_ = out[iid]
        cand = m & disc & covered
        if not cand.any(): continue
        if grid is None: grid = _zs_grid(R)
        ys, xs = np.nonzero(cand)
        behind = _z184_behind(cam, grid, xs.astype(np.float64), ys.astype(np.float64))
        if not behind.any(): continue
        cut = np.zeros(m.shape, bool); cut[ys[behind], xs[behind]] = True
        new = m & ~cut
        # crumbs the cut leaves at its edge (parts under Z184_SPECK px that touch the cut) are dropped
        L, n = _z184_label(new)
        if n:
            sz = np.bincount(L.ravel()); sz[0] = 0
            touch = np.unique(L[_z184_dil(cut, iterations=2) & new]); touch = touch[touch > 0]
            crumbs = [k for k in touch if sz[k] < Z184_SPECK]
            if crumbs: new &= ~np.isin(L, crumbs)
        if new.any(): out[iid] = (new, d_, o_)
        else: out.pop(iid, None)
    return out
