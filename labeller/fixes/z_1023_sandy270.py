# Sandy Lantern 270 super zone: logic circle minus solid blocks, posts filled (user Q11, 2026-10-23; Haiku task 6, reviewed)
# Task 6 (Haiku, 2026-10-23): skin 270 SandstormLantern (Sandy), its super's lingering zone Sandy_003_ulti (area effect data
# id 17000219; logic Sandy_def_ulti: Radius 2000, TimeMs 9000; spawned only by Sandy003UltiProjectile, so this is skin 270's zone).
# User's decisions (binding): the zone label is SOLID: the lantern posts standing on the zone are filled (Q11); only solid map
# blocks standing on the zone cut it (walls, crates, barrels: the circle-zone rule of z_1023_zone_solid.py); it is the zone's
# full logic circle (scale 1.0) from its birth (first state item, tick 357), not gated on the zone being drawn yet.
#
# Geometry: the logic circle is projected exactly (its ground points through idlabels._proj, filled polygon). The drawn opaque
# blue circuit of the zone lies inside that polygon (opaque zone-sprite px outside it: about 330 of 36000 on tick 389, 13 px at
# most, the tops of lantern posts standing on the edge), so scale 1.0 is right. The faint haze beyond it (alpha < 128, up to
# about 25 px above the top) is a soft halo and stays unlabelled. Not done through ZONE_DISC: zone_labels' ellipse (projected
# centre +- the projected radii along x and y) is off by about 32 px at this zone's far (top) edge, so this file sets the
# zone's label itself (z_1023_zone_solid's cut is reused: _zs_solid).
#
# Per pixel: a pixel of the circle is cut when it is a solid block (_zs_solid) or belongs to another label (taken: projectiles,
# spawnables). Lanterns, bushes, characters, UI and the zone's own sprites never cut it.
S270_ZONE_DID = 17000219      # area_effects_skin row Sandy_003_ulti
S270_ZONE_NAME = 'Sandy_003_ulti'
S270_ZONE_SCALE = 1.0         # drawn edge = logic radius (measured on the opaque body, ticks 359-411)
S270_RING = 1440              # ground points of the logic circle projected to the screen
S270_POST_OUT = 15            # px (holes: ZS_HOLE of z_1023): opaque zone sprites this far outside the circle, joined to it, are the posts standing on its edge
S270_SPECK = 0.01             # parts of the label under this share of its largest part are specks: dropped
# Specks: the zone's glowing sparks and white flashes lying on a solid block (a crate's window, a barrel's face) are not
# solid for z_1023's cut (ZS_SAT), so they leave islands of 80-160 px inside the solid cut, and a 1-px sliver along a block's
# edge (ticks 359, 389). Parts under S270_SPECK of the largest part (80-160 px against about 130k) are dropped.


def _s270_specks(part):
    from scipy.ndimage import label as cc
    L, n = cc(part)
    if n <= 1: return part
    sz = np.bincount(L.ravel())[1:]
    return np.isin(L, 1 + np.nonzero(sz >= S270_SPECK * sz.max())[0])


def _s270_circle(cam, x, y, rz, H, W):
    """(H, W) bool: the exact screen polygon of the logic circle (centre x, y, radius rz, on the ground plane)"""
    th = np.linspace(0.0, 2.0 * np.pi, S270_RING, endpoint=False)
    pts = [tuple(_proj(cam, x + rz * np.cos(t), y + rz * np.sin(t))) for t in th]
    im = Image.new('L', (W, H), 0); ImageDraw.Draw(im).polygon(pts, fill=1)
    return np.asarray(im, bool)


def _s270_posts(part, disc, C, cv, solid, taken):
    """the lantern posts standing on the circle's edge: opaque zone-sprite pixels within S270_POST_OUT px outside the circle,
    connected to the label through such pixels only (never across a solid block, never another label)"""
    from scipy.ndimage import label as cc, distance_transform_edt
    dout = distance_transform_edt(~disc)
    cand = (C > 0) & (cv >= CUT) & (dout <= S270_POST_OUT) & ~disc & ~solid & ~taken
    if not cand.any(): return part
    L, n = cc(cand | part)
    keep = np.unique(L[part]); keep = keep[keep > 0]
    return part | (np.isin(L, keep) & cand)


_s270_zone_labels_prev = zone_labels


def zone_labels(state, cam, R, taken, own_team=None):
    import owners
    out = _s270_zone_labels_prev(state, cam, R, taken, own_team)
    if R.get('wd') is None: return out
    H, W = R['id'][::2, ::2].shape
    solid = None
    for it in state.get('items', []):
        iid, did = int(it[0]), int(it[1])
        if did != S270_ZONE_DID: continue
        Z = owners.zone(did)
        if not Z or Z.get('Name') != S270_ZONE_NAME: continue
        if solid is None: solid = _zs_solid(cam, R)
        own = True if own_team is None else own_team(int(it[2]))
        x, y = int(it[3]), int(it[4]); rz = float(Z['logic_row'].get('Radius') or 300) * S270_ZONE_SCALE
        disc = _s270_circle(cam, x, y, rz, H, W)
        part = disc & ~solid & ~taken
        if part.any():
            part = _s270_posts(part, disc, sprite_pixels(('zone', did), own, R), R['cv'][::2, ::2], solid, taken)
            part = _zs_fill_small(part, True, solid, taken)   # pinholes at the posts' edges (not solid, not another label)
            part = _s270_specks(part)
        if part.any(): out[iid] = (part, did, int(it[2]))
        else: out.pop(iid, None)
    return out
