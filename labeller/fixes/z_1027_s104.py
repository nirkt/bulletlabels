# sweep 10-27 fix for skin 104 (Haiku fix, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s104/)
# sweep 10-27 fix for skin 104 GeishaDefault (Geisha), task s104 (Haiku fix, sweep 2026-10-27; pairs 1691, 1692, 1693 at
# ticks 371, 389, 411). Defect: Geisha's super zone (lingering damage zone 'GeishaStormVisibility', area effect data id
# 17000891, logic Radius 1500, drawn with the cyan ring and the blue floor) is labelled from its own sprites, so the bush
# patch at the lower right of the circle, drawn in front of the zone, cuts the yellow outline along the bush edges.
# The user's rule 7 (RULES.md): the zone is the full circle from the game data, cut ONLY by solid map objects (walls,
# crates, barrels); bushes, characters, UI and effects never cut it.
#
# Fix: the same rule as z_1027_s042.py (skin 042 Sandy's super zone, reviewed), keyed to this zone's logic name (every skin
# of Geisha uses the same zone, so the name, not the skin's did, selects it): the label is the exact projected logic circle
# (its ground points through idlabels._proj, filled polygon, scale 1.0) minus the solid map blocks (_zs_solid,
# z_1023_zone_solid: walls, crates, barrels, fences; Forest bushes, water, characters and UI are not solid) and minus the
# other labels (taken: projectiles, spawnables). Pinholes under ZS_HOLE px that are not solid and not another label are
# filled (_zs_fill_small); specks under 1 % of the largest part are dropped. The circle is used only once the zone's own
# sprites are labelled (the state item appears some ticks before the zone is drawn). The second zone of the same super
# ('GeishaStormDamage') has no drawn sprite and stays unlabelled as before.

S104_ZONE_NAME = 'GeishaStormVisibility'
S104_ZONE_SCALE = 1.0         # drawn edge = logic radius (checked against the cyan ring in the pairs 1691-1693)
S104_RING = 1440              # ground points of the logic circle projected to the screen
S104_SPECK = 0.01             # parts of the label under this share of its largest part are specks: dropped


def _s104_circle(cam, x, y, rz, H, W):
    """(H, W) bool: the exact screen polygon of the logic circle (centre x, y, radius rz, on the ground plane)"""
    th = np.linspace(0.0, 2.0 * np.pi, S104_RING, endpoint=False)
    pts = [tuple(_proj(cam, x + rz * np.cos(t), y + rz * np.sin(t))) for t in th]
    im = Image.new('L', (W, H), 0); ImageDraw.Draw(im).polygon(pts, fill=1)
    return np.asarray(im, bool)


def _s104_specks(part):
    from scipy.ndimage import label as cc
    L, n = cc(part)
    if n <= 1: return part
    sz = np.bincount(L.ravel())[1:]
    return np.isin(L, 1 + np.nonzero(sz >= S104_SPECK * sz.max())[0])


_s104_zone_labels_prev = zone_labels


def zone_labels(state, cam, R, taken, own_team=None):
    import owners
    out = _s104_zone_labels_prev(state, cam, R, taken, own_team)
    if R.get('wd') is None: return out
    H, W = R['id'][::2, ::2].shape
    solid = None
    for it in state.get('items', []):
        iid, did = int(it[0]), int(it[1])
        Z = owners.zone(did)
        if not Z or Z.get('Name') != S104_ZONE_NAME or owners.area(did)['kind'] != 'zone': continue
        if iid not in out: continue   # not drawn yet (the state item comes a few ticks before the zone's sprites): no label
        if solid is None: solid = _zs_solid(cam, R)
        x, y = int(it[3]), int(it[4]); rz = float(Z['logic_row'].get('Radius') or 300) * S104_ZONE_SCALE
        disc = _s104_circle(cam, x, y, rz, H, W)
        part = disc & ~solid & ~taken
        if part.any():
            part = _zs_fill_small(part, True, solid, taken)   # pinholes (not solid, not another label)
            part = _s104_specks(part)
        if part.any(): out[iid] = (part, did, int(it[2]))
        else: out.pop(iid, None)
    return out
