# sweep 10-27 fix for skin 099 (Haiku fix, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s099/)
# sweep 10-27 fix for skin 099 MeepleDefault (Meeple), task s099 (Haiku fix, sweep 2026-10-27; task folder out/sweep2/haikutasks/sweep1027/s099/)
# Confirmed defects (sweep_1027, pairs 1623-1625, ticks 371, 389, 411, row 'sw2_099 super 45 deg'): the yellow zone of Meeple's
# super (area effect data id 17000812, logic MeepleSuperArea: Radius 1000) has a thin neck and small closed loops wrapped around
# cyan arrow sparks that lie outside the logic circle (the right edge at tick 371, a detached loop beside the brawler at 389, a
# loop above the zone at 411). The label is the zone's sprite set inside a search region of 1.15 x radius + 150 (zone_labels), so
# the arrow sprites just outside the circle are labelled. The cleanup step (z_1099) does not touch zone labels.
#
# The user's rule 7 (RULES.md): the zone is the full circle from the game data, cut ONLY by solid map objects (walls, crates,
# barrels); bushes, characters, UI and effects never cut it.
#
# Fix: the same rule as z_1027_s042.py (Sandy's super zone, reviewed), keyed to this zone's logic name: the label is the exact
# projected logic circle (its ground points through idlabels._proj, filled polygon) minus the solid map blocks (_zs_solid, see
# z_1023_zone_solid) and minus the other labels (taken: projectiles, spawnables). Pinholes under ZS_HOLE px that are not solid
# and not another label are filled (_zs_fill_small); specks under 1 % of the largest part are dropped.
# Only zones named MeepleSuperArea change; every other zone keeps the earlier rules.

S099_ZONE_NAME = 'MeepleSuperArea'   # logic name of Meeple's super zone (area effect data id 17000812 in skin 099)
S099_ZONE_SCALE = 1.0                 # drawn edge = logic radius (checked by eye on ticks 359-411 of compare.png: the outline runs along the dark disc edge)
S099_RING = 1440                      # ground points of the logic circle projected to the screen
S099_SPECK = 0.01                     # parts of the label under this share of its largest part are specks: dropped
S099_MIN_PX = 5000                    # half-resolution px of the zone's own sprites before the circle is labelled


def _s099_circle(cam, x, y, rz, H, W):
    """(H, W) bool: the exact screen polygon of the logic circle (centre x, y, radius rz, on the ground plane)"""
    th = np.linspace(0.0, 2.0 * np.pi, S099_RING, endpoint=False)
    pts = [tuple(_proj(cam, x + rz * np.cos(t), y + rz * np.sin(t))) for t in th]
    im = Image.new('L', (W, H), 0); ImageDraw.Draw(im).polygon(pts, fill=1)
    return np.asarray(im, bool)


def _s099_specks(part):
    from scipy.ndimage import label as cc
    L, n = cc(part)
    if n <= 1: return part
    sz = np.bincount(L.ravel())[1:]
    return np.isin(L, 1 + np.nonzero(sz >= S099_SPECK * sz.max())[0])


_s099_zone_labels_prev = zone_labels


def zone_labels(state, cam, R, taken, own_team=None):
    import owners
    out = _s099_zone_labels_prev(state, cam, R, taken, own_team)
    if R.get('wd') is None: return out
    H, W = R['id'][::2, ::2].shape
    solid = None
    for it in state.get('items', []):
        iid, did = int(it[0]), int(it[1])
        Z = owners.zone(did)
        if not Z or Z.get('Name') != S099_ZONE_NAME or owners.area(did)['kind'] != 'zone': continue
        # the zone's body is not drawn yet (its first ticks: only a few arrow sprites, 500 px at tick 357 against 37000 px from
        # tick 359 on, half resolution): no circle, as before
        if iid not in out or int(out[iid][0].sum()) < S099_MIN_PX: continue
        if solid is None: solid = _zs_solid(cam, R)
        x, y = int(it[3]), int(it[4]); rz = float(Z['logic_row'].get('Radius') or 300) * S099_ZONE_SCALE
        disc = _s099_circle(cam, x, y, rz, H, W)
        part = disc & ~solid & ~taken
        if part.any():
            part = _zs_fill_small(part, True, solid, taken)   # pinholes (not solid, not another label)
            part = _s099_specks(part)
        if part.any(): out[iid] = (part, did, int(it[2]))
        else: out.pop(iid, None)
    return out
