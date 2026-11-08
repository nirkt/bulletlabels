# sweep 10-27 fix for skin 042 (Haiku fix, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s042/)
# Task s042 (Haiku, sweep 2026-10-27): skin 042 SandstormDefault (Sandy), its super's lingering zone Sandy_def_ulti (area
# effect data id 17000217; logic Sandy_def_ulti: Radius 2000, TimeMs 9000). Confirmed defects (sweep_1027, pairs 798-800,
# ticks 371, 389, 411): the zone's label is the sprite set (the zone's own sprites where they are the front-most draw), so the
# green bush patch at the lower right, drawn in front of the zone, is a hole and the outline runs along the bush edges.
# The user's rule 7 (RULES.md): the zone is the full circle from the game data, cut ONLY by solid map objects (walls, crates,
# barrels); bushes, characters, UI and effects never cut it.
#
# Fix: the same rule as z_1023_sandy270.py (skin 270 Sandy lantern, task 6, reviewed), keyed to this zone's did and name:
# the label is the exact projected logic circle (its ground points through idlabels._proj, filled polygon, scale 1.0) minus
# the solid map blocks (_zs_solid, z_1023_zone_solid) and minus the other labels (taken: projectiles, spawnables). Pinholes
# under ZS_HOLE px that are not solid and not another label are filled (_zs_fill_small); specks under 1 % of the largest part
# are dropped. Not done through ZONE_DISC: its axis-aligned ellipse is off-centre under perspective (see task 6).
# Only the zone with did 17000217 and logic name Sandy_def_ulti is changed; every other zone keeps the earlier rules.

S042_ZONE_DID = 17000217      # area_effects_skin row of skin 042's super zone
S042_ZONE_NAME = 'Sandy_def_ulti'
S042_ZONE_SCALE = 1.0         # drawn edge = logic radius (checked on the opaque zone sprites, see report)
S042_RING = 1440              # ground points of the logic circle projected to the screen
S042_SPECK = 0.01             # parts of the label under this share of its largest part are specks: dropped


def _s042_circle(cam, x, y, rz, H, W):
    """(H, W) bool: the exact screen polygon of the logic circle (centre x, y, radius rz, on the ground plane)"""
    th = np.linspace(0.0, 2.0 * np.pi, S042_RING, endpoint=False)
    pts = [tuple(_proj(cam, x + rz * np.cos(t), y + rz * np.sin(t))) for t in th]
    im = Image.new('L', (W, H), 0); ImageDraw.Draw(im).polygon(pts, fill=1)
    return np.asarray(im, bool)


def _s042_specks(part):
    from scipy.ndimage import label as cc
    L, n = cc(part)
    if n <= 1: return part
    sz = np.bincount(L.ravel())[1:]
    return np.isin(L, 1 + np.nonzero(sz >= S042_SPECK * sz.max())[0])


_s042_zone_labels_prev = zone_labels


def zone_labels(state, cam, R, taken, own_team=None):
    import owners
    out = _s042_zone_labels_prev(state, cam, R, taken, own_team)
    if R.get('wd') is None: return out
    H, W = R['id'][::2, ::2].shape
    solid = None
    for it in state.get('items', []):
        iid, did = int(it[0]), int(it[1])
        if did != S042_ZONE_DID: continue
        Z = owners.zone(did)
        if not Z or Z.get('Name') != S042_ZONE_NAME or owners.area(did)['kind'] != 'zone': continue
        if solid is None: solid = _zs_solid(cam, R)
        x, y = int(it[3]), int(it[4]); rz = float(Z['logic_row'].get('Radius') or 300) * S042_ZONE_SCALE
        disc = _s042_circle(cam, x, y, rz, H, W)
        part = disc & ~solid & ~taken
        if part.any():
            part = _zs_fill_small(part, True, solid, taken)   # pinholes (not solid, not another label)
            part = _s042_specks(part)
        if part.any(): out[iid] = (part, did, int(it[2]))
        else: out.pop(iid, None)
    return out
