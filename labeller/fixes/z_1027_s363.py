# sweep 10-27 fix for skin 363 (Haiku fix, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s363/)
# Skin 363 SandstormDeepsea (Sandy), task s363 (Haiku fix, sweep 2026-10-27; pairs 6833, 6834 at attack 90 deg ticks 309 and
# 311, pairs 6840, 6841, 6842 at super 45 deg ticks 371, 389 and 411). Confirmed defects and the fix:
#
# 1) Attack 90 deg (and attack 0 deg, the same projectile): Sandy006Projectile (type 6000676, a ZONE type: its label is one solid
#    region over the cloud) has no body clip; its label is made of its trail sprites. Its trail clip
#    sandy_006_atk_trail_front_wave_02 / _03 draws the pale translucent smoke wisps under and beside the clouds, and ZONE types
#    skip the TRAIL_FX name filter, so the smoke was labelled (the user: smoke is never labelled). Fix: skip the front-wave trail
#    clips of this type (TRAIL_SKIP, name prefix). Checked per clip (probe panels): the clouds are not drawn by these clips, so
#    the cloud labels are unchanged; the attack 0 and attack 90 ticks with labels (263-271, 305-311) lose only the smoke, and the
#    white light flash beside the cloud at tick 305 no longer pulls a spur into the label.
#
# 2) Super 45 deg: the lingering zone Sandy_006_ulti (area effect data id 17000222, logic Sandy_def_ulti-style zone: Radius
#    2000, the same as Sandy_def_ulti and Sandy_003_ulti) is labelled from its own sprites, so the bushes drawn in front of the
#    zone cut the yellow outline along their edges (top-left and bottom-right bushes, ticks 371, 389, 411). The user's rule 7:
#    the zone is the full logic circle, cut ONLY by solid map objects (walls, crates, barrels); bushes, characters, UI and
#    effects never cut it. Fix: the same rule as z_1027_s104.py (Geisha's super zone) and z_1023_sandy270.py, keyed to this
#    zone's logic name: the label is the exact projected logic circle (ground points through idlabels._proj, scale 1.0 =
#    the drawn edge) minus the solid map blocks (_zs_solid, z_1023_zone_solid: walls, crates, barrels; Forest bushes, water,
#    characters and UI are not solid) and minus the other labels (taken: projectiles, spawnables). Pinholes under ZS_HOLE px
#    that are not solid and not another label are filled; specks under 1 % of the largest part are dropped. The circle is used
#    only where the zone's own sprites are labelled (iid in out), so the zone keeps its earlier start (the state item comes a
#    few ticks before the zone is drawn).
#
# Types and data touched: type 6000676 (Sandy006Projectile, trail clips only) and zone 'Sandy_006_ulti' (area effect data id
# 17000222, selected by logic name). The owner tables reference both only from their own rows (no other skin of this sweep uses
# them); the change is limited to these two.

# ---- 1) attack: the smoke trail clips of Sandy006Projectile are not the attack ----
S363_PROJ = 6000676
S363_SMOKE = ('sandy_006_atk_trail_front_wave',)   # _01, _02, _03: the translucent smoke wisps
TRAIL_SKIP[S363_PROJ] = S363_SMOKE

# ---- 2) super: Sandy_006_ulti is the full logic circle minus solid map blocks ----
S363_ZONE_NAME = 'Sandy_006_ulti'
S363_ZONE_SCALE = 1.0         # drawn edge = logic radius (the same as Sandy_def_ulti and Sandy_003_ulti; checked in the pairs)
S363_RING = 1440              # ground points of the logic circle projected to the screen
S363_SPECK = 0.01             # parts of the label under this share of its largest part are specks: dropped


def _s363_circle(cam, x, y, rz, H, W):
    """(H, W) bool: the exact screen polygon of the logic circle (centre x, y, radius rz, on the ground plane)"""
    th = np.linspace(0.0, 2.0 * np.pi, S363_RING, endpoint=False)
    pts = [tuple(_proj(cam, x + rz * np.cos(t), y + rz * np.sin(t))) for t in th]
    im = Image.new('L', (W, H), 0); ImageDraw.Draw(im).polygon(pts, fill=1)
    return np.asarray(im, bool)


def _s363_specks(part):
    from scipy.ndimage import label as cc
    L, n = cc(part)
    if n <= 1: return part
    sz = np.bincount(L.ravel())[1:]
    return np.isin(L, 1 + np.nonzero(sz >= S363_SPECK * sz.max())[0])


_s363_zone_labels_prev = zone_labels


def zone_labels(state, cam, R, taken, own_team=None):
    import owners
    out = _s363_zone_labels_prev(state, cam, R, taken, own_team)
    if R.get('wd') is None: return out
    H, W = R['id'][::2, ::2].shape
    solid = None
    for it in state.get('items', []):
        iid, did = int(it[0]), int(it[1])
        Z = owners.zone(did)
        if not Z or Z.get('Name') != S363_ZONE_NAME or owners.area(did)['kind'] != 'zone': continue
        if iid not in out: continue   # not drawn yet (the state item comes a few ticks before the zone's sprites): no label
        if solid is None: solid = _zs_solid(cam, R)
        x, y = int(it[3]), int(it[4]); rz = float(Z['logic_row'].get('Radius') or 300) * S363_ZONE_SCALE
        disc = _s363_circle(cam, x, y, rz, H, W)
        part = disc & ~solid & ~taken
        if part.any():
            part = _zs_fill_small(part, True, solid, taken)   # pinholes (not solid, not another label)
            part = _s363_specks(part)
        if part.any(): out[iid] = (part, did, int(it[2]))
        else: out.pop(iid, None)
    return out
