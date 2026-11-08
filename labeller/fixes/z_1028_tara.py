# Tara's super as several classes (task q1028/tara, the user's answer 10-28, verbatim: "the super should have multiple
# classes. the eyeball should be a throwable that does no damage, the pull should be labeled as pull and the final super
# should use the circle logic we talked about earlier."). Covers ALL Tara / BlackHole super rows (each skin's own rows,
# and the shared rows of the skins that have none):
#
# 1. The flying eyeball (the super projectile of each Tara skin: 6000412 default = skin 032, 6000414 Tara002 = 246,
#    6000416 Tara003 = 267, 6000418 Tara004 = 303, 6000420 Tara005 = 314, 6000422 Tara006 = 361, 6000424 Tara007 = 409,
#    6000426 Tara008, 6001520 overcharged) = kind 'throwable' (meta kind; pink in the sheet). Its label is unchanged:
#    the drawn eye while the projectile's logic is alive (flight), nothing after it has landed.
# 2. The pull (the vortex; area rows BlackHoleUltiSuck = default and 246 / 267 / 409-style shared, and each skin's own:
#    Tara004UltiSuck, Tara005UltiSuck, Tara006UltiSuck, Tara007UltiSuck, Tara008UltiSuck, BlackHoleOverchargedUltiSuck) =
#    kind 'pull' (violet). Label = its LOGIC circle (position + Radius 1200, scale 1.0), from its birth to its end, cut only by
#    solid map blocks (walls, crates, barrels: z_1023_zone_solid._zs_solid). Characters, bushes, UI and effects never cut it.
#    The pull and the final zone start on the same tick at the same centre: the zone's disc is the zone's label (yellow)
#    and the pull's label is the ring around it (one owner per pixel: the zone wins where the two circles overlap).
# 3. The final super damage zone (area rows BlackHoleUltiExplosion = shared, Tara004UltiExplosion, BlackHoleOverchargedUltiExplosion)
#    = kind 'zone' (already the zone kind: 1400 ms) with the circle rule: its logic circle (Radius 450) from its birth to
#    its end (the state item), cut only by solid map blocks. Not the sprites: the drawn puff / smoke and the spiky eye
#    graphic are inside the circle and are not labelled separately.
#
# Measured on sw2_032 (default: skin scale 235 for the pull) and sw2_409 (Tara007: skin scale 200): the logic circle of the
# pull is 154 x 134 frame px (Radius 1200); the explosion circle 58 x 49 px (Radius 450; the spiky eye of the zone fills it).
# The overcharged rows are covered by the same rule but have no capture here (not checked).
#
# Timeline (sw2_032 super, ticks; the state items): eye in flight 344-359 (throwable); landing at 360 (the pull and the zone
# are both created at the landing point); pull 360-374 (ring, violet); zone 360-388 (disc, yellow); after 375 the zone disc
# only (the spiky eye graphic = the zone's own drawing); nothing from 389 (both logics ended).
#
# Implementation: a meta wrapper (kinds), and a zone_labels wrapper (the circle labels of the pull and the explosion rows,
# written over whatever the earlier zone code gave those items; other zones are kept, their pixels are not taken).

import numpy as np

TARA_THROW_IDS = {6000412, 6000414, 6000416, 6000418, 6000420, 6000422, 6000424, 6000426, 6001520}
TARA_PULL = {'BlackHoleUltiSuck', 'Tara004UltiSuck', 'Tara005UltiSuck', 'Tara006UltiSuck', 'Tara007UltiSuck',
             'Tara008UltiSuck', 'BlackHoleOverchargedUltiSuck'}
TARA_BOOM = {'BlackHoleUltiExplosion', 'Tara004UltiExplosion', 'BlackHoleOverchargedUltiExplosion'}
TARA_SCALE = 1.0      # the logic circle (the user's circle rule); the drawn size differs by skin, see the report


_tara_meta_prev = meta


def meta(objs, state, P):
    out = _tara_meta_prev(objs, state, P)
    import owners, bulletlabels as BL
    for o in objs:
        if o['t'] in TARA_THROW_IDS:
            for k in (BL._lid(o), o['id']):
                if isinstance(out.get(k), dict) and out[k].get('kind') == 'super':
                    out[k] = dict(out[k], kind='throwable')
    for k, (did, own) in P.get('zones', {}).items():
        if (owners.area(did) or {}).get('name') in TARA_PULL and isinstance(out.get(k), dict):
            out[k] = dict(out[k], kind='pull')
    return out


_tara_zone_prev = zone_labels


def zone_labels(state, cam, R, taken, own_team=None):
    out = _tara_zone_prev(state, cam, R, taken, own_team)
    if R.get('wd') is None: return out
    import owners
    items = []
    for it in state.get('items', []):
        iid, did = int(it[0]), int(it[1]); Z = owners.zone(did)
        if not Z: continue
        nm = Z.get('Name')
        if nm not in TARA_PULL and nm not in TARA_BOOM: continue
        rz = float(Z['logic_row'].get('Radius') or 0) * TARA_SCALE
        if rz <= 0: continue
        x, y = int(it[3]), int(it[4])
        cx, cy = _proj(cam, x, y); dx = abs(_proj(cam, x + rz, y)[0] - cx); dy = abs(_proj(cam, x, y + rz)[1] - cy)
        if dx < 1 or dy < 1: continue
        items.append((iid, did, int(it[2]), nm, cx, cy, dx, dy))
    if not items: return out
    H, W = R['wd'].shape
    mine = {it[0] for it in items}
    others = np.zeros((H, W), bool)                       # pixels of the other zones' labels: not taken here
    for k, v in out.items():
        if k not in mine: others |= v[0]
    solid = _zs_solid(cam, R)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    disc = {}
    for (iid, did, own, nm, cx, cy, dx, dy) in items:
        disc[iid] = ((xx - cx) / dx) ** 2 + ((yy - cy) / dy) ** 2 <= 1
    booms = np.zeros((H, W), bool)                        # the zone discs of this shot: they win over the pull's ring
    for (iid, did, own, nm, cx, cy, dx, dy) in items:
        if nm in TARA_BOOM: booms |= disc[iid]
    blocked = taken | others | solid
    for (iid, did, own, nm, cx, cy, dx, dy) in items:
        if nm in TARA_BOOM:                               # the final damage zone: its full circle, cut only by solid blocks
            part = disc[iid] & ~blocked
            part = _zs_fill_small(part, disc[iid], solid, taken | others)
        else:                                             # the pull: its circle minus the zone discs, cut only by solid blocks
            ring = disc[iid] & ~booms
            part = ring & ~blocked
            part = _zs_fill_small(part, ring, solid, taken | others)
        if part.any(): out[iid] = (part, did, own)
        else: out.pop(iid, None)
    return out
