# away_1104 / zones: fixes for the OWNER's final-review complaints (2026-11-03/04) on five super-zone / turret pages.
# Exec'd in idlabels' namespace after every installed fix (IDLABELS_OVERRIDE). Global names prefixed _ab4z_.
#
# COMMON CAUSE (032, 054, 102, 199): the zone discs are drawn as an AXIS-ALIGNED ELLIPSE about the projected centre, whose
# vertical radius is the NEAR-side radius (|proj(y + r) - cy|). The true ground circle projects to a slightly different curve
# (perspective): the far (upper) half is smaller. Measured on sw2_032 (r = 1200): near radius 134 px, far radius 123 px; the
# ellipse is 11 px too high at the top, 4 % too big in area, and the projected circle sticks out of the ellipse (583 px) along
# the lower flanks. The visible rings follow the PROJECTED GROUND CIRCLE: measured on the frame (steepest drop of the ring's
# outer edge along rays of the projected circle, ticks 363-411, 5-7 directions, constant over time):
#     BlackHoleUltiSuck (032 pull)           0.997 x logic radius     -> 1.0
#     ChronomancerUltiTimebend (102)         0.979                     -> 0.98
#     Chronomancer002UltiTimebend (199)      0.998                     -> 1.0
#     IceDudeUltiArea (054, the ice dome)    1.082 (dome rim, then a soft glow to 1.18) -> 1.08
# So after the installed chain, a zone label of these rows is cut to the projected ground circle of its visible radius
# (and the thin strips of that circle outside the old ellipse are added where nothing solid / no other label is).
#
# 032 (owner: "circle a little too big", super 45, tick 373): the pull's ring (BlackHoleUltiSuck, the vortex) is trimmed to
#     the projected circle. The final explosion disc (BlackHoleUltiExplosion) is not touched.
# 054 (owner: tick 411 "too big"): the Lou super's ice dome (IceDudeUltiArea, Radius 1100) was a sprite-derived label that
#     follows the cyan puddles and the soft outer glow outside the dome's rim (up to ~5 px outside at 411, ragged). Now the
#     zone is a ZONE_DISC row (the logic circle at the dome's visible radius 1.08, cut only by solid blocks like every other
#     disc zone: z_1023 / z_1028_zone184), then trimmed to the projected circle.
# 102 (owner: ticks 371 / 389 / 411 "too big"): the ring label reached ~10 px above the ring's top edge: the ellipse. Trimmed
#     to the projected circle at 0.98.
# 199 (owner: ticks 371 / 389 / 411 "too small / cut off", at the bottom): besides the ellipse, the label was cut flat ~12 px
#     above the low fence row at the bottom (tile code 13, the grey arches): z_1028_zone184 models every fence as a 590-unit
#     box (marked "unverified" in its header); the real fence top is ~310 units (its drawn top edge, measured: the box's far
#     top edge at z = 310 projects to the row where the arches' top is drawn; at 590 it projects 12 px higher and cut the ring
#     above the fence). For the two Chronomancer rows above only, the fence box is set to (310, 20) while the chain runs.
#     (Same map in 102: its bottom is cut the same way, now the label reaches the fence's top edge there too.)
#
# 045 (owner: super 45, ticks 371 / 389 / 411 "too big"; earlier, 10-28: "only the fully visible circle should be labeled, not
#     that extra part on the bottom"): the turret (SpawnerDudeTurret 16000173) of the DEFAULT SpawnerDude skin is a gold ring
#     on a dark slab; the ruling of 11-02 (whole model, no shadow) left the dark slab / ground shadow below-right of the ring
#     in the label (z_1107_turret: "the 045 ground shadow stays in the label" because a plain dark-blob cut took a chord
#     through the ring). Cut here with GEOMETRY: the dark model pixels that lie below the turret's footprint ellipse (the
#     120-unit collision circle on the ground = the gold ring's outer edge, measured on sw2_045) and outside it (2 px rim for
#     the ring's black outline). The upper parts (the dome and the knob at 411) stay. Only the skin whose super uses the
#     default SpawnerDude projectiles (types 6000695 / 6000696 / 6000697, seen earlier in the same capture: the attack rows come
#     first) is changed: the hex (291), barbell (439) and flower (369) turrets keep their accepted whole-model labels.
#
# Types / rows touched: BlackHoleUltiSuck (shared by the Tara skins whose pull has no own row), IceDudeUltiArea,
# ChronomancerUltiTimebend (102), Chronomancer002UltiTimebend (199), character 16000173 (turret of default-projectile skins).

import numpy as _ab4z_np
from PIL import Image as _ab4z_Img, ImageDraw as _ab4z_Draw
from scipy.ndimage import binary_dilation as _ab4z_dil, label as _ab4z_cc, binary_fill_holes as _ab4z_fill

_AB4Z_SCALE = {'BlackHoleUltiSuck': 1.0, 'ChronomancerUltiTimebend': 0.98, 'Chronomancer002UltiTimebend': 1.0,
               'IceDudeUltiArea': 1.08}
_AB4Z_FENCE_BOX = {'ChronomancerUltiTimebend', 'Chronomancer002UltiTimebend', 'IceDudeUltiArea'}
_AB4Z_FENCE = (310.0, 20.0)            # tile code 13: (top, footprint inset)
_AB4Z_FENCE_CODE = 13

# 054: the ice dome is a disc zone from now on (its circle at the dome's visible radius)
ZONE_DISC.add('IceDudeUltiArea')
A3_DISC_SCALE['IceDudeUltiArea'] = _AB4Z_SCALE['IceDudeUltiArea']


def _ab4z_poly(cam, x, y, rad, shape):
    """(H, W) bool: the ground circle (logic centre x, y, radius rad, z = 0) projected with the full perspective"""
    ang = _ab4z_np.linspace(0, 2 * _ab4z_np.pi, 360, endpoint=False)
    pts = [tuple(_proj(cam, x + rad * _ab4z_np.cos(a), y + rad * _ab4z_np.sin(a), 0.0)) for a in ang]
    im = _ab4z_Img.new('L', (shape[1], shape[0]), 0); _ab4z_Draw.Draw(im).polygon(pts, fill=1)
    return _ab4z_np.array(im, bool)


def _ab4z_ellipse(cam, x, y, rad, shape):
    """the (axis-aligned) ellipse the installed zone code uses"""
    H, W = shape; yy, xx = _ab4z_np.mgrid[0:H, 0:W].astype(_ab4z_np.float64)
    cx, cy = _proj(cam, x, y); dx = abs(_proj(cam, x + rad, y)[0] - cx); dy = abs(_proj(cam, x, y + rad)[1] - cy)
    return ((xx - cx) / dx) ** 2 + ((yy - cy) / dy) ** 2 <= 1


_ab4z_prev_zone_labels = zone_labels


def zone_labels(state, cam, R, taken, own_team=None):
    import owners
    items = {}
    for it in state.get('items', []):
        Z = owners.zone(int(it[1]))
        if Z and Z.get('Name') in _AB4Z_SCALE: items[int(it[0])] = (Z.get('Name'), Z, int(it[3]), int(it[4]))
    if not items: return _ab4z_prev_zone_labels(state, cam, R, taken, own_team)
    old_fence = Z184_BOX.get(_AB4Z_FENCE_CODE)
    if any(v[0] in _AB4Z_FENCE_BOX for v in items.values()): Z184_BOX[_AB4Z_FENCE_CODE] = _AB4Z_FENCE
    try:
        out = _ab4z_prev_zone_labels(state, cam, R, taken, own_team)
    finally:
        if old_fence is not None: Z184_BOX[_AB4Z_FENCE_CODE] = old_fence
    if R.get('wd') is None: return out
    solid = None
    for iid, (nm, Z, x, y) in items.items():
        if iid not in out: continue
        m, did, own = out[iid]
        rad = float(Z['logic_row'].get('Radius') or 300) * _AB4Z_SCALE[nm]
        poly = _ab4z_poly(cam, x, y, rad, m.shape)
        ell = _ab4z_ellipse(cam, x, y, rad, m.shape)
        if solid is None: solid = _zs_solid(cam, R)
        others = _ab4z_np.zeros_like(m)
        for k, v in out.items():
            if k != iid: others |= v[0]
        strip = poly & ~ell & ~solid & ~taken & ~others & _ab4z_dil(m, iterations=3)
        new = (m & poly) | strip
        if new.any(): out[iid] = (new, did, own)
        else: out.pop(iid, None)
    return out


# ---- 045: the dark slab / shadow below the turret's footprint ------------------------------------------------------------
_AB4Z_DEFAULT_PROJ = {6000695, 6000696, 6000697}        # SpawnerDudeProjectile / IndirectProjectile / UltiProjectile (default skin)
_AB4Z_SEEN = {'last': -1, 'default': False}
_AB4Z_TURRET = 16000173
_AB4Z_RIM = 2           # px: the footprint ellipse is grown by this (the ring's black outline)
_AB4Z_DARK = 120        # the slab / shadow pixels: all channels below this (the gold ring, the blue dome and the white
                        # trim are brighter; the ring's black outline lies inside the grown ellipse)
_ab4z_prev_labels = labels


def labels(objs, cam, R, state=None, ended=(), img=None):
    P = _ab4z_prev_labels(objs, cam, R, state=state, ended=ended, img=img)
    if state is None: return P
    tick = int(state.get('tick', 0))
    if tick < _AB4Z_SEEN['last']: _AB4Z_SEEN['default'] = False       # frames go back: a new capture
    _AB4Z_SEEN['last'] = tick
    if any(o['t'] in _AB4Z_DEFAULT_PROJ for o in objs): _AB4Z_SEEN['default'] = True
    if not _AB4Z_SEEN['default'] or img is None: return P
    ids, lids, share = P['ids'], P['lids'], P['share']
    for c in state['chars']:
        cid, did = int(c[0]), int(c[1])
        if did != _AB4Z_TURRET: continue
        ks = [k for k, l in enumerate(lids) if l[0] == cid and l[2] == 'spawn']
        if not ks: continue
        k = ks[0]; m = ids == k
        if not m.any(): continue
        dark = img.max(-1) < _AB4Z_DARK
        bright = m & ~dark
        if bright.sum() < 30: continue
        # the turret's lit part (gold ring, its inside, the dome): its convex hull grown by the outline's width is the
        # visible circle; the dark slab / ground shadow hangs below it (the footprint ellipse overshoots the ring by ~2 px
        # at the bottom, so the ellipse itself cannot separate the slab)
        E = _ab4z_dil(hull_mask(bright), iterations=_AB4Z_RIM)
        cy = _ab4z_np.nonzero(bright)[0].mean()
        yy = _ab4z_np.arange(ids.shape[0])[:, None]
        cut = m & ~E & (yy > cy) & dark
        if not cut.any(): continue
        new = m & ~cut
        L, n = _ab4z_cc(new)
        if n > 1:
            sz = _ab4z_np.bincount(L.ravel())[1:]
            new = _ab4z_np.isin(L, 1 + _ab4z_np.nonzero(sz >= 0.15 * sz.max())[0])
        gone = m & ~new
        ids[gone] = -1; share[gone] = 0
    P['ids'], P['share'] = ids, share
    return P
