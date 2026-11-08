# sweep 10-27 fix for skin 314 (Haiku fix, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s314/)
# sweep 10-27 fix for skin 314 BlackHoleWorldfinal (Tara, super 45 deg), task s314 (Haiku fix agent, sweep 2026-10-27).
# Pairs 5887 (tick 363) and 5888 (tick 371), extra_label: a second wavy yellow outline around the cyan splash / beam
# inside the lingering zone circle, and a tiny yellow ring on a speck above the beam.
#
# Cause: the zone is the area row 'BlackHoleUltiExplosion' (area data id 17000157, logic zone radius 450), labelled from
# the zone's own sprites that are the front-most draw (zone_labels). The cyan splash and the beam are effect sprites drawn
# over the zone, so their pixels are not zone sprites: they leave holes in the zone label, and the outline of each hole
# is the wavy / ring outline seen on the page. The zone's own outer edge is right (the logic circle, cut by the building and
# the crate at its edge, which open to the outside of the circle). Rule 7: zones are cut ONLY by solid map objects;
# effects never cut them.
#
# Fix (a wrapper of zone_labels, zones of this row only; the outer edge is unchanged):
#  1. holes of the zone label (closed regions of the label's complement, enclosed by the label) are filled where the
#     front-most draw is an effect: a blended draw that writes no depth (blend 1, dmask 0, the draw flags of R['draws']).
#     Measured on sw2_314 ticks 363 and 371: every hole pixel is such a draw (the splash, the beam, the speck rings), while
#     a solid object (walls, crates: opaque, depth-writing, blend 0 dmask 1) keeps its hole. The zone's own draw is blend 1
#     dmask 0 too. (A depth threshold against the ground was tried first and rejected: the splash is only about 0.04 units
#     nearer than the ground, so half of its pixels failed it.) Notches open to the outside (the building, the crate at the
#     edge) are not holes and stay.
#  2. specks: parts of the zone label that are smaller than 25 px and under 10 % of its largest part are dropped (the same
#     rule as the projectile clean-up; the 1-px ring centre at tick 371 and the 7-px bits inside the splash at tick 363).
#     Never drop the whole label: if the cut would keep less than half of it, the label stays as it was.
# The row is shared by the BlackHole skins that have no own explosion row (Tara 032, 246, 267, 314; see
# z_1027_s032.py). Their zones get the same hole fill. Skins with an own row (Tara004UltiExplosion, the overcharged
# BlackHoleOverchargedUltiExplosion) are not touched.

from scipy.ndimage import binary_fill_holes as _s314_fill, label as _s314_cc

_S314_ZONE_ROWS = {17000157}   # area data ids of the 'BlackHoleUltiExplosion' row (the zone of Tara's super)
_S314_SPECK = 25               # px
_S314_SPECK_FRAC = 0.1


def _s314_zone_clean(out, state, cam, R, taken):
    ids_h = R['id'][::2, ::2]
    # effect draws: blended, writing no depth (sprites of the splash, the beam, the speck rings; the zone's own sprites too)
    eff = [d for d, dr in R['draws'].items() if dr.get('blend') == 1 and dr.get('dmask') == 0]
    eff_px = np.isin(ids_h, eff)
    for iid, (m, did, own) in list(out.items()):
        if did not in _S314_ZONE_ROWS or not m.any(): continue
        holes = _s314_fill(m) & ~m                                  # enclosed by the zone's label: the effects over it
        m2 = m | (holes & eff_px & ~taken)                          # solid objects (opaque, depth-writing draws) keep their holes
        L, n = _s314_cc(m2)
        if n > 1:
            sz = np.bincount(L.ravel())[1:]
            keep = 1 + np.nonzero(~((sz < _S314_SPECK) & (sz < _S314_SPECK_FRAC * sz.max())))[0]
            m2 = np.isin(L, keep)
        if m2.sum() < 0.5 * m.sum(): continue                       # never lose the zone: as it was
        out[iid] = (m2 & ~taken, did, own)
    return out


_s314_zone_labels_prev = zone_labels


def zone_labels(state, cam, R, taken, own_team=None):
    out = _s314_zone_labels_prev(state, cam, R, taken, own_team)
    return _s314_zone_clean(out, state, cam, R, taken)
