# human review 2026-10-28 fix for skin 032 (Haiku fix, reviewed by the coordinator, installed 10-29; source out/sweep2/haikutasks/human1028/s032/fix.py; tested via IDLABELS_OVERRIDE after all fixes incl. z_1099_cleanup, hence z_1100_)
# Fix s032 (human review 2026-10-28, task human1028/s032). The user's finding on the 'super 45 deg' row, tick 363:
# "the zone shouldnt be included here, only the pull label" (while the pull is active there must be NO zone label; the zone
# circle starts only when the final explosion starts).
#
# Cause (capture sw2_032, state items): from tick 360 the state holds two items at the same centre (6240, 2340), owner 0:
#   item 3000000  area BlackHoleUltiSuck      (the pull / violet vortex)   ticks 360-374, gone from 375
#   item 3000001  area BlackHoleUltiExplosion (the final zone, yellow)     ticks 360-388
# z_1028_tara.py labels the explosion as its yellow logic circle from 360, and the pull as the ring around it (the zone
# wins the overlap). So the yellow circle sat inside the pull while it was still a pull. The explosion is on screen only
# from 375: frames 373 show the vortex alone, 375 the spiky eye (the final explosion) with the yellow circle.
#
# Fix (a zone_labels wrapper): while a pull item stands at the same centre and team as an explosion item, that explosion item
# is left out of the state the zone_labels chain sees. Its zone label is then absent, and the pull keeps its full logic circle
# (z_1028's rule without the yellow carve-out: the disc cut only by solid map blocks). The yellow circle starts on the first
# tick the pull item is gone (375 on sw2_032). Nothing else is changed: other zone items, the throwable eye and the other
# projectiles pass through untouched.
#
# Types: the pull types BlackHoleUltiSuck (default 032, shared with 246 / 267 / 409-style rows), Tara004UltiSuck (303),
# Tara005UltiSuck (314), Tara006UltiSuck (361), Tara007UltiSuck (409), Tara008UltiSuck, BlackHoleOverchargedUltiSuck; the
# explosions BlackHoleUltiExplosion (shared row 17000157 = skins 032, 246, 267, 314, 361 ...), Tara004UltiExplosion (303),
# BlackHoleOverchargedUltiExplosion. The same name sets as z_1028_tara.py. No type id is changed here; only the zone label
# timing of these area rows. Skins that were not captured in this review are not verified.

S032_PULL = {'BlackHoleUltiSuck', 'Tara004UltiSuck', 'Tara005UltiSuck', 'Tara006UltiSuck', 'Tara007UltiSuck',
             'Tara008UltiSuck', 'BlackHoleOverchargedUltiSuck'}
S032_BOOM = {'BlackHoleUltiExplosion', 'Tara004UltiExplosion', 'BlackHoleOverchargedUltiExplosion'}
S032_CENTRE_TOL = 5          # logic units: the explosion and the pull are created at the same point (sw2_032: exact)

_s032_zone_prev = zone_labels


def zone_labels(state, cam, R, taken, own_team=None):
    import owners
    items = list(state.get('items', []))
    names = [(owners.area(int(it[1])) or {}).get('name') for it in items]
    pulls = [(int(it[2]), int(it[3]), int(it[4])) for it, nm in zip(items, names) if nm in S032_PULL]
    if not pulls: return _s032_zone_prev(state, cam, R, taken, own_team)
    keep = []
    for it, nm in zip(items, names):
        if nm in S032_BOOM and any(own == int(it[2]) and abs(x - int(it[3])) <= S032_CENTRE_TOL and abs(y - int(it[4])) <= S032_CENTRE_TOL
                                   for own, x, y in pulls):
            continue                         # the final explosion has not started: its pull is still alive at this centre
        keep.append(it)
    st = dict(state); st['items'] = keep
    return _s032_zone_prev(st, cam, R, taken, own_team)
