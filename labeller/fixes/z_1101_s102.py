# hosted review 2026-10-29 fix (Haiku fix, reviewed by the coordinator, installed 10-29 LOCAL ONLY; source out/sweep2/haikutasks/site1029/s102/fix.py; tested via IDLABELS_OVERRIDE after all fixes incl. z_1100_*, hence z_1101_)
# Skin 102 (ChronomancerDefault, hosted review 2026-10-29): the Chronomancer super's time-bend ring (area zone, the ZONE_DISC
# names below) was flagged on frames 371, 389, 411 (super 45; the user's note: no reason given). Its label is the ring's
# logic disc minus whatever is drawn in front of the ground (depth test in zone_labels). Two kinds of thin protrusions
# are in the label on every ring frame (ticks 357-411):
#   1. a spur at the top right: the lit left edge of the lavender wall block (base on the rim) passes the depth test as a
#      3 px strip and sticks up above the wall's base;
#   2. fingers at the bottom: 2-5 px wide strips of the label reach 15-20 px down into the gaps between the bot and the
#      barrel (ticks 371-411; the fingers lie on the ring's bright bars).
# The wide notch where the wall covers the rim is kept: the wall tile (22, 5) stands on the disc (centre 890 < radius 900).
# Fix: opening of the zone mask with a 7x7 square (parts narrower than 7 px go; the disc boundary, the wall notch and the
# bush/crate cut are wide and stay). An opening removes nothing from a wide part and adds nothing, so no label grows.
# Only the four ZONE_DISC names of the time-bend ring are touched; other Chronomancer skins use the same area names and
# are re-checked by the coordinator.
from scipy.ndimage import binary_opening as _s102_open

_S102_TIMEBEND = {'ChronomancerUltiTimebend', 'ChronomancerOverchargedUltiTimebend', 'Chronomancer002UltiTimebend',
                  'Chronomancer003UltiTimebend'}
_S102_S7 = np.ones((7, 7), bool)
_s102_prev_zone_labels = zone_labels


def zone_labels(state, cam, R, taken, own_team=None):
    import owners
    out = _s102_prev_zone_labels(state, cam, R, taken, own_team)
    for iid, (m, did, own) in list(out.items()):
        Z = owners.zone(did)
        if not Z or Z.get('Name') not in _S102_TIMEBEND or not m.any(): continue
        p = 4
        c = _s102_open(np.pad(m, p), structure=_S102_S7)[p:-p, p:-p] & m
        if c.any(): out[iid] = (c, did, own)
    return out
