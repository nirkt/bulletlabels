# sweep 10-27 fix for skin 048 (Haiku fix, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s048/)
# Sweep 2026-10-27, skin 048 BlowerDefault, pair 892 (super 45 deg, tick 351): hole in the label of the Blower super.
# Type 6000732 = BlowerUltiProjectile (default clip gale_def_ulti_projectile): the super is a wave of translucent puffs.
# Cause: the gaps between the puffs show the floor through the cloud. Enclosed by the wave's body they stay unlabelled, so the
# label has holes: a 75 px gap (about 10 x 12 px) at tick 351, a 59 px gap at tick 353, and 1-8 px gaps at 345-353.
# The shared clean-up (z_1099) fills holes of at most 40 px only, so the 75 and 59 px gaps stay.
# Rule (the user): a label is its object's outline, one piece, filled solid, no holes. Fix: POST_BODY fills every enclosed
# hole of this type's body up to BLOWER_ULTI_HOLE px (full resolution). Holes open to the outside (notches on the edge) are
# not enclosed and stay as they are. Only type 6000732; the other Blower skins use their own type ids (6000734 .. 6000742).

from scipy.ndimage import binary_fill_holes as _bu_fill, label as _bu_cc

BLOWER_ULTI_TYPE = 6000732
BLOWER_ULTI_HOLE = 100   # px: enclosed gaps of the wave's body up to this size are filled (the largest real one seen: 75 px)


def _fix_blower_ulti_holes(body, ctx):
    h = _bu_fill(body) & ~body
    if not h.any():
        return body
    L, n = _bu_cc(h)
    if n == 0:
        return body
    sz = np.bincount(L.ravel())
    small = np.zeros(n + 1, bool)
    small[1:] = sz[1:] <= BLOWER_ULTI_HOLE
    return body | small[L]


POST_BODY[BLOWER_ULTI_TYPE] = _fix_blower_ulti_holes
