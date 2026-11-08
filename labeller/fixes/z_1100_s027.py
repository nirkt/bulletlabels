# human review 2026-10-28 fix for skin 027 (Haiku fix, reviewed by the coordinator, installed 10-29; source out/sweep2/haikutasks/human1028/s027/fix.py; tested via IDLABELS_OVERRIDE after all fixes incl. z_1099_cleanup, hence z_1100_)
# Skin 027 (CrowDefault), human review 2026-10-28, FLAGGED "holes / ragged" on row "attack 0 deg" tick 265 and row
# "attack 90 deg" ticks 305 and 307 (the user: the label must be ONE clean piece in the object's own outline, filled solid,
# smooth edges, no holes, no notches, no frayed edge).
# Type 6000308 = CrowProjectile, the default Crow attack (clip crow_def_atk_projectile, trail crow_def_atk_trail). CrowDefault
# has no Projectiles list, so its attack is this base type. Other skins with an empty Projectiles list (CrowGold, CrowSilver,
# CrowMaverick, CrowRankedProgrsesionSkinLevel1/2) may use the same type: not in this sweep, not checked.
# What was wrong: the green swirls (trail-over sprites, sprite code 4 with a green frame colour, G - B > 60) lie on the blue
# blades. The ID pass has no body under them, so each blade's label had holes and notches where a swirl sits on its edge.
# Also ragged: the swirl-notched outline (rows 265, 305).
# This fix, per projectile of type 6000308 only (each blade is its own object):
#   (a) notches of the body are closed (3x3, 2 iterations), through the blade's own body pixels only (code 1);
#   (b) the swirls (and body pixels) inside the blade's own outline (the convex hull of its body) are filled: the hidden
#       parts rule, the swirl is part of the blade. The hull only limits WHERE the fill may go; background (code 0) and
#       the cyan trail beam behind the blade (clip glow, code 2) are never added, so the notch between the white tips stays.
# Result: one solid outline per blade (checked on ticks 265-277 and 303-317, compare.png). Other types are not touched.
import numpy as np
from scipy.ndimage import binary_closing


def _crow_def_solid(body, ctx):
    if not body.any() or ctx['img'] is None: return body
    C = ctx['C']
    im = ctx['img'].astype(np.int16)
    swirl = (C == 4) & (im[..., 1] - im[..., 2] > 60)
    own = (C == 1) | swirl
    closed = binary_closing(body, structure=np.ones((3, 3), bool), iterations=2) & (C == 1)
    inside = hull_mask(body) & own
    return body | ((closed | inside) & ctx['region'])


POST_BODY[6000308] = _crow_def_solid
