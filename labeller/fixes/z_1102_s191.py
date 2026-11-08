# hosted review 2026-11-01 fix (Haiku fix, reviewed by the coordinator, installed 11-01 LOCAL ONLY; source out/sweep2/haikutasks/site1101/s191/fix.py; runs after z_1101_*)
# Hosted review 2026-11-01, skin 191 (TrickshotDudeKing_remake_01 = Rico, Rico 008), type 6000101 (Rico008Projectile, clip
# rico_008_atk_projectile). The user's finding (binding): attack 90 deg t315, t317, t323 'holes / ragged': each projectile must
# be ONE clean filled outline with smooth edges, no holes, no notches.
#
# Cause (measured on the kept capture sw2_191): the bullet is a faceted sprite with a lighter facet inside it. At t315 and t317
# that facet is not in the ID pass's body (the front-most draw there is not the bullet's), so the body had enclosed holes of
# 42-45 px inside its outline (t315: 2 bullets; t317: 2 bullets). At t315 / t317 / t323 the bottom bullet, cut by the frame's
# bottom edge, also had a U-shaped notch open to that edge, where a white sparkle lies over the bullet's tail.
# No holes or notches on any attack 0 deg or super tick of this skin; the attack 90 ticks 305-309 are convex and unchanged.
#
# Fix (type 6000101 only): per projectile, the body is closed (gaps narrower than 2 * S191_R px close; the frame's edge is
# replicated outward so a notch that opens to the bottom edge closes too), then its enclosed holes are filled. Closing only adds
# pixels in concavities of the body, so the label stays within its own outline and no glow or other object is taken in.
# This file runs after the installed fixes (IDLABELS_OVERRIDE), so it wraps whatever POST_BODY the type had before.

import numpy as _np
from scipy.ndimage import binary_closing as _s191_closing, binary_fill_holes as _s191_fill

S191_TYPE = 6000101
S191_R = 3          # px (frame resolution): gaps narrower than 2 * S191_R are closed

_s191_prev = POST_BODY.get(S191_TYPE)


def _s191_solid(body, ctx):
    if _s191_prev is not None: body = _s191_prev(body, ctx)
    if not body.any(): return body
    r = S191_R
    p = _np.pad(body, r, mode='edge')                  # the frame's edge is extended, not treated as background
    c = _s191_closing(p, iterations=r)[r:-r, r:-r]
    return _s191_fill(c | body)


POST_BODY[S191_TYPE] = _s191_solid
