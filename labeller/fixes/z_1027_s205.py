# sweep 10-27 fix for skin 205 (Haiku fix, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s205/)
# sweep 10-27 fix for skin 205 RocketGirlSteamPunk (Brock 011 projectile, type 6000081, attack 90 deg; task
# out/sweep2/haikutasks/sweep1027/s205/task.md: pair 3631 tick 305 and pair 3633 tick 309, 'hole').
# Finding: a small closed loop of unlabelled pixels inside the rocket body's label, just below the gold band. The loop is
# 66-79 px at half resolution on every tick of the row (305..321), enclosed by the label.
# Cause: the loop pixels are the rocket's own clip body (C == 1, shape 217 of effects_brawler_brock.sc, the gold band
# sprite) drawn translucent: front-most draw 102 (tick 305) / 105 (tick 309), coverage 26..127 on every tick, all under
# CUT = 128, so they are not body. The body step fills holes BEFORE its EDGE_GROW dilation closes the ring around them,
# so the loop survives. The shared clean-up (z_1099_cleanup) fills holes of at most 40 px (half resolution); these are
# 66-79 px, so it leaves them.
# Fix: after the per-type body is assembled (POST_BODY, before the hull / ONE_LABEL step, which this type has none of),
# the holes enclosed by the label are filled where they are the projectile's own clip-body pixels (C == 1). Nothing else
# is added: the outline outside the holes is unchanged, and an object's enclosed hole that is not its own clip body
# (ground showing through, another draw in front) stays open. Keyed to 6000081 only; no other type or skin is touched.
from scipy.ndimage import binary_fill_holes as _bfh_205


def _fix_6000081(body, ctx):
    if not body.any(): return body
    hole = _bfh_205(body) & ~body
    if not hole.any(): return body
    return body | (hole & (ctx['C'] == 1))


POST_BODY[6000081] = _fix_6000081
