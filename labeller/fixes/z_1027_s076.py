# sweep 10-27 fix for skin 076 (Haiku fix, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s076/)
# Beamer skin 076 (BeamerDefault), sweep 2026-10-27, pairs 1359-1362 (attack 90 deg, ticks 311-315; the same type also
# draws the attack 0 deg beam, pair 1354, which the shared clean-up already resolved).
# Defect: the BeamerProjectile (type 6001298, clip mandy_def_atk_projectile) draws a small pale spark that trails above
# the beam head. The spark uses the clip's own sprite, so the label took it in as a detached part, 4 px away from the body
# (tick 311: 39 px spark vs 596 px body; tick 313: 49 vs 704; tick 315: 57 vs 828). A trailing spark is not the attack
# body (user rule 3), so it must not be labelled.
# Fix: the projectile's body label keeps only its parts that are at least BP_SPARK_FRAC of the largest part. The spark
# ratio is 0.065-0.070; a legitimately split body would have parts of comparable size and is kept. Ticks where the body is
# one part (305, 307, 309 on this row, 263-269 on the 0 deg row) are unchanged.
BP_SPARK_TYPE = 6001298
BP_SPARK_FRAC = 0.15


def _beamer_spark(body, ctx):
    from scipy.ndimage import label as cc
    if not body.any(): return body
    L, n = cc(body)
    if n <= 1: return body
    sz = np.bincount(L.ravel())[1:]
    keep = 1 + np.nonzero(sz >= BP_SPARK_FRAC * sz.max())[0]
    return body & np.isin(L, keep)


POST_BODY[BP_SPARK_TYPE] = _beamer_spark
