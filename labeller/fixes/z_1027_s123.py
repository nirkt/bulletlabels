# sweep 10-27 fix for skin 123 (Haiku fix, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s123/)
# sweep 10-27 fix for skin 123 JesterBrawlAcademy (task out/sweep2/haikutasks/sweep1027/s123/, finding pair 1957).
# Type 6001290 = Chester006Projectile (logic JesterProjectile, sc/effects_brawler_chester.sc, skin 29001294 JesterBrawlAcademy).
# Defect (attack 90 deg, tick 315, spur_or_speck; also tick 313): the two orbs of one shot each carry small separate red
# loops on cyan spark droplets that float above the flame tops (the verifier: specks outside the projectile body; the user's
# rule 3: sparks are not labelled). Measured on sw2_123 ticks 311-315 (label components, frame px, 8-connected):
#   droplets (detached, above the flame):   46, 53, 55, 71 px       -> share of the orb's largest part 0.08-0.12
#   flame tongues (detached, attack body):  158, 164, 187, 287 px   -> share 0.24-0.52 (kept)
# The droplets share their clip shapes (3, 5) with the flame, so a shape-id rule cannot separate them; they are separated
# from the body by a background gap. Rule: a detached part of the label smaller than S123_FRAC x the largest part is a
# droplet / speck and is dropped (the largest part is always kept, so an orb never loses its label). Only type 6001290.
# Also drops the 1-9 px specks of the same type (attack 0 deg ticks 265-269, attack 90 deg tick 307): specks, not body.
# Checked on every tick 261-319 of sw2_123: the largest part of each orb keeps its exact pixel count; the flame tongues stay.
S123_TYPE = 6001290
S123_FRAC = 0.2   # a detached part below this share of the largest part is a droplet / speck


def _s123_droplets(body, ctx):
    from scipy.ndimage import label as cc
    if not body.any(): return body
    L, n = cc(body, structure=np.ones((3, 3), bool))
    if n < 2: return body
    sz = np.bincount(L.ravel())[1:]
    keep = np.concatenate([[False], sz >= S123_FRAC * sz.max()])   # label 0 = background; largest part always >= its own share
    keep[1 + int(np.argmax(sz))] = True
    return body & keep[L]


POST_BODY[S123_TYPE] = _s123_droplets
