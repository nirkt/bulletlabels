# sweep 10-27 fix for skin 293 (Haiku fix, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s293/)
# Fix sw2_293 (BarkeepDrunken, sweep 2026-10-27, task out/sweep2/haikutasks/sweep1027/s293/): the beer blob of Barkeep's attack
# (BarkeepProjectile, type 6000260, the Barley-style beer blob that flies off droplets) gets tiny detached specks.
# Findings (spur_or_speck, still present after the shared clean-up z_1099): pair 5442 (attack 0 deg, tick 265), 5443 (tick 267),
# 5444 (tick 269), 5445 (tick 281), 5451 (attack 90 deg, tick 319). Measured on the capture: the blob's body has its own separate
# droplet pieces of 2 to 22 px, away from the blob's main piece (809 px at tick 319, 659-684 px at ticks 269-281).
# Cause: z_1099_cleanup bridges gaps of 3 px before it counts parts, so a droplet a few px from the blob is still one part and
# is never dropped. The specks are then outlined as part of the label.
# Fix (POST_BODY, this type only, applied before the clean-up): the body's own pieces, counted 8-connected WITHOUT bridging,
# smaller than BARKEEP_SPECK px are removed, unless the piece is the largest piece of the body. The blob's main piece is never
# removed, so a projectile never loses its label. Nothing else of the label is changed; other types and other skins are unaffected.
BARKEEP_TYPE = 6000260     # BarkeepProjectile (Barkeep's beer blob; sw2_293 BarkeepDrunken)
BARKEEP_SPECK = 25         # px (full resolution, the body masks are full resolution): smaller pieces are droplet specks


def _barkeep_specks(body, ctx):
    import numpy as _np
    from scipy.ndimage import label as _cc
    if not body.any(): return body
    L, n = _cc(body, structure=_np.ones((3, 3)))
    if n <= 1: return body
    sz = _np.bincount(L.ravel())[1:]
    big = int(sz.argmax()) + 1
    drop = [j + 1 for j in range(n) if j + 1 != big and sz[j] < BARKEEP_SPECK]
    if not drop: return body
    return body & ~_np.isin(L, drop)


POST_BODY[BARKEEP_TYPE] = _barkeep_specks
