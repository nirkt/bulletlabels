# sweep 10-27 fix for skin 629 (Haiku fix, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s629/)
# Fix for skin 629 DoorManWindstock (sweep 2026-10-27, task s629: pair 10610 tick 267 and pair 10611 tick 269, 'merge_split';
# the same row's tick 265 shows the same split).
# DoorManWindstock's attack 0 deg (type 6001329 = DoorMan005Projectile, 'gray_005_atk_projectile'): one capsule, a cyan body with
# cream caps at both ends. Its label came out as three separate red pieces: an oval over the left cap (drawn clip body, code 1),
# a sharp-cornered rectangle over the cyan middle (drawn clip glow / shared sprites, codes 2 and 5, only partly taken in) and an
# oval over the right cap (code 1). The seams between them are where the middle was not taken in.
# Also a thin green stick leading from the neighbouring ring to the left cap (codes 2 and 4) was taken in as a spur.
# Fix (keyed to this type only, the other skins of DoorMan keep their own labels): the body is opened with a 5x5 disk (the 2-3 px
# stick does not survive, the capsule body, about 10 px tall, does). The parts that survive and are at least 20% of the largest
# (both caps and the middle; the stick is gone) are taken as the capsule, and their convex hull is the label: a capsule is convex,
# so the hull is one solid oval over both caps and the middle. The hull is built only from the capsule's own pixels after the
# opening, so the stick and the halo are not in it. If nothing survives the opening, the body is left as it is.

S629_CAPSULE = 6001329


def _s629_capsule(body, ctx):
    from scipy.ndimage import binary_opening, label as cc
    if not body.any():
        return body
    yy, xx = np.mgrid[-2:3, -2:3]
    disk2 = (yy * yy + xx * xx) <= 4                  # 5x5 disk: removes the 2-3 px stick, keeps the ~10 px capsule
    op = binary_opening(body, structure=disk2)
    L, n = cc(op)
    if not n:
        return body
    sz = np.bincount(L.ravel())[1:]
    keep = 1 + np.nonzero(sz >= 0.2 * sz.max())[0]   # the capsule's parts: both caps and the middle (a cap the opening
    core = np.isin(L, keep)                          # split off from the middle is kept; the stick is gone by now)
    return hull_mask(core)                           # one solid convex oval over both caps and the middle


POST_BODY[S629_CAPSULE] = _s629_capsule
