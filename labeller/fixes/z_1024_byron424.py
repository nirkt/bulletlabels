# Snake Oil Surfer 424 (Byron005 attack 6000870 / super 6000871): drop detached specks (Haiku task 28, reviewed 10-24)
# Task 28: skin 424 SnakeOilSurfer (Byron005). Explicit type ids only: 6000870 Byron005Projectile (attack 0 / attack 90),
# 6000871 Byron005UltiProjectile (super 45). Other types and skins untouched.
#
# 'Labels must be clean' (addendum, the user 2026-10-23): detached specks are dropped. A speck here is a part of the
# projectile's body label of 12 px or fewer that is separate from the main part (the main part is always kept).
# Seen on the bottle at attack 90 tick 313 (4 px and 1 px, 1396 -> 1391 px) and on the cube at super tick 355
# (1 px, 696 -> 695 px). No other tick has a speck, so no other tick changes.
#
# Not done (policy question, see report.txt): at attack 90 tick 317 and super tick 357 the tracker flags the projectile as
# ended on the tick it stops at its final position, and the projectile is still drawn intact there (labelled: 337 and 750
# px). The rules do not settle whether that last drawn frame is an impact animation; removing it would make a projectile
# label 0 px, so it is left as it is.


def _t28_specks(body, ctx):
    from scipy.ndimage import label as cc
    if not body.any(): return body
    L, n = cc(body, structure=np.ones((3, 3)))
    if n <= 1: return body
    sz = np.bincount(L.ravel()); sz[0] = 0
    keep = np.nonzero((sz > 12) | (sz == sz.max()))[0]
    return np.isin(L, keep) & body


POST_BODY[6000870] = _t28_specks
POST_BODY[6000871] = _t28_specks
