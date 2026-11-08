# sweep 10-27 fix for skin 369 (Haiku fix, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s369/)
# SpawnerDude005 (skin 369 SpawnerDudeMake, page 369_SpawnerDudeMake_01), sweep 2026-10-27, findings 6937 and 6938
# (attack 0 deg, ticks 267 and 269); finding 6950 (super 45 deg, tick 345) is NOT fixed here: see report.md.
#
# SpawnerDude005Projectile (6000707; rows attack 0 deg and attack 90 deg): the trail's round teal dots beside the crystal
# and thin detached cyan streaks use the same atlas shape (63, effects_brawler.sc) as the flame tongues. A dot or streak
# that lies apart from the body (gap 8 to 19 px from the crystal's part: ticks 267, 269, 305, 307, 309, 315) was labelled
# as a separate speck, which is a defect (trails and specks are not labelled; the speck is still there after the shared
# clean-up, which drops only parts under 25 px). Fix: a separate part of the body is dropped when its gap to the main
# part is at least 6 px and it is under 10 % of the main part's size. The flame tongues stay: they touch the body
# (gap 2 to 4 px at ticks 271, 311, 313, 315) and are not dropped. The main part is never dropped, so no projectile
# loses its label.


def _s369_detached_trail(body, ctx):
    """SpawnerDude005 attack: separate parts of the body at least 6 px from the main part and under 10 % of its size
    (the trail dots and streaks beside the crystal) are dropped; parts touching the body stay."""
    from scipy.ndimage import label as _cc, distance_transform_edt as _edt
    if not body.any(): return body
    L, n = _cc(body, structure=np.ones((3, 3), bool))
    if n < 2: return body
    sz = np.bincount(L.ravel()); main = int(np.argmax(sz[1:])) + 1
    gap = _edt(L != main)
    drop = [k for k in range(1, n + 1) if k != main and sz[k] < 0.1 * sz[main] and gap[L == k].min() >= 6]
    return body & ~np.isin(L, drop) if drop else body


POST_BODY[6000707] = _s369_detached_trail
