# sweep 10-27 fix for skin 111 (Haiku fix, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s111/)
# s111 (sweep 2026-10-27, MenderDefault, super 45 deg, pair 1742 tick 355, extra_label): the Mender default super
# (MenderUltiProjectile 6001756, the dark blue crystal shards in the blue glow) got a thin magenta sliver on the glow above
# the top shard, with no solid object under it (ticks 349-355). Diagnosis (ctx arrays of the three super objects): the
# shard is one large part of the label; the sliver and the specks are separate small parts. The sliver is drawn by its own
# draw call (same texture and shader, shape 20 shared with the shard), 2 px clear of the shard's top edge, so the object's
# outline has a gap there. Rule: keep the label's largest part, plus any part within 2 px of it; drop every other part
# smaller than 25 % of the largest. The shard's bright rim (rule 6) is part of the largest part and stays. Holes of at most
# 40 px are filled (as the clean-up does). Type 6001756 has no other fix file.
# Only for a full-size shard (largest part >= MENDER_ULTI_MIN_MAIN px): at spawn (sw2_111 t349) the three shards are small,
# their fragments are claimed by the neighbouring shard's pixels in the final map, and cutting them left one shard with 2 px
# (never lose a projectile). The sliver at t355 belongs to a 2275 px shard, so it is covered.
MENDER_ULTI_FRAC = 0.25
MENDER_ULTI_GAP = 2
MENDER_ULTI_MIN_MAIN = 1000


def _mender_ulti_main_part(body, ctx):
    from scipy.ndimage import label as cc, binary_dilation
    if not body.any(): return body
    L, n = cc(body, structure=np.ones((3, 3), bool))
    if n <= 1: return body
    sz = np.bincount(L.ravel())
    main = 1 + int(np.argmax(sz[1:]))
    if sz[main] < MENDER_ULTI_MIN_MAIN: return body
    mm = L == main
    near = np.unique(L[binary_dilation(mm, iterations=MENDER_ULTI_GAP) & body]); near = near[near > 0]
    keep = set(near.tolist()) | {main}
    drop = [k for k in range(1, n + 1) if k not in keep and sz[k] < MENDER_ULTI_FRAC * sz[main]]
    if not drop: return body
    b = body & ~np.isin(L, drop)
    return b | small_holes(b, 40)


POST_BODY[6001756] = _mender_ulti_main_part
