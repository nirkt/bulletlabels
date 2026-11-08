# sweep 10-27 fix for skin 244 (Haiku fix, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s244/)
# Skin 244 PocoPirate (sweep 2026-10-27, pair 4432 tick 359; the same embers are labelled at tick 357 too): the super
# (type 6000340, Poco003UltiProjectile, a ZONE type) labelled the yellow-orange embers / sparks drifting away from the
# cloud as its body. The user's rules: sparks and particles are never labelled (rule 3); the label is the cloud (green puffs).
# Cause: zone() (idlabels, ZONE types) keeps a part only if it is >= ZONE_MIN (0.15) of the largest part. While the cloud
# is big (ticks 345-355) the embers are far below that cut and dropped; once the cloud has broken into small puffs
# (ticks 357-359, largest part 335-2103 px) the 51-430 px embers pass the relative cut.
# Fix (this type only, after zone(), as a POST_BODY hook): drop each body part whose mean colour is yellow-orange
# (G/R < 0.92; the embers measure 0.76-0.86, the green puffs 2.8-14) and which does not touch a green part (G/R >= 1.3,
# within 3 px). The yellow skull of the cloud (shape 59078) touches green parts, so it stays. Measured on ticks 345-361 of
# sw2_244: only ticks 357 and 359 have such parts (357: 1 part, 428 px; 359: 12 parts, 1708 px); other ticks unchanged.
# If no green part is left at all, nothing is dropped (the label is never emptied by this rule).

_T244_PREV = POST_BODY.get(6000340)   # an earlier hook for this type, if any (none at the time of writing)


def _s244_embers(body, ctx):
    if _T244_PREV is not None:
        body = _T244_PREV(body, ctx)
    img = ctx.get('img')
    if img is None or not body.any(): return body
    from scipy.ndimage import label as cc, binary_dilation
    L, n = cc(body)
    if n < 2: return body
    im = img.astype(np.float32)
    parts = {}; gr = {}
    for k in range(1, n + 1):
        m = L == k; parts[k] = m
        rgb = im[m].mean(0); gr[k] = float(rgb[1]) / max(float(rgb[0]), 1.0)
    green = np.zeros(body.shape, bool)
    for k in range(1, n + 1):
        if gr[k] >= 1.3: green |= parts[k]
    if not green.any(): return body                    # no green part left: the label is kept as it is
    gd = binary_dilation(green, iterations=3)
    out = body.copy()
    for k in range(1, n + 1):
        m = parts[k]
        if gr[k] < 0.92 and not (m & gd).any(): out &= ~m   # yellow-orange, detached from the green cloud: an ember
    return out


POST_BODY[6000340] = _s244_embers   # Poco super (skin 244 PocoPirate): leftover embers beside the cloud are not labelled
