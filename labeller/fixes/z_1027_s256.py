# sweep 10-27 fix for skin 256 (Haiku fix, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s256/)
# Skin 256 PocoStarr super (sw2_256 super 45 deg, pair 4697, tick 359; sweep 2026-10-27): the twinkle stars that drift off
# Poco's ulti cloud were labelled as part of the super (extra_label, rule 3: twinkles and sparks are never labelled).
# Type 6000344 (Poco005UltiProjectile, clip poco_005_ulti in sc/effects_brawler.sc) draws the orange cloud (shape 63) and
# yellow twinkle stars (shape 51826) in the same clip. At tick 359 six star clusters float 40-120 px from the cloud and
# carry the magenta outline; the same stars are unlabelled at ticks 355 and 361. Ticks 345-353 show 51826 pixels only
# INSIDE the cloud (9-28% of each connected part, part of the cloud) and those stay.
# Rule: split the body into connected parts; drop a part when at least half of its pixels are shape 51826 (the stars).
# Measured at tick 359: the six star parts have 90-95% 51826 pixels; every cloud part has 0-3%. If nothing would remain,
# the body is kept unchanged (a projectile never loses its label). Keyed to the explicit type id 6000344.
POCO_ULTI_STAR_SHAPE = 51826
POCO_ULTI_FILE = 'sc/effects_brawler.sc'
POCO_ULTI_CLIPS = ('poco_005_ulti',)


def _poco_ulti_no_stars(body, ctx):
    if not body.any(): return body
    from scipy.ndimage import label as cc
    SP = _a4_shape_pixels(ctx['R'], POCO_ULTI_FILE, POCO_ULTI_CLIPS)   # shape id + 1 per pixel, 0: none of the clip's shapes
    star = body & (SP == POCO_ULTI_STAR_SHAPE + 1)
    if not star.any(): return body
    L, n = cc(body, structure=np.ones((3, 3), bool))
    if n == 0: return body
    size = np.bincount(L[body], minlength=n + 1)
    nstar = np.bincount(L[star], minlength=n + 1)
    drop = (size > 0) & (2 * nstar >= size)
    drop[0] = False
    out = body & ~drop[L]
    return out if out.any() else body


POST_BODY[6000344] = _poco_ulti_no_stars
