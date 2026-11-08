# Twins 193 / 088 super: star only, smoothed (193: round blob, user review 10-24); Haiku task 9, reviewed
# Twins supers: label only the white star core, smooth and round (task 9, the user's decision 2026-10-23; round 2 after the
# user's review: 193 "rounder, there are also some stray pixels getting labeled"). Haiku task 9. Not yet reviewed by the coordinator.
# Keyed to two projectile types; no other type or skin changes:
#   6001410 Twins003UltiProjectile (skin 193 Angel; clip twin_003_ulti_projectile and its _red export)
#       star = shape 137. Round 2: Gaussian blur of the star mask, sigma 3.0 px (frame px), threshold 0.5. The long thin spikes
#       (1-3 px wide) vanish into the round core, which is what the user asked for: the spikes are shortened to their bases, the
#       outline is round, and the 1-px spurs / strays of the earlier label (16-52 px per tick) are gone with them.
#       Measured on sw2_193, ticks 345-359: circularity 4 pi A / P^2 (edge pixels) 0.25-0.45 before, 1.0-1.5 after; spur
#       pixels (not in a 1-px opening) 16-52 before, 0-1 after; the core (2-px opening of the raw star) kept 98-100 %.
#       Other options (round 2): opening 2 px + Gaussian sigma 1.0 (lumpy, cut a notch into the core); opening 3 px (too small,
#       lost the core at ticks 349 and 357); convex hull of the opened core (core kept, but a diamond that does not follow the
#       round core); Gaussian sigma 2.0 (tips still 6-18 px); sigma 2.5 and sigma 3.0 with a rounded hull (the hull added
#       nothing: sigma 3.0 alone is the pick).
#   6001403 TwinsUltiProjectile (skin 088 Default; clip twin_def_ulti_projectile and its _red export)
#       star = shapes 137 (ticks 345-347, stretched diamond) and 61 (ticks 349-359, four-point star). Unchanged from round 1
#       (the user: "088 was fine"): closing with a 1-px cross (fills the notches between the spikes), then Gaussian sigma 0.7,
#       threshold 0.5. Shape 98 = flare specks beside the star: never labelled.
# Both types: the star's own shape pixels only (coverage >= CUT, near the predicted label). Shape 0 (the blue sphere) is never
# labelled; growth is at most 1 px beyond the raw star; one solid shape (largest piece); no hull.
TWIN_SUPER_FILE = 'sc/effects_brawler.sc'
TWIN_SUPER = {
    6001410: dict(clips=('twin_003_ulti_projectile', 'twin_003_ulti_projectile_red'), shapes=(137,), close=True, sigma=0.7),
    6001403: dict(clips=('twin_def_ulti_projectile', 'twin_def_ulti_projectile_red'), shapes=(137, 61), close=True, sigma=0.7),
}


def _twin_super_star(body, ctx):
    """the super's star core only (its clip shapes, opaque, near its predicted label), smoothed; empty when the star is not drawn"""
    from scipy.ndimage import gaussian_filter, binary_fill_holes, label as cc
    p = TWIN_SUPER[ctx['t']]
    SP = _a4_shape_pixels(ctx['R'], TWIN_SUPER_FILE, p['clips'])     # shape id + 1 per pixel, 0: none
    star = np.isin(SP, [s + 1 for s in p['shapes']]) & (ctx['cv'] >= CUT) & ctx['region']
    if not star.any(): return np.zeros_like(body)
    m = (_close(star, 1) | star) if p['close'] else star
    s = binary_fill_holes(gaussian_filter(m.astype(np.float32), p['sigma']) >= 0.5)
    s &= _dilate(star, 1)                                              # never grows more than 1 px beyond the raw star
    s &= star | (SP != 1)                                              # no sphere pixels (shape 0)
    L, n = cc(s)
    if n > 1:                                                          # one clean solid shape per projectile
        sz = np.bincount(L.ravel()); sz[0] = 0; s = L == sz.argmax()
    return s


for _t in TWIN_SUPER: POST_BODY[_t] = _twin_super_star


# The user (2026-10-24): Twins Angel's (193) super label is a convex HULL of the star. Built from the lightly smoothed star
# (round-1 settings: 1-px closing + Gaussian sigma 0.7, largest piece only), so stray pixels do not stretch the hull.
ONE_LABEL[6001410] = 'hull'
