# sweep 10-27 fix for skin 489 (Haiku fix, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s489/)
# Skin 489 SpeedyLine (Max brawler), Max002Projectile = type 6000716 (clip max_002_atk_projectile, sc/effects_brawler.sc),
# sweep 2026-10-27, Haiku task s489 (pairs 9003 tick 275, 9011 tick 315, 9012 tick 319: extra_label).
# Problem: the attack's own clip has a 32 px four-point white star (atlas polygon 3 of texture 1 of that clip). The
# same polygon also makes the blob's pointed spikes. When the projectile reaches a crate or the water edge, the star
# is drawn big as an impact flash beside the blob, and the ID pass labels it as body: a red star outline on the flash,
# which rule 4 forbids (hit / impact flashes are not labelled).
# Fix: the body pixels on that star polygon form connected parts; a part with a white core (min(R, G, B) >= 190, at
# least 12 px) is the impact flash and is dropped. The blob's spikes are on the same polygon but cyan (no white core),
# so they stay; the blob's own white twinkles are on another polygon (texture 0, polygon 2) and stay.
# Other skins: type 6000716 is the Max attack's projectile, not a skin row, so this also applies to any Max skin that
# fires it with the same clip (same brawler, fixed consistently).
MAX002_T = 6000716
MAX002_STAR_BOX = (209, 241, 3605, 3637)   # atlas px of the star polygon (texture 1), +- 2 px
MAX002_CORE_MIN = 190                      # min(R, G, B) of the white core
MAX002_CORE_PX = 12                        # white core pixels a star part needs to count as an impact flash


def _fix_6000716(body, ctx):
    from scipy.ndimage import label as _cc
    img, R, t, own = ctx['img'], ctx['R'], ctx['t'], ctx['own']
    if img is None or not body.any(): return body
    uv = R['uv']; star = np.zeros(body.shape, bool)
    x0, x1, y0, y1 = MAX002_STAR_BOX
    for (file, tex), polys in sprites(t, own).items():
        rm = region_map(t, own, file, tex); Hh, Ww = rm.shape
        for i, (poly, part, cls) in enumerate(polys):
            P = poly * [Ww, Hh]
            if not (abs(P[:, 0].min() - x0) <= 2 and abs(P[:, 0].max() - x1) <= 2 and
                    abs(P[:, 1].min() - y0) <= 2 and abs(P[:, 1].max() - y1) <= 2): continue
            ys, xs = np.nonzero(body & (uv[..., 0] >= 0))
            u = np.clip((uv[ys, xs, 0] * Ww).astype(int), 0, Ww - 1); v = np.clip((uv[ys, xs, 1] * Hh).astype(int), 0, Hh - 1)
            k = rm[v, u] == i + 1
            star[ys[k], xs[k]] = True
    if not star.any(): return body
    L, n = _cc(star, structure=np.ones((3, 3)))
    white = img.min(-1) >= MAX002_CORE_MIN
    out = body.copy()
    for k in range(1, n + 1):
        m = L == k
        if (m & white).sum() >= MAX002_CORE_PX: out &= ~m   # the impact flash: unlabelled (rule 4)
    return out


POST_BODY[MAX002_T] = _fix_6000716
