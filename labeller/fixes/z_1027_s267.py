# sweep 10-27 fix for skin 267 (Haiku fix, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s267/)
# Sweep 2026-10-27, skin 267 BlackHoleMisfortune (task s267), type 6000415 Tara003Projectile (Tara's card fan, clip
# tara_003_atk_projectile). Pairs 4915 / 4916 (attack 0 deg, ticks 271 / 273) and 4921 / 4922 (attack 90 deg, ticks 307 / 309):
# the red label took in the white four-point star flash of the impact (at the crate, below / beside the card fan; at 307 /
# 309 the flash at the left of the fan). An impact flash is never labelled (rule 4).
# Cause: the flash is drawn with a sprite of the card atlas (sc/effects_brawler.sc, texture 2560 x 4064, the 32 x 32 atlas
# polygon at uv px (210, 3606)-(242, 3638), a clip-body shape). The ID pass sees it as the projectile's body, so its pixels
# are labelled. The same polygon is also the card's dark suit symbol (its pixels darker than 100 sit inside the cards at
# ticks 265-269 and must stay: a hole in the card label otherwise).
# Fix (POST_BODY, after the per-type rules): the body pixels of that polygon whose frame brightness (min of R, G, B) is at
# least 100 and that lie within 4 px of its bright core (min channel >= 200, the flash's white centre) are removed. The core
# exists only in the impact frames (271, 273, 307, 309); the dark suit symbols and the card faces are never touched. Holes the
# removal leaves inside the body (a suit symbol's bright edge) are filled again; the flash's gap stays open to the outside.
# Other types are not touched. The polygon is looked up by its atlas box (STAR_BOX), so no other sprite is affected.

STAR_6000415_BOX = (210.0, 3606.0, 242.0, 3638.0)   # atlas px (x0, y0, x1, y1) of the flash's polygon, texture 2560 x 4064
STAR_6000415_CORE = 200   # min channel of the flash's white centre
STAR_6000415_EDGE = 100   # min channel from which the flash's pixels count (the dark suit symbol is below)
STAR_6000415_NEAR = 4     # px (half resolution) around the bright core


def _star_polys_6000415(own):
    """region-map index (+1) of the flash polygon, per (file, texture) of Tara003's sprites"""
    out = {}
    for (file, tex), polys in sprites(6000415, own).items():
        w, h = tex_size(file, tex)
        idx = []
        for i, (uvp, part, cls) in enumerate(polys):
            p = np.asarray(uvp, np.float64) * [w, h]
            b = np.r_[p.min(0), p.max(0)]
            if part == 'clip' and cls == 'body' and np.allclose(b, STAR_6000415_BOX, atol=1.5):
                idx.append(i + 1)
        if idx: out[(file, tex)] = idx
    return out


def _fix_6000415(body, ctx):
    from scipy.ndimage import binary_fill_holes
    if not body.any() or ctx['img'] is None: return body
    R = ctx['R']; uv = R['uv']; C = ctx['C']; img = ctx['img']; own = ctx['own']
    star = np.zeros(body.shape, bool)
    for (file, tex), idx in _star_polys_6000415(own).items():
        rm = region_map(6000415, own, file, tex); Hh, Ww = rm.shape
        u = np.clip((uv[..., 0] * Ww).astype(int), 0, Ww - 1); v = np.clip((uv[..., 1] * Hh).astype(int), 0, Hh - 1)
        star |= (C == 1) & np.isin(rm[v, u], idx) & (uv[..., 0] >= 0)
    if not star.any(): return body
    mn = img.min(-1)
    core = star & (mn >= STAR_6000415_CORE)
    if not core.any(): return body               # no impact flash in this frame (before the impact, or faded)
    rem = star & (mn >= STAR_6000415_EDGE) & _dilate(core, STAR_6000415_NEAR) & body
    if not rem.any(): return body
    out = body & ~rem
    hole = binary_fill_holes(out) & ~out          # holes the removal opened inside the body: filled again
    return out | (hole & _dilate(rem, 2))


POST_BODY[6000415] = _fix_6000415
