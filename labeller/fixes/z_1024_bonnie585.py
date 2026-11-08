# Cannon Girl Christmas 585 attack: the bell only, ribbon trail out (Haiku task 16, reviewed)
# task 16 fix (Haiku task 16, skin 585 CannonGirlChristmas, CannonGirl's Christmas skin: Bonnie 004 attack, attack 0 deg and
# attack 90 deg rows; no super row on this skin's page)
# User decision: the attack body only (one clean solid shape per projectile); trails, smoke and particles are not labelled.
# Type Bonnie004Projectile (6001185, own clip bonnie_004_atk_projectile / _red, sc/effects_brawler.sc).
# Cause: the label ran up the blue ribbon behind the bell (its tail of bubbles, a thin stem line and sparkles). The ribbon is
# drawn by the projectile's own clip (shapes 1, 63, 47, 11657, 11653: the stem line, its bubbles and sparkles), so the
# clip-body codes took it; the bell is shapes 11680 (the bell) and 11677 (its aqua clapper), per the clip's own layers.
# Shapes 0 and 11678 are the additive glow around the bell: outside the label already, kept outside.
# Fix: by shape (the clip's own shape ids, per pixel: _a4_shape_pixels), the bell's shapes are the body; the tail's shapes
# are dropped; pixels with no shape (sprite edges, the white flash at the stem's foot) are kept only within 1 px of the
# bell (edge=0: none kept; the shape-less pixels touching the bell were the stem's foot, a spur on its top); parts without
# bell pixels are dropped; holes filled.
_T16 = dict(file='sc/effects_brawler.sc', clips=('bonnie_004_atk_projectile', 'bonnie_004_atk_projectile_red'),
            bell=(11680, 11677), tail=(1, 63, 47, 11657, 11653), edge=0)


def _t16_bell(body, ctx):
    if not body.any(): return body
    from scipy.ndimage import binary_fill_holes, label as _cc
    p = _T16
    SP = _a4_shape_pixels(ctx['R'], p['file'], p['clips'])   # shape id + 1 per pixel, 0: none of the clip's shapes
    bell = body & np.isin(SP, [s + 1 for s in p['bell']])
    if not bell.any(): return body & ~np.isin(SP, [s + 1 for s in p['tail']])
    m = bell
    if p['edge']:   # sprite-edge pixels touching the bell (none in use: they only made the stem's foot a spur on the bell's top)
        m = bell | (body & (SP == 0) & _dilate(bell, p['edge']))
    L, n = _cc(m); keep = np.unique(L[bell]); keep = keep[keep > 0]
    return binary_fill_holes(np.isin(L, keep)) & ctx['region']


for _t in (6001185,): POST_BODY[_t] = _t16_bell
