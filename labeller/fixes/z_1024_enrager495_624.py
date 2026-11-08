# Enrager Duel 495: hull of the whole bright band; Enrager Bakugo 624 attack: no label (user 10-24; Haiku task 22, reviewed)
# Task 22 fix (2026-10-23): skin 495 EnragerDuel (type Enrager002Projectile 6000876) and skin 624 EnragerBakugo (type
# Enrager010Projectile 6000886). Both skins of the brawler Enrager; the user's rules, per projectile.
#
# 6000876 (495), coordinator decision 2026-10-23: a projectile = the WHOLE bright cyan band of its shot, with its diamonds
# and the white arrow tip. The purple / blue smoke trail behind it, sparks and the end fades stay out. Most of the band lies
# under an opaque haze draw in the ID pass (no atlas uv), so the band is taken from the frame: bright cyan pixels (G >= 180,
# B >= 235, R <= 200: the band core and its rims; smoke and blue haze are darker) inside the projectile's predicted region,
# plus the diamond outline (shape 607) and the arrow tip (shape 61) where the ID pass shows them. Closed, holes filled, and
# only the parts connected to the projectile's own shapes or predicted body are kept. ONE convex hull per projectile is
# applied after this (ONE_LABEL 'hull'). Ended projectiles have no predicted region, so they carry no label.
E002_KEEP = (607, 61)            # shape ids: the diamond's square outline and the arrow tip (clip body)
E002_G, E002_B, E002_R = 180, 235, 200


def _t22_enrager002_band(body, ctx):
    import numpy as _np
    im = ctx['img'].astype(_np.int16)   # full resolution, like the context arrays here
    bright = (im[..., 1] >= E002_G) & (im[..., 2] >= E002_B) & (im[..., 0] <= E002_R)
    SP = _a4_shape_pixels(ctx['R'], 'sc/effects_brawler.sc', ('edgar_002_atk', 'edgar_002_atk_red'))
    own = _np.isin(SP, [s + 1 for s in E002_KEEP]) & (ctx['cv'] >= CUT) & ctx['region']   # this projectile's own shapes only
    m = (bright & ctx['region']) | own
    if not m.any(): return m
    m = _close(m, 2)
    from scipy.ndimage import label as _cc, binary_fill_holes as _fh
    m = _fh(m) & ctx['region']
    L, n = _cc(m)
    if n == 0: return m
    anchor = _np.unique(L[(own | ctx['pred_body']) & m]); anchor = anchor[anchor > 0]
    return _np.isin(L, anchor) if anchor.size else m


POST_BODY[6000876] = _t22_enrager002_band
ONE_LABEL[6000876] = 'hull'


# 6000886 (624): the clip has ONE sprite, a soft radial blob (class glow). The ID pass shows it stretched over the character
# and the ground (the red outline the reviewer saw). Soft halos are not labelled (the user's rule). Coordinator decision
# 2026-10-23: NO label for this type (empty body accepted as is).
def _t22_enrager010_no_halo(body, ctx):
    import numpy as _np
    return _np.zeros_like(body)


POST_BODY[6000886] = _t22_enrager010_no_halo
