# Speedy Angel 581 (Max005Projectile 6000719, skin-specific): star + its flame wings as one hull per star (Haiku task 27, reviewed 10-24)
# Task 27 (Haiku agent t27): skin 581 SpeedyAngel (Max brawler, attack Max005Projectile, type 6000719, clip
# max_005_atk_projectile / _red in sc/effects_brawler.sc). Problem: the blue star projectiles are drawn with their shape 61
# (the textured star body, shared with the trail's yellow triangles and the glow shape 0) and with swatch fills 18306-18315
# (the star clip's own yellow-white flame wings). Labels took the star inconsistently: stars whose body was shared were
# left out when the prediction was off, and the flame wings stuck out as spurs (several parts, frayed).
# The user's decision applied (2026-10-23 rules): one clean label per projectile while it lives (one convex hull when it
# is several parts); the trail triangles, the glow (shape 0) and everything after the projectile ended stay out.
# Other skins: type 6000719 is the Max attack's projectile (Max005Projectile), not a skin-specific row, so this change
# also applies to any other Max skin that fires this attack with the same clip (BRIEF: same brawler, fixed consistently).
#
# Body (per pixel, half resolution): shape 61 of the star clip (atlas shape + 1 == 62) where the frame is NOT yellow
# (R - B < 60: the trail's yellow triangles draw the same atlas shape with a yellow colour; the white core and the blue
# star pass), plus the star clip's swatch shapes 18306-18315 (+1: 18307-18316) (the flame wings; only this clip draws them),
# at least half opaque, inside the projectile's region; components that touch its predicted body (or the largest one)
# are kept; then ONE convex hull (ONE_LABEL 'hull', applied after POST_BODY).
# Not included: shape 0 (glow, shared with the trail and the glowing dots), the trail triangles (693, 7598), the glow
# halo (coverage under half), the leftovers of ended projectiles (no prediction, so no region).
T27_TYPE = 6000719
T27_CLIPS = ('max_005_atk_projectile', 'max_005_atk_projectile_red')
T27_STAR = 62            # shape 61 + 1
T27_SWATCH = range(18307, 18317)   # shapes 18306..18315 + 1


def _t27_star_body(body, ctx):
    from scipy.ndimage import label as cc
    img = ctx['img']
    if img is None or not ctx['pred_any'].any(): return body   # no prediction (ended / hidden): nothing labelled
    SP = _a4_shape_pixels(ctx['R'], 'sc/effects_brawler.sc', T27_CLIPS)
    im = img.astype(np.int16)
    star = (SP == T27_STAR) & (im[..., 0] - im[..., 2] < 60)
    swatch = np.isin(SP, list(T27_SWATCH))
    m = (star | swatch) & (ctx['cv'] >= CUT) & ctx['region']
    if not m.any(): return body & False
    L, n = cc(m, structure=np.ones((3, 3)))
    keep = np.unique(L[ctx['pred_any'] & m]); keep = keep[keep > 0]
    if keep.size == 0:
        sz = np.bincount(L.ravel()); sz[0] = 0; keep = [int(sz.argmax())]
    return np.isin(L, keep)


POST_BODY[T27_TYPE] = _t27_star_body
ONE_LABEL[T27_TYPE] = 'hull'
