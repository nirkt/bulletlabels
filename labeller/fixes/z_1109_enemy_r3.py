# enemy-label fix 2026-11-04 (task away_1104/enemy, group r3: skins 182 and 200 are real; 166, 185, 193 are scene false positives)
# 182 RopeDudeGodzilla, Buzz007Projectile (6000994), the attack beam. z_1108_jester_rope.py builds the body from the white core plus the
#     'cyan' pixels of the object's own clip / trail-over draws that touch it, tuned on the blue art (G >= 150, B >= 150, R < 200). The red
#     art is the same sprites tinted red / orange / pink, so no pixel is 'cyan' and the enemy label shrank to the white core (area 0.28).
#     Fix: for an enemy shot (o['own'] False) the same rule with the colour test of the red art: the red tint maps blue (R, G, B) to about
#     (B, 0.8 G, R + 40), i.e. a pixel is 'cyan-like' when its R and G are high (R >= _EN4R3_R, G >= _EN4R3_G) -- the pink haze lies below.
# 200 AmbusherFantasy, Ambusher003UltiProjectile (6001459), the super seed. z_1102_s200ambusher.py takes the hull of the blue clip's shapes
#     113 and 133; the red clip lily_003_ulti_projectile_red has the same layout with shape 112 instead of 113 (133 shared) in another atlas
#     region, so no pixel was recognised and the enemy label was a few pixels. Fix: for an enemy shot the same rule with the red clip and
#     112 as the body shape.
# Own (blue) shots (o['own'] True) go through the installed functions untouched, so their labels are pixel-identical.
import numpy as _en4r3_np

_EN4R3_R = 150; _EN4R3_G = 120      # red-art colour test (blue test: G >= 150, B >= 150, R < 200)

# ---- 182
_en4r3_prev_182 = POST_BODY[6000994]


def _en4r3_182(body, ctx):
    if ctx['o'].get('own', True): return _en4r3_prev_182(body, ctx)
    from scipy.ndimage import label as _cc, binary_fill_holes as _fh
    if ctx['o'].get('finished'): return _en4r3_np.zeros_like(body)
    if not body.any(): return body
    img = ctx['img']; region = ctx['region']; C = ctx['C']
    im = img.astype(_en4r3_np.int16); Rr, Gg = im[..., 0], im[..., 1]
    like = (Rr >= _EN4R3_R) & (Gg >= _EN4R3_G)
    near = _dilate(ctx['pred_any'], 10) if ctx.get('pred_any') is not None else region
    ownc = region & near & ((C == 1) | (C == 4)) & like
    cand = ownc | body
    L, n = _cc(cand, structure=_en4r3_np.ones((3, 3), bool))
    touch = _en4r3_np.unique(L[body]); touch = touch[touch > 0]
    U = body | (_en4r3_np.isin(L, touch) & cand)
    return _fh(hull_mask(U))


POST_BODY[6000994] = _en4r3_182

# ---- 200
_en4r3_prev_200 = POST_BODY[6001459]


def _en4r3_200(body, ctx):
    if ctx['o'].get('own', True): return _en4r3_prev_200(body, ctx)
    if bool(ctx['o'].get('finished')) or not bool(ctx['o'].get('drawn', True)): return _en4r3_np.zeros_like(body)
    R = ctx['R']; region = ctx['region']; cv = ctx['cv']
    SP = _a4_shape_pixels(R, 'sc/effects_brawler_lily.sc', ('lily_003_ulti_projectile_red',)) - 1
    core = _en4r3_np.isin(SP, [112, 133]) & region & (cv >= CUT)
    if not core.any(): return _en4r3_np.zeros_like(body)
    return hull_mask(core) & region


POST_BODY[6001459] = _en4r3_200
