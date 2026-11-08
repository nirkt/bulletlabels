# 2026-11-08 enemy-label fix, group v3 (skins 019 217 288 Rico = TrickshotDude, 076 Beamer). Prefix _f3_. Installed as py/fixes/z_1118_enemy_v3.py.
# Every rule returns early for ctx['own'] objects: own (blue) labels are never touched.
import numpy as _f3_np
from scipy.ndimage import binary_dilation as _f3_dil

# ---- 019 TrickshotDefault, TrickshotDudeProjectile 6000087 (attack: the Rico ball)
# CAUSE: the RED run draws an enemy muzzle burst (a white / red puff cloud with dots) next to the shooter that the blue run does not have. Its sprites use
# atlas shapes that the type's sprite codes (1 / 4) accept, so 3-6 pieces of the puff (10-300 px) beside the ball became body (enemy label 18 % extra area
# over 14 ticks).
# 076 BeamerDefault, BeamerProjectile 6001298 (attack beam): the same red muzzle burst (pink petals right in front of the beam) put 150-250 px of petals into the
# enemy label (12-15 % extra area). Same cut (runs before the installed spark filter z_1027_s076). FIX (enemy only): the body is cut to the 4 px dilated convex hull of the predicted ball (ctx['pred_body']), as in z_1117_enemy_cocoon.
_f3_K = 4


def _f3_cut(prev):
    def post(body, ctx):
        if not ctx.get('own', True) and body.any():
            pb = ctx['pred_body']
            if pb.any(): body = body & _f3_dil(hull_mask(pb), iterations=_f3_K)
        return prev(body, ctx) if prev is not None else body
    return post


for _f3_t in (6000087, 6001298):
    POST_BODY[_f3_t] = _f3_cut(POST_BODY.get(_f3_t))


# ---- 217 TrickshotGold Rico003UltiProjectile 6000092, 288 TrickshotDudeChicken Rico005UltiProjectile 6000096 (super: a row of gems)
# CAUSE: the RED run's enemy muzzle burst (big additive star + puffs at the shooter) lies over the NEWEST gem (the one at the muzzle): the ID pass sees only a
# crescent of it (217: 250-620 px of ~930 at ticks 351-365, the own run labels all of it) and some burst pieces drawn with the type's sprite shapes became body
# (288: a 3rd gem of 540-955 px against ~410). The blue run has no burst. FIX (enemy only, ctx['own'] False): (1) body cut to the 4 px dilated convex hull of the
# predicted gem (as 019 above); (2) when less than 85 % of the predicted gem is body (hidden under the burst) and something of it is visible, the label is the
# convex hull of the exposed body and the predicted gem, cut to the region (the hidden-part rule of z_1109_enemy_pl). Visible gems and own labels untouched.
def _f3_hidden(prev):
    def post(body, ctx):
        if not ctx.get('own', True) and body.any():
            pb = ctx['pred_body']
            if pb.any():
                body = body & _f3_dil(hull_mask(pb), iterations=_f3_K)
                if body.any() and (body & pb).sum() < 0.85 * pb.sum(): body = (hull_mask(body | pb) & ctx['region']) | body
        return prev(body, ctx) if prev is not None else body
    return post


for _f3_t in (6000092, 6000096):
    POST_BODY[_f3_t] = _f3_hidden(POST_BODY.get(_f3_t))
