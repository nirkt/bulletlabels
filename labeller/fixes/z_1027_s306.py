# sweep 10-27 fix for skin 306 (Haiku fix, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s306/)
# Skin 306 GunslingerPaintball (Colt005), sweep 2026-10-27, task s306 (Haiku fix). Findings:
#  pair 5717 (attack 90 deg, tick 315) and pair 5718 (attack 90 deg, tick 317): extra_label, the teal / blue impact splash
#    where the bullet hits the water carries its own red outline (a stationary blob beside the bullet, rule 4);
#  pair 5727 (super 45 deg, tick 375): too_big, the magenta super label runs into the blue-teal splash at the water edge.
# Cause: the bullet's own clip (colt_005_atk_projectile for 6000028, colt_005_ulti_projectile for 6000029, blue export) holds
# one sprite, shape 7441, that is the water splash: drawn beside or under the bullet while it is over the water. It is an impact
# animation (rule 2: not labelled). The splash also draws shapes 427 and 143, which the bullet uses too, so the fix works on
# the pieces: the bullet's own bright head (shape 0) and body stay.
# Fix (POST_BODY of 6000028 and 6000029, after the per-type rules, before the final cleaning):
#  1. the pixels whose front-most draw shows shape 7441 are removed from the body, with the sprite-edge pixels (no shape) next
#     to them, unless those touch the bullet's own shapes (1 px);
#  2. a notch the splash leaves enclosed in the bullet is filled (the bullet's outline runs on under the splash);
#  3. a separate piece within 6 px of the removed splash, with no bright head (shape 0), is dropped when less than 30 % of it lies
#     inside the object's predicted label (dilated 3 px): the splash's own glints (427 / 143 pieces at the splash), not the bullet.
#     The bullet's main part lies inside its prediction (about 100 %), the splash pieces outside (0-14 %).
# Shape ids are stable across captures. Scope (10-28 red1028 revision): both teams. The enemy shots draw the '_red' clips
# (colt_005_atk_projectile_red / colt_005_ulti_projectile_red): the bullet's shapes (0, 63, 143, 427, 888 ...) are the same
# as in the blue clips, only the water splash differs: shapes 7440 and 7444 instead of 7441. The own (blue) result is unchanged
# (pixel-exact). Types 6000028 / 6000029 are used by the GunslingerPaintball skin only (pskin rows 28 / 29,
# the confs row GunslingerPaintball), so no other skin is changed.

SPLASH_SWF = 'sc/effects_brawler.sc'
SPLASH_CLIPS = ('colt_005_atk_projectile', 'colt_005_ulti_projectile')
SPLASH_CLIPS_RED = ('colt_005_atk_projectile_red', 'colt_005_ulti_projectile_red')   # enemy (red) export of the same clips
SPLASH_CODE = 7441 + 1          # _a4_shape_pixels codes shape id + 1
SPLASH_CODES_RED = (7440 + 1, 7444 + 1)   # the red clips draw the splash with shapes 7440 and 7444 (the blue one: 7441)
SPLASH_HEAD = (1,)              # code of shape 0, the bullet's bright head
SPLASH_NEAR = 6                 # px: pieces this close to the removed splash are candidates for the splash's leftovers
SPLASH_INPRED = 0.3             # share of a candidate piece inside the predicted label (dilated 3 px) below which it is dropped
COLT005_SPLASH = {6000028, 6000029}   # Colt005Projectile (attack), Colt005UltiProjectile (super)


def _s306_splash(body, ctx):
    from scipy.ndimage import label as cc, binary_fill_holes
    if ctx['t'] not in COLT005_SPLASH or not body.any(): return body
    own = bool(ctx['own'])    # own: blue clips, enemy: the '_red' clips (the bullet shapes 0, 63, 143, 427, 888 are the same, the splash shapes differ)
    SP = _a4_shape_pixels(ctx['R'], SPLASH_SWF, SPLASH_CLIPS if own else SPLASH_CLIPS_RED)
    spl = body & (SP == SPLASH_CODE if own else np.isin(SP, SPLASH_CODES_RED))
    if not spl.any(): return body
    # 1. the splash and the sprite-edge pixels around it that touch no bullet shape
    own_sh = body & (SP > 0) & ~spl
    edge = body & (SP == 0) & _dilate(spl, 2)
    out = body & ~(spl | (edge & ~_dilate(own_sh, 1)))
    # 2. the enclosed notch the splash leaves in the bullet's outline: filled
    out = out | (binary_fill_holes(out) & ~out & _dilate(spl, 3))
    # 3. separate pieces beside the splash without the bright head, mostly outside the predicted label: the splash's leftovers
    core = np.isin(SP, SPLASH_HEAD)
    if not (out & core).any(): return out     # no bullet head left: keep what is there (never lose the projectile)
    L, n = cc(out, structure=np.ones((3, 3)))
    if n > 1:
        has_head = np.bincount(L[out & core].ravel(), minlength=n + 1) > 0
        near_c = np.bincount(L[out & _dilate(spl, SPLASH_NEAR)].ravel(), minlength=n + 1) > 0
        sz = np.bincount(L[out].ravel(), minlength=n + 1)
        frac = np.bincount(L[out & _dilate(ctx['pred_any'], 3)].ravel(), minlength=n + 1) / np.maximum(sz, 1)
        drop = np.nonzero(near_c & ~has_head & (frac < SPLASH_INPRED))[0]
        drop = drop[drop > 0]
        if drop.size: out = out & ~np.isin(L, drop)
    return out


for _t in sorted(COLT005_SPLASH):
    _s306_prev = POST_BODY.get(_t)
    POST_BODY[_t] = (lambda prev: (lambda body, ctx: _s306_splash(prev(body, ctx) if prev else body, ctx)))(_s306_prev)
