# sweep 10-27 fix for skin 546 (Haiku fix, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s546/)
# Skin 546 UndertakerUndertakerHockey, super 45 deg (sweep 2026-10-27, findings pair 9738 tick 355 and pair 9740 tick 367).
# Type 6000302 = Mortis006UltiProjectile (logic UndertakerUltiProjectile), clip mortis_006_ulti_projectile(_red) in
# sc/effects_brawler.sc. Its trail clips draw small white four-point glints in the blue smoke: shape 98 (cosmetic_twinkle_01)
# and shape 14390 (carl_atk_line_particle_1..3, also in this type's trail clips). The label took them as tiny magenta
# marks beside the attack (rule 3 of RULES.md: twinkles, sparks and trail effects are never labelled). They bridge into the
# projectile's own shapes within 3 px, so the clean-up's speck rule (z_1099) does not drop them.
# Fix: drop the label pixels whose sprite shape is one of these two trail-only shapes (shape id + 1 from _a4_shape_pixels,
# half resolution, like the body mask). The projectile's own clip shapes (0, 61, 63, 64, 145, 635, 14617) are kept: shape 0
# is shared with the twinkle clip, so it is not dropped. Any hook already registered for this type runs first.
MORTIS_TWINKLE = {6000302: dict(
    file='sc/effects_brawler.sc',
    clips=('carl_atk_line_particle_1', 'carl_atk_line_particle_2', 'carl_atk_line_particle_3', 'cosmetic_twinkle_01',
           'gen_circle_blur_strong', 'mortis_006_ulti_iceground', 'mortis_006_ulti_projectile',
           'mortis_006_ulti_projectile_red', 'particle_dust_trail_slow', 'particle_dust_trail_slower'),
    drop=(98, 14390))}
_mortis_prev_hooks = {t: POST_BODY[t] for t in MORTIS_TWINKLE if t in POST_BODY}


def _mortis_twinkle(body, ctx):
    if not body.any(): return body
    p = MORTIS_TWINKLE[ctx['t']]
    SP = _a4_shape_pixels(ctx['R'], p['file'], p['clips'])   # shape id + 1 per pixel, 0: none of these clips' shapes
    return body & ~np.isin(SP, [s + 1 for s in p['drop']])


def _mortis_hook(body, ctx):
    t = ctx['t']
    if t in _mortis_prev_hooks: body = _mortis_prev_hooks[t](body, ctx)
    return _mortis_twinkle(body, ctx)


for _t in MORTIS_TWINKLE: POST_BODY[_t] = _mortis_hook
