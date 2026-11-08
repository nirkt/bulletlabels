# sweep 10-27 fix for skin 145 (Sonnet revision, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s145/)
# sweep 10-27 fix (2nd try) for skin 145 (Dancer199Gem), super 6001669 (Dancer002ProjectileUlti, clip mina_002_ulti_projectile
# of sc/effects_brawler_mina.sc): findings 2389-2393 (ticks 347-359): the water tornado was labelled as separate arcs per
# opaque ring segment (shapes 203-206 = the ring segments, 209 = the ring's arrow, 4/113/114/115/202 = the translucent funnel).
# The tornado is one multi-part object -> ONE convex hull of its labelled parts (as for the sibling Dancer003 super 6001673,
# skin 149, a1_agent1.py), no coverage lowering / closing (the first attempt's FADE_CUT + closing took in haze and glow and left
# notches and spurs). Only the arrow (shape 209), whose pixels are partly translucent or under the crates/brawler, so it was
# missing or cut at ticks 357-359, is added back (coverage >= 32, near the body), so the hull covers the whole ring.
# Keyed to this type only (no other skin uses 6001669).
_F6001669 = dict(file='sc/effects_brawler_mina.sc', clips=('mina_002_ulti_projectile', 'mina_002_ulti_projectile_red'),
                 arrow=(209,), near=24, cut=32)


def _fix_6001669(body, ctx):
    p = _F6001669
    if not body.any(): return body
    SP = _a4_shape_pixels(ctx['R'], p['file'], p['clips'])
    arrow = ctx['region'] & np.isin(SP, [s + 1 for s in p['arrow']]) & (ctx['cv'] >= p['cut']) & _dilate(body, p['near'])
    return body | arrow


POST_BODY[6001669] = _fix_6001669
ONE_LABEL[6001669] = 'hull'
