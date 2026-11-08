# sweep 10-27 fix for skin 194 (Haiku fix, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s194/)
# sweep 10-27 fix for skin 194 (AmbusherAngel = Lily Angel; Haiku fix task out/sweep2/haikutasks/sweep1027/s194/).
# Findings: pair 3423 (attack 0 deg, tick 265, hole), pair 3427 (attack 90 deg, tick 305, hole: an enclosed loop in the purple
# core plus a tiny closed mark above the tip), pair 3424 (attack 0 deg, tick 267, extra_label: already resolved in the new
# capture, nothing is labelled there now; the 1 px left at tick 307 goes too).
# Type 6001455 (Ambusher002Projectile, AmbusherAngel): the flame is drawn by its trail clip lily_002_atk_projectile_emitter.
# Its body is translucent (coverage 26-127) except the opaque pink rim and the opaque bits. The agent-4 rule (POST_BODY in
# a4_agent4.py: the emitter's pixels with coverage >= 128, holes filled) kept only the rim, the purple core and the wing tips,
# so the blue body, the cyan wings and the white heart were unlabelled, and the purple core was a separate labelled blob in
# the middle of them (its outline is the closed loop of the findings; the tiny mark is a second small unlabelled spot).
# Fix (POST_BODY of this type only, same region and emitter restriction as before):
#  - op = the opaque emitter pixels (the agent-4 body); if op is under 20 px the streak has faded (rule 2): nothing is labelled;
#  - the flame's silhouette = the convex hull of op; the emitter's own translucent pixels inside it are added (the body, the
#    purple core, the white heart); feathers and twinkle sprites are not emitter sprites, so they stay out, and a twinkle
#    enclosed by the body is filled as for every trail-body type; holes filled, gaps of up to 4 px closed.
# Checked on the capture: tick 265 and 305 become one solid label, no holes; tick 307 (faded, 1 px) and 267 / 309
# stay empty. Other types are untouched. Not changed (needs_user, not a finding): the yellow sparkles on the wing edges and the
# cyan twinkle in the body are emitter or enclosed sprites and stay in the solid label, as the existing rules have them.

_S194_MIN = 20   # px: opaque emitter pixels from which the streak counts as not faded


def _s194_fix_6001455(body, ctx):
    from scipy.ndimage import binary_fill_holes
    R = ctx['R']; region = ctx['region']
    em = _a4_shape_pixels(R, 'sc/effects_brawler.sc', ('lily_002_atk_projectile_emitter',)) > 0
    op = body & em
    if op.sum() < _S194_MIN: return np.zeros_like(body)
    sil = hull_mask(op)
    b = op | (sil & em & region)
    return binary_fill_holes(_close(b, 2)) & region


POST_BODY[6001455] = _s194_fix_6001455
