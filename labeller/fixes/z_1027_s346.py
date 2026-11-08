# sweep 10-27 fix for skin 346 (Haiku fix, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s346/)
# Skin 346 DeadMariachiorigins (Poco 007 skin; sweep 2026-10-27, task s346). Types 6000347 (Poco007Projectile, attack) and
# 6000348 (Poco007UltiProjectile, super); both in POCO_ATK / POCO_SUPER (ZONE, trail-body types).
#
# 1. Attack leftovers (pairs 6466 tick 283, 6474 tick 323): the wave group's logic objects stay alive for a few ticks after
#    the wave graphic is gone (no clip body drawn in their region: 0 px at ticks 279-283 and 319-323, 864-2456 px while the
#    wave is drawn). Their trail particles (the music notes) were still labelled as the attack, red-filled. Rule 4: the label
#    goes with the wave, so a trail-only region carries no label. Gate: fewer than 20 clip-body pixels in the region.
# 2. Super streaks (pairs 6478 tick 347, 6479 tick 349, 6480 tick 353): the super's body is several strokes per streak-clef
#    group, with gaps between them. Rule 6: one solid convex hull per attack. Groups = connected parts of the body (2 px
#    dilation: the strokes of one group touch; the groups are apart, 2 to 4 groups per tick, checked at ticks 345-353). Each
#    group's convex hull is its label. Gate: at least 1000 clip-body pixels (the streaks are drawn); the notes left at ticks
#    355-359 (about 100 clip pixels) keep their current labels.
# Only these two type ids change; no other skin's frames are touched (the super hull applies only with the streak clip drawn).
SKIN346_CLIP_MIN = 20          # px: clip-body pixels in the region below which the wave is gone
SKIN346_SUPER_MIN_CLIP = 1000  # px: clip-body pixels in the region from which the super's streaks are on screen
SKIN346_SUPER_LINK = 2         # px (half resolution): dilation that joins the strokes of one streak-clef group
SKIN346_SPECK = 20             # px (half resolution): groups smaller than this are dropped


def _skin346_attack(body, ctx):
    """Poco 007 attack: no label on trail particles once the wave graphic is gone (see 1 above)"""
    C, reg, cv = ctx['C'], ctx['region'], ctx['cv']
    if body.any() and int(((C == 1) & reg & (cv >= CUT)).sum()) < SKIN346_CLIP_MIN:
        return body & False
    return body


def _skin346_super(body, ctx):
    """Poco 007 super: one solid convex hull per streak-clef group (see 2 above)"""
    from scipy.ndimage import label as cc
    C, reg, cv = ctx['C'], ctx['region'], ctx['cv']
    if not body.any() or int(((C == 1) & reg & (cv >= CUT)).sum()) < SKIN346_SUPER_MIN_CLIP:
        return body
    L, n = cc(_dilate(body, SKIN346_SUPER_LINK))
    out = np.zeros_like(body)
    for k in np.unique(L[body]):
        m = body & (L == k)
        if m.sum() < SKIN346_SPECK: continue
        out |= hull_mask(m)
    return out


POST_BODY[6000347] = _skin346_attack
POST_BODY[6000348] = _skin346_super
