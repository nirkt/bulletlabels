# sweep 10-27 fix for skin 192 (Haiku fix, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s192/)
# Skin 192 ClusterBombDudeSB (brawler TICK, attack 0 deg), tick 265, pair 3376 (sweep 2026-10-27, Haiku task s192):
# extra_label. The bomb of Tick007Projectile (type 6000539, clip tick_007_atk1_projectile, sc/effects_brawler.sc) has a
# sub-clip with a blue puff (shape 63) drawn at its origin. At the launch tick (265) that puff sits over the character's
# head, 15-25 px left of the grey bomb, and its right part was labelled as body with the bomb. The puff is a launch /
# impact effect on the character, not the projectile (rules 2 and 4: hit, impact and burst effects are not labelled).
# The same puff is also labelled at the launch of the attack 90 deg row (ticks 305 and 307, same type, same shape; not in the
# finding, removed by the same rule: the blue cloud over the head is the launch effect there too, and 307 has a 23 px speck
# of it). No other tick of the capture has shape 63 in a label (checked over all frames of sw2_192). Other objects and
# all other frames are unchanged (checked: identical label maps at every tick but 265, 305, 307). The fix is keyed to the
# type id and the shape id (both stable across captures): the grey bomb's own shapes (38363, 38364, 38365, 38366 and
# its other shapes 93, 138, 828, 1013, 3684, 47) stay exactly as they are, and no-shape edge pixels are dropped only where
# they touch the puff and no kept shape.
TICK007_PUFF = {6000539: 63}   # type -> the puff's shape id (sc/effects_brawler.sc)
TICK007_FILE = 'sc/effects_brawler.sc'


def _tick007_puff(body, ctx):
    """body without the launch puff (shape TICK007_PUFF) and the edge pixels that only the puff touches"""
    if not body.any(): return body
    t = ctx['t']; T = projtab.table()[t]
    names = tuple(sorted({n for f, n in _trail_clips(t) if f == TICK007_FILE} | {n for n in (T['blue'], T['red']) if n}))
    SP = _a4_shape_pixels(ctx['R'], TICK007_FILE, names)   # shape id + 1 per pixel, 0: none of the clip's shapes
    pid = TICK007_PUFF[t] + 1
    puff = body & (SP == pid)
    if not puff.any(): return body
    kept = body & (SP > 0) & (SP != pid)
    edge = body & (SP == 0) & _dilate(puff, 2) & ~_dilate(kept, 2)
    return body & ~(puff | edge)


for _t in TICK007_PUFF: POST_BODY[_t] = _tick007_puff
