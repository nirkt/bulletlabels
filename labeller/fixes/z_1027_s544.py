# sweep 10-27 fix for skin 544 (Haiku fix, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s544/)
# sweep 10-27 fix for skin 544 (RollerUniversity, Haiku task s544, sweep1027; task folder out/sweep2/haikutasks/sweep1027/s544/)
# Fix sw2_544 pair 9712 (attack 0 deg, tick 271, extra_label): a detached cyan bubble with a white highlight, lying between the
# Roller 003 shell and the cyan impact splash at the right edge of the frame, is outlined red as part of the attack. Rule 4:
# impact splashes and the bubbles of an impact are never labelled.
# Cause (measured on the capture, sw2_544 attack 0 deg, ticks 263-273, type 6000908 Roller003Projectile, clip
# sc/effects_brawler.sc): the attack's label takes in shape 19042 (the cyan bubble sprite of this type's own clip), which at tick 271
# is a separate 41 px piece about 6 px past the shell's front (x 780-791, the splash's bubble). Shape 19042 is also the cyan tail at
# the shell's rear edge at ticks 267-269 (16-29 px, touching the shell): that part IS the attack and stays.
# Fix (POST_BODY, this type only, before the shared clean-up): shape-19042 pieces of the body that do not touch the rest of the body
# (within S544_GAP px, full logic resolution) are removed. The shell's own pieces, the rear tail (touching) and every other shape are
# not changed, so a projectile never loses its label to this rule unless its only piece is the bubble.
# Other types and other skins are unaffected (keyed to 6000908 only).
S544_TYPE = 6000908          # Roller003Projectile (Roller 003 shell; sw2_544 RollerUniversity)
S544_FILE = 'sc/effects_brawler.sc'
S544_BUBBLE = 19042          # shape id of the cyan bubble sprite (white highlight), clip of this type
S544_GAP = 1                 # px (half resolution): a bubble piece within this distance of the rest of the body is kept


def _s544_bubble(body, ctx):
    import numpy as _np
    from scipy.ndimage import label as _cc, binary_dilation as _bd
    if not body.any(): return body
    t = ctx['t']; T = projtab.table()[t]
    names = tuple(sorted({n for f, n in _trail_clips(t) if f == S544_FILE} | {n for n in (T['blue'], T['red']) if n}))
    SP = _a4_shape_pixels(ctx['R'], S544_FILE, names)   # shape id + 1 per pixel, 0: none (half resolution, like body)
    bub = body & (SP == S544_BUBBLE + 1)
    if not bub.any(): return body
    rest = body & ~bub
    if not rest.any(): return body                      # the bubble is the only piece: keep the label (never lose it)
    L, n = _cc(bub, structure=_np.ones((3, 3), bool))
    touch = _bd(rest, structure=_np.ones((3, 3), bool), iterations=S544_GAP)
    drop = [j + 1 for j in range(n) if not (touch & (L == j + 1)).any()]
    if not drop: return body
    return body & ~_np.isin(L, drop)


POST_BODY[S544_TYPE] = _s544_bubble
