# sweep 10-27 fix for skin 416 (Haiku fix, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s416/)

# Fire Dude Fantasy 416 (brawler Amber), type 6000837 FireDude005Projectile: the blue flame's label took in a glow bulb
# under the flame's dark tip (sweep 2026-10-27, skin 416 'sw2_416 attack 90 deg', ticks 311/313/315; pairs 7841-7843).
# Cause: the bulb is shape 0 of the attack's own clip (amber_005_atk_projectile, paint layer 'glow', the floor glow under the
# flame). It is connected to the body by a neck, so the connected-part rule of z_1024 (_t15_body) kept it.
# Fix: wrap z_1024's POST_BODY for 6000837: drop the pixels of that clip shape (shape 0), then close, fill holes and drop
# specks again, exactly as z_1024 does. The flame's own shapes and the glow fringe along its sides are kept.
# Amber clips only; the super (6000838, amber_005_ulti_projectile) is untouched.

_S416_CLIP = 'sc/effects_brawler_amber.sc'
_S416_BULB_SHAPES = {6000837: (0,)}   # clip shape ids that are the floor-glow bulb under the flame (not the body)
_s416_prev_body = POST_BODY.get(6000837)


def _s416_body(body, ctx):
    t, R, region, cv = ctx['t'], ctx['R'], ctx['region'], ctx['cv']
    m = _s416_prev_body(body, ctx) if _s416_prev_body is not None else body
    if not m.any():
        return m
    SP = _a4_shape_pixels(R, _S416_CLIP, _T15_CLIPS[t])   # shape id + 1 per pixel (half resolution), 0 = no shape
    bulb = np.isin(SP, [s + 1 for s in _S416_BULB_SHAPES[t]])
    m = m & ~bulb
    if not m.any():
        return m
    # the glow under the flame's dark tip (sprite code 2 = clip glow) lies below the body's lowest body-coded row
    C = ctx.get('C')
    if C is not None and C.shape == m.shape:
        tipr = np.nonzero((m & (C == 1)).any(1))[0]
        if tipr.size:
            below = np.zeros(m.shape, bool); below[tipr.max() + 2:] = True
            m = m & ~((C == 2) & below)
    if not m.any():
        return m
    m = _close(m, 1) & region
    m = binary_fill_holes(m) & region
    if m.any():
        L, n = _cc(m)
        if n > 1:
            sz = np.bincount(L.ravel())[1:]; m = np.isin(L, 1 + np.nonzero(sz >= 0.2 * sz.max())[0])
    return m


POST_BODY[6000837] = _s416_body
