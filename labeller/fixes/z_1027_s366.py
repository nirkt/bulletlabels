# sweep 10-27 fix for skin 366 (Haiku fix, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s366/)
# sweep 10-27 fix for skin 366 (BarkeepPirate), task s366 (sweep 2026-10-27, Haiku fix agent).
# Types: 6000262 = Barley008Projectile (the 0 deg and 90 deg ball, clip barley_008_projectile, sc/effects_brawler.sc);
#        6000263 = Barley008UltiProjectile (the super balls, same clip).
# Defects (pairs 6879 tick 269, 6880 tick 281, 6886 tick 319; pair 6892 super at tick 369, nearest captured ticks 367/370):
# detached blue puff specks beside the ball, the blue drip (shape 407) with thin streaks off the spout end (a spur), and the
# super label reaching into the blue splash inside the yellow zone (shape 407). Trail smoke, drips and splash are not the
# ball / balls (rules 1, 2, 3). Clip shape ids measured on the kept capture sw2_366 (mean frame colour of each shape):
#   ball = shapes 1 (white sleeve), 92 (blue side), 19846 (cyan highlight and spout), 21409 (yellow star), 0 (dark rim),
#          681 (blue rim)
#   407 = blue drip / splash (83,137,246), 63 = blue puff sprite (82,133,248), 61 / 62 = puff sprites
#   'none' (shape -1) = sprite edge pixels, 1-2 px around the shapes
# Attack (6000262): candidate = the label's ball-shape pixels, its puff pixels inside the convex hull of the ball (a puff
#   drawn over the ball hides it, the ball is there), its shape-less edge pixels within 2 px of the ball; holes filled.
#   Thin parts (the streaks and drip threads) are opened away, the solid ball shapes (white, blue, yellow star) kept whole;
#   the largest part kept, so detached specks never stay. The cyan spout stays while it is thick.
# Super (6000263): the super's ball = the largest connected part of its ball-shape pixels, plus its shape-less edge pixels
#   within 2 px, holes filled. No convex hull (the splash lies inside the hull between the balls): the splash and the puffs
#   are never part of the super label.
BARLEY008 = 6000262
BARLEY008_ULTI = 6000263
BARLEY008_CLIP = ('barley_008_projectile',)
BARLEY008_BALL = (1, 92, 19846, 21409, 0, 681)
BARLEY008_PUFF = (407, 63, 61, 62)
BARLEY008_SOLID = (1, 92, 21409)
BARLEY008_OPEN = 2   # px (half resolution): thin parts thinner than about 2 x this are opened away


def _fix_6000262(body, ctx):
    if not body.any(): return body
    from scipy.ndimage import binary_fill_holes, binary_opening, label as cc
    SP = _a4_shape_pixels(ctx['R'], 'sc/effects_brawler.sc', BARLEY008_CLIP) - 1   # clip shape id per pixel, -1: none
    ball = body & np.isin(SP, BARLEY008_BALL)
    if not ball.any(): return body
    hull = hull_mask(ball)
    puff = body & np.isin(SP, BARLEY008_PUFF) & hull
    edge = body & (SP < 0) & _dilate(ball, 2)
    out = binary_fill_holes(ball | puff | edge)
    op = binary_opening(out, iterations=BARLEY008_OPEN)
    keep = op | (out & np.isin(SP, BARLEY008_SOLID))
    L, n = cc(keep)
    if n > 1:
        sz = np.bincount(L.ravel())[1:]; keep = L == (1 + int(np.argmax(sz)))
    return binary_fill_holes(keep)


def _fix_6000263(body, ctx):
    if not body.any(): return body
    from scipy.ndimage import binary_fill_holes, label as cc
    SP = _a4_shape_pixels(ctx['R'], 'sc/effects_brawler.sc', BARLEY008_CLIP) - 1   # clip shape id per pixel, -1: none
    ball = body & np.isin(SP, BARLEY008_BALL)
    if not ball.any(): return body
    L, n = cc(ball)
    if n > 1:
        sz = np.bincount(L.ravel())[1:]; ball = L == (1 + int(np.argmax(sz)))
    edge = body & (SP < 0) & _dilate(ball, 2)
    out = binary_fill_holes(ball | edge)
    L, n = cc(out)
    if n > 1:
        sz = np.bincount(L.ravel())[1:]; out = L == (1 + int(np.argmax(sz)))
    return out


POST_BODY[BARLEY008] = _fix_6000262
POST_BODY[BARLEY008_ULTI] = _fix_6000263
