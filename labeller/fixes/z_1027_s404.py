# sweep 10-27 fix for skin 404 (Haiku fix, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s404/)
# Sweep 2026-10-27, skin 404 RuffsSandsOfTimes (pairs 7601, 7615, 7616, 7617). Two Ruffs006 types, both shared with the other
# Ruffs skins that use the same attack / super (see report.md).
#
# (A) Ruffs006Projectile (6000898), attack 0 deg, pair 7601 (tick 265, 267):
#     the label's left flank bulged out over the soft glow sprites of the bullet (clip shapes 145 and 2598 of
#     sc/effects_brawler.sc, classed 'glow' as in the predicted labels). The aligned-prediction rule took those glow pixels into
#     the body, so the outline ran about 6 px beyond the blue spiky core (rule 2: no soft glow / halo). The glow pixels (front-most
#     draw shows one of those two shapes) are removed from the body, then the body is filled back in where the glow is enclosed by
#     the body (its highlights on the core stay). Separate pieces made only of the trail's dots drawn over the predicted body (code
#     4, no clip shape: the orange dots in the smoke, a speck below the bullet at tick 267) are not the bullet (rule 3) and are dropped.
#     The clip shapes of the body (47, 63, 143, 1013, 2596, 3784) keep their pixels exactly (checked on ticks 265, 267).
#
# (B) Ruffs006UltiProjectile (6000899), super 45 deg, pairs 7615, 7616, 7617 (ticks 353, 355, 359): the yellow wings (clip shape 47
#     of sc/effects_brawler.sc, translucent: coverage median 67) lay outside the outline, though they are the same yellow as the
#     core (rule 1: the visible object, filled). Shape 47 is the whole round yellow sprite, its low-coverage edge is the teal glow,
#     so the wing pixels are taken only where they are warm-coloured (red >= 120 and red >= green: yellow, not teal) and only
#     the warm pixels joined to the body; the teal glow stays out (a plain coverage cut, FADE_CUT, drew a disc with the glow in it).
#     The added wing pixels are opened by a 3x3 cross-free square (1-2 px strips go: the yellow sparkle's spur above the head uses
#     shape 47 too, and the wing's own parts are 3 px and more). Holes are filled inside the region.
from scipy.ndimage import binary_fill_holes as _s404_fill, label as _s404_cc, binary_opening as _s404_open

RUFFS006_CLIPS = ('ruffs_006_atk_projectile', 'ruffs_006_atk_projectile_red')
RUFFS006_GLOW_SHAPES = (145, 2598)
RUFFS006ULTI_CLIPS = ('ruffs_006_ulti_projectile', 'ruffs_006_ulti_projectile_red')
RUFFS006ULTI_WING_SHAPE = 47
RUFFS006ULTI_CV = 32


def _s404_ruffs006(body, ctx):
    if not body.any(): return body
    SP = _a4_shape_pixels(ctx['R'], 'sc/effects_brawler.sc', RUFFS006_CLIPS)   # shape id + 1, 0: none of its shapes
    C = ctx['C']
    m = body & ~np.isin(SP, [s + 1 for s in RUFFS006_GLOW_SHAPES])
    if not m.any(): return body
    m = _s404_fill(m) & body                          # glow enclosed by the body (highlights on the core) is back in
    L, n = _s404_cc(m, structure=np.ones((3, 3), bool))
    if n > 1:
        for k in range(1, n + 1):
            c = L == k
            if (SP[c] == 0).all() and (C[c] == 4).all(): m = m & ~c   # trail dots apart from the bullet
    return m


def _s404_ruffs006ulti(body, ctx):
    if not body.any(): return body
    SP = _a4_shape_pixels(ctx['R'], 'sc/effects_brawler.sc', RUFFS006ULTI_CLIPS)
    img = ctx['img'].astype(np.int16)
    warm = (img[..., 0] >= 120) & (img[..., 0] >= img[..., 1])
    cand = (SP == RUFFS006ULTI_WING_SHAPE + 1) & (ctx['cv'] >= RUFFS006ULTI_CV) & warm & ctx['region'] & ~body
    # thin strips of the wing shape (1-2 px: the sparkle's spur above the head, which uses shape 47 too) are opened away;
    # the wing's own parts (3 px and more) stay. The body's own pixels are not opened.
    add = _s404_open(cand, np.ones((3, 3), bool))
    m = body | add
    L, n = _s404_cc(m, structure=np.ones((3, 3), bool))
    keep = np.unique(L[body]); keep = keep[keep > 0]
    m = np.isin(L, keep) & m
    # a 1-2 px spur (the body's own pixels too: the sparkle's spur above the head at tick 355) is opened away: about 2% of the label
    m = _s404_open(m, np.ones((3, 3), bool))
    return _s404_fill(m) & ctx['region']


POST_BODY[6000898] = _s404_ruffs006
POST_BODY[6000899] = _s404_ruffs006ulti
