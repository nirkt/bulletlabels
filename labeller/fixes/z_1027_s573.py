# sweep 10-27 fix for skin 573 (Haiku fix, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s573/)
# s573 (sweep 2026-10-27, PowerLevelerOnceupon, Surge 007): Surge007Projectile (6000804), the attack at 0 and 90 deg.
# Defect 1 (findings 10020, 10021, 10025, 10026, 10027): the label of the blast had holes over its bright white core
# (inner red loops). The core is drawn by translucent layers of the attack's own clip (coverage 32..127, under CUT) and by
# sprite-less pixels, so the body's enclosed gaps are the attack itself. Fix: every enclosed hole of the body is filled.
# Defect 2 (small specks seen while checking the fix on ticks 267-311): a few-px green fleck beside the blast (tick 269,
# tick 311) stays as a separate bit of the label. Fix: parts of this body smaller than 40 px and under 10% of its largest
# part are dropped (the attack's own flame is the largest part; the flame's tongue stays).
# Only this type is touched (its clip surge_007_proj_sword_01 is used by no other type).
def _s573_fix_6000804(body, ctx):
    from scipy.ndimage import binary_fill_holes, label as cc
    if not body.any(): return body
    b = binary_fill_holes(body) & ctx['region']
    L, n = cc(b)
    if n > 1:
        sz = np.bincount(L.ravel())[1:]
        drop = 1 + np.nonzero((sz < 40) & (sz < 0.1 * sz.max()))[0]
        b = b & ~np.isin(L, drop)
    return b
POST_BODY[6000804] = _s573_fix_6000804
