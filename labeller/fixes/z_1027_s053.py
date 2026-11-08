# sweep 10-27 fix for skin 053 (Haiku fix, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s053/)
# FireDude (Amber) super, skin 053 FireDudeDefault (sweep 2026-10-27, pair 973 spur_or_speck; rows "sw2_053 super 45 deg").
# Type 6000829 FireDudeUltiProjectile (super orb, clip amber_def_ulti_projectile / _red, sc/effects_brawler.sc).
# Cause: the label took the orb's flame wisps and the blue droplets of its trail. Per pixel, the orb's own sprite is shape
# 64136 (the disc, ~830 px); shape 407 is the flame wisps drawn on its upper-right and lower-left (a spur above-right on
# tick 353, a blob beside the orb on 345/347), shape 63 is the small blue droplets above it. The trail's shapes (143, 888)
# are not in the label. Fix: the label is the pixels of the orb's own shape (per pixel, _a4_shape_pixels), plus pixels with
# no shape (sprite edges) within 2 px of it, then its convex hull (the rim a wisp covers is restored; the disc adds nothing
# outside its silhouette), wisp pixels touching the rim (1 px), one-pixel nubs opened away, holes filled, one part. The orb
# is never dropped (if its shape is absent the label is left as it was). Only type 6000829 is changed.
FIRE_ULTI_ORB = {6000829: dict(file='sc/effects_brawler.sc', clips=('amber_def_ulti_projectile', 'amber_def_ulti_projectile_red'),
                               orb=(64136,), edge=2)}


def _fire_ulti_orb(body, ctx):
    if not body.any(): return body
    from scipy.ndimage import binary_fill_holes, label as cc
    p = FIRE_ULTI_ORB[ctx['t']]
    SP = _a4_shape_pixels(ctx['R'], p['file'], p['clips'])   # shape id + 1 per pixel, 0: none of the clip's shapes
    orb = body & np.isin(SP, [s + 1 for s in p['orb']])
    if not orb.any(): return body
    other = body & (SP == 0) & _dilate(orb, p['edge'])   # sprite edge pixels next to the orb
    # the orb's rim where a flame wisp (shape 407) is drawn over it is cut out of its shape: its convex hull (the orb is a
    # round disc) fills that notch in the silhouette and adds nothing outside it, so the wisps sticking out stay out
    disc = hull_mask(orb)
    m = (disc | other | (_dilate(disc, 1) & (SP > 0))) & ctx['region']   # wisp pixels touching the rim are the rim (1 px)
    from scipy.ndimage import binary_opening
    op = binary_opening(m) & ctx['region']   # single-pixel nubs of the edge pixels out (the disc has none)
    if op.any(): m = op
    m = binary_fill_holes(m) & ctx['region']
    L, n = cc(m)
    if n > 1:
        sz = np.bincount(L.ravel())[1:]; m = L == (1 + int(np.argmax(sz)))
    return m


for _t in FIRE_ULTI_ORB: POST_BODY[_t] = _fire_ulti_orb
