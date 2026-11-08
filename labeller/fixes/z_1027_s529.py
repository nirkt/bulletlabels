# sweep 10-27 fix for skin 529 (Haiku fix, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s529/)
# sweep 10-27 Haiku fix for skin 529 (BarrelBotCupcake, Darryl 007, attack 90 deg), task out/sweep2/haikutasks/sweep1027/s529/
# Finding pair 9541, tick 311 (hole): the three-swirl train's label had a closed loop around a small purple core inside the
# middle swirl. The core is the swirl's own opaque draw (a separate lilac overlay draw, cv 255, not one of the clip's
# sprites, so its sprite code is 0), and it lies in a notch of the label, so the solid label was not solid there.
# The same purple core is a 48 px enclosed hole in the left swirl at tick 309 (same skin, same row, same draw kind; the
# clean-up fills holes of at most 40 px only and missed it). Both are the projectile's own purple core, not ground:
#  - enclosed unlabelled pieces of at most DARRYL_CORE_MAX px (full res) inside the body are filled (tick 309);
#  - unlabelled purple pixels (R >= 80, B >= R, B - G >= 45, G <= R + 8: the core and its darker rim; the cyan spark
#    beside it, the blue glow and the dark violet ground around the train are not purple by this test) that lie in a notch of
#    the body (inside its closing by 3 px) are taken in, in pieces of at most DARRYL_CORE_MAX px (tick 311). The purple haze
#    outside a convex edge of a swirl stays out. The cyan spark stays out.
# Keyed to the type id 6000433 only (Darryl007Projectile: the same projectile in every Darryl skin that shows the purple core).
# Checked on ticks 263-277 (attack 0 deg) and 303-317 (attack 90 deg) of sw2_529: changes only at 309 and 311 (see report).
DARRYL_CORE = {6000433}
DARRYL_CORE_MAX = 60   # px (full resolution): largest unlabelled piece taken into the body


def _darryl_core(body, ctx):
    from scipy.ndimage import label as cc, binary_fill_holes
    if not body.any() or ctx['img'] is None: return body
    im = ctx['img'].astype(np.int16); r, g, b = im[..., 0], im[..., 1], im[..., 2]
    purple = (r >= 80) & (b >= r) & (b - g >= 45) & (g <= r + 8)
    region = ctx['region']
    out = body.copy()
    # enclosed unlabelled pieces (the core of tick 309)
    enc = binary_fill_holes(body) & ~body & region
    if enc.any():
        L, n = cc(enc); sz = np.bincount(L.ravel())
        for j in range(1, n + 1):
            if sz[j] <= DARRYL_CORE_MAX: out |= L == j
    # purple pieces in a notch of the body (the core of tick 311): inside the body's closing, so the haze that lies
    # outside a convex edge (the purple light above a swirl) stays out
    cand = purple & ~out & region & _close(out, 3)
    if cand.any():
        L, n = cc(cand); sz = np.bincount(L.ravel())
        touch = np.unique(L[_dilate(out, 1) & cand])
        for j in touch:
            if j > 0 and sz[j] <= DARRYL_CORE_MAX: out |= L == j
    return out


for _t in DARRYL_CORE: POST_BODY[_t] = _darryl_core
