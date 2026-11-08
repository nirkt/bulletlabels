# sweep 10-27 fix for skin 625 (Sonnet revision, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s625/)
# Skin 625 LuchadorMHA, 2nd attempt (task s625; pairs 10572/10574/10575/10577, attack 90 deg ticks 305-315, and the same
# type's 0 deg row ticks 263-277): the label of Primo014Projectile (type 6000410, skin 625 only) was a comb of spikes:
# the fist + striped arm is drawn from blended sprites (and an opaque mesh at the fist), with the black slash streaks on top;
# the streak sprites poke out of the arm over the brawler's head / torso and leave interior lines in the label.
# The first attempt used a plain convex hull, which cut across the crate drawn over the fist and took in head, torso, glow
# and background. This one keeps the object's own outline (no hull, so the notch around the crate stays): the body pixels
# are opened with a disk wider than the streaks (they are <= ~8 px wide, the arm / fist 25+ px) which removes the spikes,
# the result is closed and filled solid; the fist's soft crescent halo (shape 0 of the clip) is dropped first, and everything stays restricted to the body's own pixels (+ its filled interior).
_LU_OPEN = 6
_LU_CLOSE = 5
def _lu_disk(k):
    yy, xx = np.mgrid[-k:k + 1, -k:k + 1]
    return (xx ** 2 + yy ** 2) <= k * k
_LU_FILE = 'sc/effects_brawler_elprimo.sc'
_LU_CLIPS = ('primo_014_atk_projectile', 'primo_014_atk_projectile_red')
_LU_GLOW_SHAPE = 0     # shape id of the soft crescent halo round the fist (clip primo_014_atk_projectile); not the object
def _luch_body(body, ctx):
    from scipy.ndimage import binary_fill_holes, binary_closing, binary_opening, binary_dilation, label as cc
    if not body.any(): return body
    orig = body
    try:
        SP = _a4_shape_pixels(ctx['R'], _LU_FILE, _LU_CLIPS)   # shape id + 1 per pixel
        body = body & (SP != _LU_GLOW_SHAPE + 1)
    except Exception: pass
    if body.sum() < 0.5 * orig.sum(): body = orig
    p = 16
    B = np.pad(body, p)
    B = binary_fill_holes(binary_closing(B, structure=_lu_disk(2)))
    core = binary_opening(B, structure=_lu_disk(_LU_OPEN))
    if core.sum() < 0.3 * body.sum(): return orig
    L, n = cc(core)
    if n > 1:
        sz = np.bincount(L.ravel())[1:]; core = np.isin(L, 1 + np.nonzero(sz >= 0.2 * sz.max())[0])
    # restore the edge the opening rounded off: body pixels (not thin streaks) within the opening radius of the core
    core = core | (B & binary_dilation(core, structure=_lu_disk(_LU_OPEN)) & binary_opening(B, structure=_lu_disk(2)))
    core = binary_fill_holes(binary_closing(core, structure=_lu_disk(_LU_CLOSE)))
    return core[p:-p, p:-p]
POST_BODY[6000410] = _luch_body
