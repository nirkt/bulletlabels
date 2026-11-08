# sweep 10-27 fix for skin 313 (Haiku fix, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s313/)
# Haiku fix task s313 (sweep 2026-10-27), skin 313 ArtilleryDudeZombie, finding pair 5864 (too_big, super 45 deg, tick 389).
# Finding: the white landing marker (kind 'target') of the turret bomb is drawn about 28 px taller than the drawn blue / orange
# ring it marks (in the 1200 x 540 frame: about 78 px tall and centred about 11 px above the ring, against 58 px). Its top
# rises above the ring and takes in Bot 9's head and the bushes behind.
# Cause (z_1024_penny_classes.labels): the marker circle is the ground circle at the bomb's logic end point, with radius the
# 98th percentile of the marker sprite's pixels in ground units, drawn as a screen ellipse from the camera's ground projection
# (_proj of x +- 100, y +- 100). The drawn ring is flatter than that projection (its vertical radius is about 0.78 of the
# projected one, its width about 0.92) and its centre lies about 11 px below the logic end point, so the ellipse overshoots at
# the top. Measured on sw2_313 ticks 379-401: the marker's own pixels span x 479..571 and y 156..212 (centre 525, 184.5; radii 46,
# 28), the label ellipse was centre (528, 173), radii (50, 39). The same offset shows on sw2_430 (Penny008 turret, same sprite).
# Approaches considered:
#   1. scale the projected ellipse's vertical radius by a constant (the A3_DISC_SCALE way): rejected, the centre is off by about
#      11 px too, so a scale alone cannot place the ring;
#   2. move the centre to the logic point and fit the radii to the marker pixels: rejected, the centre is the error;
#   3. fit the centre and both radii to the marker sprite's own pixels (0.5 and 99.5 percentiles of its x and y), per frame:
#      chosen. It measures the drawn ring itself, is the same measure for every turret type (one shared marker sprite), and the
#      extents were identical over the 23 ticks checked on sw2_313 and sw2_430. It relies on the marker being visible on its
#      extremes; a ring cut by an occluder in some frame would be smaller there (the marker is not cut by occluders in the checked
#      ticks; the bbox held).
# Keyed to the 8 Penny-style turret projectiles (PENNY_TURRET: the same marker sprite gadget_primo_meteor_target). Verified on
# 6000460 (sw2_313, the zombie skin) and 6000468 (sw2_430). The label is still the whole marker circle, solid, cut only by solid
# map blocks (_zs_solid), and it never takes pixels of another label.

_s313_prev_labels = labels
S313_PCT = (0.5, 99.5)   # percentiles of the marker's own pixels: robust to a stray pixel


def _s313_marker_ellipse(M):
    """centre x, centre y, radius x, radius y (1200 x 540 frame px) of the marker's own pixels"""
    ys, xs = np.nonzero(M)
    x0, x1 = np.percentile(xs, S313_PCT); y0, y1 = np.percentile(ys, S313_PCT)
    return (x0 + x1) / 2.0, (y0 + y1) / 2.0, max(1.0, (x1 - x0) / 2.0), max(1.0, (y1 - y0) / 2.0)


def labels(objs, cam, R, state=None, ended=(), img=None):
    P = _s313_prev_labels(objs, cam, R, state=state, ended=ended, img=img)
    if not P.get('targets'): return P
    import bulletlabels as BL
    ids, lids, share = P['ids'], list(P['lids']), P['share']; H, W = ids.shape
    try: solid = _zs_solid(cam, R)
    except Exception: solid = np.zeros((H, W), bool)
    cv = R['cv'][::2, ::2]; yy, xx = np.mgrid[0:H, 0:W]
    for o in objs:
        if o['t'] not in PENNY_TURRET or not o.get('drawn') or not o.get('target'): continue
        oid = o['id'] + PENNY_TARGET_ID
        ks = [q for q, l in enumerate(lids) if l[0] == oid and l[2] == 'target']
        if not ks: continue                       # no marker label this frame (the prev labels found no marker pixels)
        k = ks[0]
        M = (sprite_pixels(('pre', o['t']), o['own'], R) > 0) & (cv >= PENNY_MARK_CV)
        if M.sum() < PENNY_MARK_MIN: continue
        cx, cy, rx, ry = _s313_marker_ellipse(M)
        disc = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1
        old = ids == k
        new = disc & ~solid & ((ids < 0) | old)  # free pixels, or the marker's own old pixels; never another label's
        if not new.any(): continue
        ids[old] = -1; share[old] = 0
        ids[new] = k; share[new] = BL.CUT
    P['ids'] = ids; P['share'] = share; P['lids'] = lids
    return P
