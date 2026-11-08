# sweep 10-27 fix for skin 605 (Sonnet revision, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s605/)
# Skin 605 MummySurfer, Emz007Projectile (type 6000689): the attack 0 deg (ticks 263-287) and attack 90 deg (ticks 303-327)
# clouds. Sweep 2026-10-27 findings 10361 (tick 267, too_small: the bright right-hand lobe and a notch in it were outside the
# outline) and 10368 (tick 305, holes: gone after the shared clean-up). Second version (fix2) after fix.py was rejected: it
# left notches, slivers and loops. Causes, measured on W/runs/gc/sw2_605:
#  1. The cloud's translucent clip edge (coverage 32..127, a 2-6 px band along the lobes' outline) lies under the body cut
#     (CUT = 128): not body. The attack's own clip pixels (shapes of the type's clip only: the trail haze has other shapes) up to
#     S605_EDGE px outside the label, coverage >= S605_EDGE_CV, grow the label. (A global lower FADE_CUT pulled in the dark shadow
#     under the cloud: rejected.)
#  2. z_1024_emz_attack takes only pixels that are not 'solid world' (_zs_solid: ID-pass depth inside a solid map tile). The cloud
#     draws no depth, so where a crate / wall stands BEHIND a lobe the depth is the crate's and the lobe's opaque pixels counted as
#     world: whole interior parts of a lobe stayed unlabelled and open to the outside (283: ~600 px, 267: the notch, 313-325: the
#     gaps). A pixel whose front-most draw shows the type's own clip shape is the cloud, in front of whatever is behind it (a
#     wall in front would be the front-most draw itself, not the clip), so those pixels are not world here: the opaque clip
#     pixels connected to the label are body.
#  3. Enclosed holes of the attack's own silhouette (seams between lobes, gaps between overlapping clouds) are filled. A hole is
#     filled when no other label owns a pixel of it (real world pixels inside a small hole too, like the shared clean-up).
# Other types are untouched. The shared clean-up (z_1099) stays valid after this (nothing here depends on it).
S605_TYPES = {6000689}
S605_EDGE = 2                    # px: the translucent clip edge is taken this far outside the label
S605_EDGE_CV = 32                # coverage (0..255) from which the clip edge is body
S605_OPAQUE_CV = 128             # coverage from which clip pixels are opaque body (CUT)
S605_OPAQUE_CV = 128             # coverage from which clip pixels are opaque body (CUT)
S605_REACH = 6                   # px: opaque clip pixels are taken at most this far from the label of the per-type rules
S605_HULLPAD = 4                 # px: the hull of the label is taken this much wider
S605_BLOB = 20                   # px: smallest such blob
S605_RING = 8                    # px: the translucent ring of a taken blob
S605_SIL = 7                     # px: the attack's silhouette = its label closed by this, filled
S605_SLIT = 2                    # px: dark seams between the lobes (up to 2 x this wide) are the clip's own body
S605_FINAL = 2                   # px: final closing of the outline (inlets up to 2 x this wide)
S605_HOLE_MAX = 1000             # px: an enclosed hole larger than this stays open

_s605_prev_labels = labels


def labels(objs, cam, R, state=None, ended=(), img=None):
    import bulletlabels as BL
    from scipy.ndimage import binary_fill_holes, binary_dilation, label as cc
    P = _s605_prev_labels(objs, cam, R, state=state, ended=ended, img=img)
    want = {(BL._lid(o), -1, 'body') for o in objs if o['t'] in S605_TYPES}
    if not want: return P
    ids, lids, share = P['ids'], P['lids'], P['share']
    ks = [k for k, l in enumerate(lids) if l in want and (ids == k).any()]
    if not ks: return P
    try: world = _zs_solid(cam, R)
    except Exception: world = np.zeros(ids.shape, bool)
    cv = R['cv'][::2, ::2]
    clip = np.zeros(ids.shape, bool)       # the clip's own shapes (blue / red export; the trail clip is not body), front-most
    for o in objs:
        if o['t'] not in S605_TYPES: continue
        T = projtab.table()[o['t']]
        names = tuple(n for n in (T.get('blue'), T.get('red')) if n)
        if names: clip |= _a4_shape_pixels(R, T['file'], names) > 0
    edge = clip & (cv >= S605_EDGE_CV)
    opaque = clip & (cv >= S605_OPAQUE_CV)
    s8 = np.ones((3, 3), bool)
    for k in ks:
        m = ids == k
        # a pixel whose front-most draw shows the clip's own shape is the cloud, in front of what stands behind it
        free = (ids < 0) & (~world | clip)       # nobody's pixel, not a block in front of the cloud
        reach = _dilate(m, S605_REACH)
        for _ in range(S605_REACH + 1):   # the clip's own opaque pixels connected to the label (shared sprites beyond the
            grow = binary_dilation(m, s8) & opaque & free & reach & ~m   # prediction are not body in the per-type rule)
            if not grow.any(): break
            m = m | grow
        # opaque clip blobs (a lobe's gradient highlight / lower bulge, split off by a translucent ring) lying inside the
        # convex hull of the label: body, with the translucent ring of their own sprite (S605_RING px). Blobs outside the hull
        # (the trail's small bubbles beside the cloud) are not taken.
        core = opaque & free & ~m
        if core.any():
            hull = hull_mask(_dilate(m, S605_HULLPAD))
            L, n = cc(core, structure=s8)
            sz = np.bincount(L.ravel()); inh = np.bincount(L.ravel(), weights=hull.ravel().astype(float))
            ok = (sz >= S605_BLOB) & (inh >= 0.95 * sz); ok[0] = False
            if ok.any():
                blob = ok[L]; m = m | blob
                for _ in range(S605_RING):
                    grow = binary_dilation(m) & edge & free & ~m & _dilate(blob, S605_RING)
                    if not grow.any(): break
                    m = m | grow
        for _ in range(S605_EDGE):   # the translucent clip edge, outside the label, connected to it
            grow = binary_dilation(m) & edge & free & ~m
            if not grow.any(): break
            m = m | grow
        # the translucent clip body inside the attack's silhouette (between lobes, in front of crates: coverage 30-95)
        sil = binary_fill_holes(_close(m, S605_SIL)) & (free | m)
        inner = sil & edge & free & ~m
        m = m | inner
        slit = _close(m, S605_SLIT) & clip & free & ~m    # the clip's own seams, open to the outside
        m = m | slit
        hole = binary_fill_holes(m) & ~m
        if hole.any():   # enclosed holes of the attack's own silhouette
            L, n = cc(hole, structure=s8)
            sz = np.bincount(L.ravel())
            oth = np.bincount(L.ravel(), weights=((ids >= 0) & (ids != k)).ravel().astype(float))
            ok = (sz <= S605_HOLE_MAX) & (oth == 0); ok[0] = False
            m = m | ok[L]
        # last: narrow inlets / slits of the outline (up to 4 px wide) closed over pixels nobody owns, then enclosed holes again
        m = m | (_close(m, S605_FINAL) & ((ids < 0) & ~world))
        hole = binary_fill_holes(m) & ~m
        if hole.any():
            L, n = cc(hole, structure=s8)
            sz = np.bincount(L.ravel())
            oth = np.bincount(L.ravel(), weights=((ids >= 0) & (ids != k)).ravel().astype(float))
            ok = (sz <= S605_HOLE_MAX) & (oth == 0); ok[0] = False
            m = m | ok[L]
        new = m & (ids != k)
        ids[new] = k; share[new] = BL.CUT
    return P
