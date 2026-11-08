# sweep 10-27 fix for skin 178 (Haiku fix, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s178/)
# sweep 10-27 fix for skin 178 (CactusPoop, Haiku task s178, finding pair 3100: sw2_178 super 45 deg, tick 349, spur_or_speck).
# Type 6000134 Spike008UltiProjectile (logic CactusUltiProjectile, the skin's super cloud, blue clip spike_008_ulti_projectile,
# effects_brawler.sc). The label of the cloud took in small brown debris dots on its edge: the two outline rings right of the
# cloud at tick 349 (x 674-687, y 249-257: one bump on the cloud edge, one detached), a dot on the upper-left edge at tick 349
# (x 616-621, y 230-235), and dots on the top edge at ticks 345 and 351-359. Debris is never labelled (rule 3).
# The dots are drawn with the cloud's own clip shape 63 (effects_brawler.sc), the same shape id as parts of the blue body, so
# a cut by shape id alone is too coarse. Checked on the capture: the brown debris enclosed by the cloud (tick 347, 70 px; tick
# 349, 30 px) stays (solid label), and the blue body's shape-63 parts are blue (mean R - B below 0), so the rule touches only the dots.
# Rule, only on this type's labels, per 8-connected part of the label's shape-63 pixels: a part is removed when it is at most
# S178_MAX px, touches the OUTSIDE of the label (an edge dot, not an enclosed one) and is warm (mean R - B >= S178_WARM: brown).
# Then, per label: parts detached from its largest part and at most S178_SPECK px are dropped (edge bits left by a removed dot,
# and a grey 4-px fleck above the cloud at tick 345). The largest part is never dropped.
# Runs after the shared clean-up (z_1099), on the final label map. Needs the frame colours (img): callers without img keep the
# label map as it was.
# Verified on every frame of sw2_178 (scan): changed ticks 345 (-40 px), 349 (-61 px), 351 (-33), 353 (-17), 355 (-26), 357 (-35),
# 359 (-58); unchanged 343, 347, 361-365 (other labels, not this cloud) and the zone rows 371-411. Pixels removed are debris
# and edge bits only; the body's part keeps its size minus those pixels.
# Only type 6000134 is touched. Other skins: projtab has no other type with this clip; no other capture was checked for this id.

S178_TYPE = 6000134
S178_SHAPE = 63     # shape id in effects_brawler.sc of the cloud's own clip (spike_008_ulti_projectile)
S178_MAX = 40       # px: largest debris part removed
S178_WARM = 30      # mean R - B of a brown part (blue body parts are below 0)
S178_SPECK = 6      # px: a detached part of the cloud's label this small is a speck (edge bits of a removed dot too)


def _s178_fix(P, objs, R, img):
    from scipy.ndimage import label as cc, binary_dilation
    import bulletlabels as BL
    T = projtab.table()[S178_TYPE]
    star = {BL._lid(o) for o in objs if o['t'] == S178_TYPE}
    ks = [k for k, l in enumerate(P['lids']) if l[0] in star and l[2] == 'body']
    if not ks or img is None: return P
    SP = _a4_shape_pixels(R, T['file'], tuple(c for c in (T.get('blue'), T.get('red')) if c))
    ids = P['ids'].copy(); share = P['share'].copy(); changed = False
    S8 = np.ones((3, 3), bool)
    for k in ks:
        m = ids == k
        if not m.any(): continue
        sh = m & (SP == S178_SHAPE + 1)
        if sh.any():
            edge_out = binary_dilation(m, S8) & ~m
            L, n = cc(sh, structure=S8)
            for j in range(1, n + 1):
                mm = L == j
                if mm.sum() > S178_MAX: continue
                if not (binary_dilation(mm, S8) & edge_out).any(): continue  # enclosed debris stays
                ys, xs = np.nonzero(mm)
                warm = img[ys, xs, 0].astype(int).mean() - img[ys, xs, 2].astype(int).mean()
                if warm < S178_WARM: continue                                 # blue part of the body
                ids[mm] = -1; share[mm] = 0.0; changed = True
        # specks: parts of this label that are detached from its largest part and at most S178_SPECK px (the one-pixel edge
        # bits left by a removed dot, and the grey 4-px fleck above the cloud at tick 345); the largest part is never dropped
        L, n = cc(ids == k, structure=S8)
        if n > 1:
            sz = np.bincount(L.ravel()); sz[0] = 0; big = int(sz.argmax())
            for j in np.nonzero((sz > 0) & (sz <= S178_SPECK))[0]:
                if j == big: continue
                ids[L == j] = -1; share[L == j] = 0.0; changed = True
    if changed:
        P = dict(P); P['ids'] = ids; P['share'] = share
    return P


_s178_labels_prev = labels


def labels(objs, cam, R, state=None, ended=(), img=None):
    P = _s178_labels_prev(objs, cam, R, state=state, ended=ended, img=img)
    return _s178_fix(P, objs, R, img)
