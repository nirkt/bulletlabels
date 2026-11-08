# sweep 10-27 fix for skin 370 (Sonnet revision, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s370/)
# sweep 10-27 fix for skin 370 BlowerDeepsea (Haiku fix task out/sweep2/haikutasks/sweep1027/s370/).
# Pairs 6968 (attack 90 deg, tick 317), 6971 .. 6974 and 6976 (super 45 deg, ticks 345-359).
#
# 1. Attack, type 6000739 (Gale005Projectile, clip gale_005_atk_projectile). At tick 317 the bubble row stands on the
#    translucent ground shadow of the bubbles (one dark blob, every pixel under half opaque). The shadow's pixels are glow
#    pixels of the bubble's own clip inside its aligned prediction, so they were labelled: two loops hanging below the row,
#    detached from the bubble outlines. Rule: a label pixel drawn by a draw that has no opaque pixel over the whole frame
#    (under 5 % with coverage >= CUT) is a translucent ground effect and is dropped. Computed per frame from the ID pass
#    (not keyed to a draw number). A coverage cut on the pixels themselves was rejected: the bubbles' own edges sit at
#    coverage 60-70 on the other ticks (measured), so it would cut good labels.
#
# 2. Super, type 6000740 (Gale005UltiProjectile, its clip is empty: the cloud is its trail, a wave). The label was a band
#    along the cloud's edge with holes around each bubble head (the heads are not trail-body pixels) and gaps between the
#    puffs. Rule: the object's own outline, filled solid: a 3 px closing of its body (joins the puffs), then every enclosed
#    hole (the heads) filled, inside the object's search region only. No convex hull (the user prefers the object's own outline).
#    3. After the logic end ('finished' flag, tick 355 on), only the largest part of the super's body is kept; parts under 25 %
#       of it are the drifted puffs (the splash droplets and the green wedge of tick 359, finding 6976).
# 4. A label of type 6000739 left under 25 px by the shadow removal is dropped (the shadow-only bubbles at tick 317); only
#    objects that lost shadow pixels in the frame are checked, so the opaque bubble edges of ticks 305-313 stay as they are.
#
# Side effect, same type, not a finding: attack 0 (ticks 265-271, Gale 005 bubbles) loses the shadow bits that were
# attached to the bubble sides (19-43 px per object), the outlines now hug each bubble.
#
# Only these two type ids. Other skins of the Gale family use other type ids; the shared 6000739 clip is also used by other
# Blower skins (see the report).

from scipy.ndimage import binary_fill_holes as _s370_fill, find_objects as _s370_find

GALE005_ATK = 6000739          # Gale 005 attack (BlowerDeepsea 370 and the other skins using this type)
GALE005_ULTI = 6000740         # Gale 005 super (BlowerDeepsea 370 and the other skins using this type)
GALE005_OPAQUE = 0.05          # a draw with fewer opaque pixels than this share over the whole frame is translucent
GALE005_MAIN = 0.2             # parts under this share of the largest connected part are particles beside the cloud
GALE005_PARTICLE_SHAPES = (98, 143)   # atlas shape ids of the sparkle flowers and the cyan droplets (effects_brawler.sc)
GALE005_LIME = 80              # G - B from which a pixel is the lime splash (the cloud is cyan / white: B >= G)
GALE005_HEAD_R = 60            # red below this: the bubble heads' green (R about 0), not the lime splash (R about 120+)
GALE005_OPEN = 3               # px: necks thinner than about twice this are cut off the cloud
GALE005_ULTI_CLOSE = 6         # px (half resolution): gaps between the super's puffs closed before the fill
GALE005_DARK = 100             # median brightness (max of R, G, B) below which a draw part is the dark ground shadow
GALE005_JOIN = 1               # px: pixels of the cloud this close are one connected cloud


def _fix_gale005_shadow(body, ctx):
    ids_h = ctx['ids_h']; cv = ctx['cv']
    if not body.any(): return body
    dd = ids_h.ravel().astype(np.int64)
    n = np.bincount(dd)
    k = np.bincount(dd, weights=(cv.ravel() >= CUT).astype(np.float64), minlength=len(n))
    share = np.where(n > 0, k / np.maximum(n, 1), 0.0)
    bad = [int(d) for d in np.unique(ids_h[body]) if d > 0 and share[d] < GALE005_OPAQUE]
    if not bad: return body
    import bulletlabels as BL
    _S370_SHADOW.add(BL._lid(ctx['o']))   # this object lost shadow pixels in this frame (see labels below)
    out = body & ~np.isin(ids_h, bad)
    # what is left under 25 px is a speck (the clean-up's speck size): an object whose only label was the shadow loses it
    return out if out.sum() >= 25 else np.zeros_like(body)


def _s370_trail_clips():
    out = {}
    for f, n in _trail_clips(GALE005_ULTI): out.setdefault(f, []).append(n)
    return {f: tuple(sorted(set(v))) for f, v in out.items()}


def _fix_gale005_ulti(body, ctx):
    """super cloud: (1) once the logic projectile has finished (range reached, its 'finished' flag) the label is gone: what
    is drawn is the end splash, the droplets and the pool of an ended attack (rules 2 and 4); (2) before: the cloud's own
    outline, filled solid: every opaque trail-body pixel of the cloud connected to the object (not only the strip of the
    prediction's search region), without the dark ground-shadow parts, gaps closed, holes (bubble heads) filled."""
    from scipy.ndimage import label as _cc
    o = ctx['o']
    if o.get('finished'): return np.zeros_like(body)
    if not body.any(): return body
    C = ctx['C']; cv = ctx['cv']; img = ctx['img']; ids_h = ctx['ids_h']
    cand = (C == 3) & (cv >= CUT)
    # splash particles flying by the cloud are trail sprites too: the lime splash blobs (green, no blue) and the sparkle flowers /
    # cyan droplets (their own atlas shapes, effects_brawler.sc) are not the cloud
    bad = np.zeros_like(cand)
    if img is not None:
        im = img.astype(np.int16); gb = (im[..., 1] - im[..., 2] >= GALE005_LIME)
        bad |= gb & (im[..., 0] >= GALE005_HEAD_R)       # lime splash: red high; the bubble heads' green has no red (R < 40)
    for swf, names in _s370_trail_clips().items():
        SP = _a4_shape_pixels(ctx['R'], swf, names)
        bad |= np.isin(SP, [k + 1 for k in GALE005_PARTICLE_SHAPES])
    bad = _dilate(bad, 1)
    cand &= ~bad; body = body & ~bad
    # dark ground shadow: connected parts of one draw whose median brightness is low (the puff sprite tinted dark)
    if img is not None:
        v = img.max(-1)
        for d in np.unique(ids_h[cand]):
            L, n = _cc(cand & (ids_h == d), structure=np.ones((3, 3)))
            for k in range(1, n + 1):
                m = L == k
                if np.median(v[m]) < GALE005_DARK: cand &= ~m; body = body & ~m
    # the cloud: opaque trail-body pixels connected (within a 2 px gap) to the object's own body
    L, n = _cc(_dilate(cand | body, GALE005_JOIN), structure=np.ones((3, 3)))
    sz = np.bincount(L.ravel()); sz[0] = 0
    cloud = (cand | body) & np.isin(L, np.nonzero(sz >= GALE005_MAIN * sz.max())[0])   # droplets / sparkles beside it: out
    filled = _s370_fill(_close(cloud, GALE005_ULTI_CLOSE)) | cloud
    # the bubble heads standing on the cloud (green discs, no red) are part of the object: those lying mostly inside the
    # cloud's footprint are added (they stick out of its edge, so no hole fill can take them)
    if img is not None:
        hd = gb & (im[..., 0] < GALE005_HEAD_R); L2, n2 = _cc(_dilate(hd, 1), structure=np.ones((3, 3)))
        foot = _dilate(filled, 4); add = np.zeros_like(filled)
        szs = np.bincount(L2.ravel()); fr = _s370_find(L2)
        for k in range(1, n2 + 1):
            if szs[k] < 40: continue
            sl = fr[k - 1]; m = _s370_fill(L2[sl] == k)
            if m.sum() >= 60 and (m & foot[sl]).sum() >= 0.5 * m.sum(): add[sl] |= m
        if add.any(): cloud = cloud | add; filled = _s370_fill(_close(cloud, GALE005_ULTI_CLOSE)) | cloud
    # thin necks (a particle touching the cloud) opened away, only the main part kept, its edge pixels given back
    from scipy.ndimage import binary_opening as _op
    op = _op(np.pad(filled, 4), iterations=GALE005_OPEN)[4:-4, 4:-4]
    L, n = _cc(op, structure=np.ones((3, 3)))
    if n:
        sz = np.bincount(L.ravel())[1:]; op = np.isin(L, 1 + np.nonzero(sz >= 0.25 * sz.max())[0])
        filled = op | (cloud & _dilate(op, 2))
    return filled


POST_BODY[GALE005_ATK] = _fix_gale005_shadow
POST_BODY[GALE005_ULTI] = _fix_gale005_ulti

# The shadow-only bubbles (objects 2000025 and 2000027 at tick 317) share bubble 2's pixels, which the nearest-object claim
# gives to bubble 2's object: their own remainder is a few pixels, a speck. Only the objects that lost shadow pixels in
# this frame are checked: a final label of such an object under 25 px is dropped (the clean-up's speck size; it drops
# specks only among several parts). Other tiny labels (opaque bubble edges on ticks 305-313) are not touched.
_S370_SHADOW = set()
_s370_labels_prev = labels


def labels(objs, cam, R, state=None, ended=(), img=None):
    import bulletlabels as BL
    _S370_SHADOW.clear()
    P = _s370_labels_prev(objs, cam, R, state=state, ended=ended, img=img)
    ids = P['ids']; lids = P['lids']
    for k in np.unique(ids[ids >= 0]):
        if lids[k][0] in _S370_SHADOW and (ids == k).sum() < 25:
            m = ids == k; ids[m] = -1; P['share'][m] = 0
    return P
