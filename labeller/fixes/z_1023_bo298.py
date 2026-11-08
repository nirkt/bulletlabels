# Bo Wasp 298 attack: each arrow one hull, back stripe to stinger (user Q20, 2026-10-23; Haiku task 3, reviewed)
# task 3 fix: Bo's wasp arrow (BowDudeWasp, skin 298; type Bo006Projectile 6000367, the attack). User's decision (2026-10-23):
# the whole arrow, from its front (stinger tip) to its back end (last striped segment), as ONE label per arrow: one convex
# hull of that arrow's own sprites. No trail particles, glows or hit effects; neighbouring arrows are separate labels.
# Cause: the clip draws each arrow as ONE atlas sprite (shape 52745 of sc/effects_brawler.sc: stinger and striped segments;
# the thin blue shaft between the stripes is the same sprite), so the label came out as separate bright pieces with the
# shaft between them left out (agent 6's THIN_OFF cut the shaft). Behind the back-most yellow stripe the same sprite
# continues as a blue streak (no stripe): the user's back end is the last striped segment, so it is not in the label.
# Fix: the arrow's own pixels are the shape-52745 pixels, opaque (coverage >= CUT), within BO_WASP grow px of the arrow's
# own predicted label, and not inside another Bo wasp arrow's predicted label (the frame's other predictions, computed
# by the wrapper below: at 307 / 309 / 311 / 313 the neighbour arrows lie side by side and touch). The part touching the
# own prediction is kept. Along the arrow's flight direction (object 'angle', projected to the screen with the camera)
# they are cut at the back-most yellow pixel (R - B >= 50: the striped segments) minus 1 px. The label is the convex hull
# of those pixels, minus the trail particle sprites (shapes 63 / 1012 / 7152 / 7426 of its trail clips) lying in the background
# gaps between the stripes (the user's 'no trail particles in the label'); particles over the arrow's own stripes are kept
# (see below). The hull is made here, not by
# ONE_LABEL, so that the removed particles are not put back.
BO_WASP = dict(file='sc/effects_brawler.sc', clips=('bo_006_atk_projectile', 'bo_006_atk_projectile_red'), shape=52745,
               type=6000367, grow=4, yellow=50, touch=2)
BO_WASP_FRAME = {}   # per frame: the Bo wasp objects and their predicted labels (ids, lids), set by the wrapper below


def _bo_wasp_dir(ctx):
    """unit screen vector of the arrow's flight direction (the object's angle, logic x / y, projected with the camera)"""
    o = ctx['o']
    if o.get('angle') is None or o.get('logic') is None: return None
    a = np.deg2rad(float(o['angle'])); lx, ly = o['logic']
    p0 = np.array(_proj(ctx['cam'], lx, ly)); p1 = np.array(_proj(ctx['cam'], lx + 100 * np.cos(a), ly + 100 * np.sin(a)))
    d = p1 - p0; n = np.linalg.norm(d)
    return d / n if n > 0 else None


def _bo_wasp_others(ctx):
    """(H, W) bool: the predicted labels of the frame's other Bo wasp arrows (not this one)"""
    import bulletlabels as BL
    H, W = ctx['pred_any'].shape; out = np.zeros((H, W), bool)
    if 'ids' not in BO_WASP_FRAME: return out
    lids = BO_WASP_FRAME['lids']; ids = BO_WASP_FRAME['ids']
    for q in BO_WASP_FRAME['objs']:
        if q['t'] != BO_WASP['type'] or q['id'] == ctx['o']['id']: continue
        ks = [k for k, l in enumerate(lids) if l[0] == BL._lid(q)]
        if ks: out |= np.isin(ids, ks)
    return out


def _bo_wasp_trail(ctx):
    """(H, W) bool: pixels whose front-most draw shows a trail particle sprite of this type (not the arrow's own shape)"""
    names = tuple(sorted({n for f, n in _trail_clips(BO_WASP['type']) if f == BO_WASP['file']}))
    SPt = _a4_shape_pixels(ctx['R'], BO_WASP['file'], names)
    return (SPt > 0) & (SPt != BO_WASP['shape'] + 1)


def _bo_wasp_arrow(body, ctx):
    from scipy.ndimage import label as cc
    SP = _a4_shape_pixels(ctx['R'], BO_WASP['file'], BO_WASP['clips'])   # shape id + 1 per pixel, 0: none
    near = ctx['pred_any'] if BO_WASP['grow'] == 0 else _dilate(ctx['pred_any'], BO_WASP['grow'])
    arrow = (SP == BO_WASP['shape'] + 1) & (ctx['cv'] >= CUT) & near & ~_bo_wasp_others(ctx)
    if not arrow.any(): return body
    L, n = cc(arrow); keep = np.unique(L[ctx['pred_any'] & arrow]); keep = keep[keep > 0]
    arrow = np.isin(L, keep) & arrow
    d = _bo_wasp_dir(ctx); core = arrow
    if d is not None and ctx.get('img') is not None:
        im = ctx['img'].astype(np.int16); yel = arrow & (im[..., 0] - im[..., 2] >= BO_WASP['yellow'])   # the striped segments
        if yel.any():
            ysy, ysx = np.nonzero(yel); back = (ysx * d[0] + ysy * d[1]).min()   # the back-most striped segment (along d)
            core = _proj_mask(arrow, d, back - 1.0)
    # a particle blob touching the arrow's own pixels stays (it is drawn over the arrow: the arrow is there under it; clean labels
    # fill holes from effects drawn over the attack); a particle blob in a background gap of the hull is cut out
    from scipy.ndimage import binary_dilation
    tr = _bo_wasp_trail(ctx) & hull_mask(core); Lt, nt = cc(tr)
    if nt:
        touch = np.unique(Lt[binary_dilation(core, iterations=BO_WASP['touch']) & tr]); touch = touch[touch > 0]
        free = tr & ~np.isin(Lt, touch)
    else: free = tr
    return hull_mask(core) & ~free


def _proj_mask(m, d, lo):
    """pixels of m whose projection on d is at least lo"""
    H, W = m.shape; yy, xx = np.mgrid[0:H, 0:W]
    return m & ((xx * d[0] + yy * d[1]) >= lo)


_labels_before_bo = labels


def labels(objs, cam, R, state=None, ended=(), img=None):
    """wrapper: the frame's Bo wasp arrows and their predicted labels (BO_WASP_FRAME), for the arrow fix above"""
    import bulletlabels as BL
    if any(o['t'] == BO_WASP['type'] for o in objs):
        P0 = BL.labels(objs, cam, cut=0.01)
        BO_WASP_FRAME.update(ids=P0['ids'], lids=P0['lids'], objs=objs)
    else:
        BO_WASP_FRAME.clear()
    return _labels_before_bo(objs, cam, R, state=state, ended=ended, img=img)


POST_BODY[6000367] = _bo_wasp_arrow   # the hull is made here (see above); no ONE_LABEL for this type
