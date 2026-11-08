# Frank DJ 242: one hull per wave, attack and super, no holes (user Frank rule 10-23; Haiku task 11, reviewed, carve off)
# Task 11: skin 242 HammerDudeDJ (Frank002, brawler Frank, skin 002). User's decision (2026-10-23, same as skin 321 /
# Frank 321 model z_1023_frank321.py): ONE label per wave while the wave's logic lives: the convex hull of the whole
# wave's tiles (attack 6000477, super 6000478), no per-tile labels, no holes. After the wave's logic has ended: no label.
#
# Shapes of sc/effects_brawler.sc (per pixel, _a4_shape_pixels on the type's trail clips; checked on sw2_242 with shapediag
# at ticks 281 and the super rows):
#   shape 1      (tex 1, 32 x 32 units): the light dance-floor tile body, a plain white quad tinted per tile. Small
#                effects use the same quad (yellow spark ticks, a white star burst): removed by the thickness cut below.
#   shape 55279  (tex 0, 37 x 18 units): listed by the bar clips; not drawn in these ticks (kept, harmless).
#   shape 0      (tex 1, 32 x 32 units): the soft yellow glow flare (big disc drawn over the tiles), listed by the bar clips too.
#                Dropped: a tile lying under a flare loses its covered part (no label there), the hull fills the rest.
#   shape 7600   (particle_stars_small_blink): flare sparkles. Dropped.
#   shapes 689-694 (shocked_yellow_delay sparks), 763-765, 59290-59292 (groundFrag cracks, super): dropped.
# No-shape pixels (sprite edges) are kept only within 2 px of a kept shape.
FRANK242_KEEP = {6000477: (1, 55279),   # attack: tile bodies
                 6000478: (1, 55279)}   # super: tile bodies
FRANK242_FILE = 'sc/effects_brawler.sc'
FRANK242_CARVE = False   # the user's Frank rule (10-23): one hull per wave, NO holes (objects in front are not carved; coordinator 10-24)
FRANK242_THIN = 3.0   # px: a tile piece's inscribed radius must reach this (sparks and slivers are thinner)
FRANK242_NEAR = 8    # px (half resolution): a tile part further than this from the wave's own predicted label (pred_any) is not the wave


def _t11_names(t):
    T = projtab.table()[t]
    return tuple(sorted({n for f, n in _trail_clips(t) if f == FRANK242_FILE} | {n for n in (T['blue'], T['red']) if n}))


def _t11_clean(body, ctx):
    """body pixels of the tile shapes only (plus no-shape edge pixels touching them), in the parts that lie within
    FRANK242_NEAR px of the wave's own predicted label"""
    if not body.any(): return body
    from scipy.ndimage import label as cc, distance_transform_edt
    t = ctx['t']
    SP = _a4_shape_pixels(ctx['R'], FRANK242_FILE, _t11_names(t))   # shape id + 1 per pixel, 0: none
    keep = np.isin(SP, [s + 1 for s in FRANK242_KEEP[t]]) & body
    m = keep | (body & (SP == 0) & _dilate(keep, 2))
    if not m.any(): return m
    # shape 1 is a plain white quad: the tiles' body, but also small effects drawn with the same quad (yellow spark ticks
    # at the crates, sparks in the burst): pieces narrower than 6 px (inscribed radius < FRANK242_THIN) are not tiles
    from scipy.ndimage import distance_transform_edt as edt
    L0, n0 = cc(m, structure=np.ones((3, 3), bool))
    for k in range(1, n0 + 1):
        mk = L0 == k
        if edt(mk).max() < FRANK242_THIN: m &= ~mk
    if not m.any(): return m
    pa = ctx['pred_any']
    if not pa.any(): return m & False
    dist = distance_transform_edt(~pa)
    L, n = cc(m, structure=np.ones((3, 3), bool))
    near = np.zeros(n + 1, bool)
    near[1:] = np.array([dist[L == k].min() <= FRANK242_NEAR for k in range(1, n + 1)])
    return m & near[L]


for _t in FRANK242_KEEP: POST_BODY[_t] = _t11_clean


# ONE label per wave: the labels of a wave group (its members share one label id) are one convex hull over all their tiles,
# not one hull per member. labels() is wrapped: the group's body label is replaced by the hull of its own pixels (only
# pixels nothing else claimed, so other objects' labels stay as they are). A group with an ended object (the wave's logic
# has ended: Tracker.gone) gets no label at all (rule 4, the wave's logic is over).
_t11_labels_orig = labels


def labels(*args, **kwargs):
    import bulletlabels as BL
    P = _t11_labels_orig(*args, **kwargs)
    objs = args[0] if args else kwargs['objs']
    cam = args[1] if len(args) > 1 else kwargs['cam']
    R = args[2] if len(args) > 2 else kwargs['R']
    lids, ids, share = P['lids'], P['ids'], P['share']
    cv = R['cv'][::2, ::2]
    groups = {}
    for o in objs:
        if o['t'] in FRANK242_KEEP: groups.setdefault(o.get('label_id', o['id']), []).append(o)
    for gid, mem in groups.items():
        key = (gid, -1, 'body')
        if key not in lids: continue
        k = lids.index(key); cur = ids == k
        if any(o.get('ended') for o in mem):
            ids[cur] = -1; share[cur] = 0; continue
        if not cur.any(): continue
        h = hull_mask(cur) & (ids < 0)
        # occluders: a wall, crate or character drawn in front of the wave is not the wave (the pipeline's rule for every
        # label): hull pixels nearer the camera than the wave's own depth there (wd < anchor depth - DEPTH_TOL) stay out
        wd = R.get('wd')
        if FRANK242_CARVE and wd is not None:
            wref = R['meta'].get('wref', 0.0)
            a0 = next(o['anchor'] for o in mem if o.get('anchor') is not None)   # the wave's ground plane (a flat floor)
            gd = ground_depth(cam, a0[0], -a0[1], 1000, wref)                   # per pixel: the depth of the floor there
            # only world geometry (opaque draws that write depth: walls, crates, characters, bushes) occludes; blended effect
            # sprites (glows, flares, sparks: no depth write) lie over the tiles and do not hide them
            opaque = [d for d, dr in R['draws'].items() if dr.get('dmask') == 1 and not dr.get('blend')]
            occ = np.isin(R['id'][::2, ::2], opaque) & (R['cv'][::2, ::2] >= CUT) & (wd < gd - DEPTH_TOL)
            h &= ~occ
        ids[h] = k; share[h] = np.maximum(cv[h] / 255.0, BL.CUT)
    return P
