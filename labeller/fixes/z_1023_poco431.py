# Poco Surfer 431: discs only, one label per disc (user Q2, 2026-10-23; Haiku task 4, reviewed)
# task 4 fix (Haiku, 2026-10-23): PocoSurferOvercharge, skin 431 (brawler Poco, skin 010). Types 6000353 (Poco010Projectile,
# the attack) and 6000354 (Poco010UltiProjectile, the super). The user's decision (binding, 2026-10-23): label the DISCS only:
# each disc one clean solid label, no lightning bolt, no speed wedges, no music notes, no empty ground between them. For this
# skin only: the other Poco skins keep their ZONE labels (their types are other ids). If attack discs overlap or touch, each keeps
# its own label (one per projectile object). Once a projectile has ended, its leftover effects are unlabelled.
#
# Cause: both types are in ZONE (one solid blob over the whole cloud). The clip draws the glowing disc plus, as clip body, a
# lightning bolt behind each attack disc (attack) and cyan speed wedges behind each super disc (super); the trail clips are
# music notes. The wave grouping (bulletlabels.frame_objects, wave.groups) gave the discs of one burst one shared label id, so
# zone() joined them into one blob.
#
# Method (checked on sw2_431 at ticks 267, 271, 277, 305, 307, 345, 349, 353 with overlays):
#   - the disc is the object's own predicted disc: the clip polygons of the disc shapes (P431), rasterised (masks.raster_poly).
#     The prediction covers the disc where notes are drawn over it, so the label is the whole disc, not its visible crescent;
#   - a pixel shared by several discs (stacked discs of one burst) goes to the NEAREST disc (smallest eye depth of the anchor:
#     the one in front; the draw order of the objects does not show it: tick 305). An object keeps its disc minus the nearer ones;
#   - walls / bushes in front of a disc are NOT cut (the depth test exists, P431_DEPTH, but is off: see the note in _p431_disc).
#     A disc behind a crate keeps its predicted disc (tick 313: a few px of the lower edge under the crate); a defect left.
#   - a projectile the game no longer shows (drawn False: finished by range or hit, hidden 0.1 s after its finish) is not
#     labelled: what is left on screen is its end effect (rule 2 and rule 4 of the brief).
# Clean-up: holes and parts under P431_SPECK px are dropped, and the result is opened by one px (no 1-px spurs).
#
# Shapes of the clips (shape ids of the prediction and of the drawn pixels agree for these; see the overlays):
#   attack 6000353: 0 = the disc body (outer disc, bright rim), 22 = inner ring, 767 / 768 = its highlights. Not labelled: the
#                   lightning bolt (769 / 782 / 763 / 10 / 1), its streak (31), the notes (11 / 77 / 78 trail).
#   super  6000354: 8 = lime centre, 21 = thin orange outer edge, 24 = white rim, 22 = teal ring, 23 = purple ring. Not
#                   labelled: the speed wedges and their tips (0 / 2 / 20 / 26; pixel shape 64), the notes (1 / 11 / 77).
P431_FILE = 'sc/effects_brawler_poco.sc'
P431 = {
    6000353: dict(clips=('poco_010_atk_projectile', 'poco_010_atk_projectile_red'), shapes=(0, 22, 767, 768)),
    6000354: dict(clips=('poco_010_ulti_projectile', 'poco_010_ulti_projectile_red'), shapes=(8, 21, 22, 23, 24)),
}
P431_DEPTH = False
P431_ADD = 0.2     # additive colour (0..1) from which a pixel of a bright additive layer is the disc
P431_SPECK = 10    # px: holes up to this, and parts up to this, are specks (not the disc)
DISC431 = {}       # object id -> (eye depth of the anchor, (H, W) bool predicted disc at label resolution); set per frame by labels() below


def _p431_raster(o, cam, shapes):
    """(H, W) bool at label resolution (540 x 1200): the object's predicted clip polygons of the given shapes"""
    import masks as MK
    H, W = 540, 1200
    M = np.zeros((2 * H, 2 * W), bool); S = PD.sc(o['file'])
    for q in o['parts'].get('clip', []):
        if q.get('shape') not in shapes: continue
        r = MK.raster_poly(S, q, cam)
        if r is None: continue
        x0, y0, C, T = r
        m = (C.max(-1) >= P431_ADD) | ((1 - T).max(-1) >= 0.5)   # additive layers (the bright ring) show as colour, alpha ones as coverage
        y1 = min(2 * H, y0 + m.shape[0]); x1 = min(2 * W, x0 + m.shape[1])
        if y1 <= y0 or x1 <= x0: continue
        M[max(0, y0):y1, max(0, x0):x1] |= m[max(0, -y0):y1 - y0, max(0, -x0):x1 - x0]
    return M[::2, ::2]


def _p431_disc(body, ctx):
    from scipy.ndimage import binary_fill_holes, binary_opening, label as cc
    o = ctx['o']
    if o['id'] not in DISC431: return body & False           # not drawn (finished: its end effect), or not one of these discs
    dz0, D = DISC431[o['id']]
    nearer = np.zeros_like(D)
    for oid, (dz2, D2) in DISC431.items():
        if dz2 < dz0: nearer |= D2                            # nearer the camera (smaller eye depth): in front where they overlap
    m = D & ~nearer
    wd = ctx['R'].get('wd')
    # depth test (walls, bushes in front) is off: the discs are drawn above the ground, so the ground under their lower half is
    # nearer than the anchor and the test cut a flat edge across every disc (tick 307). P431_DEPTH = True to test again.
    if P431_DEPTH and wd is not None and o.get('anchor') is not None:
        dz = _eye_depth(ctx['cam'], o['anchor'], ctx['R']['meta'].get('wref', 0.0))[0]
        m = m & (wd >= dz - DEPTH_TOL)
    if not m.any(): return m
    m = binary_fill_holes(m)
    m = binary_opening(m, structure=np.ones((3, 3), bool))   # 1-px spurs out
    L, n = cc(m)
    if n > 1:
        sz = np.bincount(L.ravel()); keep = sz >= P431_SPECK; keep[0] = False; m = keep[L]   # detached crumbs out
    holes = binary_fill_holes(m) & ~m
    if holes.any():
        L2, n2 = cc(holes)
        if n2:
            sz2 = np.bincount(L2.ravel()); fill = sz2 <= P431_SPECK; fill[0] = False; m = m | fill[L2]   # specks filled
    return m


for _t in P431: POST_BODY[_t] = _p431_disc


# labels: one label per projectile object (the wave grouping is off for these two types: each disc its own label id), no
# group region for them, no pseudo object of an ended attack, and each drawn disc's prediction for the frame (DISC431).
# Other types are untouched.
_t4_prev_labels = labels


def labels(objs, cam, R, state=None, ended=(), img=None):
    keep = []
    for o in objs:
        if o['t'] in P431:
            if o.get('ended'): continue          # the wave group's pseudo object of an ended attack
            if o['t'] in P431_HULL: GROUP431[o['id']] = o.get('label_id', o['id'])   # attack: one label per shot (its wave group)
            else: o['label_id'] = o['id']
            o['attack'] = None
        keep.append(o)
    DISC431.clear()
    wref = R['meta'].get('wref', 0.0)
    for o in keep:
        if o['t'] not in P431 or not o.get('drawn') or not o.get('parts') or not o['parts'].get('clip'): continue
        dz = _eye_depth(cam, o['anchor'], wref)[0]   # the disc nearer the camera is the one in front (checked at ticks 267, 305)
        DISC431[o['id']] = (float(dz), _p431_raster(o, cam, P431[o['t']]['shapes']))
    return _t4_prev_labels(keep, cam, R, state=state, ended=ended, img=img)


# The user (2026-10-23): Poco's ATTACK is one hull over all discs of a shot, not a label per disc (the super keeps one label
# per disc). The shot's discs keep their wave group's label id (GROUP431); each member's body is the convex hull of the union
# of the clean discs of its group that are drawn this frame (finished discs are not drawn: not in DISC431).
P431_HULL = {6000353}
GROUP431 = {}


def _p431_shot(body, ctx):
    o = ctx['o']; g = GROUP431.get(o['id'], o['id'])
    if o['id'] not in DISC431: return body & False
    m = np.zeros_like(body)
    for oid in DISC431:
        if GROUP431.get(oid, oid) == g:
            c = dict(ctx); c['o'] = dict(o, id=oid); m |= _p431_disc(body, c)
    return hull_mask(m) if m.any() else m


for _t in P431_HULL: POST_BODY[_t] = _p431_shot
