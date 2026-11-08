# Poco Surfer 431 (6000353 attack hull / 6000354 super discs): cut by solid map blocks in front (discs are depth-tested: hidden behind walls) (user 10-26 Q5; Haiku task 38, reviewed)
# task 38 (Haiku, 2026-10-26): skin 431 PocoSurferOvercharge (brawler Poco, skin 010). Loads after z_1023_poco431.py (it
# replaces the POST_BODY hooks of 6000353 / 6000354 set there, keeping their disc and hull logic).
# The user's decision (binding, 2026-10-26): find out whether the disc sprites are drawn ON TOP of walls and crates or
# HIDDEN behind them. Finding (sw2_431, the depth test of the ID pass against the ground plane under each disc): the discs
# are depth-tested. Where a solid map block (wall, crate, barrel: _zs_solid) is the front-most OPAQUE draw, the disc is
# hidden behind it (ticks 273, 313, 317: 67-315 px per tick). Where the disc is nearer, it is drawn over the block (tick 351:
# the super disc over an orange crate; the things in front there are blended sparks, which never cut). So the disc label
# is cut only by the solid map blocks that are drawn in front of it: pixels whose front-most draw is opaque world geometry
# and whose surface lies over a solid tile. Bushes, characters and blended effects never cut. A disc with no block in
# front keeps its full ellipse (no change). Attack (6000353): the shot's hull is cut by the same blocks.
#
# Mechanism: _zs_solid(cam, R) (z_1023_zone_solid.py) gives the solid-tile pixels above ground; intersected with the
# opaque draws (blend 0) of the ID pass. Cleanup as in _p431_disc: holes and parts up to P431_SPECK px, 1-px spurs.
# Holes made by the cut stay cut (not refilled).

_T38_CACHE = [None, None]   # [R, mask]: the frame's solid-block-in-front mask (computed once per frame)


def _t38_block_front(ctx):
    """(H, W) bool at label resolution: pixels where a solid map block is the front-most opaque draw"""
    R = ctx['R']
    if _T38_CACHE[0] is R: return _T38_CACHE[1]
    ids_h = R['id'][::2, ::2]
    opaque = [d for d, v in R['draws'].items() if v.get('blend', 0) == 0]
    cut = _zs_solid(ctx['cam'], R) & np.isin(ids_h, opaque) & (ids_h > 0)
    _T38_CACHE[0] = R; _T38_CACHE[1] = cut
    return cut


def _t38_tidy(m, cut):
    """m without the block pixels (cut), holes that are not block pixels filled, 1-px spurs and parts < P431_SPECK px dropped"""
    from scipy.ndimage import binary_fill_holes, binary_opening, label as cc
    m = binary_fill_holes(m & ~cut) & ~cut
    if not m.any(): return m
    m = binary_opening(m, structure=np.ones((3, 3), bool))
    if not m.any(): return m
    L, n = cc(m)
    if n > 1:
        sz = np.bincount(L.ravel()); keep = sz >= P431_SPECK; keep[0] = False; m = keep[L]
    holes = binary_fill_holes(m) & ~m & ~cut
    if holes.any():
        L2, n2 = cc(holes)
        if n2:
            sz2 = np.bincount(L2.ravel()); fill = sz2 <= P431_SPECK; fill[0] = False; m = m | fill[L2]
    return m


def _t38_disc(body, ctx):
    """a super disc (6000354): the disc of z_1023_poco431 minus the solid blocks drawn in front of it"""
    m = _p431_disc(body, ctx)
    if not m.any(): return m
    cut = _t38_block_front(ctx)
    if not (m & cut).any(): return m                     # no block in front: the full ellipse, unchanged
    return _t38_tidy(m, cut)


def _t38_shot(body, ctx):
    """an attack (6000353): the shot's hull (z_1023_poco431 _p431_shot), minus the solid blocks drawn in front of its discs"""
    o = ctx['o']; g = GROUP431.get(o['id'], o['id'])
    if o['id'] not in DISC431: return body & False
    m = np.zeros_like(body)
    for oid in DISC431:
        if GROUP431.get(oid, oid) == g:
            c = dict(ctx); c['o'] = dict(o, id=oid); m |= _p431_disc(body, c)
    if not m.any(): return m
    h = hull_mask(m)
    cut = _t38_block_front(ctx)
    if not (h & cut).any(): return h                     # no block in front of the shot: the hull, unchanged
    return _t38_tidy(h, cut)


POST_BODY[6000353] = _t38_shot    # attack: the disc hull, cut by the blocks in front of it (skin 431 only: its types are skin 010's)
POST_BODY[6000354] = _t38_disc    # super: each disc, cut by the blocks in front of it
