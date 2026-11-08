# human review 2026-10-28 fix for skin 041 (Haiku fix, reviewed by the coordinator, installed 10-29; source out/sweep2/haikutasks/human1028/s041/fix.py; tested via IDLABELS_OVERRIDE after all fixes incl. z_1099_cleanup, hence z_1100_)
# Skin 041 ArcadeDefault_01 (human review 2026-10-28, page 041_ArcadeDefault_01.png, page note by the user: "for the attack,
# the should be a hull per projectile"; decision for this skin: each projectile gets ONE convex hull of its own, not one hull
# over the whole attack).
#
# Type 6000551 = ArcadeProjectile, the default skin's attack (sc/effects_brawler.sc, clip arcade_def_projectile). Shared by the
# skins whose projectile is the brawler's default one (ArcadeClassic / ArcadeSilver / ArcadeGold; the other Arcade skins have their
# own types 6000560 .. 6000570 and are not touched). Each projectile object is a chain of 5 tiles: its clip holds, per tile, a
# tile sprite (shape 1, normal blend, fading along the chain: bright near the brawler's start, dim at the far end) and an additive
# halo sprite of the same tile (shape 709, a slightly larger glow: not labelled), plus one long soft streak (shape 0, glow).
# Before: the label was only the bright tiles that the ID pass sees (the additive / faded tiles are invisible to it), in many
# small pieces per projectile (see the page outlines), dim tiles missing.
# Fix:
#  1. POST_BODY: the body of a projectile = its own tile sprites (shape 1), rasterised from the game's geometry (masks.raster_poly,
#     the same projection as the predicted labels), a pixel is tile where the tile's own alpha is >= 1/2 (the CUT rule). Plus the
#     drawn clip-body pixels of the ID pass within 2 px of those tiles (the drawn edge is 1-2 px outside the geometry at the
#     tiles' corners: measured, the same projectile's pixels). Halos (shape 709) and the streak (shape 0) are left out: the user's
#     rule 'soft halos are not part of the attack'. The trail sparks that share a tile's atlas shape (e.g. the 4 px glint above
#     the top tile at tick 315) are not body (rule 3).
#  2. ONE_LABEL 'hull': one convex hull per projectile object (hull_mask per object, not over the attack).
#  3. labels() wrapper: where one projectile's hull fill touches another projectile's tiles (the chains of two shots meet, or
#     two projectiles of one shot lie next to each other), the fill is kept one px away from the other tiles, so the two hulls
#     read as two outlines (one object = one label; no tile pixel is taken from its projectile). Tiles themselves are never cut.
# Nothing else changes: other types, other objects and the super (6000552) are untouched.

ARC6551_TYPE = 6000551
ARC6551_FILE = 'sc/effects_brawler.sc'
ARC6551_EXPORT = 'arcade_def_projectile'
ARC6551_TILE_SHAPE = 1          # clip shape of a tile (its body; normal blend)
ARC6551_TILES = {}              # object id -> tile mask (half resolution) of the current frame (set by POST_BODY)


def _arc6551_tiles(o, cam):
    """(540, 1200) bool: the projectile's own tiles, the game's geometry: tile sprite polygons (shape 1) rasterised, alpha >= 0.5 of
    the tile's own alpha (faded tiles count at their own shape, not at their fade)"""
    import masks
    S = PD.sc(o['file']); m = np.zeros((masks.H_FULL, masks.W_FULL), bool)
    for q in o['parts'].get('clip', []):
        if q['shape'] != ARC6551_TILE_SHAPE or q.get('blend') != 'normal': continue
        a = float(q['mul'][3]) / 255.0
        if a <= 0: continue
        r = masks.raster_poly(S, q, cam)
        if r is None: continue
        x0, y0, C, T = r
        mm = (1.0 - T[..., 0]) / a >= 0.5           # normal blend: 1 - T = tile alpha x layer alpha
        m[y0:y0 + mm.shape[0], x0:x0 + mm.shape[1]] |= mm
    return m[::2, ::2]


def _arc6551_body(body, ctx):
    o = ctx['o']
    if o.get('file') != ARC6551_FILE or o.get('export') != ARC6551_EXPORT:
        return body                                  # another clip of this type id: left as it is
    m = _arc6551_tiles(o, ctx['cam'])
    # the drawn tile pixels (ID pass, clip body) at the tiles' edges: the same projectile's, kept where the geometry is 1-2 px short
    m = m | (ctx['mine'] & _dilate(m, 2))
    ARC6551_TILES[o['id']] = m
    return m


POST_BODY[ARC6551_TYPE] = _arc6551_body
ONE_LABEL[ARC6551_TYPE] = 'hull'                     # one convex hull per projectile object (labels(): per object)

_arc6551_prev_labels = labels


def labels(objs, cam, R, state=None, ended=(), img=None):
    import bulletlabels as _BL6551
    from scipy.ndimage import binary_dilation
    ARC6551_TILES.clear()
    P2 = _arc6551_prev_labels(objs, cam, R, state=state, ended=ended, img=img)
    ids = P2['ids']; share = P2['share']; lids = P2['lids']
    lid_obj = {_BL6551._lid(o): o['id'] for o in objs if o['t'] == ARC6551_TYPE}
    T_of = {}                                        # label index -> the projectile's own tiles
    for k, l in enumerate(lids):
        if l[2] == 'body' and lid_obj.get(l[0]) in ARC6551_TILES:
            T_of[k] = ARC6551_TILES[lid_obj[l[0]]]
    if len(T_of) > 1:
        for k, T in T_of.items():
            mine = ids == k
            if not mine.any(): continue
            others = np.zeros(mine.shape, bool)
            for k2, T2 in T_of.items():
                if k2 != k: others |= T2
            if not others.any(): continue
            ring = binary_dilation(others, structure=np.ones((3, 3), bool))
            cut = mine & ring & ~T                  # this projectile's hull fill beside another projectile's tiles
            if cut.any(): ids[cut] = -1; share[cut] = 0
    P2['ids'] = ids; P2['share'] = share
    return P2
