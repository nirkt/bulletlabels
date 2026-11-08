# sweep 10-27 fix for skin 562 (Sonnet revision, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s562/)
# sweep 10-27 fix (2nd version) for skin 562 PercenterCony (task out/sweep2/haikutasks/sweep1027/s562/, row "sw2_562 attack 90 deg",
# pairs 9909 (tick 305: hole, too_small), 9910 (307), 9911 (309), 9912 (311), 9913 (313): hole).
#
# Type 6000825 = Colette007Projectile (clip colette_007_atk_projectile): the cony bunny of this skin (attack shot: one
# projectile per shot, drawn from its clip sprites: body shape 635, ears / arms 47, 2777, 65, hearts of its trail = trail code).
#
# What was wrong (checked on every tick 303-315 of the attack 90 row):
#  a. holes: the bunny's face (eyes and mouth, drawn in front of the body with no sprite of this type) is a hole of 45-51 px
#     on ticks 309 / 313 (the clean-up fills holes up to 40 px only);
#  b. tick 305 (pair 9909 too_small): besides the bunny that flies (the tracked projectile, lower) a second, larger cyan bunny is
#     drawn under the brawler's green ring, in a different pose (lying, head to the left). It is made of the projectile's own
#     clip sprites (body shape 635), fully opaque, and is not a flash or a glow. By rule 3 (every visible attack object is
#     labelled) it is labelled. Before, the label leaked onto a small part of it only. JUDGEMENT, see the report: if the user
#     says that bunny is only the launch visual of the throw, remove ADOPT_BLOB below and nothing else changes.
#  c. the label stopped inside the bunny's ears / arms / feet (outside the predicted footprint) and was frayed there (thin
#     bars at the ear tips, detached loops, a small loop above the ear on tick 313).
#
# Approaches considered:
#   1. 3D / drawn-pixel rule via EXTEND_BODY (the existing mechanism: connected opaque clip-body pixels of the label's own draws,
#      wherever they reach) -- chosen as the base for the ears / arms / feet: it uses the sprite codes (stable), not colours.
#   2. a colour threshold on the frame (cyan vs dark background): rejected, the soft blue glow and the trail hearts (similar blue)
#      make the edge depend on the frame.
#   3. a size threshold on parts / a position test against the prediction (the first attempt removed the second bunny): rejected,
#      it turns the visible attack into a missing label.
#   4. a time rule (first tick of a projectile): rejected, the label function has no age.
# Chosen: EXTEND_BODY + adoption of whole clip-body blobs near the projectile (so tick 305's second bunny is labelled whole, also
# if the leak that touched it were missing) + fill of the enclosed holes (faces) + close one-pixel gaps + drop tiny detached bits.
#
# Keyed to type 6000825 only. Shared type: any other skin using Colette007Projectile gets the same rule (not checked: only
# this capture is available).

COLETTE007_TYPE = 6000825
COLETTE007_CLIP = 'colette_007_atk_projectile'
COLETTE007_HEART = 65     # shape id of the trail hearts inside the clip's atlas
COLETTE007_BLOB = 100      # half-resolution px: an opaque own-clip blob of at least this size near the projectile is body
COLETTE007_NEAR = 50       # half-resolution px around the predicted footprint where such a blob is searched
COLETTE007_HOLE = 150      # holes up to this size (px) inside the bunny are the face, filled (ground showing through is bigger)
COLETTE007_GROW = 3        # px: edge growth into the bunny's bright cyan
COLETTE007_CYAN = 0xb0     # green channel from which a pixel next to the label is the bunny's own colour (hearts: <= 0xa8)
COLETTE007_CLOSE = 3       # px: gaps between the bunny's body and its ears / arms / feet (sprite edges) are closed
COLETTE007_SPECK = 0.1     # detached bits (after closing 1 px gaps) smaller than this share of the main part are dropped

EXTEND_BODY.add(COLETTE007_TYPE)


def _fix_colette007_bunny(body, ctx):
    from scipy.ndimage import label as _c562_cc, binary_fill_holes as _c562_fill
    if not body.any():
        return body
    C, cv = ctx['C'], ctx['cv']
    st = np.ones((3, 3), bool)
    # ADOPT_BLOB: opaque pixels of its own clip sprites forming a blob near the projectile
    # the trail's hearts (clip shape 65, drawn like the other clip shapes, partly over the bunny's neck) are not the bunny
    SP = _a4_shape_pixels(ctx['R'], 'sc/effects_brawler.sc', (COLETTE007_CLIP,))
    heart = SP == COLETTE007_HEART + 1
    body = body & ~heart
    cand = (C == 1) & (cv >= CUT) & ~heart
    near = _dilate(ctx['pred_any'], COLETTE007_NEAR)
    L, n = _c562_cc(cand | body, structure=st)
    if n:
        sz = np.bincount(L.ravel())
        for k in range(1, n + 1):
            if (body & (L == k)).any() or (sz[k] >= COLETTE007_BLOB and (near & (L == k)).any()):
                body = body | (L == k)
    # sprite edges: the label stops a pixel or two inside the bunny's silhouette (ear / arm tips, lit rims) where the ID pass gives
    # no sprite code. Grow it up to COLETTE007_GROW px into the bunny's own bright cyan (the trail hearts are a flatter, darker
    # blue, G < 0xb0, and are never reached through bright cyan pixels).
    img = ctx['img']
    if img is not None:
        im = img.astype(np.int16)
        bright = (im[..., 1] >= COLETTE007_CYAN) & (im[..., 2] >= 0xe0) & (im[..., 0] <= 0x90) & near & ~_dilate(heart, 1)
        for _ in range(COLETTE007_GROW):
            body = body | (_dilate(body, 1) & bright)
    # one-pixel gaps closed, enclosed holes (the face) filled
    body = _close(body, COLETTE007_CLOSE)
    hl, hn = _c562_cc(_c562_fill(body) & ~body)
    if hn:
        hs = np.bincount(hl.ravel())
        for k in range(1, hn + 1):
            if hs[k] <= COLETTE007_HOLE:
                body = body | (hl == k)
    # tiny detached bits (ear-tip crumbs, loops) are dropped, the main part always stays
    L, n = _c562_cc(body, structure=st)
    if n >= 2:
        sz = np.bincount(L.ravel())
        main = 1 + int(np.argmax(sz[1:]))
        keep = np.zeros(n + 1, bool)
        keep[main] = True
        for k in range(1, n + 1):
            if k != main and sz[k] >= COLETTE007_SPECK * sz[main]:
                keep[k] = True
        body = body & keep[L]
    return body


POST_BODY[COLETTE007_TYPE] = _fix_colette007_bunny
