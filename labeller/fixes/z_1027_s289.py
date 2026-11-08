# sweep 10-27 fix for skin 289 (Haiku fix, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s289/)
# Skin 289 TntDudeToxic: Mike007Projectile (6000279), the cheese-shard attack (attack 0 and attack 90 rows).
# Sweep 2026-10-27, pairs 5368 (tick 317) and 5369 (tick 321): the right-hand shard's label had a thin red spur running up from
# its top edge toward a teal glow speck, which is not part of the shard. Per pixel the spur (a 12 px wedge in fact) is the shard's
# trail drawn over the glow (sprite code 4: trail over, not body), which the body rule takes in from the aligned prediction.
# The shard is an opaque solid: its drawn body (clip sprite, code 1) starts at its top row, so trail-over pixels above that row
# are candidates. But on ticks 305-309 the glow covers the shard's top face, and that face is also trail-over above the first
# clip-body row, so the row test alone would cut the face (about 30 % of the label on 307 and 309). The two cases differ in
# colour: glow over the dark background is darker than the shard's own body (0.5-0.7 of its median luminance on ticks
# 317-321), glow over the lit face is not (0.9-1.2 on 307-309). So a candidate is cut only as a whole connected part whose mean
# luminance is below CUT_LUM of the object's clip-body median luminance. Everything else is left as it is.
MIKE_CHEESE_TYPE = 6000279
MIKE_CHEESE_CUT_LUM = 0.8


def _fix_mike_cheese_glow(body, ctx):
    from scipy.ndimage import label as cc
    if not body.any(): return body
    C = ctx['C']; img = ctx['img']
    top = (C == 1) & body
    if not top.any(): return body
    y_top = np.nonzero(top)[0].min()
    cand = body & (C == 4) & (np.arange(C.shape[0])[:, None] < y_top)
    if not cand.any(): return body
    lum = 0.3 * img[..., 0].astype(float) + 0.59 * img[..., 1].astype(float) + 0.11 * img[..., 2].astype(float)
    kl = np.median(lum[top])
    if kl <= 0: return body
    L, n = cc(cand, structure=np.ones((3, 3), bool))
    out = body.copy()
    for k in range(1, n + 1):
        m = L == k
        if lum[m].mean() / kl < MIKE_CHEESE_CUT_LUM: out &= ~m
    return out


POST_BODY[MIKE_CHEESE_TYPE] = _fix_mike_cheese_glow
