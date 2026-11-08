# Task 23, skin 183 BeeSniperVirus (Bea), the user's rules (one clean label per projectile; additive rims in; no holes):
# the bee (body, eyes, both translucent wings, outer cyan rim) is ONE convex hull. Haiku draft (t23/fix.py) took the clip's
# body sprites (code 1, also the translucent wings, cv >= 32) and hulled them, but cut glow pixels out of the hull (holes)
# and missed the 1-2 px antialiased rim. Coordinator (2026-10-23): no cut inside the hull; grow the hull into pixels drawn by
# the bee's own draws (codes 0/1 only, so never the trail squares 3/4 or the glow 2, even when batched), then 1 px into the
# bee's additive cyan rim (rule: additive rims in), then hull again.
BEE_SKIN183 = (6000656,   # Bea006Projectile (attack)
               6000658)   # Bea006UltiProjectile (super)
BEE183_GROW = 2           # px


def _bee_skin183(body, ctx):
    C, cv, ids = ctx['C'], ctx['cv'], ctx['ids_h']
    bee = (C == 1) & ctx['region'] & (cv >= 32)          # body, eyes, wings (translucent)
    m = body | bee
    if not m.any(): return body
    hull = hull_mask(m)
    own = np.isin(ids, np.unique(ids[m & (C == 1)])) & (C <= 1) & (cv >= 16) & _dilate(ctx['region'], BEE183_GROW)
    for _ in range(BEE183_GROW): hull = hull | (_dilate(hull, 1) & own)
    img = ctx['img']
    if img is not None:   # the bee's additive cyan rim (own draw, no sprite code): 1 px, then hull again
        im = img.astype(np.int16)
        rim = (C == 0) & (im[..., 1] >= 150) & (im[..., 2] >= 150) & (im[..., 0] <= 170) & (cv >= 64)
        hull = hull | (_dilate(hull, 1) & rim)
    return hull_mask(hull)


for _t in BEE_SKIN183: POST_BODY[_t] = _bee_skin183
