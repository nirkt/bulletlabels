# sweep 10-28 round 3 fix for skin 107 (DancerDefault), super 6001664 (DancerProjectileUlti, clip mina_def_ulti_projectile of
# sc/effects_brawler_mina.sc), sw2_107 super ticks 345-359 (pairs 1711 / 1713: holes at the dark centre and around the rims).
# Cause: only the opaque bright rim crescents (shape 223, coverage >= CUT 128) were labelled; the cyan funnel body (the clip's
# shapes 4/113/114/115/202/237 at coverage 34-112) and the dark centre were not. The tornado is one multi-part translucent
# object -> body = every pixel of the clip's own shapes (any non-trivial coverage) near the predicted body, minus the lilac
# music notes (drawn into the same effect clip family, not the projectile), then ONE solid convex hull (as the sibling
# Dancer supers 6001673 / 6001669). Keyed to this type only (6001664 is used by Dancer skin 107 and any recoloured Dancer
# skin of the same default clip; see report).
_F6001664 = dict(file='sc/effects_brawler_mina.sc', clips=('mina_def_ulti_projectile', 'mina_def_ulti_projectile_red'),
                 near=10, cut=24)


def _fix_6001664(body, ctx):
    from scipy.ndimage import label as cc
    p = _F6001664
    if not ctx['pred_body'].any(): return body
    img = ctx['img'].astype(int)
    SP = _a4_shape_pixels(ctx['R'], p['file'], p['clips'])
    note = (img[..., 2] >= 200) & (img[..., 0] >= 130) & (img[..., 0] <= 225) & (img[..., 1] >= 90) & (img[..., 1] <= 185) & (img[..., 0] > img[..., 1] + 25)
    note = _dilate(note, 1)
    cand = ctx['region'] & (SP > 0) & (ctx['cv'] >= p['cut']) & ~note
    cand |= body & ~note
    cand &= _dilate(ctx['pred_any'], p['near'])
    if not cand.any(): return body
    L, n = cc(_close(cand, 2) | cand); sz = np.bincount(L.ravel()); sz[0] = 0
    keep = np.nonzero(sz >= 0.1 * sz.max())[0]
    out = hull_mask(np.isin(L, keep) & cand)
    # solid world geometry / characters drawn IN FRONT of the tornado (depth-writing obstacle / character draws, fs 623557017, front-most
    # there: the crate at t359, the brawler at t349) are not part of it
    R = ctx['R']; ids = R['id'][::2, ::2]; dr = R['draws']
    solid = [d for d in np.unique(ids[out]) if d > 0 and d in dr and dr[d].get('dmask') == 1 and dr[d].get('fs') == 623557017]
    if solid:
        sm = np.isin(ids, solid) & out
        from scipy.ndimage import binary_dilation
        ring = binary_dilation(out, iterations=2) & ~out
        Ls, ns = cc(sm | (binary_dilation(sm, iterations=1) & ring))
        touch = np.unique(Ls[ring & binary_dilation(sm, iterations=2)]); touch = touch[touch > 0]
        out = out & ~(np.isin(Ls, touch) & sm)
        L, n = cc(out); sz = np.bincount(L.ravel()); sz[0] = 0
        out = L == int(sz.argmax())
    return out


POST_BODY[6001664] = _fix_6001664
