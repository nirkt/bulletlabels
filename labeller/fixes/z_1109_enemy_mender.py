# 2026-11-04 enemy-label fix, group misc2 (skins 085, 088, 111, 115). exec'd in idlabels' namespace (IDLABELS_OVERRIDE).
#
# ---- skin 111 MenderDefault, type 6001755 MenderProjectile (the shining ball on its tether) ----
# CAUSE: z_1100_s111 finds the ball's white core with min(R,G,B) >= 150 and the disc around it with G >= 200 and B >= 200 (the cyan
# disc of the blue art). The red (enemy) ball is a white core inside a PINK disc: min(R,G,B) >= 150 also takes in the pink halo
# (its G/B are 130-190), and G >= 200 / B >= 200 only holds in the white core itself, so the enemy label was the core only
# (76-134 px against 320-380 px of the blue disc).
# FIX (enemy only, ctx['own'] False): the same rule with red-art colour tests: core = min(R,G,B) >= 205, disc = R >= 224, G >= 120,
# B >= 150 within DISC_R of the core centre; fill by point reflection (hidden by the tether) and the convex hull, exactly as the blue
# rule. Own labels call the installed function unchanged.
_EN4M_CORE_T = 205
_EN4M_DISC = (224, 120, 150)     # R, G, B minimum of the pink disc (the pink halo outside the disc has R < 224 or G < 120)
_en4m_prev_disc = POST_BODY.get(6001755)


def _en4m_disc(body, ctx):
    if ctx.get('own'): return _en4m_prev_disc(body, ctx)
    from scipy.ndimage import label as cc
    img = ctx['img']
    if img is None: return body
    if ctx['pred_any'].any():
        core = ctx['region'] & (img.min(-1) >= _EN4M_CORE_T) & _dilate(ctx['pred_any'], 6)
    else:
        if ctx['o']['id'] not in _MENDER_END_NOW: return _en4_np_zeros_like(body)
        ex, ey = _proj(ctx['cam'], ctx['o']['logic'][0], ctx['o']['logic'][1], 0)
        H0, W0 = body.shape; yy0, xx0 = np.mgrid[0:H0, 0:W0]
        core = (yy0 - ey) ** 2 + (xx0 - ex) ** 2 <= 40 ** 2
        core &= (img.min(-1) >= _EN4M_CORE_T) & (ctx['C'] > 0)
    if not core.any(): return _en4_np_zeros_like(body)
    L, n = cc(core, structure=np.ones((3, 3), bool)); sz = np.bincount(L.ravel())
    main = 1 + int(np.argmax(sz[1:])); ys, xs = np.nonzero(L == main)
    cy, cx = ys.mean(), xs.mean()
    H, W = body.shape; yy, xx = np.mgrid[0:H, 0:W]
    near = (yy - cy) ** 2 + (xx - cx) ** 2 <= DISC_R ** 2
    r, g, b = _EN4M_DISC
    bright = (img[..., 0] >= r) & (img[..., 1] >= g) & (img[..., 2] >= b) & near
    L2, n2 = cc(_close(bright, 1), structure=np.ones((3, 3), bool))
    ks = np.unique(L2[(L == main) | (core & ~(L == 0))]); ks = ks[ks > 0]
    if not ks.size: return _en4_np_zeros_like(body)
    vis = np.isin(L2, ks)
    py_, px_ = np.nonzero(vis)
    my, mx = np.round(2 * cy - py_).astype(int), np.round(2 * cx - px_).astype(int)
    ok = (my >= 0) & (my < H) & (mx >= 0) & (mx < W)
    full = vis.copy(); full[my[ok], mx[ok]] = True
    full &= near | vis
    return hull_mask(full)


def _en4_np_zeros_like(a):
    import numpy as _n
    return _n.zeros_like(a)


POST_BODY[6001755] = _en4m_disc


# Skins 085, 088, 115: NO code (checker false positives from the scene, see the report): 085 spawn:CocoonerCocoon is a stray model draw of the
# water edge at the screen bottom taken for the cocoon (spawn_labels, CHAR_NEAR 80 px) because the enemy cocoon stood 78 px from it; 088
# volley pellets are labelled by the same rule in both runs (only their short labelled windows differ with hit effects); 115 body is SAME
# on the two re-captures (the first capture's own waffle was just drawn much larger).
