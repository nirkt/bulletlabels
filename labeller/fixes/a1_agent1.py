# Label-fix agent 1 overrides (2026-10-20): exec'd in idlabels' namespace (IDLABELS_OVERRIDE). Per-type, per-skin fixes.
import numpy as np


def _a1_own_draws(body, ctx, frac=0.5, grow=4):
    """keep only the draws of the body that are this projectile's own: at least `frac` of the draw's labelled pixels lie
    within `grow` px of its predicted body. Drops hit / end effects of earlier projectiles of the same burst that are
    drawn with the projectile's own sprite shapes next to it (rule 2: hit / end animations are not labelled). When no
    draw passes (offset prediction), the draw with the largest overlap with the predicted label stays."""
    if not body.any(): return body
    ids_h, pb, pa = ctx['ids_h'], ctx['pred_body'], ctx['pred_any']
    if not pb.any(): return body
    near = _dilate(pb, grow)
    keep = np.zeros_like(body); best = (0, None)
    for d in np.unique(ids_h[body]):
        m = body & (ids_h == d); n = m.sum(); ov = (m & near).sum()
        if ov >= frac * n: keep |= m
        ova = (m & _dilate(pa, grow)).sum()
        if ova > best[0]: best = (ova, m)
    if not keep.any() and best[1] is not None: keep = best[1]
    # small sparkle draws enclosed by the kept body (glints over it) stay
    from scipy.ndimage import binary_fill_holes
    keep |= body & binary_fill_holes(keep)
    return keep


# Gunslinger (Colt) skins 143 Colt011 (2025 ranked), 230 Colt002 (Hanbok), 258 Colt004 (Player), 272 Colt006 (Badass),
# 348 Colt008 (DarkAngel), 355 Colt009 (Mecha): attack and super. The live projectile took the hit bursts / end stars
# of earlier projectiles of its burst (same atlas shapes, within NEAR): e.g. sw2_143 super tick 349 (burst at the crate),
# ticks 365 / 375 (end stars at range end), attack 90 tick 323
A1_COLT = (6000040, 6000041, 6000022, 6000023, 6000026, 6000027, 6000030, 6000031, 6000034, 6000035, 6000036, 6000037)
for _t in A1_COLT: POST_BODY[_t] = _a1_own_draws


def _a1_rim(body, ctx, thr, r=RIM_R, cool=False):
    """BRIGHT_RIM after the own-draw filter (so rims of dropped hit effects stay out): frame pixels with min(R, G, B) >=
    thr grown from the body at most r px, in its region; cool: only white / cyan pixels (blue >= red: never the warm
    white smoke puffs of broken boxes drawn over it)"""
    img = ctx['img']
    if img is None or not body.any(): return body
    bright = ctx['region'] & (img.min(-1) >= thr)
    if cool: bright &= img[..., 2].astype(int) >= img[..., 0]
    for _ in range(r): body = body | (_dilate(body, 1) & bright)
    return body


# Gunslinger 258 (Colt004 Player) super 6000027: the star's white rim is additive (invisible to the ID pass), the label
# was only the cyan inner star (sw2_258 super t349: rim min(R,G,B) 200-255 over 3 px, the blue comet glow < 150)
POST_BODY[6000027] = lambda body, ctx: _a1_rim(_a1_own_draws(body, ctx), ctx, 150, cool=True)


def _a1_bright_pred(body, ctx, thr, grow=3, close=2):
    """attacks whose visible body is drawn additively / translucent over a big translucent glow sprite (the ID pass sees
    the glow, not the body): body = the frame's bright pixels (max(R, G, B) >= thr) within `grow` px of the predicted
    body, gaps closed, holes filled, the parts touching the predicted body"""
    img = ctx['img']; pb = ctx['pred_body']
    if img is None or not pb.any(): return body
    from scipy.ndimage import binary_fill_holes, label as cc
    near = _dilate(pb, grow)
    bb = near & (img.max(-1) >= thr)
    bb = binary_fill_holes(_close(bb, close)) & near
    L, n = cc(bb); keep = np.unique(L[bb & pb]); keep = keep[keep > 0]
    return np.isin(L, keep) if keep.size else body


# MechaDude 465 (Mecha007 Cybernoodles overcharge skin) attack 6001083 (Mecha007UltiProjectile, the cubes): the cube
# is drawn translucent / additive over its glow ellipse (shared sprites, coverage < 128): only slivers were labelled
# (sw2_465 attack 90 t309: predicted body max(R, G, B) median 253, glow around 79)
POST_BODY[6001083] = lambda body, ctx: _a1_bright_pred(body, ctx, 200)


def _a1_solid(body, ctx, close=2):
    """one solid body: gaps up to `close` px closed, holes filled (translucent stacked layers left holes where the top
    layer is under half opaque)"""
    if not body.any(): return body
    from scipy.ndimage import binary_fill_holes
    return binary_fill_holes(_close(body, close)) & ctx['region']


# MechaDude 602 (Mecha006 Angel) attack 6001081 (Mecha006UltiProjectile, the glowing drops): translucent stacked
# layers, the label was the drop's outline with many holes inside (sw2_602 attack 90 t315): its drawn body plus the
# bright pixels of its predicted body (white core, lilac flame: max(R, G, B) >= 170; glow around ~80), made solid
POST_BODY[6001081] = lambda body, ctx: _a1_solid(body | _a1_bright_pred(np.zeros_like(body), ctx, 170, grow=1), ctx)



def _a1_gus007_atk(body, ctx, grow=2):
    """Gus007 attack: the opaque pixels of the body's own draws (part of the box is drawn with sprites its trail shares),
    minus the dark-blue smoke wisp of its clip: core = the light-blue fist (max(R, G, B) >= 0xa0) or the tan box (red >
    blue); body = core, holes filled, plus body pixels within `grow` px of it (the fist's darker rim)"""
    if not body.any() or ctx['img'] is None: return body
    from scipy.ndimage import binary_fill_holes, label as cc
    ids_h, cv, img = ctx['ids_h'], ctx['cv'], ctx['img'].astype(np.int16)
    own = np.unique(ids_h[body]); own = own[own > 0]
    cand = body | (np.isin(ids_h, own) & (cv >= CUT) & ctx['region'])
    core = cand & ((img.max(-1) >= 0xa0) | (img[..., 0] > img[..., 2]))
    # the rest of the tan box next to it (grown <= 6 px from the core), also where a translucent smoke puff of its trail
    # is drawn over it (grey-tan there: red >= blue - 4; the night ground and the blue glow are clearly blue)
    tan = ctx['region'] & (img[..., 0] >= img[..., 2] - 4) & (img.max(-1) >= 0x70)
    core = core | (tan & _dilate(core, 6))
    # the light fist where a translucent puff of its trail is the front-most draw (in its predicted body)
    core = core | (ctx['region'] & _dilate(ctx['pred_body'], 1) & (img.max(-1) >= 0xb0) & _dilate(core, 4))
    core = binary_fill_holes(_close(core, 1)) & (cand | tan)
    out = core | (cand & _dilate(core, grow))
    out = binary_fill_holes(_close(out, 2)) & ctx['region']
    L, n = cc(out); keep = np.unique(L[out & body]); keep = keep[keep > 0]
    if not keep.size: return body
    sz = np.bincount(L.ravel()); keep = [k for k in keep if sz[k] >= 0.15 * max(sz[j] for j in keep)]
    return np.isin(L, keep)


# SoulCollector 454 (Gus007 MHA) attack 6001252: the dark-blue smoke wisp above the box (a sprite of its own clip) was
# labelled, and part of the box (sprites shared with its trail) was not (sw2_454 attack 90 t309: wisp max(R,G,B) < 0xa0
# and blue, fist >= 0xa0, box tan)
POST_BODY[6001252] = _a1_gus007_atk


def _a1_dancer003_ulti(body, ctx, cut=32, frac=0.5, grow=3):
    """Dancer003 super, the water tornado: one big translucent funnel sprite (clip body, partly sprites its trail shares)
    of which only scattered fragments were at least half opaque. Body = the pixels of its own draws (>= frac of the draw
    within `grow` px of the predicted body) showing its clip or shared sprites at coverage >= cut, as one solid label
    (convex hull: the funnel's rings with the water between them)"""
    ids_h, cv, C, pb = ctx['ids_h'], ctx['cv'], ctx['C'], ctx['pred_body']
    if not pb.any(): return body
    near = _dilate(pb, grow); reg = ctx['region']
    cand = reg & ((C == 1) | (C == 5)) & (cv >= cut)
    out = np.zeros_like(body)
    for d in np.unique(ids_h[cand]):
        if d <= 0: continue
        m = cand & (ids_h == d)
        if (m & near).sum() >= frac * m.sum(): out |= m
    if not out.any(): return body
    from scipy.ndimage import label as cc
    L, n = cc(_close(out, 2)); sz = np.bincount(L.ravel()); sz[0] = 0
    keep = np.nonzero(sz >= 0.15 * sz.max())[0]
    return hull_mask(np.isin(L, keep) & out)


# Dancer 149 (Dancer003 Fairies) super 6001673: the tornado was labelled as scattered fragments of its translucent
# funnel (sw2_149 super t349 / t355; Haiku: 'partly and too small'). Drawn as many translucent rings -> one hull label
POST_BODY[6001673] = _a1_dancer003_ulti


# TrickshotDude 443 (Rico011 Brawlentine) attack 6000109 and super 6000110: the white bounce / hit star of an earlier
# projectile (same atlas, clip-body shapes) was taken by the next live projectile (sw2_443 super t349 / t357 / t361
# at the crate): own-draw filter as for the Colt skins
for _t in (6000109, 6000110): POST_BODY[_t] = _a1_own_draws
