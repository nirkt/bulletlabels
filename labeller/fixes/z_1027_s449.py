# sweep 10-27 fix for skin 449 (Sonnet revision, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s449/)
# sweep 10-27 fix (2nd attempt) for skin 449 (ShamanAprilFools, Nita's orange slime blob; task s449, pairs 8421-8424,
# attack 90 deg ticks 309-315). Type 6000246 Shaman013Projectile only (no other skin uses it; no per-attack decision
# in RULES.md concerns it; the super 6000247 is not touched).
#
# What was wrong (the label is a stack of effects of one blob drawn by ONE front-most draw plus a few small ones):
#  - holes / diamonds / notches at the edges and inside (sprites of the trail clip share the blob's shapes, so parts of the
#    blob carry the 'trail over' code and are dropped; the overlap seams between the blob's own sprites have a low
#    coverage in the ID pass, and the prediction fills in diamond-shaped bits);
#  - too_small at tick 315: the left lobe with its white highlight lies behind a low-coverage seam and is cut off;
#  - too_big: low-coverage BLUE glow pixels (the soft glow ring under the blob, B - R > 0, coverage < CUT) were part of the
#    label (the bulge at the lower left); and the hazy translucent bubble of the trail near the head at ticks 309-311.
#
# Fix: the blob = the pixels of its own draw(s) (the draws carrying at least a quarter of the old label's pixels; the
# ID pass tells which draw is front-most at each pixel, so the glow ring under it and the trail drawn in other draws are
# out by construction):
#  1. core = own-draw pixels with coverage >= CUT, plus old label pixels within 1 px of the own draws (edge pixels), minus
#     glow pixels (blue minus red > 0 and coverage < CUT);
#  2. seams: own-draw pixels with coverage >= 30 and not glow-coloured within 3 px of the core are bridged by a closing of 3
#     (the thin low-coverage lines where two sprites of the blob overlap, which cut the lobes off); nothing else is added;
#  3. opening of 2 on that (drops thin spurs); keep the old label's edge pixels, keep only the part connected to the old
#     label; fill enclosed holes (the blob's own highlight ring, the small diamonds).
# Old-label pixels farther than 1 px from the own draws (the trail's hazy bubble beside the head) are dropped, unless they are
# enclosed by the blob (they are then holes and get filled). Ticks 318+ (attack ended): no body, nothing changes.
S449_FRAC = 0.25     # a draw is the blob's own when it holds this fraction of the old label's pixels in its largest draw
S449_LOWCV = 30      # coverage from which a seam pixel of an own draw may be bridged
S449_CLOSE = 3       # px: seam bridge
S449_OPEN = 2        # px: opening that drops thin spurs


def _s449_fix_6000246(body, ctx):
    from scipy.ndimage import binary_fill_holes, binary_opening, binary_closing, binary_dilation, label as cc
    img = ctx['img']
    if img is None or not body.any(): return body
    cv = ctx['cv']; reg = ctx['region']; ids = ctx['ids_h']
    bl = img[..., 2].astype(np.int16) - img[..., 0].astype(np.int16)    # blue minus red
    glow = (bl > 0) & (cv < CUT)
    C = ctx['C']
    spr = body & (ids > 0) & (C > 0)       # label pixels showing the blob's own sprites (a mesh / predicted-only body: untouched)
    if spr.sum() < 0.5 * body.sum(): return body
    dd, n = np.unique(ids[spr], return_counts=True)
    if not dd.size: return body
    own = dd[n >= S449_FRAC * n.max()]
    ow = np.isin(ids, own) & reg
    edge = body & ~glow & binary_dilation(ow, iterations=1)
    core = (ow & (cv >= CUT)) | edge
    ok = ow & (cv >= S449_LOWCV) & (bl <= 0)
    U = core | (binary_closing(core | (binary_dilation(core, iterations=S449_CLOSE) & ok), iterations=S449_CLOSE) & ok)
    L, k = cc(U); keep = np.unique(L[body]); U = np.isin(L, keep[keep > 0])
    U = edge | binary_opening(U, iterations=S449_OPEN)
    L, k = cc(U); keep = np.unique(L[edge]); U = np.isin(L, keep[keep > 0])
    out = binary_fill_holes(U) & reg
    return out if out.any() else body
POST_BODY[6000246] = _s449_fix_6000246
