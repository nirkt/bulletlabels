# hosted review 2026-10-29 fix (Haiku fix, reviewed by the coordinator, installed 10-29 LOCAL ONLY; source out/sweep2/haikutasks/site1029/s130/fix2.py; tested via IDLABELS_OVERRIDE after all fixes incl. z_1100_*, hence z_1101_)
# s130 fix2 (CrossBomberFranken, skin 130; hosted review 2026-10-29, THE USER + coordinator review): super 45 =
# Grom003UltiProjectile (type 6000968). Replaces fix.py's hook for this type (load it INSTEAD of fix.py: IDLABELS_OVERRIDE=fix2.py).
# Coordinator's review of fix.py at tick 345: (1) an inner line runs through the flask and its right rim (the dark outlined round
# edge) is outside the label; (2) a notch at the bottom between the blob and the lower streak. Required: at 345 ONE smooth filled
# outline around the cyan blob + the whole flask including its rim; the bottom notch filled; no yellow muzzle rays, no soft glow.
# Also: the solid blue disc at the bottom of the body (ticks 355 and on; the eye of the same projectile) is part of the body
# (solid, saturated, moves with the body, the same eye that is labelled at 357-363): its white top rim is included.
#
# Cause (ID pass, per pixel): at 345 the cyan blob's body is draw 95 (translucent additive, no clip shape; coverage median 74
# inside the prediction), and the flask is runtime-texture model draws (MODEL_TEX, draw 72) batched with the tan box; the flask's
# right part (its rim) has coverage < 128 and was left out by the coverage cut. Background draws (texture 1d304abe: the scene
# under the projectile) and the HUD bars (opaque UI draws above the blob) are never added.
#
# Rules (POST_BODY for 6000968, after the per-type rules, before ONE_LABEL / ended-drop):
#  A. spawn phase (only where the uncoded body draws make >= 20 px): uncoded (code 0) pixels of the draws that use the label's
#     own body texture hash, inside the predicted region grown 2 px and within 12 px of the label; opaque (cov >= CUT) or bright
#     (max RGB >= 150; the dim soft fringe of the same draw stays out).
#  B. whole flask: every model-drawn pixel (any coverage) right of the grown label's centre column, within 12 px of it, not in
#     the label's left (the tan box is left of the blob: out).
#  C. one smooth silhouette (spawn phase only): closing radius 4 of the union, holes filled, clipped to the predicted region;
#     the closing may not add background-texture pixels (1d304abe) or pixels that are not part of the union's neighbourhood.
#  D. the blue disc (all ticks where it is drawn): unlabelled bright white pixels (min RGB >= 170) within 3 px of a solid
#     saturated blue pixel (R <= 90, G <= 170, B >= 200, B - R >= 130) and inside the predicted region are added (its rim).
#     The blue pixels themselves are already labelled (about 92%, checked on 349-363).
# Verified (fix2 vs before): see report2.md (tick list, changed pixels per tick).
from scipy.ndimage import binary_fill_holes as _s130_fill2


def _fix2_6000968(body, ctx):
    if not body.any(): return body
    ids_h, C, cv, region, pa, img = ctx['ids_h'], ctx['C'], ctx['cv'], ctx['region'], ctx['pred_any'], ctx['img']
    draws = ctx['R']['draws']
    BG = {d for d, dr in draws.items() if str(dr.get('th', '')).startswith('1d304abe')}   # scene background texture
    out = body.copy()
    # A. spawn-phase body: the uncoded draws of the label's body texture
    bd = [int(d) for d in np.unique(ids_h[body & (C == 1)]) if d > 0]
    th = {draws[d].get('th') for d in bd if d in draws}
    tex = [d for d, dr in draws.items() if th and dr.get('th') in th]
    spawn = False
    if tex and img is not None:
        solid = (cv >= CUT) | (img.max(-1) >= 150)
        a = np.isin(ids_h, tex) & (C == 0) & solid & _dilate(pa, 2) & _dilate(body, 12) & ~body
        if a.sum() >= 20:
            spawn = True; out = out | a
    if spawn:
        # B. the whole flask: model draws right of the centre column, near the grown label, any coverage
        model = [d for d, dr in draws.items() if dr.get('th') == MODEL_TEX]
        ys, xs = np.nonzero(out); cx = xs.mean()
        yy, xx = np.mgrid[0:out.shape[0], 0:out.shape[1]]
        out = out | (np.isin(ids_h, model) & _dilate(out, 12) & ~out & (xx > cx))
        # C. one smooth silhouette: closing r=4, holes filled, no background pixels added, inside the predicted region
        bg = np.isin(ids_h, list(BG)) if BG else np.zeros(out.shape, bool)
        # the convex hull of the union, clipped to the union and to bright cyan pixels (the body's lower band, draw 95 and
        # its kin; blue-channel high, max RGB >= 150): fills the bottom notch, keeps yellow rays and dim glow out
        cyan = (img[..., 2] >= 180) & (img[..., 2] >= img[..., 0].astype(int) + 10) & (img.max(-1) >= 150) if img is not None else np.zeros(out.shape, bool)
        hull = hull_mask(out)
        out = _s130_fill2(hull & (out | (cyan & _dilate(out, 12)))) & region & (~bg | out)
    # D. the solid blue disc's white top rim (any tick the disc is drawn)
    im = img.astype(np.int16) if img is not None else None
    if im is not None:
        blue = (im[..., 2] >= 200) & (im[..., 0] <= 90) & (im[..., 1] <= 170) & (im[..., 2] - im[..., 0] >= 130)
        if BG: blue &= ~np.isin(ids_h, list(BG))
        if blue.sum() >= 30 and (blue & out).any():
            white = (im.min(-1) >= 170) & region & ~out
            model = [d for d, dr in draws.items() if dr.get('th') == MODEL_TEX]   # never the tan box / flask model draws
            rim = white & _dilate(blue, 3) & _dilate(out, 3) & ~np.isin(ids_h, model)
            out = out | rim
    return out


POST_BODY[6000968] = _fix2_6000968
