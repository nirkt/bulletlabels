# hosted review 2026-11-01 fix (Haiku fix, reviewed by the coordinator, installed 11-01 LOCAL ONLY; source out/sweep2/haikutasks/site1101/g_ambusher/fix.py; runs after z_1101_*)
# Hosted review 2026-11-01, task g_ambusher: Ambusher (Lily) skins 200 AmbusherFantasy and 206 AmbusherSandsOfTime.
# Both pages were marked bad WITHOUT a note: the coordinator asked for every frame to be compared raw vs label. Found,
# per type (exec'd in idlabels' namespace; keyed to the explicit type ids below; no other skin uses these types):
#  (a) after the logic projectile has finished, the wave's label stayed on for ~6 ticks (the object is still in the
#      client state, finished and hidden): the leftover shards, sparkles, feathers and the faded body carried a label.
#      Rules 2 and 4 (fades and leftovers after the end are not labelled). Fix: the body is empty while the object is
#      finished or hidden. Types 6001458 (skin 200 attack 0 / 90 wave: ticks 267-271, 307-311 cleared), 6001461 (skin 206
#      attack 0 / 90: 267-269, 307-309 cleared), and the supers 6001459 / 6001462 (ticks 351 cleared).
#  (b) the wave body was labelled in pieces: the arrow's pale back and its tip were cut off (their coverage is 24-127,
#      translucent), the blue discs on the skin 206 arrow head were holes, the green energy streak across the arrow left a
#      slit, and the label took in twinkle sparkles / the brawler's bubbles (skin 206 tick 265 and 305: outlines on the
#      brawler, the swatch of the same atlas). Fix (6001458 and 6001461): the body is the arrow's OWN clip shapes (found per
#      pixel by atlas shape, _a4_shape_pixels), one connected piece reaching into the object's reach (80 px past its
#      prediction, the prediction is offset from the drawn arrow), the solid swatch shape 25 only where it is bright,
#      translucent and connected to the arrow (skin 200 only: the arrow's pale back), the green streak (shape 110, skin 200)
#      not body, closed (6 px), holes filled, a 1 px opening, and a 2 px anti-aliased edge of the arrow's own draws grown.
#      Beam, sparkles (shape 108), feathers and brawler parts are not in the clip, so they are out.
#  (c) super 200 (6001459): its outline had a sparkle sticking out of its lower-left edge (labelled, shape 108) with a
#      bay beside it, and the body was cut around the sparkle. Fix: the seed is the hull of its own shapes 113 (body) and
#      133 (green highlight) at the same coverage (CUT), no glow: the seed is convex, so the hull closes the bay.
#  Skin 206 super (6001462) is left as it is: its label was already the whole seed without glow (checked on 345-349).

import numpy as np

_AMB_SWIRL = 'sc/effects_brawler_lily.sc'
_AMB_SWATCH = 25           # shape id of the solid colour swatch (atlas corner) shared by the Lily clips
_AMB_CLOSE = 6             # px: gaps closed in the arrow body (the green streak crossing it leaves a slit of ~10 px)
_AMB_EDGE = 2              # px: the translucent anti-aliased edge of the arrow's own draws grows the body (tip, rims)
_AMB_EDGE_CV = 16          # coverage from which that edge counts
_AMB_OVERLAY = (110,)      # shape ids of the arrow's clip that are an energy streak drawn over it (green, 200 wave): not body
_AMB_CORE_CV = 24          # coverage from which the arrow's own shapes are body (the sprite's translucent edge reaches the tip)
_AMB_SW_CV = 32            # coverage from which the swatch (the arrow's translucent pale back half) is the arrow's body
_AMB_SW_MIN = 100          # the swatch counts as the arrow's pale back only where the frame colour is bright (min channel >= this)
_AMB_OPEN = 1              # px: thin spurs of the arrow's own streak sprite cut off (opening)
_AMB_REACH = 80            # px: the arrow's drawn shape reaches this far past its predicted region (the prediction is offset)


def _amb_finished(ctx):
    o = ctx['o']
    return bool(o.get('finished')) or not bool(o.get('drawn', True))


def _amb_wave_body(ctx, clip, arrow_only):
    """the wave's body from its own clip's shapes: one connected piece that reaches into the object's reach (the arrow
    is one piece; its tip and back are not cut at the predicted region's edge), holes filled, closed, edge grown."""
    from scipy.ndimage import binary_fill_holes, binary_opening, label as cc
    R = ctx['R']; cv = ctx['cv']; img = ctx['img']; pa = ctx['pred_any']
    reach = _dilate(pa, _AMB_REACH) if pa.any() else ctx['region']
    SP = _a4_shape_pixels(R, _AMB_SWIRL, (clip,)) - 1           # shape id, -1 for none
    core = (SP >= 0) & (SP != _AMB_SWATCH) & ~np.isin(SP, _AMB_OVERLAY) & (cv >= _AMB_CORE_CV)
    if not core.any(): return np.zeros_like(reach)
    if arrow_only:
        A = core
    else:
        sw = (SP == _AMB_SWATCH) & (cv >= _AMB_SW_CV) & (img.min(-1) >= _AMB_SW_MIN)
        A = core | sw
    L, n = cc(A, structure=np.ones((3, 3), bool))
    keep = np.unique(L[core & reach]); keep = keep[keep > 0]
    m = np.isin(L, keep) & A
    m = _close(m, _AMB_CLOSE)
    m = binary_fill_holes(m)
    m = binary_opening(m, structure=np.ones((3, 3), bool), iterations=_AMB_OPEN) if _AMB_OPEN else m
    m = m & reach
    # the sprite's translucent anti-aliased edge (coverage 16-23, no shape) at the tip: its own draws, 2 px at most
    ids_h = R['id'][::2, ::2]; own_d = np.unique(ids_h[m & core]); own_d = own_d[own_d > 0]
    for _ in range(_AMB_EDGE):
        m = m | (_dilate(m, 1) & np.isin(ids_h, own_d) & (cv >= _AMB_EDGE_CV) & reach)
    return m


def _amb_fix_6001458(body, ctx):
    # Ambusher003Projectile (skin 200, lily_003 wave): the arrow = lily_003_atk_projectile's shapes (the swatch counts
    # only where bright and connected); twinkle and feather sprites are other clips and are left out.
    if _amb_finished(ctx) or not body.any(): return np.zeros_like(body)
    return _amb_wave_body(ctx, 'lily_003_atk_projectile', arrow_only=False)


def _amb_fix_6001461(body, ctx):
    # Ambusher004Projectile (skin 206, lily_004 wave): the arrow head = lily_004_atk_projectile's own colour shapes
    # (blue, with the discs and rings); the swatch there is the beam above / behind the arrow and the brawler's bubbles
    # (not the projectile): left out.
    if _amb_finished(ctx) or not body.any(): return np.zeros_like(body)
    return _amb_wave_body(ctx, 'lily_004_atk_projectile', arrow_only=True)


def _amb_fix_6001459(body, ctx):
    # Ambusher003UltiProjectile (skin 200, the seed super): hull of its own shapes 113 and 133 at coverage CUT (no glow);
    # the sparkles on its edge (shape 108) are not body and the hull closes the bay they leave.
    if _amb_finished(ctx): return np.zeros_like(body)
    R = ctx['R']; region = ctx['region']; cv = ctx['cv']
    SP = _a4_shape_pixels(R, _AMB_SWIRL, ('lily_003_ulti_projectile',)) - 1
    core = np.isin(SP, [113, 133]) & region & (cv >= CUT)
    if not core.any(): return np.zeros_like(body)
    m = hull_mask(core) & region    # the seed is convex: its hull closes the bay a sparkle leaves on its edge
    return m


POST_BODY[6001458] = _amb_fix_6001458
POST_BODY[6001461] = _amb_fix_6001461
POST_BODY[6001459] = _amb_fix_6001459
