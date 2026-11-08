# 2026-11-08 enemy-label fix v2, group v1 (Belle skins 059 300 308 417). Prefix _f1_. Installed as py/fixes/z_1118_enemy_v1.py.
# exec'd in idlabels' namespace (after every installed fix). Own (blue) labels are never touched: every hook returns the installed
# result for ctx['own'] True, and the clip-layer patch only renames layers of a RED clip (never drawn in an own shot).
#
# 059 ElectroSniperDefault   6000913 attack / 6000915 super: FALSE POSITIVE (no change)
# 308 ElectroSniperCoral     6000925 super:                  FALSE POSITIVE (no change)
# 300 ElectroSniperPlanthunter 6000920 attack: REAL, fixed   (section 1)
# 417 ElectroSniperFantasy   6000929 attack, 6000931 super: REAL, fixed (sections 2, 3)
import numpy as _f1_np


# ---- 1. 300 ElectroSniperPlanthunter attack, Belle003Projectile 6000920 -----------------------------------------------------
# CAUSE: the red clip 'belle_003_atk_bullet_red' draws shape 65 (the soft flame tongue above the bullet) once more as a normal-blend,
# high-alpha layer (clip layer (0, 32)); the blue clip draws shape 65 only as additive glow layers. glow.clip_layers therefore classes
# shape 65 of the red clip 'body', so the PREDICTED body (bulletlabels, via glow.layer_class) and the drawn tongue pixels (idlabels: glow
# code pixels inside the aligned prediction) made a tall chimney above the egg in the enemy label; the blue label is the egg only.
# FIX (red clip only): every layer of shape 65 of that red clip is 'glow', as in the blue clip.
_F1_RED_GLOW = {('sc/effects_brawler.sc', 'belle_003_atk_bullet_red'): {65}}
if not getattr(glow.clip_layers, '_f1', False):
    _f1_prev_cl = glow.clip_layers
    _f1_cl_cache = {}

    def _f1_clip_layers(file, export):
        sh = _F1_RED_GLOW.get((file, export))
        if not sh: return _f1_prev_cl(file, export)
        if (file, export) not in _f1_cl_cache:
            kind, cl = _f1_prev_cl(file, export)
            _f1_cl_cache[(file, export)] = (kind, {k: ('glow' if k[1] in sh else v) for k, v in cl.items()})
        return _f1_cl_cache[(file, export)]
    _f1_clip_layers._f1 = True
    glow.clip_layers = _f1_clip_layers


# ---- 2. 417 ElectroSniperFantasy attack, Belle006Projectile 6000929 ---------------------------------------------------------
# CAUSE: a2_agent2's crystal rule takes the pixels with min(R, G, B) >= 100 near the prediction. For the blue crystal that is the
# crystal itself (the cyan glow has R < 100); for the red crystal min(R, G, B) is the green channel, which stays >= 100 well into the
# pink halo, so the enemy label was a hull of crystal + halo (+ the white muzzle burst of the first tick merged into it).
# FIX (enemy only): the whiteness threshold is 170 (crystal rim ~ 180+, halo edge ~ 140); the cluster is cut to within 4 px of the
# predicted core (20 px at the first tick of a shot, where the core is only the leading half of the crystal) so that a white muzzle
# burst merged into it is cut away; the core itself is always part of the label (hidden under the burst); convex hull as before.
_F1_P929 = 6000929
_F1_MINV_RED = 170
_f1_prev929 = POST_BODY.get(_F1_P929)


def _f1_b929(body, ctx):
    prev = _f1_prev929(body, ctx) if _f1_prev929 else body
    if ctx['own'] or ctx['img'] is None: return prev
    from scipy.ndimage import label as cc
    pb = ctx['pred_body']
    bb = ctx['region'] & (ctx['img'].min(-1) >= _F1_MINV_RED) & _dilate(ctx['pred_any'], 4)
    if not bb.any() or not pb.any(): return prev
    bb = _close(bb, 1); L, n = cc(bb)
    keep = _f1_np.unique(L[bb & _dilate(pb, 2)]); keep = keep[keep > 0]
    if not keep.size: return prev
    # first tick of a shot: the predicted core is only the leading half of the crystal (< 50 px), the crystal reaches ~17 px further
    m = _f1_np.isin(L, keep) & _dilate(pb, 4 if pb.sum() >= 50 else 20)
    if not m.any(): return prev
    return hull_mask(m | pb) & ctx['region']


POST_BODY[_F1_P929] = _f1_b929


# ---- 3. 417 ElectroSniperFantasy super, Belle006UltiProjectile 6000931 ------------------------------------------------------
# CAUSE: at the first drawn tick the red chevron is hidden under a big white additive muzzle burst (the own shot has none); the
# green-channel test of a2_agent2 then keeps only a fragment of the chevron (e.g. 5 of 170 predicted-body px).
# FIX (enemy only): when less than a quarter of the predicted body is in the label (hidden), the label is the hull of the exposed
# fragment near the prediction plus the predicted body (the hidden-part rule of z_1109_enemy_pl), cut to the region, plus the
# fragment. On visible ticks the strokes are kept within 6 px of the prediction, bright specks (< 12 px) dropped, so a burst
# shard beside the chevron cannot stretch the hull (ticks 2-3 of the 2nd capture: a triangle towards the burst).
_F1_P931 = 6000931
_f1_prev931 = POST_BODY.get(_F1_P931)
_F1_NEAR931 = 6


def _f1_big(m, frac=0.02, minpx=12):
    """the connected parts of m of at least frac of its largest part and minpx px (specks dropped)"""
    from scipy.ndimage import label as cc
    L, n = cc(_close(m, 1))
    if n <= 1: return m
    sz = _f1_np.bincount(L.ravel())[1:]
    return m & _f1_np.isin(L, 1 + _f1_np.nonzero(sz >= max(minpx, frac * sz.max()))[0])


def _f1_b931(body, ctx):
    r = _f1_prev931(body, ctx) if _f1_prev931 else body
    if ctx['own'] or not r.any() or not ctx['pred_body'].any(): return r
    pb = _f1_big(ctx['pred_body']); r = _f1_big(r)
    if (r & pb).sum() < 0.25 * pb.sum():      # hidden under the burst: fragment near the prediction + predicted body, hull
        return hull_mask((r & _dilate(pb, 12)) | pb) & ctx['region'] | r
    return r & _dilate(pb, _F1_NEAR931)       # the strokes near the prediction (burst shards further away stay out); hulled by ONE_LABEL


POST_BODY[_F1_P931] = _f1_b931
