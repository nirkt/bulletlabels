# Label-fix agent 0 overrides (2026-10-20). Exec'd in idlabels' namespace (IDLABELS_OVERRIDE). Keyed to explicit type ids.
import numpy as np


def _cc_keep(m, min_px=0, frac=0.0):
    """connected parts of m with at least min_px pixels and at least frac of the largest part"""
    from scipy.ndimage import label as cc
    L, n = cc(m)
    if not n: return m
    sz = np.bincount(L.ravel())[1:]
    keep = 1 + np.nonzero((sz >= min_px) & (sz >= frac * sz.max()))[0]
    return np.isin(L, keep)


def _cc_touch(m, t):
    """connected parts of m that touch t (the projectile's own predicted body: one instance per projectile)"""
    from scipy.ndimage import label as cc
    L, n = cc(m)
    if not n: return m
    k = np.unique(L[t & m]); return np.isin(L, k[k > 0])


# ---- Poco009 (DeadMariachiGoodRandoms, piano skin) -------------------------------------------------------------------
# Attack 6000351: each projectile is a row of piano-key squares (clip body, translucent: coverage 25-130), with a soft
# trail glow ellipse and blue lightning spikes behind it. The generic rules labelled scraps of it (keys under CUT) and
# glow. Body = its clip-body key squares at any coverage >= 20, gaps between the keys closed, holes filled: one solid
# keyboard per projectile (ZONE, as Poco's attack on every skin); small parts (single flying keys of the break-up
# animation) dropped.
POCO009_SUPER_B = 245   # blue channel of the keyboards' bright keys and band (water and bushes below it)
def _poco009_atk(body, ctx):
    C, cv, region = ctx['C'], ctx['cv'], ctx['region']
    from scipy.ndimage import binary_fill_holes
    k = (C == 1) & (cv >= 20)
    if ctx['img'] is not None:   # keys of keyboards overlapping each other at launch (under the front one's glow): bright
        k = k | ((ctx['img'][..., 2] >= POCO009_SUPER_B) & _dilate(C == 1, 4))   # saturated-blue pixels beside its key sprites
    k = k & region & _dilate(ctx['pred_any'], 6)
    if not k.any(): return k
    f = binary_fill_holes(_close(k, 3)) & region
    # the break-up animation's flying hollow squares (outlines filled by the hole fill): parts whose drawn key pixels
    # cover under half of the filled part are dropped
    from scipy.ndimage import label as cc
    L, n = cc(f)
    if n:
        a = np.bincount(L.ravel(), minlength=n + 1); b = np.bincount(L[k], minlength=n + 1)
        f = np.isin(L, 1 + np.nonzero(b[1:] >= 0.5 * a[1:])[0])
    return _cc_keep(f, min_px=400)   # (its predicted body is offset at launch: no _cc_touch)
POST_BODY[6000351] = _poco009_atk


# Super 6000352: the curved keyboards. Their bright keys and blue band are drawn additively in most frames (the ID pass
# sees only the soft trail discs and the dome glow around them; the generic rule labelled the dome glow beside the
# keyboard). Body = the frame's saturated-blue pixels (blue >= 245: the keyboard band and keys; water and bushes stay
# below) near its own sprites and its predicted body, gaps closed, holes filled, small parts (notes) dropped.
def _poco009_super(body, ctx):
    img, C, region = ctx['img'], ctx['C'], ctx['region']
    if img is None: return body
    from scipy.ndimage import binary_fill_holes
    k = region & (img[..., 2] >= POCO009_SUPER_B) & _dilate(C > 0, 3) & _dilate(ctx['pred_body'], 8)
    if not k.any(): return k
    k = binary_fill_holes(_close(k, 3)) & region
    return _cc_touch(_cc_keep(k, min_px=150, frac=0.15), _dilate(ctx['pred_body'], 6))
POST_BODY[6000352] = _poco009_super



# ---- Beamer003 (Beamerharvest) ------------------------------------------------------------------------------------------
# Super 6001303: a crystal cube flying with a trail of small cubes that use the cube's own clip sprite (clip body, so the
# generic rule labelled each of them as dots along the path, Haiku's 'dotted outline'). Only the cube itself: parts
# smaller than BODY_ONLY_FRAC of the largest part are dropped (the BODY_ONLY mechanism, as for Gale's snowballs)
BODY_ONLY.add(6001303)


# ---- Beamer006 (BeamerSCStore) ------------------------------------------------------------------------------------------
# Attack 6001308: the green ball; its clip also draws the translucent flame/smoke tail above it, whose few opaque texels
# became specks of label up the smoke (Haiku's 'outlines on smoke trail fragments'). Only the ball: BODY_ONLY
BODY_ONLY.add(6001308)


# ---- Mike004 (TntDudeMecha) ---------------------------------------------------------------------------------------------
# Attack 6000273: two little robot dynamites drawn as ONE instanced 3D mesh draw. The draw can have one owner only
# (labels(): one draw per projectile), so the second robot got no label (attack 90: 313-321). Body |= the parts of the
# mesh draws under its own predicted body (opaque, not sprites) that touch it and lie mostly inside it (never the thrower)
def _mike004_atk(body, ctx):
    from scipy.ndimage import label as cc
    R, ids_h, pb, region = ctx['R'], ctx['ids_h'], ctx['pred_body'], ctx['region']
    if not pb.any(): return body
    cvh = ctx['cv']; uvh = R['uv'][..., 0]; spr = np.unique(ids_h[ctx['C'] > 0])
    pbd = _dilate(pb, 4); add = np.zeros_like(body)
    for d, n in zip(*np.unique(ids_h[pb & (ids_h > 0)], return_counts=True)):
        if n < 4 or d in spr: continue
        md = ids_h == d
        if (uvh[md] >= 0).mean() >= 0.5: continue   # an atlas sprite draw (ground, UI), not a mesh
        m = md & (cvh >= CUT) & region; L, k = cc(m)
        for p in np.unique(L[pb & m]):
            if p == 0: continue
            part = L == p
            if (part & pbd).sum() >= 0.6 * part.sum(): add |= part
    return body | add
POST_BODY[6000273] = _mike004_atk


# ---- Tick006 (ClusterBombDudeGodzilla) ----------------------------------------------------------------------------------
# Attack 6000536: three glowing orbs; the clip also draws a thin white speed streak behind them (clip body, 2-3 px wide),
# labelled as red slivers (267-279). Thin parts opened away (THIN_OFF, as Bulletstorm's motion strips); the orbs stay
THIN_OFF[6000536] = 2
