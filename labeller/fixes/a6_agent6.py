# label-fix agent 6 overrides (JetpackGirl, RopeDude, ShieldTank, Twins, BowDude, Flea, Meeple, FireDude, CannonGirl)
# exec'd in idlabels' namespace after load

import numpy as _np


def _own_fade(cut):
    """POST_BODY factory: a 3D-mesh projectile whose mesh fades along its length (the ID pass's coverage drops below CUT
    toward its back while still clearly drawn): the connected pixels of its own body draws with coverage >= cut, in its
    region, are body too"""
    def fn(body, ctx):
        if not body.any(): return body
        from scipy.ndimage import label as cc
        ids_h, cv = ctx['ids_h'], ctx['cv']
        own = _np.unique(ids_h[body]); own = own[own > 0]
        if not own.size: return body
        cand = _np.isin(ids_h, own) & (cv >= cut) & ctx['region']
        L, n = cc(cand | body); keep = _np.unique(L[body]); keep = keep[keep > 0]
        return body | (_np.isin(L, keep) & cand)
    return fn


# RopeDude 005 (Buzz Lollipop) attack: the rainbow streak is a mesh fading toward its back; its rainbow bands stay
# clearly visible down to coverage ~64 (fixdiag 317 t269 / t309), below that only a faint glow
POST_BODY[6000990] = _own_fade(64)


# (wave-group KEEP_GROUP wrapper for Buzz SB removed 2026-10-23: idlabels.labels keeps a wave group while any member lives)


# RopeDude 008 (Buzz SB) attack: the big bubbles (trail clip buzz_008_atk_projectile, emitted along the path) are the
# attack. Its other trail clips are not: the small bubbles floating off (jessie_008_atk_bubble_*) and the water splash at
# the hand (liquid_atk_*; its sprite is the same atlas image as the bubble's inner blob, shape 63, so a part counts only
# when it contains the bubble's own rim sprites 829 / 47). A bubble is one solid convex disc (its ID-pass pixels are its
# rim and highlights). A popping bubble (the clip's pop frames: flash shapes 577 / 6671 / 92 on it) is an end
# animation: dropped whole; the white glint stars (shape 137) over a bubble are not the bubble: dropped.
TRAIL_SKIP[6000998] = ('liquid_atk', 'jessie_008_atk_bubble')
BUZZ_SB = dict(file='sc/effects_brawler.sc', drop=(137,), rim=(829, 47), pop=(577, 6671, 92))


def _shape_pixels(R, ids_h, file, sids):
    """(H, W) bool: pixels whose front-most draw shows one of the shapes (atlas polygon test on the draw's uv)"""
    import projdraw as PD
    from matplotlib.path import Path
    S = PD.sc(file); uv = R['uv']; out = _np.zeros(ids_h.shape, bool); draws = R['draws']
    for sid in sids:
        for tex, suv, xy in S.shapes[sid]:
            tw, th = tex_size(file, tex)
            sel = [d for d, r in draws.items() if (r.get('tw'), r.get('tht')) == (tw, th)]
            m = _np.isin(ids_h, sel) & (uv[..., 0] >= 0)
            if not m.any(): continue
            a = _np.asarray(suv, _np.float64) / 65535.0
            c = a.mean(0); ang = _np.arctan2(a[:, 1] - c[1], a[:, 0] - c[0]); poly = a[_np.argsort(ang)]
            ys, xs = _np.nonzero(m); inside = Path(poly).contains_points(uv[ys, xs, :2])
            out[ys[inside], xs[inside]] = True
    return out


def _fix_6000998(body, ctx):
    if not body.any(): return body
    from scipy.ndimage import label as cc
    R, ids_h = ctx['R'], ctx['ids_h']
    body = body & ~_shape_pixels(R, ids_h, BUZZ_SB['file'], BUZZ_SB['drop'])
    rim = _shape_pixels(R, ids_h, BUZZ_SB['file'], BUZZ_SB['rim'])
    pop = _dilate(_shape_pixels(R, ids_h, BUZZ_SB['file'], BUZZ_SB['pop']), 3)
    L, n = cc(_close(body, 3) | body); out = _np.zeros_like(body)
    for k in range(1, n + 1):
        m = L == k
        if (m & rim & body).sum() >= 3 and not (m & pop).any(): out |= hull_mask(m & body)
    for _ in range(2):   # parts of one bubble hulled apart (rim arc, highlight) touch: one hull
        L, n = cc(_close(out, 2) | out); out = _np.zeros_like(out)
        for k in range(1, n + 1): out |= hull_mask(L == k)
    return out & ctx['region']


POST_BODY[6000998] = _fix_6000998

# ShieldTank 002 (Buster Chainsaw) attack: the spiky ball's clip body is drawn at a constant coverage 115 for its first
# ticks (141 later): clearly drawn, under CUT (315 t265-307 unlabelled)
FADE_CUT[6001257] = 96

# ShieldTank 004 (Buster SupercellMake) attack: an atom, a core with thin orbit rings (rule 5: thin lines -> one hull of
# its own lines; its body is only the clip's rings and core, the roman-numeral trail and the halo are not in it)
ONE_LABEL[6001261] = 'hull'


def _own_part(r):
    """POST_BODY factory: only the body pixels within r px of the object's own predicted labels (its neighbours' sprites,
    side by side in one volley, are in its search region too; a hull must be of its own lines only)"""
    def fn(body, ctx):
        return body & _dilate(ctx['pred_any'], r) if body.any() else body
    return fn


POST_BODY[6001261] = _own_part(2)

# ShieldTank 005 (Buster Angel) attack: a feathered wing crescent drawn as many thin bright strokes (rule 5: one hull of
# its own strokes; its neighbours in the volley and the blue ring trail are not in it)
ONE_LABEL[6001263] = 'hull'
POST_BODY[6001263] = _own_part(2)

# Twins 003 (Angel) pet attack (Twins003ShotgunProjectile): a fan of thin translucent feather strokes, labelled as
# fragments (193 t389): rule 5, one hull per projectile of its own strokes
ONE_LABEL[6001409] = 'hull'
POST_BODY[6001409] = _own_part(2)

# BowDude 006 (Bo Wasp) attack: its clip draws a long thin motion streak behind the stinger (the same body sprite), the
# label ran back along it to Bo (298 t269 / t309): the thin streak cut off the body (THIN_OFF, as Bulletstorm's strips)
THIN_OFF[6000367] = 2
