# installed 11-01 from out/sweep2/haikutasks/siteai2/misc_c/fix.py (2nd AI sweep: 267 card stack solid, 361 card hulls, 452 beam one hull per shot; single-skin types)
# misc_c fix, AI-review sweep 2026-11-01 (skins 267, 361, 452). Exec'd in idlabels' namespace (IDLABELS_OVERRIDE).
# Every change is keyed to an explicit type id. Each wrapper calls the installed hook first (z_1027_s267 for 6000415,
# a2_agent2 for 6001758) and only adds to it.
#
# 267 BlackHoleMisfortune, Tara003Projectile 6000415 (attack 90, tick 307; the flagged frame is the card fan under the
#   white flash of the impact). The label had a thin spur (flash ray, no card shape: 'none' pixels) trailing down-left
#   from the bottom card, a detached lobe (the card under the flash) and an interior notch between the lobe and the
#   stack. Hidden-part rule: the card under the flash keeps its own silhouette. Fix: 'none' pixels further than 2 px
#   from the card's own shapes are dropped (the spur); the object's pixels are opened by 1 px (the spur's root) and
#   closed by 2 px (the notch), holes filled, pieces under 40 px dropped (specks). Card pixels are never removed.
#
# 361 BlackHoleStreetNinja, Tara006Projectile 6000421 (attack 90, ticks 305 and 307). Holes where the card's emblem is
#   drawn, a cyan trail bar touching a card's top edge (trail pixels taken in), and a card whose left part is covered
#   by the cyan streak (hidden part). Each card is one convex hull over its own card pixels (a card is convex; the
#   hull fills the emblem holes and the covered edge, and never leaves the card). Trail pixels are not card pixels.
#
# 452 Mender199Gem, Mender002Projectile 6001758 (attack 0 deg, ticks 265 and 269). The water beam's left end (about
#   475 px of its glowbert dust sprite, x 590-606 at tick 265) was outside the label (too small), and the shards and
#   spray of the same shot were separate outlines. Fix: the dust sprite pixels (shape 2 of glowbert_def_atk_dust) at
#   cv >= CUT within 30 px of the object's label join the body, then one convex hull of the whole shot (ONE_LABEL
#   'hull', owner rule: one outline around the whole shining area). Only this type; no other skin uses 6001758.

import numpy as np

# 267 -----------------------------------------------------------------------------------------------------------------
_F267 = 'sc/effects_brawler.sc'
_C267 = ('tara_003_atk_projectile', 'tara_003_atk_projectile_red')
_T267 = 6000415
_prev_267 = POST_BODY.get(_T267)
_MC267_SPUR = 2     # px: 'none' pixels within this of the card's own shapes are kept (sprite edges)
_MC267_MIN = 40     # px: pieces of the object's label smaller than this (specks) are dropped


def _mc267(body, ctx):
    from scipy.ndimage import binary_fill_holes, binary_opening, binary_closing, label as cc
    b = _prev_267(body, ctx) if _prev_267 else body
    if not b.any(): return b
    SP = _a4_shape_pixels(ctx['R'], _F267, _C267)
    card = SP > 0
    b = b & (card | _dilate(card, _MC267_SPUR))          # the flash ray (no card shape) far from the cards goes
    b = binary_opening(b, structure=np.ones((3, 3)), iterations=1) | (b & card)   # root of the spur, 1 px
    b = _close(b, 2) | b                                  # notch between the lobe and the stack
    b = binary_fill_holes(b)
    L, n = cc(b, structure=np.ones((3, 3)))
    if n > 1:
        sz = np.bincount(L.ravel())[1:]
        b = np.isin(L, [k + 1 for k in range(n) if sz[k] >= _MC267_MIN])
    return b


POST_BODY[_T267] = _mc267


# 361 -----------------------------------------------------------------------------------------------------------------
_F361 = 'sc/effects_brawler.sc'
_C361 = ('tara_006_atk_projectile', 'tara_006_atk_projectile_red')
_T361 = 6000421
_MC361_NEAR = 2     # px: 'none' pixels within this of the card's own shapes (sprite edges, emblem edge) are kept


def _mc361(body, ctx):
    from scipy.ndimage import label as cc
    if not body.any(): return body
    SP = _a4_shape_pixels(ctx['R'], _F361, _C361)
    card = SP > 0
    keep = body & (card | _dilate(card, _MC361_NEAR))     # trail pixels (the bar, the streak) are not card pixels
    if not keep.any(): return body
    L, n = cc(keep, structure=np.ones((3, 3)))
    if n > 1:
        sz = np.bincount(L.ravel())[1:]
        keep = np.isin(L, [k + 1 for k in range(n) if sz[k] >= 0.5 * sz.max()])   # the card, not a stray piece
    return keep


POST_BODY[_T361] = _mc361
# the hull is taken AFTER the pixel ownership (labels() wrapper): each card's hull covers only its own claimed pixels and
# never takes a pixel another card owns; a hull is convex, so it stays inside its own card
_prev_labels_361 = labels


def labels(objs, cam, R, state=None, ended=(), img=None):
    from scipy.ndimage import label as cc
    import bulletlabels as BL
    P = _prev_labels_361(objs, cam, R, state=state, ended=ended, img=img)
    tl = {BL._lid(o) for o in objs if o['t'] == _T361}
    ids, lids = P['ids'], P['lids']
    ks = [k for k, l in enumerate(lids) if l[0] in tl and l[2] == 'body']
    if not ks: return P
    cv = R['cv'][::2, ::2]
    for k in ks:
        m = ids == k
        if not m.any(): continue
        L, n = cc(m, structure=np.ones((3, 3)))
        if n > 1:
            sz = np.bincount(L.ravel())[1:]; m = L == (int(np.argmax(sz)) + 1)
        add = hull_mask(m) & (ids < 0)          # the hull grows only into unclaimed pixels
        if add.any():
            ids[add] = k
            P['share'][add] = np.maximum(cv[add] / 255.0, BL.CUT)
    return P


# 452 -----------------------------------------------------------------------------------------------------------------
_F452 = 'sc/effects_brawler_glowbert.sc'
_C452 = ('gen_particle_empty', 'glowbert_def_atk_dust')
_T452 = 6001758
_prev_452 = POST_BODY.get(_T452)
_MC452_NEAR = 30    # px: dust-sprite pixels within this of the object's label join the shot
_MC452_BEAM = 400   # dust-sprite pixels an object needs to count as the water beam (attack 0: about 1200-2700 px)


def _horizontal(m):
    """a flat, horizontal shot: bounding box at least 1.5 x as wide as tall (attack 0 beam; attack 90 is a vertical wave)"""
    ys, xs = np.nonzero(m)
    if not len(xs): return False
    w = xs.max() - xs.min() + 1; h = ys.max() - ys.min() + 1
    return w >= 1.5 * h


def _mc452(body, ctx):
    b = _prev_452(body, ctx) if _prev_452 else body
    if not b.any() or not _horizontal(b): return b
    SP = _a4_shape_pixels(ctx['R'], _F452, _C452)
    dust = (SP > 0) & (ctx['cv'] >= CUT)
    if (dust & b).sum() < _MC452_BEAM: return b          # not the beam (its dust sprites are the beam's own pixels)
    return b | (dust & _dilate(b, _MC452_NEAR))


POST_BODY[_T452] = _mc452
# the one hull of the shot is taken after the ownership (labels() wrapper, as for 361), and it never takes world
# geometry (the brawler, crates, walls: opaque depth-writing draws) nor a character's band (z_1026's character band)
_prev_labels_452 = labels


def labels(objs, cam, R, state=None, ended=(), img=None):
    from scipy.ndimage import label as cc
    from scipy.ndimage import binary_fill_holes
    import bulletlabels as BL
    P = _prev_labels_452(objs, cam, R, state=state, ended=ended, img=img)
    tl = {BL._lid(o) for o in objs if o['t'] == _T452}
    ids, lids = P['ids'], P['lids']
    ks = [k for k, l in enumerate(lids) if l[0] in tl and l[2] == 'body']
    if not ks: return P
    cv = R['cv'][::2, ::2]
    idh = R['id'][::2, ::2]
    H, W = ids.shape
    # the hull must not reach a character: each character's band (its feet to its head, plus a margin for the
    # turret / gun) is kept out of the hull's added pixels (the floor is opaque too, so a depth mask is not usable)
    world = np.zeros((H, W), bool)
    if state is not None and cam is not None:
        yy, xx = np.mgrid[0:H, 0:W]
        for c in state['chars']:
            cx, cy = _proj(cam, int(c[3]), int(c[4])); hx, hy = _proj(cam, int(c[3]), int(c[4]), 450)
            mx, my = (cx + hx) / 2.0, (cy + hy) / 2.0
            world |= ((((xx - mx) / 40.0) ** 2 + ((yy - my) / (abs(cy - hy) / 2.0 + 22.0)) ** 2) <= 1)
    for k in ks:
        m = ids == k            # all pieces of the shot (the shards too): one hull per shot
        if not m.any() or not _horizontal(m): continue
        dust = (_a4_shape_pixels(R, _F452, _C452) > 0) & (cv >= CUT)
        if (dust & m).sum() < _MC452_BEAM: continue      # not the beam: no hull
        add = hull_mask(m) & (ids < 0) & ~world
        if add.any():
            ids[add] = k
            P['share'][add] = np.maximum(cv[add] / 255.0, BL.CUT)
    return P

