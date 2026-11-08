# installed 11-01 from out/sweep2/haikutasks/siteai2/barrel_luchador/fix.py (2nd AI sweep: Darryl 493/529/600 hole fill per projectile, Primo 609 fist under spark bar, 625 brawler arm/torso out; single-skin types)
# AI-review sweep 2 (2026-11-01), task barrel_luchador (skins 493, 529, 600, 609, 625; 511 checked, no fix).
# Exec'd in idlabels' namespace (IDLABELS_OVERRIDE). Every change is keyed to explicit type ids below.
#
# 493 BarrelBotLaser (a90 t309/t311, a0 t267-t271) and 600 BarrelBotGreekGod (a90 t307-t313, a0 t265-t273):
#   the Darryl hull per projectile (z_1103_barrelbot.py) leaves tiny enclosed holes and 1-2 px seams where neighbouring
#   projectile hulls of one volley meet (493 t311: four holes of 2-5 px in the cluster, the flagged "small red interior
#   loop"). Rule: holes / ragged, fill the union solid. Fix: the labelled pixels of each group are filled where enclosed
#   by labels (holes up to HOLE_MAX px), and each filled pixel takes the label of its nearest labelled neighbour, so the
#   projectile partition is unchanged. The per-projectile hull stays as accepted (separate projectiles keep separate
#   labels). Type ids: 6000430 (493), 6000435 (600). 6000433 (529 pearl) is in the same group: the pearl ring hole of t275
#   (529) is filled the same way.
#
# 529 BarrelBotCupcake (a0 t275): the lower pearl's label is a ring with a hole (the pale-cyan centre of the pearl is
#   not labelled, the flagged "reaches into the flash" part is this ring); the hole is hidden-part fill (rule: a flash
#   covering part of a projectile keeps the full shape). Covered by the group fill above (6000433).
#
# 625 LuchadorMHA (a90 t305-t315): the label of Primo014Projectile (6000410) runs up over the brawler's striped torso
#   above the blue flames (the user: "it should cover the flames only"). Those torso pixels come from blended sprites of
#   the attack's striped arm (not from the opaque fist mesh), so they are cut by colour: a pixel is kept when it is
#   cyan-blue flame (G >= 140, B >= 180, R <= 170) or belongs to the opaque fist mesh (texture hash 20e53de0... blend 0)
#   and is not yellow / pink (R <= 150). Opened (spurs and streaks removed), largest part kept, holes filled. The
#   z_1027_s625 body fix runs first and stays in place. Type 6000410 (skin 625 only).
#
# 609 LuchadorCarreta (a90 t311-t313): the fist's label has interior lines where the yellow spark bar lies over the fist
#   (t313) and a notch between the top cap and the fist (t311); the hidden part under the bar is filled, as the
#   hidden-parts rule says (the fist's own silhouette, closed within its convex hull, nothing outside the hull). The
#   bar's tip that sticks out of the silhouette stays out. A 5 px cyan spark object (t309, a separate tiny label) is
#   dropped (sparks are never labelled). OPEN nuance (owner question still open): the bar is filled as hidden part, not
#   excluded. Type 6000409 (skin 609).
#
# 511 LuchadorCorazon (t269 a0, t309 a90): NOT changed. The lobe over the glove at t269 (object 2000006) is drawn with
#   the blast sprite (same texture as the wave, its own draw) at the fist on its first tick: this is the open Q1
#   (El Primo first-tick fist sprite), so it is left as it is and reported as needs_user.

import numpy as np

# ---------------------------------------------------------------------------------------------------------------
# 1. labelled pixels of the Darryl volley types: enclosed holes filled (nearest label takes them)
# ---------------------------------------------------------------------------------------------------------------
HOLE_GROUP = {6000430,   # Darryl004Projectile (skin 493, BarrelBotLaser)
              6000435,   # Darryl009Projectile (skin 600, BarrelBotGreekGod)
              6000433}   # Darryl007Projectile (skin 529, BarrelBotCupcake: pearls)
HOLE_MAX = 300           # px (full resolution): enclosed holes this small or smaller are filled; bigger holes are not
_bb_labels_prev = labels


def labels(objs, cam, R, state=None, ended=(), img=None):
    from scipy.ndimage import binary_fill_holes, label as _cc, distance_transform_edt
    P = _bb_labels_prev(objs, cam, R, state=state, ended=ended, img=img)
    gids = {BL._lid(o) for o in objs if o['t'] in HOLE_GROUP}
    if not gids: return P
    ids = P['ids']; lids = P['lids']
    ks = [k for k, l in enumerate(lids) if l[2] == 'body' and l[0] in gids]
    if not ks: return P
    U = np.isin(ids, ks)
    H = binary_fill_holes(U) & ~U
    if not H.any(): return P
    L, n = _cc(H)
    sz = np.bincount(L.ravel())
    small = np.isin(L, 1 + np.nonzero(sz[1:] <= HOLE_MAX)[0]) & H
    if not small.any(): return P
    _, (iy, ix) = distance_transform_edt(~U, return_indices=True)
    ids = ids.copy(); share = P['share'].copy()
    sy, sx = iy[small], ix[small]
    ids[small] = ids[sy, sx]; share[small] = share[sy, sx]
    P2 = dict(P); P2.update(ids=ids, share=share)
    return P2


# ---------------------------------------------------------------------------------------------------------------
# 2. 625 LuchadorMHA: Primo014 (6000410) label = the flames only (no torso above them)
# ---------------------------------------------------------------------------------------------------------------
_LU625_MESH_TH = '20e53de0a7693f5d'     # texture hash of the opaque fist mesh (blend 0), stable across captures
_lu625_prev = POST_BODY.get(6000410)


def _lu625_disk(k):
    yy, xx = np.mgrid[-k:k + 1, -k:k + 1]
    return (xx ** 2 + yy ** 2) <= k * k


def _lu625_flames(body, ctx):
    from scipy.ndimage import binary_fill_holes, binary_closing, binary_opening, label as _cc
    if _lu625_prev is not None: body = _lu625_prev(body, ctx)
    if not body.any(): return body
    img = ctx['img']; ids_h = ctx['ids_h']; R = ctx['R']
    if img is None: return body
    mesh_ids = [d for d, v in R['draws'].items() if v.get('th') == _LU625_MESH_TH and v.get('blend') == 0]
    mesh = np.isin(ids_h, mesh_ids) if mesh_ids else np.zeros(body.shape, bool)
    r, g, b = img[..., 0].astype(int), img[..., 1].astype(int), img[..., 2].astype(int)
    flame = (g >= 140) & (b >= 180) & (r <= 170)              # cyan-blue flame
    keep = body & (flame | (mesh & (r <= 150)))              # fist mesh, not yellow / pink
    if keep.sum() < 0.3 * body.sum(): return body
    p = 8
    K = binary_opening(np.pad(keep, p), structure=_lu625_disk(2))
    L, n = _cc(K)
    if n > 1:
        sz = np.bincount(L.ravel())[1:]
        K = np.isin(L, 1 + np.nonzero(sz >= 0.25 * sz.max())[0])
    K = binary_fill_holes(binary_closing(K, structure=_lu625_disk(2)))[p:-p, p:-p]
    return K & body


POST_BODY[6000410] = _lu625_flames


# ---------------------------------------------------------------------------------------------------------------
# 3. 609 LuchadorCarreta: fist (6000409) one solid silhouette (spark bar hidden part filled), no specks
# ---------------------------------------------------------------------------------------------------------------
_lf609_prev = POST_BODY.get(6000409)


def _lf609_fist(body, ctx):
    from scipy.ndimage import binary_fill_holes, binary_closing
    if _lf609_prev is not None: body = _lf609_prev(body, ctx)
    if not body.any(): return body
    if body.sum() < 12: return body & False           # a 5 px spark label beside the fist (t309): not the fist
    p = 6
    cl = binary_fill_holes(binary_closing(np.pad(body, p), structure=_lu625_disk(3)))[p:-p, p:-p]
    add = cl & ~body & hull_mask(body) & ctx['region']
    return body | add


POST_BODY[6000409] = _lf609_fist
