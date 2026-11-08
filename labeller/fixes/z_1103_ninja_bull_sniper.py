# AI-review batch 2026-11-01 fix (Haiku fix, reviewed by the coordinator, installed 11-01 LOCAL ONLY; source out/sweep2/haikutasks/siteai/ninja_bull_sniper/fix.py)
# Fix file, AI-review findings 2026-11-01, task ninja_bull_sniper (skins 441, 472, 466, 613, 480, 595).
# Exec'd in idlabels' namespace (IDLABELS_OVERRIDE), after all installed fixes. Keyed to explicit type ids only.
#
# 441 (Leon010Projectile 6000582, Leon's green cloud puffs): the outline of each cloud is ragged, with notches and holes
#   (attack 0 t277, attack 90 t315). Each cloud keeps its own label (the four clouds of a stream are spawned on four
#   different ticks: separate projectiles, so no hull). Closed by 3 px, holes filled, inside the region; touching clouds
#   get a 1 px gap so each outline is visible; specks dropped.
# 472 (NinjaShark 6000576): the blob's lower body has a notch (attack 0 t277). Closed by 4 px, holes filled, 1 px gap to
#   touching blobs, specks dropped. A shallow notch of the crescent's mouth (t269-t273) is the object's own silhouette and
#   is kept.
# 466 (BullGuyHair 6000048, the cyan shards of Bull's spray): five shards spawned on one tick, each its own projectile
#   (Bull-like pellets: own outline, no hull over a spread). Ragged blade edges (attack 0 t269, attack 90 t307): closed by
#   2 px, holes filled, specks dropped. No gap (the shards' pixels interleave: seams would appear).
# 480 (SniperValentines 6000376, the winged heart): one object; a V-shaped slit cut from the top between the two wings,
#   and a loop at the bottom (attack 90 t313-t315). Closed by 6 px, holes filled: the slit is filled to a shallow V dip.
#   Not a hull (the wings' gap is background; a hull would take it in).
# 595 (Piper009Projectile 6000382: the attack 0 beam and the attack 90 orb / needle share this type): (a) the blue core
#   disc drawn over the projectile (sprite code 5, moves with it) is part of it (hidden parts filled in); (b) a streak
#   (bounding box at least 3x as long as wide: the attack 0 beam at t265, the needle at t305) is one hull over its body
#   instead of the torn zigzag; the round orb keeps its own outline; (c) closed by 3 px, holes filled. The installed fix
#   z_1027_s595 (hole fill) runs first and is kept. Other skins using 6000382 change too.
# 613 (BullGuyStrange 6000052, the Bull pellets): each pellet is one object and already has its own label. The outlines
#   of touching pellets were drawn as one merged wavy contour. Where two pellet labels touch, the contact pixels of the
#   higher-numbered label are left unlabelled (a 1 px gap), and detached specks under 20 % of a label's largest piece are
#   dropped (attack 90 t313 squiggle). No hull.

# ---------------------------------------------------------------------------------------------------------------------
# 441 / 472 / 466 / 480 / 595: closing by r px and hole fill inside the region (ONE_LABEL with an integer radius)
ONE_LABEL.update({6000582: 3,    # 441 Leon010Projectile: cloud puffs
                  6000576: 4,    # 472 NinjaShark: shark blob notch (attack 0 t277)
                  6000048: 2,    # 466 BullGuyHair: shard blade edges
                  6000376: 6,    # 480 SniperValentines: wings' notch (V slit)
                  6000382: 3})   # 595 Piper009Projectile: orb notch around the core disc, beam streak

# ---------------------------------------------------------------------------------------------------------------------
# 595: 6000382 post-body hook: (a) the blue core disc (sprite code 5) touching the body is part of it (hidden parts);
#      (b) a streak (bounding box >= 3 x as long as wide) becomes the convex hull of its body (one streak, one outline).
_nbs_prev_post = POST_BODY.get(6000382)


def _nbs_fix_6000382(body, ctx):
    from scipy.ndimage import label as _cc
    if _nbs_prev_post is not None:
        body = _nbs_prev_post(body, ctx)          # installed fix z_1027_s595: holes of the body filled (region only)
    if not body.any():
        return body
    C, region = ctx['C'], ctx['region']
    # (a) the core disc: sprite code 5 (a clip sprite its trail shows too) inside the region, connected to the body
    M = body | ((C == 5) & region)
    L, n = _cc(M)
    if n > 1:
        keep = np.unique(L[body]); keep = keep[keep > 0]
        body = np.isin(L, keep) & M
    # (b) a streak: the hull of the body, only when its bounding box is at least 3 x as long as wide
    ys, xs = np.nonzero(body)
    h, w = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
    if max(h, w) >= 3 * min(h, w):
        body = hull_mask(body) & (region | body)
    return body


POST_BODY[6000382] = _nbs_fix_6000382


# ---------------------------------------------------------------------------------------------------------------------
# 441 / 472 / 613: projectiles that touch are drawn as one merged outline: a 1 px gap at each contact (the contact
# pixels of the higher-numbered label are left unlabelled). 466's shards are not gapped: their claimed pixels interleave
# (a grid of seams), it keeps its closing and speck rule only. Specks (detached pieces under 20 % of the label's largest
# piece) are dropped for the same types, the largest piece always stays.
import numpy as np
_nbs_prev_labels = labels
_NBS_GAP_TYPES = {6000582, 6000576, 6000052}
_NBS_SPECK_TYPES = {6000048} | _NBS_GAP_TYPES
_NBS_SPECK_FRAC = 0.2


def labels(objs, cam, R, state=None, ended=(), img=None):
    import bulletlabels as BL
    from scipy.ndimage import label as _nbs_cc
    P = _nbs_prev_labels(objs, cam, R, state=state, ended=ended, img=img)
    if not any(o['t'] in _NBS_SPECK_TYPES for o in objs):
        return P
    ids, lids, share = P['ids'], P['lids'], P['share']
    ltype = {}
    for o in objs:
        ltype.setdefault(BL._lid(o), o['t'])
    gap_k = np.zeros(len(lids) + 1, bool)            # index k+1: label k is a body label of a gap type
    spk_k = np.zeros(len(lids) + 1, bool)
    for k, l in enumerate(lids):
        if l[2] != 'body': continue
        if ltype.get(l[0]) in _NBS_GAP_TYPES: gap_k[k + 1] = True
        if ltype.get(l[0]) in _NBS_SPECK_TYPES: spk_k[k + 1] = True
    if gap_k.sum() >= 2:
        shifted = np.where(ids >= 0, ids + 1, 0)     # 0: unlabelled
        own = gap_k[shifted]
        rm = np.zeros(ids.shape, bool)
        for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0)):
            nb = np.roll(shifted, (dy, dx), (0, 1))
            contact = own & (nb > 0) & (nb != shifted) & gap_k[nb]
            rm |= contact & (ids > nb - 1)           # the higher-numbered label gives up its contact pixels
        if rm.any():
            ids[rm] = -1; share[rm] = 0
    for k in range(len(lids)):
        if not spk_k[k + 1]: continue
        m = ids == k
        if not m.any(): continue
        L, n = _nbs_cc(m)
        if n > 1:
            sz = np.bincount(L.ravel()); sz[0] = 0
            drop = m & (sz[L] < _NBS_SPECK_FRAC * sz.max())
            if drop.any():
                ids[drop] = -1; share[drop] = 0
    return P
