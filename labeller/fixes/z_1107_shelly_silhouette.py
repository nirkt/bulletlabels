# installed 11-02 from out/sweep2/haikutasks/siteai5/shelly_silhouette/fix.py (owner ruling shelly_seams: touching pellets = one solid silhouette; seam steps of z_1104_shelly / z_1105_shelly_poco_gale switched off)
# Owner ruling shelly_seams (2026-11-02, RULINGS_1102.md second batch; task shelly_silhouette), binding:
#   "for Shelly pellets that TOUCH / overlap each other, draw ONE SILHOUETTE over the touching cluster (no seam lines between
#    pellets). Pellets that do not touch keep their own outline. No convex hull ... the silhouette is the union of the pellets'
#    own shapes, filled solid." It reverts the seam lines of z_1104_shelly.py (364 attack 6000004, 240 super 6000007,
#    341 super 6000013) and of z_1105_shelly_poco_gale.py (240 attack 6000006: seams after the per-pellet split; 280 attack
#    6000010 and 280 super 6000011: seams).
#
# Why the outline showed seams: the review page outlines the union of an attack's labels, so touching pellets read as one
# outline unless a pixel is removed along their shared edge. Those removals (the seam steps) are the only cause, so they are
# switched off here. Each seam step is named where it is switched off:
#   1. z_1104 labels wrapper: it removes the farther pellet's pixel along each shared edge of the SHF_TYPES (364 attack,
#      240 super, 341 super). SHF_TYPES is emptied, so that wrapper returns the labels unchanged.
#   2. z_1105 labels wrapper, step A (240 attack 6000006): the seam function _spg_seam is replaced by one with no seam.
#      Step A still splits the cluster's pixels by nearest pellet anchor (ownership only): no pixel of the union is dropped.
#   3. z_1105 labels wrapper, step B (280 attack 6000010 and super 6000011): its seam step runs only if R has a depth
#      buffer 'wd'. The wrapper is called with R without 'wd' (so no seam), while the chain below it still gets the full R
#      (the depth test of idlabels and of the fix2 discs needs it). Step C (Gale) and D (Poco) are unchanged.
# Kept from those files: the per-draw split of each pellet's body (z_1104 _shf_per_draw, POST_BODY: solid pellets, no holes),
# the per-pellet closing / fill / specks (z_1103 _sh_pellet), the 280 attack per-pellet disc (fix2, FX2_TYPES incl. 6000010).
# Added (4, end of file): one solid silhouette per touching cluster (1-px channels between touching pellets bridged, enclosed
# holes of the union filled). No hull is used. Type ids touched: 6000004, 6000006, 6000007, 6000010, 6000011, 6000013 (each
# is one Shelly skin, see owners.projectile). Not touched: 6000001 (015 BanditGirlDefault super), 6000002 (223), 6000012 and
# 6000014 (fix2 discs), 6000016 / 6000017 (362 ShotgunGirlSquad): their label code has none of the seam steps.

import numpy as _shs_np

# 1. z_1104: seam step off (its POST_BODY per-draw split of the pellets stays: it was registered at import time)
SHF_TYPES.clear()


# 2. z_1105 step A: no seam between the pellets of the 240 attack (the function is looked up at call time)
def _spg_seam(owner, dist, H, W):
    return _shs_np.zeros((H, W), bool)


# 3. z_1105 step B: the wrapper gets R without its depth buffer (no seam for 6000010 / 6000011); the chain it calls gets the full R
_shs_chain = labels                  # the installed top labels (z_1105_shelly_poco_gale: steps A, B, C, D)
_shs_inner = _spg_prev_labels        # the chain under z_1105 (z_1104, z_1103, fix2, the base labels)
_shs_full = {}


def _spg_prev_labels(objs, cam, R, state=None, ended=(), img=None):
    return _shs_inner(objs, cam, _shs_full.get('R', R), state=state, ended=ended, img=img)


# 4. one silhouette: the union of the Shelly pellets' labels in a frame is made solid. A 1-px channel of unlabelled pixels
#    between two labelled pixels of the cluster (where two touching pellets meet) is bridged, and an enclosed hole of the
#    union is filled when it is no bigger than SHS_HOLE_SHARE pellet shares (one pellet's share = union px / number of
#    pellets labelled in the frame). Label maps are at frame resolution (540 x 1200). Only pixels that no label holds are
#    added, each taking the label of its nearest cluster pixel. The steps repeat until stable (a bridge can close a notch
#    into a new hole). No hull: the union keeps the pellets' own edges.
_SILH_TYPES = {6000004, 6000006, 6000007, 6000010, 6000011, 6000013}
SHS_HOLE_SHARE = 1.0


def _shs_silhouette(P, objs, R):
    import bulletlabels as BL
    from scipy.ndimage import binary_fill_holes as _shs_fill, label as _shs_cc, distance_transform_edt as _shs_edt
    cl = {BL._lid(o) for o in objs if o['t'] in _SILH_TYPES}
    ids, lids = P['ids'], P['lids']
    ks = [k for k, l in enumerate(lids) if l[0] in cl and l[2] == 'body']
    if not ks: return P
    cv = R['cv'][::2, ::2]
    ids2 = ids.copy(); share2 = P['share'].copy(); changed = False
    for _ in range(4):                                  # a bridge can close a notch into a new hole: repeat until stable
        U = _shs_np.isin(ids2, ks)
        if not U.any(): break
        hor = _shs_np.zeros(U.shape, bool); hor[:, 1:-1] = (~U[:, 1:-1]) & U[:, :-2] & U[:, 2:]
        ver = _shs_np.zeros(U.shape, bool); ver[1:-1, :] = (~U[1:-1, :]) & U[:-2, :] & U[2:, :]
        holes = _shs_fill(U) & ~U
        if holes.any():
            L, n = _shs_cc(holes)
            sz = _shs_np.bincount(L.ravel())[1:]
            share = U.sum() / max(1, len(set(ids2[U].tolist())))   # one pellet's share of the union (px)
            holes = _shs_np.isin(L, 1 + _shs_np.nonzero(sz <= SHS_HOLE_SHARE * share)[0])
        add = (hor | ver | holes) & (ids2 < 0)
        if not add.any(): break
        _, (iy, ix) = _shs_edt(~U, return_indices=True)     # nearest cluster pixel of every pixel
        ids2[add] = ids2[iy[add], ix[add]]
        share2[add] = _shs_np.maximum(cv[add] / 255.0, BL.CUT)
        changed = True
    if not changed: return P
    P2 = dict(P); P2['ids'] = ids2; P2['share'] = share2
    return P2


def labels(objs, cam, R, state=None, ended=(), img=None):
    _shs_full['R'] = R
    Rn = dict(R)
    Rn.pop('wd', None)               # z_1105 step B reads only this key: off
    P = _shs_chain(objs, cam, Rn, state=state, ended=ended, img=img)
    return _shs_silhouette(P, objs, R)
