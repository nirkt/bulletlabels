# AI-review batch 2026-11-01 fix (Haiku fix, reviewed by the coordinator, installed 11-01 LOCAL ONLY; source out/sweep2/haikutasks/siteai/poco_dm/fix.py)
# AI-reviewer findings on Poco (DeadMariachi) skins 274, 311, 373 (task poco_dm, 2026-11-01).
# Rule (coordinator, 2026-11-01): Poco's attack = ONE solid zone over the whole cloud, stopping when the logic ends
# (same as the default skin, z_1100_s028).
# Findings fixed here:
#   274 attack 0 t273: the cloud was several separate outlines (one per star group) -> one zone over the cloud.
#   311 attack 0 t275: holes / ragged outline -> one solid zone, no holes, no ragged edge;
#   311 attack 90 t315: the label missed the right part of the cloud (too small) -> the whole cloud is labelled;
#   373 attack 0 t275 and attack 90 t309: holes / ragged outline and an inner circle line -> one solid zone, no inner line.
# Not changed here (see the report): 346 attack (sound-wave strands: a zone or a hull over the wave group runs into empty
# space and into crates), 346 super, and 390 attack (keyboard rows of one shot: rows stay split, slits remain). Their
# labels stay as they were.
#
# Types (attack of each skin; keyed to these ids only):
#   6000341 Poco004Projectile  (skin 274 DeadMariachiPunk)
#   6000345 Poco006Projectile  (skin 311 DeadMariachiOutlaw)
#   6000349 Poco008Projectile  (skin 373 DeadMariachiOnceupon)
# The super types are not touched by this file.
#
# Mechanism (the default skin's zone mechanism, z_1100_s028, applied to these types):
#  - every drawn cloud pixel belongs to its NEAREST drawn member of the group (screen Voronoi cell; member anchor projected
#    as in z_1100_s028); the label is the union of the cells of the RUNNING members (finished == 0, not ended). A finished
#    member's cloud is never labelled and the boundary to it is a straight line.
#  - cloud = the body of the per-type rules plus the clip's own shapes inside the group's region (widened by 6 px), cut
#    to the cells;
#  - closing (disc 3 px), fill holes, opening (disc 3 px): strips, spurs and slivers go; parts of at least 15 % of the largest
#    stay, so a cloud split by a block stays whole;
#  - the cut by solid map blocks drawn in front (_zs_solid, opaque draws); the envelope is 2 px around the drawn cloud.
# Other types, skins and the default skin 028 are untouched. The z_1028_poco swept-region cut is not applied to these types
# (the cells replace it, as for 028).
import projtab as _ptP
import numpy as _npP
from scipy.ndimage import (binary_closing as _bcP, binary_fill_holes as _bfhP, binary_opening as _boP,
                           binary_dilation as _bdP, label as _ccP)

PF_TYPES = (6000341, 6000345, 6000349)
PF_WIDE = 6      # px: margin around the group's region
PF_CLOSE = 3     # px: disc radius of the closing
PF_OPEN = 3      # px: disc radius of the opening
PF_MIN = 0.15    # parts of the opened cloud below this share of the largest are dropped
PF_ENV = 2       # px: the label may lie at most this far outside the drawn cloud
PF_FLASH = 200   # every channel of a flash pixel at least this bright
PF_FLASH_R = 4   # px: a flash pixel counts as cloud when it lies this close to the cloud
_PF_CTX = {'objs': None, 'cam': None}


def _pf_disc(r):
    yy, xx = _npP.mgrid[-r:r + 1, -r:r + 1]
    return (yy * yy + xx * xx) <= r * r


def _pf_cells(g, shape):
    """(H, W) bool: the pixels whose nearest DRAWN member of group g is a RUNNING member (None: no members known)"""
    objs, cam = _PF_CTX['objs'], _PF_CTX['cam']
    if objs is None or cam is None: return None
    mem = [m for m in objs if int(m['t']) in PF_TYPES and m.get('label_id', m['id']) == g and m.get('drawn', True) and 'anchor' in m]
    if not mem: return None
    H, W = shape
    yy, xx = _npP.mgrid[0:H, 0:W]
    best = _npP.full((H, W), _npP.inf); who = _npP.full((H, W), -1)
    for k, m in enumerate(mem):
        px, py = _proj(cam, float(m['anchor'][0]), -float(m['anchor'][1]), 0.0)
        d = (xx - px) ** 2 + (yy - py) ** 2
        upd = d < best; best[upd] = d[upd]; who[upd] = k
    run = _npP.array([(m.get('finished', 0) == 0 and not m.get('ended')) for m in mem], bool)
    return (who >= 0) & run[_npP.clip(who, 0, None)]


def _pf_body(t, body, ctx):
    o = ctx['o']
    if o.get('finished', 0) or o.get('ended') or not o.get('drawn', True): return body
    R, cam, region, ids_h, img = ctx['R'], ctx['cam'], ctx['region'], ctx['ids_h'], ctx.get('img')
    g = o.get('label_id', o['id'])
    cells = _pf_cells(g, ids_h.shape)
    if cells is None: return body
    wide = _bdP(region, structure=_pf_disc(PF_WIDE)) if region.any() else region
    T = _ptP.table()[t]; names = tuple(sorted({n for n in (T.get('blue'), T.get('red')) if n}))
    SP = _a4_shape_pixels(R, T['file'], names) if names else _npP.zeros(ids_h.shape, _npP.int32)
    cloud = (body | ((SP > 0) & wide)) & wide & cells
    if not cloud.any(): return _npP.zeros(cloud.shape, bool)
    flash = ((img.min(-1) >= PF_FLASH) & (ids_h > 0) & wide & cells & _bdP(cloud, structure=_pf_disc(PF_FLASH_R))) \
        if img is not None else _npP.zeros(cloud.shape, bool)
    m = cloud | flash
    m = _bcP(m, structure=_pf_disc(PF_CLOSE)) | m
    m = _bfhP(m) & wide & cells
    opaque = [d for d, v in R['draws'].items() if v.get('blend', 0) == 0]
    cut = _zs_solid(cam, R) & _npP.isin(ids_h, opaque) & (ids_h > 0)
    m = m & ~cut
    if not m.any(): return m
    op = _boP(m, structure=_pf_disc(PF_OPEN))
    if not op.any(): return op
    L, n = _ccP(op)
    if n > 1:
        sz = _npP.bincount(L.ravel()); keep_l = sz >= PF_MIN * sz[1:].max(); keep_l[0] = False
        op = keep_l[L]
    keep = m & _bdP(op, structure=_pf_disc(1))
    out = _bfhP(keep) & ~cut
    return out & _bdP(cloud | flash, structure=_pf_disc(PF_ENV))


for _tP in PF_TYPES:
    POST_BODY[_tP] = (lambda t: (lambda body, ctx: _pf_body(t, body, ctx)))(_tP)


# the group's objects and the camera for the screen cells (the labels wrapper keeps the call's own arguments)
_pf_prev_labels = labels


def labels(objs, cam, R, state=None, ended=(), img=None):
    _PF_CTX['objs'] = objs; _PF_CTX['cam'] = cam
    return _pf_prev_labels(objs, cam, R, state=state, ended=ended, img=img)


# the finished members' swept-region cut (z_1028_poco) is replaced by the cells for these types
_pf_prev_exclude = _p6_exclude


def _p6_exclude(P, cam, pairs):
    if any(int(o['t']) in PF_TYPES for _, fin, _ in pairs for o in fin):
        return P
    return _pf_prev_exclude(P, cam, pairs)
