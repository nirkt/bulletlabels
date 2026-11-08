# human review 2026-10-28 fix for skin 028 (Haiku fix, reviewed by the coordinator, installed 10-29; source out/sweep2/haikutasks/human1028/s028/fix3.py; runs after z_1099_cleanup like the tests)
# Fix task s028 v3 (human review 2026-10-28; coordinator 2026-10-29): skin 028 DeadMariachiDefault (Poco).
# Type 6000335 (DeadMariachiProjectile, the Poco attack) ONLY; the super 6000336 is not touched.
# Point of this version: tick 275 must be ONE clean outline of the RUNNING member's drawn cloud (its own blobs, closed,
# holes filled, smoothed), cut only by the barrel / crates drawn in front; no inner seam, no second loop. The drawn cloud of
# the finished members (their logic ended, still drawn) stays unlabelled, the boundary to them is one straight line.
#
# Mechanism:
#  - each member's screen position: _proj(cam, anchor_x, -anchor_y) (the y sign of the anchor is flipped for the screen:
#    checked on capture sw2_028 tick 275, the four members land on their blobs). Members drawn False are ignored.
#  - every drawn cloud pixel (the body of the ID pass or the cloud's own clip shapes, in the logic region widened a little)
#    belongs to its NEAREST drawn member (Voronoi cell on the screen). The label of the group is the union of the cells of
#    the RUNNING members (finished == 0 and not ended). The cell borders are straight lines: one clean boundary between
#    a running and a finished cloud, no loop.
#  - the same clean-up as fix2: a small closing (gaps up to about 6 px), hole fill, opening radius 3 (strips and spurs
#    go; parts of at least 15% of the largest stay, so a cloud split by a block stays whole), reconstruction 1 px, then the cut
#    by solid map blocks drawn in front (_zs_solid and opaque draws; holes made by the cut stay cut).
#  - the flash rule of fix2 (a white sparkle next to the cloud is cloud) and the envelope (2 px around the drawn cloud).
#  - z_1028_poco _p6_exclude is not applied for these types: the cells replace it (its swept-region cut made the seams).
import projtab as _pt3
import numpy as _np3
from scipy.ndimage import (binary_closing as _bc3, binary_fill_holes as _bfh3, binary_opening as _bo3,
                           binary_dilation as _bd3, label as _cc3)

SZ3_TYPES = (6000335,)
SZ3_WIDE = 6        # px: margin around the logic region
SZ3_CLOSE = 3       # px: disc radius of the closing
SZ3_OPEN = 3        # px: disc radius of the opening
SZ3_MIN = 0.15      # parts of the opened cloud smaller than this share of the largest are dropped
SZ3_ENV = 2         # px: the label may lie at most this far outside the drawn cloud
SZ3_FLASH = 200     # every channel of a flash pixel at least this bright
SZ3_FLASH_R = 4     # px: a flash pixel counts as cloud when it lies this close to the cloud

_SZ3_CTX = {'objs': None, 'cam': None}


def _sz3_disc(r):
    yy, xx = _np3.mgrid[-r:r + 1, -r:r + 1]
    return (yy * yy + xx * xx) <= r * r


def _sz3_cells(ctx, g):
    """(H, W) bool: the pixels whose nearest DRAWN member of group g is a RUNNING member (None: no members known)"""
    objs, cam = _SZ3_CTX['objs'], _SZ3_CTX['cam']
    if objs is None or cam is None: return None
    mem = [m for m in objs if int(m['t']) in SZ3_TYPES and m.get('label_id', m['id']) == g and m.get('drawn', True)]
    if not mem: return None
    H, W = ctx['ids_h'].shape
    yy, xx = _np3.mgrid[0:H, 0:W]
    best = _np3.full((H, W), _np3.inf); who = _np3.full((H, W), -1)
    for k, m in enumerate(mem):
        px, py = _proj(cam, float(m['anchor'][0]), -float(m['anchor'][1]), 0.0)
        d = (xx - px) ** 2 + (yy - py) ** 2
        upd = d < best; best[upd] = d[upd]; who[upd] = k
    run = _np3.array([(m.get('finished', 0) == 0 and not m.get('ended')) for m in mem], bool)
    return (who >= 0) & run[_np3.clip(who, 0, None)]


def _sz3_body(t, body, ctx):
    if not body.any(): return body
    R, cam, region, ids_h, img = ctx['R'], ctx['cam'], ctx['region'], ctx['ids_h'], ctx.get('img')
    o = ctx['o']
    if o.get('finished', 0) or o.get('ended') or not o.get('drawn', True): return body
    T = _pt3.table()[t]; names = tuple(sorted({n for n in (T.get('blue'), T.get('red')) if n}))
    if not names: return body
    g = o.get('label_id', o['id'])
    cells = _sz3_cells(ctx, g)
    if cells is None: return body
    wide = _dilate(region, SZ3_WIDE)
    SP = _a4_shape_pixels(R, T['file'], names)
    cloud = (body | ((SP > 0) & wide)) & wide & cells
    if not cloud.any(): return _np3.zeros(cloud.shape, bool)
    flash = ((img.min(-1) >= SZ3_FLASH) & (ids_h > 0) & wide & cells & _bd3(cloud, structure=_sz3_disc(SZ3_FLASH_R))) if img is not None else _np3.zeros(cloud.shape, bool)
    m = cloud | flash
    m = _bc3(m, structure=_sz3_disc(SZ3_CLOSE)) | m
    m = _bfh3(m) & wide & cells
    opaque = [d for d, v in R['draws'].items() if v.get('blend', 0) == 0]
    cut = _zs_solid(cam, R) & _np3.isin(ids_h, opaque) & (ids_h > 0)
    m = m & ~cut
    if not m.any(): return m
    op = _bo3(m, structure=_sz3_disc(SZ3_OPEN))
    if not op.any(): return op
    L, n = _cc3(op)
    if n > 1:
        sz = _np3.bincount(L.ravel()); keep_l = sz >= SZ3_MIN * sz[1:].max(); keep_l[0] = False
        op = keep_l[L]
    keep = m & _bd3(op, structure=_sz3_disc(1))
    out = _bfh3(keep) & ~cut
    out = out & _bd3(cloud | flash, structure=_sz3_disc(SZ3_ENV))
    return out


for _t3 in SZ3_TYPES:
    POST_BODY[_t3] = (lambda t: (lambda body, ctx: _sz3_body(t, body, ctx)))(_t3)


# the group's objects and the camera, for the screen cells (the labels wrapper keeps the call's own arguments)
_z3_prev_labels = labels


def labels(objs, cam, R, state=None, ended=(), img=None):
    _SZ3_CTX['objs'] = objs; _SZ3_CTX['cam'] = cam
    return _z3_prev_labels(objs, cam, R, state=state, ended=ended, img=img)


# the finished members' swept-region exclusion (z_1028_poco) is replaced by the cells for these types
_z3_prev_exclude = _p6_exclude


def _p6_exclude(P, cam, pairs):
    if any(int(o['t']) in SZ3_TYPES for _, fin, _ in pairs for o in fin):
        return P
    return _z3_prev_exclude(P, cam, pairs)
