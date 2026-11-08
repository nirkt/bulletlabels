# AI-review batch 2026-11-01 fix (Haiku fix, reviewed by the coordinator, installed 11-01 LOCAL ONLY; source out/sweep2/haikutasks/siteai/poco/fix.py)
# Poco task (AI review 2026-11-01): skins 231 PocoValentine attack, 256 PocoStarr attack, 244 PocoPirate super (Poco zone clouds). Skin 431 not changed here (see report.md).
# Poco rule (task brief): attack and super = ONE solid zone over the whole cloud, stopping when the logic ends.
#
# Mechanism (the cell rule of z_1100_s028, copied here with per-type parameters so skin 028 is not touched):
#  - every drawn cloud pixel of the wave group belongs to its NEAREST DRAWN member (Voronoi cell on the screen);
#  - the label is the union of the cells of the RUNNING members (finished == 0 and not ended): a finished member's drawn
#    cloud (its logic has ended) is unlabelled; the boundary to it is a straight line;
#  - cut only by solid map blocks in front (walls, crates, barrels: _zs_solid), flash over the cloud kept (hidden parts);
#  - closing, hole fill, opening (spurs out), parts < 15 % of the largest dropped; bridging (closing of the solid cloud, per
#    type) fills the gaps between puffs of one cloud so it is one solid zone.
#  - z_1028_poco's exclusion of a finished member's swept region is skipped for these types: it cut straight strips through
#    the running cloud (skin 231 tick 271) and left loops (tick 275).
import numpy as _pz_np
from scipy.ndimage import (binary_closing as _pz_bc, binary_fill_holes as _pz_fill, binary_opening as _pz_bo,
                           binary_dilation as _pz_bd, label as _pz_cc)
import projtab as _pz_pt

# type -> (opening radius px, bridging closing radius px); 0 = no bridging
PZ_TYPES = {6000337: (3, 0),    # skin 231 PocoValentine attack (Poco002Projectile)
            6000343: (3, 0),    # skin 256 PocoStarr attack (Poco005Projectile)
            6000340: (3, 8)}    # skin 244 PocoPirate super (Poco003UltiProjectile): several puffs of one shot
PZ_WIDE = 6
PZ_CLOSE = 3
PZ_MIN = 0.15
PZ_ENV = 2
PZ_FLASH = 200
PZ_FLASH_R = 4

_PZ_CTX = {'objs': None, 'cam': None}


def _pz_disc(r):
    yy, xx = _pz_np.mgrid[-r:r + 1, -r:r + 1]
    return (yy * yy + xx * xx) <= r * r


def _pz_cells(g):
    """(H, W) bool: pixels whose nearest DRAWN member of wave group g is a RUNNING member (None: no members known)"""
    objs, cam = _PZ_CTX['objs'], _PZ_CTX['cam']
    if objs is None or cam is None: return None
    mem = [m for m in objs if int(m['t']) in PZ_TYPES and m.get('label_id', m['id']) == g and m.get('drawn', True)]
    if not mem: return None
    H, W = 540, 1200
    yy, xx = _pz_np.mgrid[0:H, 0:W]
    best = _pz_np.full((H, W), _pz_np.inf); who = _pz_np.full((H, W), -1)
    for k, m in enumerate(mem):
        px, py = _proj(cam, float(m['anchor'][0]), -float(m['anchor'][1]), 0.0)
        d = (xx - px) ** 2 + (yy - py) ** 2
        upd = d < best; best[upd] = d[upd]; who[upd] = k
    run = _pz_np.array([(m.get('finished', 0) == 0 and not m.get('ended')) for m in mem], bool)
    return (who >= 0) & run[_pz_np.clip(who, 0, None)]


def _pz_body(t, body, ctx):
    o = ctx['o']
    if not body.any() or o.get('finished', 0) or o.get('ended') or not o.get('drawn', True): return body
    open_r, bridge_r = PZ_TYPES[t]
    R, cam, region, ids_h, img = ctx['R'], ctx['cam'], ctx['region'], ctx['ids_h'], ctx.get('img')
    T = _pz_pt.table()[t]; names = tuple(sorted({n for n in (T.get('blue'), T.get('red')) if n}))
    if not names: return body
    g = o.get('label_id', o['id'])
    cells = _pz_cells(g)
    if cells is None: return body
    wide = _dilate(region, PZ_WIDE)
    SP = _a4_shape_pixels(R, T['file'], names)
    cloud = (body | ((SP > 0) & wide)) & wide & cells
    if not cloud.any(): return _pz_np.zeros(cloud.shape, bool)
    flash = ((img.min(-1) >= PZ_FLASH) & (ids_h > 0) & wide & cells & _pz_bd(cloud, structure=_pz_disc(PZ_FLASH_R))) if img is not None else _pz_np.zeros(cloud.shape, bool)
    m = cloud | flash
    m = _pz_bc(m, structure=_pz_disc(PZ_CLOSE)) | m
    m = _pz_fill(m) & wide & cells
    opaque = [d for d, v in R['draws'].items() if v.get('blend', 0) == 0]
    cut = _zs_solid(cam, R) & _pz_np.isin(ids_h, opaque) & (ids_h > 0)
    m = m & ~cut
    if not m.any(): return m
    op = _pz_bo(m, structure=_pz_disc(open_r))
    if not op.any(): return op
    L, n = _pz_cc(op)
    if n > 1:
        sz = _pz_np.bincount(L.ravel()); keep_l = sz >= PZ_MIN * sz[1:].max(); keep_l[0] = False
        op = keep_l[L]
    keep = m & _pz_bd(op, structure=_pz_disc(1))
    out = _pz_fill(keep) & ~cut
    if bridge_r:
        out = _pz_fill(_pz_bc(out, structure=_pz_disc(bridge_r)) | out) & wide & cells & ~cut
    return out


for _tp in PZ_TYPES:
    POST_BODY[_tp] = (lambda t: (lambda body, ctx: _pz_body(t, body, ctx)))(_tp)


_pz_prev_labels = labels


def labels(objs, cam, R, state=None, ended=(), img=None):
    _PZ_CTX['objs'] = objs; _PZ_CTX['cam'] = cam
    return _pz_prev_labels(objs, cam, R, state=state, ended=ended, img=img)


_pz_prev_exclude = _p6_exclude


def _p6_exclude(P, cam, pairs):
    keep = [p for p in pairs if not any(int(o['t']) in PZ_TYPES for o in p[1])]
    return _pz_prev_exclude(P, cam, keep) if keep else P

