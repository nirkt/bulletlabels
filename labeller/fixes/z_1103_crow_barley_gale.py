# AI-review batch 2026-11-01 fix (Haiku fix, reviewed by the coordinator, installed 11-01 LOCAL ONLY; source out/sweep2/haikutasks/siteai/crow_barley_gale/fix.py)
# Task crow_barley_gale (AI-reviewer findings 2026-11-01), Crow daggers of skins 216 (CrowWhite) and 425 (CrowBrawloween).
# Owner rule: Crow daggers are separate projectiles (no hull over the three). Each dagger keeps its own outline. Where
# daggers overlap (a cluster: the dagger's own clip body lies well outside its predicted silhouette, i.e. the neighbours'
# draws are in its region), its outline is torn into several pieces with interior lines (the reviewer's findings: 216
# attack 90 tick 305; 425 attack 90 tick 305). Such a dagger becomes ONE solid piece in its own silhouette.
# Single daggers (no neighbour in the region) keep their labels unchanged.
#
# Type 6000317 = Crow003Projectile (skin 216, attack 0 and attack 90 rows). Template: the dagger's predicted silhouette
# (ctx pred_body: the clip placed at the object's own logic transform). On separated daggers it matches the label (IoU
# 0.89-0.93 at tick 267). Cluster label = predicted silhouette, holes filled, plus the object's own drawn body within
# 2 px of it (the real edge). Nothing outside that 2 px band is added, so the neighbours' draws are not taken.
#
# Type 6000327 = Crow009Projectile (skin 425, attack 0 and attack 90 rows). Its predicted silhouette does not follow the
# dark jaw and its fangs (IoU 0.5-0.6 on separate daggers), so the template is its own drawn body: the body (no glow,
# sprite code 2 excluded) within 2 px of its predicted label, one convex hull per dagger (the splintered jaw and fangs of
# one object = one hull per object, the owner's rule for splintered attacks), holes filled. Each dagger keeps its own
# hull; overlapping pixels go to the nearest predicted label (the existing ownership).
#
# Both: an object with no usable body or prediction keeps its label unchanged, so nothing is lost.
import numpy as np
from scipy.ndimage import binary_fill_holes as _cr_fill, binary_dilation as _cr_dil
from PIL import Image as _cr_Image, ImageDraw as _cr_Draw
from scipy.spatial import ConvexHull as _cr_hull

CROW003_TYPE = 6000317     # Crow003Projectile (skin 216; clip crow_003_atk_projectile)
CROW009_TYPE = 6000327     # Crow009Projectile (skin 425; clip crow_009_atk_projectile)
CR_MIN_PRED = 50           # px: smallest predicted silhouette trusted as the template
CR_CLUSTER_PX = 100        # px: the object's own clip-body pixels outside its predicted silhouette (+3 px) from which a neighbour dagger is in its region
CR_ASPECT = 1.5            # predicted silhouette height / width from which the dagger points down the screen (the 90-deg row: 2.0-3.9; the 0-deg row 0.3-0.8)


def _cr_hull_mask(m):
    """convex hull of a mask's pixels, as a mask (the same construction as idlabels.hull_mask)"""
    ys, xs = np.nonzero(m)
    if len(ys) < 3: return m
    pts = np.c_[xs, ys].astype(float)
    try: h = _cr_hull(pts)
    except Exception: return m
    I = _cr_Image.new('L', (m.shape[1], m.shape[0]), 0)
    _cr_Draw.Draw(I).polygon([tuple(pts[k]) for k in h.vertices], fill=1, outline=1)
    return np.array(I).astype(bool)


def _cr_cluster(ctx):
    """True while a neighbour dagger's clip body is in this dagger's region (the daggers overlap), and the dagger points
    down the screen (its predicted silhouette is at least CR_ASPECT times taller than wide: the 90-deg row; the 0-deg row,
    where the daggers lie side by side, is not changed)"""
    pa, reg, C = ctx['pred_any'], ctx['region'], ctx['C']
    ys, xs = np.nonzero(pa)
    if len(ys) == 0: return False
    if (ys.max() - ys.min() + 1) < CR_ASPECT * (xs.max() - xs.min() + 1): return False
    near = _cr_dil(pa, iterations=3)
    return int(((C == 1) & reg & ~near).sum()) > CR_CLUSTER_PX


def _crow003_solid(body, ctx):
    pb, reg = ctx['pred_body'], ctx['region']
    if not body.any() or pb is None or int(pb.sum()) < CR_MIN_PRED or not _cr_cluster(ctx): return body
    near = _cr_dil(pb, iterations=2)
    out = _cr_fill(pb | (body & near)) & reg
    return out if out.any() else body


def _crow009_solid(body, ctx):
    pa, reg, C = ctx['pred_any'], ctx['region'], ctx['C']
    if not body.any() or pa is None or int(pa.sum()) < CR_MIN_PRED or not _cr_cluster(ctx): return body
    near = _cr_dil(pa, iterations=2)
    m = body & (C != 2) & near & reg
    if not m.any(): return body
    out = _cr_hull_mask(m) & reg & (C != 2)
    out = _cr_fill(out) & reg
    return out if out.any() else body


POST_BODY[CROW003_TYPE] = _crow003_solid
POST_BODY[CROW009_TYPE] = _crow009_solid


# ---------------------------------------------------------------------------------------------------------------------
# Skin 254 BlowerTrader (Gale 002), super 45 deg: type 6000734 = Gale002UltiProjectile (clip gale_002_ulti_projectile).
# Finding (AI reviewer 2026-11-01, ticks 345-353): the snow-wind wave is drawn as several separate pieces (two or three
# labels with a dent of background between them, a hole where the crate shows through). Owner rule (Gale super = one hull
# over the snow-wind wall, as skin 048 BlowerDefault, py/fixes/z_1101_s048.py): ONE convex hull over the whole wave.
# The wave's four objects of one tick (one shot) share one label: the hull of the union of their labels (the wave's
# extent as the labels already show it: its puffs, translucent edge included), cut only where a bush / crate stands in
# front (world geometry: depth-writing, not blended, a prop) and where green glow lies, largest piece kept. The label
# goes to the first wave object with a label; the others are empty. Only type 6000734 is touched. (A shape-63 band like
# z_1101_s048 was tried first: on this skin the wave's translucent puffs are below the opaque threshold, so the band was
# only the bright lower edge.)
F254_TYPE = 6000734
F254_PROP_PX = 15000       # px (frame resolution): a depth-writing draw this big or less is a prop (bush, crate), not the floor
F254_GREEN = 60            # G - B from which a pixel is green glow
_f254_prev_labels = labels


def labels(objs, cam, R, state=None, ended=(), img=None):
    import bulletlabels as BL
    from scipy.ndimage import label as _f254_cc
    from scipy.spatial import ConvexHull as _f254_CH
    P = _f254_prev_labels(objs, cam, R, state=state, ended=ended, img=img)
    if img is None: return P
    on = sorted([o for o in objs if o['t'] == F254_TYPE], key=lambda o: o['id'])
    if not on: return P
    ids = P['ids']; lids = list(P['lids'])
    wave = [(o, lids.index((BL._lid(o), -1, 'body'))) for o in on if (BL._lid(o), -1, 'body') in lids]
    wave = [(o, k) for o, k in wave if (ids == k).any()]
    if not wave: return P
    ks = [k for _, k in wave]
    U = np.isin(ids, ks)
    ys, xs = np.nonzero(U)
    H = U.copy()
    if len(ys) >= 3:
        hull = _f254_CH(np.stack([xs, ys], 1).astype(np.float64))
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        gy, gx = np.mgrid[y0:y1, x0:x1]
        inside = np.ones(gx.shape, bool)
        for a, b, c in hull.equations:
            inside &= (a * gx + b * gy + c) <= 1e-9
        H = np.zeros(U.shape, bool); H[y0:y1, x0:x1] = inside
    # cut only where a prop stands in front (depth-writing, not blended, small) and where green glow lies
    cv = R['cv'][::2, ::2]; idh = R['id'][::2, ::2]; cntd = np.bincount(idh.ravel())
    occ = [d for d, dr in R['draws'].items() if dr.get('dmask') and not dr.get('blend') and d < len(cntd) and 0 < cntd[d] <= F254_PROP_PX]
    if occ: H &= ~np.isin(idh, occ)
    im = img.astype(np.int16)
    H &= ~((im[..., 1] - im[..., 2] >= F254_GREEN) & (im[..., 1] - im[..., 0] >= 10))
    from scipy.ndimage import binary_fill_holes as _f254_fill
    L, n = _f254_cc(H, structure=np.ones((3, 3)))
    if n:
        cnt = np.bincount(L.ravel()); cnt[0] = 0
        H = L == int(np.argmax(cnt))
    H = _f254_fill(H)              # the leaves cut out at the wave's edge leave no holes in the label
    if not H.any(): return P
    ids = ids.copy(); share = P['share'].copy()
    ids[np.isin(ids, ks)] = -1
    owner_k = wave[0][1]
    take = H & (ids < 0)
    ids[take] = owner_k; share[take] = np.maximum(cv[take] / 255.0, BL.CUT)
    P2 = dict(P); P2['ids'] = ids; P2['share'] = share
    return P2
