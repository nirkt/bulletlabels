# installed 11-01 from out/sweep2/haikutasks/siteai2/gale_crow/fix.py (2nd AI sweep: Crow 216 dagger ownership + gaps, 375 spurs; Gale 283/370/442 one solid piece per snowball; all types single-skin)
# AI-review sweep 2 (2026-11-01), group gale_crow: fixes for the reviewer's findings on skins 216, 283, 370, 375, 442.
# Exec'd in idlabels' namespace (IDLABELS_OVERRIDE). Keyed to explicit type ids. Each part names the finding it fixes.
#
# A. Gale snowball / bullet columns, attack 0 (types 6000737 = 283 BlowerBusters, 6000739 = 370 BlowerDeepsea,
#    6000741 = 442 BlowerSteamPunk; stage angle 180 only). Findings: 283 t265 (t267, t269 the same column: 'split into
#    several small separate outlines'); 370 t265 ('stacked column of blobs drawn as several small separate outlines':
#    its middle snowballs had no label at all); 442 t267 and t269 ('one notched zigzag contour').
#    Checked on the captures: each column object is one snowball / bullet with its own object id and its own anchor (the
#    six anchors project onto the six blobs, 9-10 px apart). The base labels were wrong because the translucent bodies
#    were below the coverage cut (370: middle snowballs empty, 283: shreds) and because a pixel went to the first object
#    whose prediction reached it (stacked neighbours took each other's pieces). Fix, only while the column is alive (>= 3
#    of its objects labelled by the base): each object's label = convex hull of its own sprite pixels (coverage >= 32;
#    442: >= 64) inside an ellipse around its anchor (283: 14 x 7 px, 370: 11 x 7 px, 442: 13 x 5.5 px), ownership by the
#    nearest normalised anchor, holes filled. Each snowball stays its own object and label (owner rule: separate snowballs
#    keep separate outlines). The reviewer's 'one hull per shot' is NOT applied: the six are six objects. The attack-90
#    rows (stage angle 270) and the ended / fading column (279, 319) are not changed.
#
# B. Crow Harvest fire dagger (type 6000325 = 375 CrowHarvest, Crow007), attack 90 row only (angle 60..120). Finding:
#    375 t307 'spiky squiggle with interior lines' (also the spur at t309). The heads had thin spurs on the tongue tops and
#    ragged edges. Fix: after the existing rules (z_1027_s375 flash removal) an opening with a 5 x 5 square removes thin
#    spurs, the largest part is kept, holes filled, then the body pixels within 2 px of that part are restored, so the
#    pointed head tips stay (an opening alone also cut the tips). Attack 0 (pointed tips, angle ~180) is unchanged.
#
# C. Crow003 daggers (type 6000317 = 216 CrowWhite, attack 0 and attack 90). Findings: 216 t265 ('merged into one hull:
#    three daggers in one shape': the three daggers lie one above the other and the first-come ownership cut the cluster
#    into a wedge and two bands) and t305 (small red loops inside the cluster: enclosed gaps between neighbouring daggers).
#    Fix: a cluster's pixels are owned by the nearest dagger anchor (three stacked bands at 265, three columns at 305, the
#    same owners as before on separated daggers at 267, 307); enclosed gaps of <= 40 px between this type's labels, and
#    enclosed holes of one dagger, are filled to the neighbour with the most shared border (the z_1027_s310 limit). Only
#    labels of type 6000317 change.
import numpy as _gc_np
import masks as _gc_masks
from scipy.ndimage import label as _gc_cc, binary_fill_holes as _gc_fill, binary_dilation as _gc_dil
from scipy.ndimage import binary_opening as _gc_open
from scipy.spatial import ConvexHull as _gc_hull
from PIL import Image as _gc_Image, ImageDraw as _gc_Draw


def _gc_anchor(o, cam):
    a = _gc_masks.project(cam, o['anchor'][None])[0][0]
    return _gc_np.array([a[0] / 2.0, a[1] / 2.0])   # the projection is in the 2400 x 1080 frame; labels are 1200 x 540


def _gc_hullmask(pts, shape):
    I = _gc_Image.new('L', (shape[1], shape[0]), 0)
    if len(pts) < 3:
        for x, y in pts: I.putpixel((int(x), int(y)), 1)
        return _gc_np.array(I).astype(bool)
    try:
        h = _gc_hull(_gc_np.asarray(pts, float))
    except Exception:
        return None
    _gc_Draw.Draw(I).polygon([tuple(map(float, pts[k])) for k in h.vertices], fill=1, outline=1)
    return _gc_np.array(I).astype(bool)


# ---- A: Gale snowball / bullet columns ----------------------------------------------------------------------------------
GC_COL = {6000737: (14.0, 7.0, 32),    # 283 BlowerBusters attack 0: round snowballs, 20-27 px wide
          6000739: (11.0, 7.0, 32),    # 370 BlowerDeepsea attack 0: round snowballs, about 20 px wide (11: not the green dot beside the last one)
          6000741: (13.0, 5.5, 64)}    # 442 BlowerSteamPunk attack 0: capsule bullets, about 24-28 px long, 8 px tall (64: not the pale smoke under the last one)
_gc_prev_labels = labels


def labels(objs, cam, R, state=None, ended=(), img=None):
    import bulletlabels as BL
    P = _gc_prev_labels(objs, cam, R, state=state, ended=ended, img=img)
    # attack 0 only (stage angle 180; the attack-90 rows, stage angle 270, keep the base labels), and only while the
    # column is alive: at least three of its objects are labelled by the base (the fading column after its end is not)
    on = [o for o in objs if o['t'] in GC_COL and not o.get('ended') and int(o.get('stage_angle', -1)) % 360 == 180]
    if not on: return P
    ids, lids, share = P['ids'].copy(), list(P['lids']), P['share'].copy()
    base_n = 0
    for o in on:
        lid = (BL._lid(o), -1, 'body')
        if lid in lids and (ids == lids.index(lid)).sum() >= 15: base_n += 1
    if base_n < 3: return P
    cv = R['cv'][::2, ::2]
    H, W = ids.shape
    yy, xx = _gc_np.mgrid[0:H, 0:W]
    dmin = _gc_np.full((H, W), _gc_np.inf, _gc_np.float32); own = _gc_np.full((H, W), -1, _gc_np.int32)
    Ct = _gc_np.zeros((H, W), _gc_np.int8)
    ks = []
    for j, o in enumerate(on):
        a = _gc_anchor(o, cam); rx, ry, _ = GC_COL[o['t']]
        d = _gc_np.sqrt(((xx - a[0]) / rx) ** 2 + ((yy - a[1]) / ry) ** 2)
        m = d < dmin; dmin[m] = d[m]; own[m] = j
        c = sprite_pixels(o['t'], o['own'], R)
        Ct = _gc_np.where((Ct == 0) & (c > 0), c, Ct)
        lid = (BL._lid(o), -1, 'body')
        if lid not in lids: lids.append(lid)
        ks.append(lids.index(lid))
    old = _gc_np.isin(ids, ks)
    ids[old] = -1; share[old] = 0
    for j, o in enumerate(on):
        cut = GC_COL[o['t']][2]
        sel = (Ct > 0) & (cv >= cut) & (own == j) & (dmin <= 1.0)
        if sel.sum() < 3: continue
        pts = _gc_np.c_[_gc_np.nonzero(sel)[1], _gc_np.nonzero(sel)[0]]
        hm = _gc_hullmask(pts, (H, W))
        if hm is None: continue
        hm = _gc_fill(hm & (own == j) & (dmin <= 1.0)) & (own == j)
        if not hm.any(): continue
        ids[hm & (ids < 0)] = ks[j]
        share[hm] = _gc_np.maximum(cv[hm] / 255.0, BL.CUT)
    P2 = dict(P); P2['ids'] = ids; P2['share'] = share; P2['lids'] = lids
    return P2


# ---- B: Crow Harvest fire dagger (type 6000325), spur removal ----------------------------------------------------------
_gc_b_prev = POST_BODY.get(6000325)   # z_1027_s375 (flash removal) runs first


def _gc_fire_body(body, ctx):
    if _gc_b_prev is not None: body = _gc_b_prev(body, ctx)
    if not body.any(): return body
    ang = float(ctx['o'].get('angle', 0)) % 360
    if not (60 <= ang <= 120): return body      # the 90-deg row only (the 0-deg row's pointed tips are real and thin)
    m = _gc_open(body, structure=_gc_np.ones((5, 5), bool))
    if not m.any(): return body
    L, n = _gc_cc(m)
    if n > 1:
        sz = _gc_np.bincount(L.ravel())[1:]
        m = L == (int(_gc_np.argmax(sz)) + 1)
    m = _gc_fill(m)
    m = m | (body & _gc_dil(m, iterations=2))   # the pointed tips next to the body stay (2 px); long thin spurs go
    m = _gc_fill(m) & body
    return m if m.any() else body


POST_BODY[6000325] = _gc_fire_body


# ---- C: Crow003 daggers (type 6000317), ownership by anchor + enclosed gaps filled ------------------------------------
GC_DAGGER = 6000317
GC_POCKET = 40                     # px: largest enclosed gap between daggers that is filled
_gc_c_prev_labels = labels


def labels(objs, cam, R, state=None, ended=(), img=None):   # noqa: F811  (wraps the wrapper above)
    import bulletlabels as BL
    P = _gc_c_prev_labels(objs, cam, R, state=state, ended=ended, img=img)
    on = [o for o in objs if o['t'] == GC_DAGGER and not o.get('ended')]
    if not on: return P
    ids, lids, share = P['ids'].copy(), list(P['lids']), P['share'].copy()
    cv = R['cv'][::2, ::2]
    ks, ob = [], []
    for o in on:
        lid = (BL._lid(o), -1, 'body')
        if lid in lids: ks.append(lids.index(lid)); ob.append(o)
    if not ks: return P
    U = _gc_np.isin(ids, ks)
    if len(ks) > 1 and U.any():
        A = _gc_np.array([_gc_anchor(o, cam) for o in ob])
        ys, xs = _gc_np.nonzero(U)
        d = _gc_np.sqrt((xs[:, None] - A[None, :, 0]) ** 2 + (ys[:, None] - A[None, :, 1]) ** 2)
        owner = d.argmin(1)
        ids[U] = -1
        for j, k in enumerate(ks):
            s = owner == j
            ids[ys[s], xs[s]] = k
    M = _gc_np.isin(ids, ks)
    Hm = _gc_fill(M) & ~M
    L, n = _gc_cc(Hm)
    for j in range(1, n + 1):
        hj = L == j
        if hj.sum() > GC_POCKET: continue
        if (ids[hj] >= 0).any() and not _gc_np.isin(ids[hj], ks).all(): continue
        ring = _gc_dil(hj) & ~hj
        nb = ids[ring]
        if nb.size == 0 or (nb < 0).any() or not _gc_np.isin(nb, ks).all(): continue
        cnt = _gc_np.bincount(nb, minlength=int(ids.max()) + 1)
        k = int(cnt.argmax())
        ids[hj] = k; share[hj] = BL.CUT
    took = _gc_np.isin(ids, ks)
    share[took] = _gc_np.maximum(cv[took] / 255.0, BL.CUT)
    P2 = dict(P); P2['ids'] = ids; P2['share'] = share; P2['lids'] = lids
    return P2
