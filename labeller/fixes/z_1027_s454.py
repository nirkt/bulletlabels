# sweep 10-27 fix for skin 454 (Sonnet revision, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s454/)
# Skin 454 (SoulCollectorMHA, Gus007 MHA), attack 6001252 (Gus007Projectile), sweep 2026-10-27 task s454 (pair 8511),
# second version (the first, fix.py, was rejected: after it, tick 309 had a V notch + spur at the box's right side, a new
# spur at its top-right corner and the box's upper-right corner was still cut diagonally). Exec'd in idlabels' namespace
# after the other fixes (IDLABELS_OVERRIDE).
#
# The projectile: a tan box in front of a blue cloud, with a dark-blue trail (wisps, strands, puffs) on the box's far side.
# attack 90 deg: box above the cloud (trail above the box); attack 0 deg: box left of the cloud (trail to its left).
# Defects of the type (the pair's glove / strap claim is gone in the new capture, but these remain):
#  - the trail (smoke wisps) is labelled; green halo pixels beside the box are labelled as spurs (rules 2, 3);
#  - a V notch in the cloud's translucent edge where the box meets it (bright pixels, coverage 32-127, below CUT);
#  - the box is cut diagonally / has a V dip on its top: the cloud's soft edge is drawn over the box's corner and the
#    id pass gives those pixels to another draw.
# Fix (on top of a1's POST_BODY, kept):
#  1. green halo pixels are never labelled;
#  2. the box = the tan-ish part (r - b >= -15, g >= b, not dark) of the RAW frame, 4-connected to the largest tan part
#     of the label, found within 12 px (so the covered corner is found too); accepted when rectangle-like (fill >= 0.85);
#     its convex hull is added to the label (the box is a rectangle, drawn solid);
#  3. the trail = label pixels beyond the box's far edge (axis = box-to-cloud direction snapped to y or x): cut;
#  4. the cloud's edge: bright-blue pixels (coverage >= 32, own draws, joined to the cloud, inside the prediction's
#     region) strictly on the cloud side of the box's cloud-facing edge are added inside the cloud's hull (nothing is
#     added beside the box: that was the glow strip / spurs);
#  5. 1-px stair notches of the outline closed (3x3 closing, not into halo / trail / outside the prediction's region);
#  6. 1-2 pixel specks dropped.
# Only type 6001252 is touched (Gus007 ulti 6001253 and Gus006 types 6001250 / 6001251 are other projectiles).

_S454_GUS_PREV = POST_BODY.get(6001252)


def _s454_gus007_atk(body, ctx):
    from scipy.ndimage import label as _lab
    if _S454_GUS_PREV is not None:
        body = _S454_GUS_PREV(body, ctx)
    img = ctx['img']
    if not body.any() or img is None:
        return body
    im = img.astype(np.int16)
    r, g, b = im[..., 0], im[..., 1], im[..., 2]
    mx = im.max(-1)
    tan = (r > b + 8) & (r > g - 30) & (mx >= 0x70) & (g > b)                 # the tan box
    boxish = (r - b >= -15) & (g >= b) & (mx >= 0x60)                          # the box incl. its glow-covered corner
    green = (g > r + 30) & (g > b + 30)                                         # the halo around the brawler
    bright_blue = (b > r + 25) & (mx >= 0x80)                                   # the cloud's colour, not background
    body = body & ~green                                                        # 1. halo: never labelled
    L, n = _lab(body & tan)
    if n == 0:
        return body
    sz = np.bincount(L.ravel())[1:]
    seed = L == (1 + int(np.argmax(sz)))
    # 2. the whole box from the raw frame (rectangle-like), else the label's own tan part
    Lx, nx = _lab(boxish & _dilate(seed, 12))
    ids = np.unique(Lx[seed]); ids = ids[ids > 0]
    ext = np.isin(Lx, ids)
    ye, xe = np.nonzero(ext)
    fill = ext.sum() / float((ye.max() - ye.min() + 1) * (xe.max() - xe.min() + 1)) if ext.any() else 0.0
    box = ext if (fill >= 0.85 and ext.sum() >= seed.sum()) else seed
    body = body | (hull_mask(box) & ~green)
    rim = _dilate(box, 1)
    H, W = body.shape
    YY, XX = np.mgrid[0:H, 0:W]
    yb, xb = np.nonzero(box)
    cb = np.array([yb.mean(), xb.mean()])
    rest = body & ~rim
    Lb, nb_ = _lab(rest & bright_blue)                                          # the cloud: the largest bright part
    if nb_:
        sb = np.bincount(Lb.ravel())[1:]; cl = Lb == (1 + int(np.argmax(sb)))
    else:
        cl = rest
    if not cl.any():
        return body
    yc, xc = np.nonzero(cl)
    d = cb - np.array([yc.mean(), xc.mean()])                                   # from the cloud to the box: the trail's side
    if np.hypot(*d) < 1.0:
        return body
    d = np.array([np.sign(d[0]), 0.0]) if abs(d[0]) >= abs(d[1]) else np.array([0.0, np.sign(d[1])])   # axis-aligned box
    proj = (YY - cb[0]) * d[0] + (XX - cb[1]) * d[1]
    pb = proj[box]
    far, near = float(pb.max()), float(pb.min())                                # the box's far edge / cloud-facing edge
    trail = proj > far + 0.5
    body = body & ~trail                                                        # 3. the trail on the box's far side
    # 4. the cloud's edge, where the box meets it (translucent bright pixels of its own draws, inside its convex hull)
    cloud = body & ~rim
    if cloud.any():
        ids_h, cv = ctx['ids_h'], ctx['cv']
        own_d = np.unique(ids_h[cloud]); own_d = own_d[own_d > 0]
        cand = (bright_blue & (cv >= 32) & ctx['region'] & ~rim & ~trail & (proj < near - 0.5) & np.isin(ids_h, own_d))
        Lc, nc = _lab(cand | cloud)
        touch = np.unique(Lc[cloud]); touch = touch[touch > 0]
        cand = cand & np.isin(Lc, touch)
        body = body | (hull_mask(cloud | cand) & cand)
    # 5. 1-px stair notches of the outline closed (3x3 closing, never into green halo / trail / outside the prediction)
    from scipy.ndimage import binary_closing as _bcl
    cl3 = _bcl(np.pad(body, 3), structure=np.ones((3, 3), bool))[3:-3, 3:-3]
    body = body | (cl3 & ~green & ~trail & ctx['region'])
    # 6. single pixels touching only diagonally: specks, dropped
    L8, n8 = _lab(body, structure=np.ones((3, 3), int))
    if n8 > 1:
        sz8 = np.bincount(L8.ravel()); small = np.nonzero(sz8 < 3)[0]; small = small[small > 0]
        if small.size:
            body = body & ~np.isin(L8, small)
    return body


POST_BODY[6001252] = _s454_gus007_atk
