# human review 2026-10-28 fix for skin 119 (Haiku fix, reviewed by the coordinator, installed 10-29; source out/sweep2/haikutasks/human1028/s119/fix2.py; tested via IDLABELS_OVERRIDE after all fixes incl. z_1099_cleanup, hence z_1100_)
# fix2 for skin 119 (JesterLoki, review 2026-10-28, coordinator's second round): Chester003Projectile (type 6001275).
# Loaded after fix.py (IDLABELS_OVERRIDE=fix.py:fix2.py). Each crystal hexagon is convex, so its label is taken as the convex
# hull of its own main draw (its face and rim, the pixels its draw shows at full coverage), kept to within 2 px of that hull.
# This removes what stuck out of the hexagon: the cyan zap spark at the left edge of the right hex (tick 313, a spur), and the
# small bumps (tick 315 left hex top right, tick 309 right hex top). Pixels of the hull that nothing else labels are added
# (smooth outline). Pixels of another object's label are never taken. Only type 6001275.
# Not done here: the hidden part of the back hex at tick 305 (it is under the front hex; the label format allows one owner
# per pixel, see py/bulletlabels.py), so the back hex's label keeps its visible part only (needs the user's decision).
import numpy as np

S119B_TYPES = {6001275}
_s119b_prev_labels = labels
S119B_KEEP = 2   # px: the label may reach this far outside the hull of its own draw (the rim); nothing beyond it


def labels(objs, cam, R, state=None, ended=(), img=None):
    import bulletlabels as BL
    from scipy.ndimage import label as cc, binary_dilation, binary_fill_holes
    P = _s119b_prev_labels(objs, cam, R, state=state, ended=ended, img=img)
    mine = [o for o in objs if o['t'] in S119B_TYPES]
    if not mine: return P
    ids, lids, share = P['ids'], P['lids'], P['share']
    ids_h = R['id'][::2, ::2]; cv = R['cv'][::2, ::2]
    kk = {}
    for o in mine:
        lb = (BL._lid(o), -1, 'body')
        if lb in lids: kk[o['id']] = lids.index(lb)
    for oid, k in kk.items():
        m = ids == k
        if m.sum() < 30: continue
        dv, cnt = np.unique(ids_h[m & (ids_h > 0)], return_counts=True)
        if not dv.size: continue
        main = dv[cnt.argmax()]
        vis = (ids_h == main) & (cv >= 128)
        L, n = cc(vis, structure=np.ones((3, 3)))
        if n > 1:
            sz = np.bincount(L.ravel()); sz[0] = 0; vis = L == sz.argmax()
        if vis.sum() < 30: continue
        ys, xs = np.nonzero(vis)
        from scipy.spatial import ConvexHull
        from PIL import Image, ImageDraw
        pts = np.c_[xs, ys].astype(float)
        try: h = ConvexHull(pts)
        except Exception: continue
        I = Image.new('L', (ids.shape[1], ids.shape[0]), 0)
        ImageDraw.Draw(I).polygon([tuple(pts[v]) for v in h.vertices], fill=1, outline=1)
        S = np.asarray(I, bool) | vis
        keep = binary_dilation(S, iterations=S119B_KEEP)
        # the label: its own pixels within the kept hull, plus the hull's pixels nothing else labels
        drop = (ids == k) & ~keep
        ids[drop] = -1; share[drop] = 0
        add = S & ((ids < 0) | (ids == k)) & (cv >= 128)
        add |= binary_fill_holes(((ids == k) | add)) & (ids < 0)
        ids[add] = k; share[add] = np.maximum(cv[add] / 255.0, BL.CUT)
    P['ids'] = ids; P['share'] = share
    return P
