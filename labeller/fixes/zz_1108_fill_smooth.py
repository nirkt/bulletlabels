# fill_smooth v2 (smooth group, 2026-11-05): GLOBAL last step, meant to REPLACE py/fixes/zz_1108_fill_smooth.py (same role: runs last).
# Owner, many pages: "too ragged / a little too ragged / has holes / make the label smoother" (175, 216, 219, 222, 237, 240, 241, 249,
# 263, 273, 290, 310, 341, 349, 355, 364) and 2026-11-03: "holes in the middle ... only an outer outline", "labels connected even if a few
# pixels between", "insanely ragged attacks: smoothen them out". v1 only smoothed labels above a raggedness threshold (0.12) with one fixed
# blur and left stray pixels, shallow notches and frayed fringes; v2 treats EVERY plain body / spawn label the same way, scaled to its size:
#  1. HOLES: filled (outer outline only). Pixels of another label inside a hole stay that label's.
#  2. PIECES: pieces of one label <= SV2_GAP px apart are joined; remaining detached specks (<= 6 px) and detached pieces smaller than SV2_REL of the main
#     piece that lie on ANOTHER label (a label's strays on a neighbouring bullet) are dropped; other detached parts (a thin arrow head, a tip) stay.
#  3. NOTCHES: shallow concavities of the outline (convex-hull pockets of depth <= D px and area <= SV2_POCKET of the label, D = 0.04 sqrt(area)
#     clipped 1.5..2.5 px) are filled, then a disc closing of radius SV2_RC closes the remaining slits.
#  4. SPURS / FRAYS: opening with a disc of radius SV2_RO removes thin short protrusions; thin parts longer than SV2_SPUR px are kept.
#  5. A Gaussian blur thresholded at 0.5 (sigma = 0.045 sqrt(area) clipped 0.8..2 px) rounds the outline; growth is limited to 2 px beyond
#     the closed label; a result that loses more than 30 % of the area is rejected (the closed label is used).
#  6. Pixels that lost their label but are enclosed by labelled pixels are given to the majority neighbour (no 1-px holes between labels).
#  Labels that are already convex (hulls: Nita / Sandy / Janet / ShieldTank / Frank attacks, Mortis / Frank / Mender supers, ONE_LABEL hull
#  types) are only hole-filled. Zones ('area'), hits and 'ground' labels are not touched. Pixels are only added where no other label is.
import numpy as _sv2_np
from scipy.ndimage import (binary_fill_holes as _sv2_fill, gaussian_filter as _sv2_gauss, label as _sv2_cc, binary_dilation as _sv2_dil,
                           binary_closing as _sv2_bclose, binary_opening as _sv2_bopen, distance_transform_edt as _sv2_edt)
from scipy.spatial import ConvexHull as _sv2_CH
from PIL import Image as _sv2_I, ImageDraw as _sv2_D
import bulletlabels as _SV2BL

SV2_KINDS = ('body', 'spawn')
SV2_GAP = 3          # px: pieces of one label up to this far apart are joined
SV2_REL = 0.25       # detached pieces smaller than this fraction of the label's largest piece are dropped (min SV2_SPECK px)
SV2_SPECK = 6
SV2_ON_OTHER = 0.35  # ... only when they lie on other labels (>= this fraction of their 2-px ring); smaller specks (<= SV2_SPECK px) always
SV2_MIN = 20         # labels smaller than this (px) are only hole-filled
SV2_POCKET = 0.2     # a notch is filled when its area is at most this fraction of the label ...
SV2_KD, SV2_KS = 0.04, 0.045      # ... and its depth <= clip(SV2_KD sqrt(area), 1.5, 2.5); blur sigma = clip(SV2_KS sqrt(area), 0.8, 2)
SV2_RC = 2.0         # disc radius of the closing (px)
SV2_RO = 1.5         # disc radius of the spur-removing opening
SV2_SPUR = 10        # thin protrusions with more pixels than this are kept (real thin parts)
SV2_GROW = 4
SV2_CONVEX = 0.01    # labels whose hull deficiency is below this fraction are treated as convex (hulls): only filled
_sv2_prev_labels = labels


def _sv2_dilate(m, r): return _sv2_dil(m, iterations=int(r)) if m.any() else m


def _sv2_disc(r):
    k = int(_sv2_np.ceil(r)); y, x = _sv2_np.ogrid[-k:k + 1, -k:k + 1]
    return (x * x + y * y) <= r * r + 1e-6


def _sv2_dclose(m, r):
    if not m.any(): return m
    p = int(_sv2_np.ceil(r)) + 2
    return _sv2_bclose(_sv2_np.pad(m, p), structure=_sv2_disc(r))[p:-p, p:-p]


def _sv2_dopen(m, r):
    if not m.any(): return m
    p = int(_sv2_np.ceil(r)) + 2
    return _sv2_bopen(_sv2_np.pad(m, p), structure=_sv2_disc(r))[p:-p, p:-p]


def _sv2_hull(m):
    ys, xs = _sv2_np.nonzero(m)
    if len(ys) < 3: return m
    pts = _sv2_np.c_[xs, ys].astype(float)
    try: h = _sv2_CH(pts)
    except Exception: return m
    I = _sv2_I.new('L', (m.shape[1], m.shape[0]), 0); _sv2_D.Draw(I).polygon([tuple(pts[k]) for k in h.vertices], fill=1, outline=1)
    return _sv2_np.asarray(I, bool) | m


def _sv2_pockets(f, H, D, frac):
    P = H & ~f
    if not P.any(): return f
    de = _sv2_edt(_sv2_np.pad(H, 1))[1:-1, 1:-1]
    lab, k = _sv2_cc(P, structure=_sv2_np.ones((3, 3))); out = f.copy(); n = f.sum()
    for i in range(1, k + 1):
        c = lab == i
        if c.sum() <= frac * n and de[c].max() <= D: out |= c
    return out


def _sv2_one(m, other):
    f = _sv2_fill(m)
    L, k = _sv2_cc(f)
    if k > 1:
        f = _sv2_fill(f | (_sv2_dclose(f, SV2_GAP) & _sv2_dilate(f, SV2_GAP)))
        L, k = _sv2_cc(f)
        if k > 1:
            sz = _sv2_np.bincount(L.ravel())[1:]; big = sz.argmax(); keep = _sv2_np.zeros_like(f)
            for i in range(k):
                comp = L == i + 1
                if i == big or (sz[i] > SV2_SPECK and not (sz[i] < SV2_REL * sz[big] and other[_sv2_dilate(comp, 2) & ~comp].mean() >= SV2_ON_OTHER)): keep |= comp
                elif sz.max() < 3 * SV2_SPECK: keep |= comp          # tiny labels keep all their pieces
            f = keep
    n = int(f.sum())
    if n < SV2_MIN: return f
    H = _sv2_hull(f)
    if (H & ~f).sum() < SV2_CONVEX * n: return f
    r = _sv2_np.sqrt(n); sigma = float(_sv2_np.clip(SV2_KS * r, 0.8, 2.0)); D = float(_sv2_np.clip(SV2_KD * r, 1.5, 2.5))
    c = _sv2_fill(_sv2_pockets(f, H, D, SV2_POCKET))
    c = _sv2_fill((_sv2_dclose(c, SV2_RC) & _sv2_dilate(f, SV2_GROW)) | c)
    o = _sv2_dopen(c, SV2_RO)
    if o.any():
        rem = c & ~_sv2_dilate(o, 1)
        lab, k = _sv2_cc(rem, structure=_sv2_np.ones((3, 3)))
        keep = _sv2_np.zeros_like(c)
        for i in range(1, k + 1):
            comp = lab == i
            if comp.sum() > SV2_SPUR: keep |= comp
        c2 = o | (_sv2_dilate(o, 1) & c) | keep
        if c2.sum() >= 0.8 * c.sum(): c = c2
    g = _sv2_gauss(_sv2_np.pad(c, 6).astype(_sv2_np.float32), sigma)[6:-6, 6:-6] >= 0.5
    s = _sv2_fill(g) & _sv2_dilate(c, 2)
    if s.sum() < 0.7 * n: return c
    # thin bridges (an arrow's shaft) and thin separate parts must survive the blur: pieces of c the blur removed that
    # join two or more pieces of s, or touch none, are put back (a spur touches exactly one piece and stays removed)
    Ls, ns_ = _sv2_cc(s, structure=_sv2_np.ones((3, 3)))
    gone = c & ~_sv2_dilate(s, 1)
    if gone.any():
        lab, k = _sv2_cc(gone, structure=_sv2_np.ones((3, 3)))
        for i in range(1, k + 1):
            comp = lab == i
            t = _sv2_np.unique(Ls[_sv2_dilate(comp, 2) & (Ls > 0)])
            if len(t) >= 2 or (len(t) == 0 and comp.sum() > SV2_SPUR // 2): s = s | comp | (_sv2_dilate(comp, 1) & c)
        s = _sv2_fill(s)
    return s


def _sv2_orphans(pre, out, need=5, rounds=2):
    out = out.copy()
    for _ in range(rounds):
        o = (out < 0) & (pre >= 0)
        if not o.any(): break
        ys, xs = _sv2_np.nonzero(o)
        for y, x in zip(ys, xs):
            if y < 1 or x < 1 or y >= out.shape[0] - 1 or x >= out.shape[1] - 1: continue
            nb = out[y - 1:y + 2, x - 1:x + 2].ravel(); nb = nb[nb >= 0]
            if len(nb) >= need: out[y, x] = _sv2_np.bincount(nb).argmax()
    return out


def _sv2_apply(ids, lids, share, cv):
    """ids / share / cv at label resolution -> (ids, share)"""
    ks = [k for k in _sv2_np.unique(ids) if k >= 0 and lids[k][2] in SV2_KINDS]
    if not ks: return ids, share
    out = ids.copy(); sh = share.copy(); H, W = ids.shape
    for k in ks:
        mf = ids == k
        if not mf.any(): continue
        ys, xs = _sv2_np.nonzero(mf)
        y0, y1, x0, x1 = max(0, ys.min() - 12), min(H, ys.max() + 13), max(0, xs.min() - 12), min(W, xs.max() + 13)   # work on the label's box only
        sl = (slice(y0, y1), slice(x0, x1))
        m = mf[sl]
        f = _sv2_one(m, ((ids >= 0) & (ids != k))[sl])
        free = (out[sl] < 0) | (out[sl] == k)
        add = f & free & ~m
        o_ = out[sl]; s_ = sh[sl]; c_ = cv[sl]
        o_[add] = k; s_[add] = _sv2_np.maximum(c_[add] / 255.0, _SV2BL.CUT)
        drop = m & ~f
        o_[drop] = -1; s_[drop] = 0
    fin = _sv2_orphans(ids, out)
    sh[(fin >= 0) & (out < 0)] = _SV2BL.CUT
    return fin, sh


def labels(objs, cam, R, state=None, ended=(), img=None):
    P = _sv2_prev_labels(objs, cam, R, state=state, ended=ended, img=img)
    ids, sh = _sv2_apply(P['ids'], P['lids'], P['share'], R['cv'][::2, ::2])
    P2 = dict(P); P2.update(ids=ids, share=sh)
    return P2
