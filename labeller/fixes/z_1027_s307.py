# sweep 10-27 fix for skin 307 (Sonnet revision, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s307/)
# sw2_307 SilencerPharaoh (sweep 2026-10-27, task s307, 2nd attempt): Otto002Projectile, type 6001199, attack 90 deg.
# Exec'd in idlabels' namespace after load (IDLABELS_OVERRIDE). Touches type 6001199 only. Works on the FINAL label map (a
# labels() wrapper): the body before the ownership step also holds neighbours' pixels.
#
# 1. Impact puddle labelled as the projectile (ticks 317, 319; pair 5741). A landed projectile leaves a dark puddle at the
#    water edge drawn with its own body sprite; the live projectile's region takes its pixels. Rule (as in the first attempt,
#    which the reviewer confirmed clean): body pixels of a projectile more than 4 px from its predicted body, in connected
#    blobs >= 100 px, are dropped (puddle 163 to 529 px; tail slivers of a live orb <= 62 px stay).
#
# 2. Two projectiles of one shot merged into one outline (tick 305; pair 5736) and touching with a ragged join (tick 309).
#    The first attempt deleted every pixel along the contact, which left holes / specks on ragged tail edges. Now the join
#    is re-cut as a clean line and smoothed, only for pairs of distinct projectiles of this type whose labels touch by >= 12 px
#    (8-neighbour; 305: 14 px, 309: 55 and 59 px; every other tick has <= 10 px and stays untouched):
#      a. in the contact zone (within 3 px of both labels) each pixel goes to the projectile whose predicted body centre is
#         nearer (perpendicular bisector of the two orbs: the tail of the one behind no longer runs along the other's side);
#      b. within 14 px of the partner each label is closed (disk r=4), kept 2 px off the partner (this is the gap between the
#         two outlines), holes filled, opened (5x5) so the join edge is one smooth line, no notch, no spur, no strip;
#      c. leftovers smaller than 25 px of the processed labels are dropped.
#    Outside that 14 px band the labels are unchanged.
#
# 3. Hairs (tick 317: spur at the upper right, stub on the left tail): every label of this type is opened with a 3x3 square;
#    what the opening removes is dropped only when it is a nib (connected piece <= 9 px) and dropping it cuts nothing off; larger removed parts
#    (prongs, tails) are restored, so flame tips stay.
import numpy as np

S307_TYPES = {6001199}
S307_FAR = 4        # px from the predicted body: beyond this a body pixel is not the projectile's own
S307_BLOB = 100     # px: a far blob at least this big is a landed projectile's puddle
S307_CONTACT = 12   # px of contact between two projectiles' labels that makes the join be re-cut
S307_ZONE = 3       # px: contact zone in which pixels are re-assigned by the bisector
S307_BAND = 14      # px around the partner in which the label is smoothed
S307_CLOSE = 4      # closing disk radius in the band
S307_OPEN = 5       # opening square in the band
S307_SPECK = 25     # px: leftovers of processed labels below this are dropped
S307_HAIR = 9      # px: a nib removed by the 3x3 opening (single-pixel spurs, stubs, hooks up to 9 px) is dropped

_s307_prev_labels = labels


def _s307_disk(r):
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    return (y * y + x * x) <= r * r + r // 2


def labels(objs, cam, R, state=None, ended=(), img=None):
    from scipy.ndimage import (binary_dilation, binary_closing, binary_opening, binary_fill_holes,
                               distance_transform_edt, label as cc)
    import bulletlabels as BL
    P = _s307_prev_labels(objs, cam, R, state=state, ended=ended, img=img)
    ids, lids, share = P['ids'], P['lids'], P['share']
    ones3 = np.ones((3, 3), bool)
    # body masks of the objects of this type, one per label id
    mine = {}
    for o in objs:
        if o['t'] not in S307_TYPES: continue
        lid = BL._lid(o)
        if lid in mine: continue
        bk = [k for k, L in enumerate(lids) if L[0] == lid and L[2] == 'body']
        if bk: mine[lid] = bk
    if not mine: return P
    masks, kmain = {}, {}
    for lid, bk in mine.items():
        m = np.isin(ids, bk)
        if not m.any(): continue
        masks[lid] = m
        kmain[lid] = max(bk, key=lambda k: int((ids == k).sum()))
    if not masks: return P
    old = {lid: m.copy() for lid, m in masks.items()}
    # predicted bodies (cut 0.01), as in the first attempt
    P0 = BL.labels(objs, cam, cut=0.01)
    pids, plids, pshare = P0['ids'], P0['lids'], P0['share']
    pb = {}
    for lid in masks:
        pk = [k for k, L in enumerate(plids) if L[0] == lid and L[2] == 'body']
        pb[lid] = (np.isin(pids, pk) & (pshare >= BL.CUT)) if pk else np.zeros(ids.shape, bool)
    # 1. puddle
    for lid in list(masks):
        if not pb[lid].any(): continue
        far = masks[lid] & (distance_transform_edt(~pb[lid]) > S307_FAR)
        if not far.any(): continue
        L, n = cc(far, structure=ones3)
        sz = np.bincount(L.ravel())
        big = np.nonzero(sz >= S307_BLOB)[0]
        big = big[big > 0]
        if big.size: masks[lid] = masks[lid] & ~np.isin(L, big)
    # 2. joins of touching projectiles
    keys = sorted(l for l in masks if masks[l].any())
    partners = {l: [] for l in keys}
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            if (masks[a] & binary_dilation(masks[b], ones3)).sum() >= S307_CONTACT:
                partners[a].append(b); partners[b].append(a)
    proc = [l for l in keys if partners[l]]
    yy, xx = np.mgrid[:ids.shape[0], :ids.shape[1]]
    def centre(l):
        m = pb[l] if pb[l].any() else masks[l]
        y, x = np.nonzero(m); return y.mean(), x.mean()
    new = {l: masks[l].copy() for l in proc}
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            if b not in partners[a]: continue
            A, B = new[a], new[b]
            Z = (binary_dilation(A, ones3, iterations=S307_ZONE) & binary_dilation(B, ones3, iterations=S307_ZONE))
            if not Z.any(): continue
            ca, cb = centre(a), centre(b)
            toA = ((yy - ca[0]) ** 2 + (xx - ca[1]) ** 2) <= ((yy - cb[0]) ** 2 + (xx - cb[1]) ** 2)
            new[a] = (A & ~Z) | (A & Z & toA)
            new[b] = (B & ~Z) | (B & Z & ~toA)
    cur = dict(new)
    pad = S307_CLOSE + 3
    for a in proc:
        others = np.zeros(ids.shape, bool)
        for b in partners[a]: others |= cur[b]
        A = cur[a]
        if not others.any() or not A.any(): continue
        N = distance_transform_edt(~others) <= S307_BAND
        C = binary_closing(np.pad(A, pad), _s307_disk(S307_CLOSE))[pad:-pad, pad:-pad]
        C &= ~binary_dilation(others, ones3)
        C = binary_fill_holes(C)
        O = binary_opening(np.pad(C, 2), np.ones((S307_OPEN, S307_OPEN), bool))[2:-2, 2:-2]
        res = np.where(N, O, A)
        L, n = cc(res, structure=ones3)
        sz = np.bincount(L.ravel()); sz[0] = 0
        keep = np.nonzero(sz >= S307_SPECK)[0]
        keep = keep[keep > 0]
        if keep.size: res = np.isin(L, keep)
        new[a] = res
    for a in proc: masks[a] = new[a]
    # 3. hairs
    for a in list(masks):
        m = masks[a]
        if not m.any(): continue
        op = binary_opening(np.pad(m, 2), ones3)[2:-2, 2:-2]
        rem = m & ~op
        if not rem.any(): continue
        L, n = cc(rem, structure=ones3)
        sz = np.bincount(L.ravel()); sz[0] = 0
        small = np.nonzero((sz > 0) & (sz <= S307_HAIR))[0]
        ncomp = cc(m, structure=ones3)[1]
        for i in small:      # a nib is dropped only when that does not cut a piece off the label (no new specks)
            m2 = m & ~(L == i)
            if m2.any() and cc(m2, structure=ones3)[1] <= ncomp: m = m2
        masks[a] = m
    # write back
    anyold = np.zeros(ids.shape, bool)
    for lid in old: anyold |= old[lid]
    ids[anyold] = -1; share[anyold] = 0
    for lid in masks:
        w = masks[lid] & (ids < 0)
        ids[w] = kmain[lid]; share[w] = 1.0
    return P
