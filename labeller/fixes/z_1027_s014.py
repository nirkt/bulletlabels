# sweep 10-27 fix for skin 014 (Sonnet revision fix2.py after a rejected Haiku fix, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s014/)
# Skin 014 RosaWindstock, Rosa008Projectile (type 6000591, only used by the RosaWindstock skin): the two or three white-tailed
# balls of the attack. Sweep 2026-10-27 findings 278 (tick 309) and 281 (tick 315), kind 'hole': small red diamond loops
# inside the label over the white tails (review: first fix left them at tick 315 and left thin diagonal spurs at tick 309).
#
# Cause: the tails are drawn as thin soft strips, so the label (the front-most draw pixels of the body) has 1-2 px wide
# slits between the tails that are open to the outside only through a 1-px channel or a diagonal touch, plus 1-2 px wide
# diagonal spur strips hanging off the lower edge. No pocket is closed by one label alone, so the shared clean-up (fills closed
# holes, drops specks) leaves them; they draw as diamond loops / spurs.
#
# Fix (this type only, each ball's own label, never touching other objects' pixels): close the label with a disc of
# S014_CLOSE_R px (slits, notches and diagonal pinches go), fill holes, open with a 2x2 square (1 px wide diagonal strips go),
# keep the biggest piece, close by 1 px; a strip dropped from one ball that lies flush against the other ball's label goes
# to that one (its tail edge); unlabelled pockets <= S014_HOLE px enclosed by the balls' labels together are filled. A ball
# never goes below half its pixels. Checked on sw2_014 attack 90 (ticks 305-315) and attack 0 (265-275): every tick has the same
# number of labels, no hole (4- or 8-connected), no diamond loop, tail tips lose 1-3 px, labels gain 100-240 px (closed notches).
S014_TYPE = 6000591
S014_CLOSE_R = 2
S014_HOLE = 40
_s014_prev_labels = labels


def _s014_disc(r):
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r + 1


def labels(objs, cam, R, state=None, ended=(), img=None):
    from scipy.ndimage import binary_fill_holes, binary_closing, binary_opening, binary_dilation, label as cc
    import bulletlabels as BL
    P = _s014_prev_labels(objs, cam, R, state=state, ended=ended, img=img)
    rosa = {BL._lid(o) for o in objs if o['t'] == S014_TYPE}
    if not rosa: return P
    ids, lids, share = P['ids'], P['lids'], P['share']
    ks = [int(k) for k in np.unique(ids[ids >= 0]) if lids[k][0] in rosa and lids[k][2] == 'body']
    H, Wd = ids.shape
    pad = S014_CLOSE_R + 3
    SQ3 = np.ones((3, 3), bool)
    D3, D1, SQ = _s014_disc(S014_CLOSE_R), _s014_disc(1), np.ones((2, 2), bool)
    dropped = np.zeros(ids.shape, bool); before = ids.copy()
    for k in ks:
        m = ids == k
        ys, xs = np.nonzero(m)
        y0, y1, x0, x1 = max(0, ys.min() - pad), min(H, ys.max() + pad + 1), max(0, xs.min() - pad), min(Wd, xs.max() + pad + 1)
        sm = m[y0:y1, x0:x1]; sid = ids[y0:y1, x0:x1]
        other = (sid >= 0) & ~sm
        # work in a zero border so closing / opening do not see the crop edge
        b = np.pad(sm, pad); ob = np.pad(other, pad)
        c = binary_closing(b, D3) | b
        c = (binary_fill_holes(c) | c) & ~ob
        c = binary_opening(c, SQ)
        L, n = cc(c)
        if n > 1:
            sz = np.bincount(L.ravel())[1:]
            c = L == (1 + int(np.argmax(sz)))
        c = (binary_closing(c, D1) | c) & ~ob
        c = (binary_fill_holes(c) | c) & ~ob
        c = c[pad:-pad, pad:-pad]
        if int(c.sum()) < 0.5 * int(sm.sum()): continue   # safety: never lose the projectile
        sh = share[y0:y1, x0:x1]
        gone = sm & ~c
        add = c & ~sm & (sid < 0)
        sid[gone] = -1; sh[gone] = 0; dropped[y0:y1, x0:x1] |= gone
        sid[add] = k; sh[add] = BL.CUT
    # a strip dropped from one ball's label that lies flush against the other ball's label is that ball's tail edge: give it over
    for k in ks:
        give = dropped & (before != k) & (ids < 0) & binary_dilation(ids == k, SQ3)
        ids[give] = k; share[give] = BL.CUT
    # unlabelled pockets enclosed by the balls' labels together (between two balls): taken into the label touching them most
    kk = set(ks)
    lab = ids >= 0
    hh = binary_fill_holes(lab) & ~lab
    if hh.any():
        L, n = cc(hh)
        for j in range(1, n + 1):
            hj = L == j
            if int(hj.sum()) > S014_HOLE: continue
            nb = ids[binary_dilation(hj) & ~hj]; nb = nb[nb >= 0]
            if nb.size == 0 or not set(np.unique(nb).tolist()) <= kk: continue
            vals, cnt = np.unique(nb, return_counts=True)
            ids[hj] = int(vals[np.argmax(cnt)]); share[hj] = BL.CUT
    return P
