# sweep 10-27 fix for skin 075 (Sonnet revision, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s075/)
# fix2 (after the reviewer rejected fix.py): fix.py left the bar's two edge rows (full coverage, but sprite code 0 / glow 2) out, so the
# hull of what was left was chamfered at the right end; now every fully covered pixel connected to the bar is taken in as well.
# Fix for skin 075 DoorManDefault (sweep 2026-10-27, finding pair 1341: attack 0 deg, tick 269, 'merge_split';
# the same row's tick 267 shows the same left shift of the label).
# DoorManProjectile (type 6001295, gray default attack): the attack is one bright bar. Its label took in the translucent glow
# beside the bar (coverage 106-109, glow code 2, taken in through the aligned prediction) and missed the bar's bright tip,
# which the trail sprite draws over the bar (coverage 113-255, trail-over code 4). The trail bars behind the bar stay
# unlabelled (trail, rule 3).
# Fix: the glow (code 2, coverage < 128) is dropped from the label; the bar's tip and body (codes 1 and 4, coverage >= 100)
# are then added where they connect to what is left of the label, and never to the left of its left edge (so the trail bars
# behind the bar stay out); the bar is then taken as its convex hull (a solid rectangle: the one-pixel steps and the tip's
# missing corners are closed; hull_mask inside the hook, so the other shot is not touched). Keyed to this type only; if nothing is left of the label, the label is left as it is.

S075_BAR = 6001295


def _s075_bar(body, ctx):
    from scipy.ndimage import label as cc, binary_fill_holes
    C, cv, region = ctx['C'], ctx['cv'], ctx['region']
    ys, xs = np.nonzero(body)
    if not ys.size or xs.var() <= 2 * ys.var():     # only the horizontal bar (0 deg, spread along x); the vertical one
        return body                                 # (90 deg, spread along y) is left as it is
    base = body & ~((C == 2) & (cv < 128)) & (cv >= 100)   # the glow (and the bar's faint outer fringe row, coverage < 100) the prediction took in is not the bar
    good = np.nonzero(base.sum(0) >= 4)[0]          # the bar's left edge: a column with at least 4 of its pixels
    if not good.size:
        return body
    x_lo = int(good.min())
    base[:, :x_lo] = False                          # stray pixels left of the edge (a row's corner) are not the bar
    grow = (((C == 1) | (C == 4)) & (cv >= 100) | (cv >= 200)) & region   # cv >= 200 of any code: the bar's solid edge rows (code 0 / 2)
    grow[:, :x_lo] = False
    cand = grow | base
    L, n = cc(cand)                                  # 4-connected parts
    keep = np.unique(L[base]); keep = keep[keep > 0]
    out = binary_fill_holes(np.isin(L, keep) & cand)
    if not out.any():
        return body
    h = hull_mask(out)                               # the bar is a solid rectangle: the hull closes its steps and corners
    yy, xx = np.nonzero(h)
    if h.sum() >= 0.93 * (np.ptp(yy) + 1) * (np.ptp(xx) + 1):    # an axis-aligned bar: its bounding rectangle (the corners of the
        h[yy.min():yy.max() + 1, xx.min():xx.max() + 1] = True    # tip, faint at the sprite's rounded end, are part of the bar)
    return h


POST_BODY[S075_BAR] = _s075_bar
