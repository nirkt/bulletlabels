# sweep 10-27 fix for skin 009 (Sonnet revision fix2.py after a rejected Haiku fix, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s009/)
# Fix sw2_009 (RosaCoco, sweep 2026-10-27), 2nd attempt: the ball cluster of Rosa 003's attack (type 6000586) has
# saw-tooth spikes along its left (rear) edge that follow the blue trail streaks, not the solid balls.
# Findings: pair 190 (attack 0 deg tick 269) and pair 193 (tick 275), spur_or_speck. Review of the 1st fix asked for:
# no notch at tick 273, no peaks at tick 309, no waviness at 269 / bulge at 275, no hull bulge, nothing lost.
#
# Cause: Rosa003Projectile is a 3D mesh (three balls + a lime-green bar across their rear), one draw per ball, no atlas
# shape. The mesh draws also include the soft blue streak fins behind the bar (fully opaque next to the bar, so no
# coverage threshold separates them), and the draws are what the label is made of (cv >= CUT).
# Separation by colour: the bar is the only lime-green part (G - B > 20 in the frame); the streak fins are blue and touch
# the balls only through the bar. So per projectile body (POST_BODY, this type only, no hull):
#   bar  = body pixels with G - B > ROSA_GB;
#   rest = body without the bar, holes (eyes) filled; balls = opening of rest with a disk of radius 4 (the fins, at most
#          ~9 px wide and attached to the bar, are gone; the balls are far bigger, their rim stays);
#   label = bar + balls, then body pixels within 1 px of it (seams, highlight wedges at the ball / bar junction), a
#          closing (r=3) restricted to body pixels, once more the 1 px growth, holes filled, 1-2 px spurs opened away
#          (restored 1 px so the edge stays put). Nothing is added that is not the mesh's own pixel (no hull).
# If nothing of the balls survives the opening, the body is left as it was (a projectile never loses its label).
import os as _os
ROSA_OPEN = 4      # px: opening radius that removes the trail fins from the balls
ROSA_GROW = 1      # px: seam pixels next to the result that are body again
ROSA_GB = 20       # G - B above which a pixel is the lime-green bar
ROSA_CLOSE = 3     # px: closing radius (body pixels only)
ROSA_SPUR = 3      # px: opening radius that removes spurs at the end


def _rosa_disk(k):
    yy, xx = np.mgrid[-k:k + 1, -k:k + 1]
    return (xx ** 2 + yy ** 2) <= k * k


def _rosa003_body(body, ctx):
    from scipy.ndimage import binary_fill_holes, binary_opening, binary_closing, binary_dilation
    if not body.any() or ctx.get('img') is None: return body
    img = ctx['img'].astype(int); gb = img[..., 1] - img[..., 2]
    bar = body & (gb > ROSA_GB)
    rest = binary_fill_holes(body & ~bar)
    k = ROSA_OPEN; p = k + 6
    bo = binary_opening(np.pad(rest, p), structure=_rosa_disk(k))[p:-p, p:-p]
    if not bo.any(): return body
    m = bar | bo
    g = _rosa_disk(ROSA_GROW)
    m = m | (body & binary_dilation(m, structure=g))
    m = m | (body & binary_closing(np.pad(m, p), structure=_rosa_disk(ROSA_CLOSE))[p:-p, p:-p])
    m = m | (body & binary_dilation(m, structure=g))
    m = binary_fill_holes(m)
    sp = binary_opening(np.pad(m, p), structure=_rosa_disk(ROSA_SPUR))[p:-p, p:-p]
    if sp.any():
        cut = m & ~(sp | (m & binary_dilation(sp, structure=g)))   # the spurs
        m = sp | (m & binary_dilation(sp, structure=g))
        if cut.any():
            # a wedge of this ball squeezed between the balls (not a spur): the cut pixels touching another ball's mesh draw stay
            ids_h, R = ctx['ids_h'], ctx['R']
            mine = np.unique(ids_h[body]); mine = mine[mine > 0]
            oth = [int(d) for d in np.unique(ids_h[binary_dilation(body, structure=_rosa_disk(4))]) if d > 0 and d not in mine and d in R['draws']
                   and R['draws'][d].get('blend') == 1 and (R['uv'][..., 0][ids_h == d] >= 0).mean() < 0.5]
            if oth: m = m | (cut & binary_dilation(np.isin(ids_h, oth), structure=_rosa_disk(2)))
    return m


POST_BODY[6000586] = _rosa003_body


# The balls of one shot touch each other, each label is made on its own: a small unlabelled gap between two labels
# (a highlight wedge of one ball squeezed in at the seam) is a hole of the union. Fill holes of the union of this type's
# labels (<= ROSA_HOLE px, only unlabelled pixels) into the neighbouring label.
ROSA_HOLE = 40
_rosa_prev = labels


def labels(objs, cam, R, state=None, ended=(), img=None):
    from scipy.ndimage import binary_fill_holes, binary_dilation, label as cc
    import bulletlabels as BL
    P = _rosa_prev(objs, cam, R, state=state, ended=ended, img=img)
    ltype = {}
    for o in objs: ltype.setdefault(BL._lid(o), o['t'])
    ids, lids, share = P['ids'], P['lids'], P['share']
    ks = [k for k in np.unique(ids[ids >= 0]) if lids[k][2] == 'body' and ltype.get(lids[k][0]) == 6000586]
    if len(ks) < 2: return P
    U = np.isin(ids, ks)
    hole = binary_fill_holes(U) & (ids < 0)
    if not hole.any(): return P
    L, n = cc(hole); sz = np.bincount(L.ravel())
    for j in range(1, n + 1):
        if sz[j] > ROSA_HOLE: continue
        hj = L == j
        nb = ids[binary_dilation(hj, structure=_rosa_disk(1)) & U]
        if not nb.size: continue
        ids[hj] = np.bincount(nb).argmax(); share[hj] = BL.CUT
    return P
