# sweep 10-27 fix for skin 002 (Sonnet revision fix2.py after a rejected Haiku fix, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s002/)
# Duplicator Villain 002 (Lolla003Projectile 6001093, Lola's electric attack; attack 0 deg ticks 265-281, attack 90 deg ticks 303-331;
# sw2_002 pairs 47, 48, 49, 55, 58; second attempt after review1.md): each live projectile = ONE clean solid label of its body.
# The projectile is the green cylinder (clip shape 35441) with its ring cap (clip shape 28), plus the bright square sprite on or
# beside it (trail-clip shapes 575 / 642, edgar_pixel: the user-visible glowing core square). The orange ground streaks (58595),
# the rotating rings (866) and every other trail piece around it are effects and are NOT labelled (rules 3, 4): this removes the
# tangle of loops around the cones (pair 55). Free squares not touching the body are leftover sparks (pair 58): unlabelled.
# The label is the object's own outline: its own pixels (plus 1-2 px of no-shape edge) closed over gaps of 3 px, holes filled
# solid, crumbs dropped (review1 problems 1, 3, 4: loops/holes/speckle on the cylinders).
# Only clip pixels of the cylinder (and the squares) are used, so the burst / shards of a projectile that hit the wall (t269+) are out.
import numpy as _np
L27_ATK = 6001093
L27_FILE = 'sc/effects_brawler.sc'
L27_CLIP = ('lolla_003_atk_projectile', 'lolla_003_atk_projectile_red')
L27_CYL = 35441           # clip shape of the green cylinder
L27_SQ_DIST = 32          # px (half resolution) a square may sit away from its cylinder
L27_SQ_MIN = 40           # px: the real square is ~90 px, specks of the same shapes are not
import os as _os
L27_DBG = bool(_os.environ.get("L27_DBG"))
L27_SOLID = 0.78          # a group less solid than this (area / convex hull area) is drawn as its hull
L27_SQUARE = (575, 642)    # trail-clip shapes of the glowing square sprite


def _l27_atk(body, ctx):
    if not body.any(): return body
    from scipy.ndimage import binary_fill_holes, binary_opening, label as cc
    R = ctx['R']; ONE = _np.ones((3, 3))
    trail = tuple(n for f, n in _trail_clips(L27_ATK) if f == L27_FILE)
    TS = _a4_shape_pixels(R, L27_FILE, trail)
    OWNS = _a4_shape_pixels(R, L27_FILE, L27_CLIP)
    OWN = OWNS > 0
    TR = TS > 0
    cyl = (body | ctx['region']) & (OWNS == L27_CYL + 1)                       # the cylinder itself
    if not cyl.any(): cyl = body & OWN
    if not cyl.any(): return cyl
    sqs = body & _np.isin(TS, [s + 1 for s in L27_SQUARE])
    Lq, nq = cc(sqs, structure=ONE)
    # one group per cylinder (cylinders of one shot are separate projectiles): closing / filling never bridges two of them
    Lc, nc = cc(_dilate(cyl, 2), structure=ONE)
    out = _np.zeros_like(body)
    for j in range(1, nc + 1):
        cj = cyl & (Lc == j)
        if cj.sum() < 25 and nc > 1: continue                # crumb of a cylinder, not a projectile
        # its ring cap (clip shape 28) only where it sits on the cylinder; other ring strokes of the clip far from it are effects
        m = cj | (body & OWN & _dilate(cj, 4))
        det = []                                              # the glowing square sprite on / beside the cylinder (it can be drawn
        for q in _np.unique(Lq[sqs & _dilate(m, L27_SQ_DIST)]):   # offset from it; free sparks of the same shapes are too small)
            if q <= 0 or (Lq == q).sum() < L27_SQ_MIN: continue
            det.append(Lq == q)
        m |= body & ~TR & ~OWN & _dilate(m, 2)                # 1-2 px of antialiased edge
        m = binary_fill_holes(_close(m, 6))                   # ring-cap strokes are thin diagonals: closed to one solid piece
        mo = binary_opening(_np.pad(m, 3), iterations=2)[3:-3, 3:-3]   # spurs / frayed bumps off
        m = mo if mo.sum() >= 0.85 * m.sum() else m
        hm = hull_mask(m)
        if L27_DBG: print('L27 solidity', ctx['o']['id'], round(float(m.sum() / hm.sum()), 3))
        if m.sum() < L27_SOLID * hm.sum(): m = hm              # a cylinder partly hidden / faded (ragged): its convex hull
        out |= m
        for d in det: out |= binary_fill_holes(_close(d, 2))   # each square: its own solid piece (never bridged to the cylinder)
    return out if out.any() else cyl


POST_BODY[L27_ATK] = _l27_atk
