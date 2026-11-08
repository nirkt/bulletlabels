# human review 2026-10-28 fix for skin 002 (Haiku fix, reviewed by the coordinator, installed 10-29; source out/sweep2/haikutasks/human1028/s002/fix2.py; tested via IDLABELS_OVERRIDE after all fixes incl. z_1099_cleanup, hence z_1100_)
# Skin 002 DuplicatorVillain, human review 2026-10-28 (task out/sweep2/haikutasks/human1028/s002, the user's note on the page:
# "some attack effects get labeled along with the bullet and the label is sometimes jagged"; flagged frames attack 0 ticks
# 267, 269, 273, 277 and attack 90 ticks 309, 315, 317; "microscopic hole in the first bullet" at tick 277).
# Type 6001093 Lolla003Projectile (Lola's electric cylinder shot). This file REPLACES fix.py (same type id; this file is
# exec'd after the installed fixes and sets POST_BODY, so its entry wins). Install this one, not fix.py.
# What the projectile is: a green cylinder = its own clip shape 35441, with its ring cap (shape 28) and the glowing square
# sprites (shapes 575, 642) that sit on it. Each cylinder is one label, and its label is taken from those shapes only:
# - the rotating rings (trail shape 866), the orange ground streaks (58595) and any pixel without one of the attack's shapes
#   (the cyan glow next to the cylinder, sparks, halos) are effects, not the bullet (the user: effects labelled along with it);
# - a ring or streak drawn over the cylinder hides part of it in the ID pass: the gaps are closed and the holes filled, so
#   a square or a ring showing through the cylinder is inside the label (the user's hole at tick 277);
# - a cylinder outline still notched after closing is replaced by its convex hull (a cylinder is convex): one smooth,
#   solid, clean piece in the object's own outline (the user's jagged edges).
# Review 2026-10-28 (coordinator, on fix.py), three leftovers, fixed here:
# (1) tick 315, upper cylinder (attack 90): the body and its tapered lower part were two pieces with a 1 px unlabelled
#     row between them (a rotating ring crossing it), so an inner closed loop showed. The cylinder's labelled groups that
#     meet across a gap of 1-2 px are one piece: the union of the labels is closed (gaps <= 2 px filled).
# (2) tick 313, bottom-left cylinder: a notch at the top-right where a green glowing square overlaps the rim. The hidden-parts
#     rule: the cylinder's outline continues under the square, so the cylinder's own convex outline is filled within
#     S02_FILL px of the square and of the cap ring (the square itself is not added unless it is mostly inside the cylinder).
# (3) tick 277, the cylinder in the crate: the front ring (cap, shape 28) is one piece with the body; its bottom arc that
#     stuck out below the body (the thin piece with a seam) is filled into the cylinder's own outline by the same rule as (2).
#     The green glow ring beside the cylinder is an effect (its colour is the glow's, not the cap's) and stays out.
# Pieces of fewer than 25 px are crumbs and are dropped; a cylinder that is not visible at all (hidden in a hit / explosion)
# gets no label, as in z_1027_s002 (the pipeline's body is not used: it held the explosion's electric shards, tick 279).
import numpy as _np
S02_ATK = 6001093
S02_FILE = 'sc/effects_brawler.sc'
S02_CLIP = ('lolla_003_atk_projectile', 'lolla_003_atk_projectile_red')
S02_CYL = 35441          # clip shape of the green cylinder
S02_CAP = 28             # its ring cap (clip shape)
S02_SQ = (575, 642)      # glowing square sprites (trail-clip shapes) that sit on a cylinder
S02_SQ_MIN = 40          # px: the real square is ~90 px at frame resolution, specks of the same shapes are not
S02_CAP_NEAR = 4         # px: a ring-cap piece this close to its cylinder belongs to it
S02_SOLID = 0.92         # a closed outline less solid than this (area / convex hull area) is drawn as its hull
S02_CRUMB = 25           # px: cylinder pieces smaller than this are crumbs
S02_SQ_IN = 0.5          # a square at least this share inside its cylinder's outline is part of the bullet, whole
S02_CLOSE = 10           # px: concavities up to about 2 x this closed (8-connected: diagonal slits close too)
S02_FILL = 3             # px: the cylinder's own outline is filled this far under a square / ring piece on its rim (review 2026-10-28)
S02_JOIN = 2             # px: unlabelled gaps this narrow between the labels of one cylinder are closed (review 2026-10-28)
S02_MERGE = 10           # px: cylinder groups this close in y (and overlapping in x, S02_JOIN_X) are one cylinder (review 2026-10-28)
S02_JOIN_X = 0.6         # share of the narrower group's width that two groups must overlap in x to be one cylinder
S02_MAX_H = 44           # px: the height of a merged cylinder (one cylinder is at most ~40 px tall at frame resolution)


def _close8(m, r):
    from scipy.ndimage import binary_closing
    return binary_closing(_np.pad(m, r), structure=_np.ones((3, 3)), iterations=r)[r:-r, r:-r] if m.any() else m


def _s02_atk(body, ctx):
    from scipy.ndimage import binary_fill_holes, label as cc
    R = ctx['R']; region = ctx['region']; ONE = _np.ones((3, 3))
    trail = tuple(n for f, n in _trail_clips(S02_ATK) if f == S02_FILE)
    SP = _a4_shape_pixels(R, S02_FILE, S02_CLIP)     # clip shape id + 1 per pixel, 0: none of the clip's shapes
    TS = _a4_shape_pixels(R, S02_FILE, trail)        # trail clips' shape id + 1 per pixel
    cyl = (SP == S02_CYL + 1) & region
    if not cyl.any(): return _np.zeros_like(body)   # no cylinder visible (hidden in a hit / explosion): no label, as before
    sq_mask = _np.isin(TS, [s + 1 for s in S02_SQ]) & region
    Lq, nq = cc(sq_mask, structure=ONE)
    Lc, nc = cc(_dilate(cyl, 2), structure=ONE)      # one group per cylinder: cylinders of one shot never bridge
    # review 2026-10-28 (1): groups of one cylinder split by a ring crossing its body (the upper and lower part of one cylinder,
    # side by side in x, 1-10 px apart in y) are one group; cylinders of one shot stacked in a column are far enough apart
    from scipy.ndimage import find_objects
    sl = find_objects(Lc); par = list(range(nc + 1))
    def _root(a):
        while par[a] != a: a = par[a]
        return a
    for a in range(1, nc + 1):
        for b in range(a + 1, nc + 1):
            if sl[a - 1] is None or sl[b - 1] is None: continue
            ya0, ya1, xa0, xa1 = sl[a - 1][0].start, sl[a - 1][0].stop - 1, sl[a - 1][1].start, sl[a - 1][1].stop - 1
            yb0, yb1, xb0, xb1 = sl[b - 1][0].start, sl[b - 1][0].stop - 1, sl[b - 1][1].start, sl[b - 1][1].stop - 1
            ov = min(xa1, xb1) - max(xa0, xb0) + 1
            narrow = min(xa1 - xa0 + 1, xb1 - xb0 + 1)
            gap = max(ya0, yb0) - min(ya1, yb1) - 1
            tall = max(ya1, yb1) - min(ya0, yb0) + 1      # one cylinder is ~ 26-40 px tall: two stacked cylinders are not one
            if ov >= S02_JOIN_X * narrow and 0 <= gap <= S02_MERGE and tall <= S02_MAX_H:
                par[_root(b)] = _root(a)
    lut = _np.array([_root(a) for a in range(nc + 1)])
    nparts = _np.bincount(lut[1:], minlength=nc + 1)  # how many parts a merged cylinder has (1: not merged)
    Lc = lut[Lc]
    groups = [int(g) for g in _np.unique(Lc) if g > 0]
    out = _np.zeros_like(body)
    for j in groups:
        cj = cyl & (Lc == j)
        if cj.sum() < S02_CRUMB and len(groups) > 1: continue
        cap = (SP == S02_CAP + 1) & region & _dilate(cj, S02_CAP_NEAR)
        core = cj | cap
        core |= body & (SP == 0) & (TS == 0) & _dilate(core, 1)  # 1 px of antialiased edge of the cylinder itself
        core = binary_fill_holes(_close8(core, S02_CLOSE))        # gaps / slits where a ring or streak is drawn over it, holes filled
        hc = hull_mask(core)                                      # the cylinder's own convex outline
        m = core.copy()
        # its glowing squares: one that lies at least half inside that outline sits on the bullet and is part of it, whole
        # (a notch or hole where it sticks out is a defect); a square mostly outside it is an effect beside it, not the bullet
        for q in _np.unique(Lq[sq_mask & hc]):
            sq = Lq == q
            if q <= 0 or sq.sum() < S02_SQ_MIN: continue
            if (sq & hc).sum() >= S02_SQ_IN * sq.sum(): m |= sq
        m = binary_fill_holes(_close8(m, 2))                       # 1-2 px notches of the outline (a slit at the cap) closed
        # review 2026-10-28 (2)/(3): the outline continues under a square or a ring piece that sits on its rim, inside its own
        # convex outline (a notch there, or a ring's arc sticking out below the body, is a defect)
        rim = (cap | (sq_mask & hc)) & hc
        if rim.any(): m |= hc & _dilate(rim, S02_FILL) & region
        if m.sum() < S02_SOLID * hull_mask(m).sum(): m = hull_mask(m)   # notched outline: its hull (no more than the cylinder and its squares)
        if nparts[j] > 1:   # two parts of one cylinder (split by a ring crossing its body): the body is a straight column, its width
            # from the top part extruded down to the lowest part (the hidden body under the ring is filled in, review 2026-10-28 (1))
            rows = _np.nonzero(core.any(1))[0]
            top = core[rows.min():rows.min() + 13]; xs = _np.nonzero(top.any(0))[0]
            if xs.size:
                rect = _np.zeros_like(core); rect[rows.min():rows.max() + 1, xs.min():xs.max() + 1] = True
                m = binary_fill_holes(m | (rect & region))
        if m.sum() > 200:   # review 2026-10-28 (3): spurs narrower than 3 px (a ring arc or cap tip sticking out) are cut off
            from scipy.ndimage import binary_opening
            mo = binary_opening(m, structure=ONE)
            if mo.sum() >= 0.9 * m.sum(): m = mo
        out |= m
    if out.any():   # review 2026-10-28 (1): the labels of one cylinder that meet across a 1-2 px unlabelled gap are one piece
        out = binary_fill_holes(_close8(out, S02_JOIN)) & region
    return out


POST_BODY[S02_ATK] = _s02_atk
