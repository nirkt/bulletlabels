# sweep 10-27 fix for skin 255 (Sonnet revision, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s255/)
# sweep 10-27 fix (2nd attempt) for skin 255 SandstormGiftShop (Sandy's attack, Sandy002Projectile 6000668; task out/sweep2/haikutasks/sweep1027/s255/)
# Findings: pair 4659 (attack 0 deg, tick 271) and pair 4662 (attack 90 deg, tick 305): the sand cloud has broken into cyan loops and
# streaks; its label has a deep open notch next to the dark centre of the loops (tick 271) and a notch in the left lobe (tick 305).
# Rules: Sandy's attack is one solid zone label over the whole cloud in the cloud's own outline (the user's decision); no notches or
# holes (rule 1); not the background or other objects (rule 2).
# Why the 1st attempt (fix.py) was rejected: a diamond (4-connected) closing bridges gaps with straight diagonal chords, it bridged the
# two separate parts of tick 311 and the cyan sliver of tick 307, and its 'opaque draws that are not this attack' test treated the
# crate / barrel draws as the attack's own (a few of their pixels are in the body), so it spilled over them.
# Fix, per connected part of the cloud (parts under 10 % of the main one are crumbs / slivers and stay as they are, never bridged):
#  - the part is closed with a ROUND (disk) structuring element, which leaves arcs instead of chords, and holes are filled;
#    radius S255_R_LO px (frame is 540 x 1200) for a compact part, S255_R_HI px for a loop-state part (solidity under S255_SOLID:
#    the cloud has opened into loops, tick 271 and the left part of tick 311); parts are closed separately, never joined to each other;
#  - pixels of opaque world draws (crates, barrels, walls: every opaque draw except the ground, whose fragment shader is the one of
#    the largest opaque draw in the cloud's region) are never added.
# Only additions: the original label is kept whole. A cloud whose main part is under S255_MIN_MAIN px (the dispersing end burst)
# is left alone; the end-burst drop of fixes/a2_agent2.py (_sandy_end) runs first and is kept (chained).
# Not done: a convex hull of the loop state: it takes in the dark background (rule 2).
import numpy as np

S255_ATK = 6000668            # Sandy002Projectile (skin 255's Sandy attack, clip sandy_002_atk_projectile)
S255_R_LO = 8                 # px: disk radius closing a compact part
S255_R_HI = 12                # px: disk radius closing a loop-state part
S255_SOLID = 0.7              # part area / convex hull area below which a part is a loop state
S255_MIN_MAIN = 500           # px: the cloud's main part; below it the body is a dispersing crumb, unchanged
S255_OPEN = 3                 # px: disk radius of the opening that finds a compact part's thick core
S255_REACH = 6                # px: pieces of a compact part farther than this from its core and under 5 % of it are slivers
S255_POCKET = 600             # px: enclosed holes up to this are filled
S255_PART_FRAC = 0.1          # parts under this share of the main part are crumbs (the same share as the clean-up)


def _s255_disk(r):
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    return x * x + y * y <= r * r + r * 0.5


def _s255_close(m, r):
    from scipy.ndimage import binary_dilation, binary_erosion
    p = r + 2; s = _s255_disk(r)
    return binary_erosion(binary_dilation(np.pad(m, p), s), s)[p:-p, p:-p]


def _s255_solidity(m):
    from scipy.spatial import ConvexHull
    ys, xs = np.nonzero(m)
    try: h = ConvexHull(np.stack([xs, ys], 1).astype(float))
    except Exception: return 1.0
    return m.sum() / (h.volume + 0.5 * h.area + 1)


def _s255_fill(body, ctx):
    from scipy.ndimage import binary_fill_holes, binary_opening, binary_dilation, label as cc
    if not body.any(): return body
    ids_h = ctx['ids_h']; region = ctx['region']; R = ctx['R']
    L, n = cc(_dilate(body, 3)); sz = np.bincount(L[body], minlength=n + 1); sz[0] = 0
    if sz.max() < S255_MIN_MAIN: return body
    opq = {}
    for d in np.unique(ids_h[region & (ids_h > 0)]):
        if R['draws'].get(int(d), {}).get('blend') == 0: opq[int(d)] = int((ids_h[region] == d).sum())
    ground = R['draws'][max(opq, key=opq.get)].get('fs') if opq else None
    world = np.zeros(body.shape, bool)                                # opaque draws that are not the ground
    for d in opq:
        if R['draws'][d].get('fs') != ground: world |= (ids_h == d)
    out = body.copy()
    for k in np.nonzero(sz >= S255_PART_FRAC * sz.max())[0]:
        m = body & (L == k)
        r = S255_R_HI if _s255_solidity(m) < S255_SOLID else S255_R_LO
        inp = m
        if r == S255_R_LO:       # compact part: a thin sliver hanging off it (tick 307) is not bridged to the body
            core = binary_opening(m, _s255_disk(S255_OPEN))
            if core.sum() >= 0.5 * m.sum():
                tips = m & ~binary_dilation(core, iterations=S255_REACH)
                TL, tn = cc(tips); ts = np.bincount(TL.ravel(), minlength=tn + 1)
                inp = m & ~(np.isin(TL, [j for j in range(1, tn + 1) if ts[j] < 0.05 * m.sum()]) & tips)
        out |= binary_fill_holes(_s255_close(inp, r) | inp) & region & ~world
    # a small hole left inside the cloud by an opaque draw seen through it (the crate pocket in the left lobe, tick 305)
    H, nh = cc(binary_fill_holes(out) & ~out); hs = np.bincount(H.ravel(), minlength=nh + 1)
    for j in range(1, nh + 1):
        if hs[j] <= S255_POCKET: out |= (H == j)
    return out


# chained: the end-burst drop of fixes/a2_agent2.py (_sandy_end) runs first and empties the body at the burst (ticks 273-275, 315)
_s255_prev = POST_BODY.get(S255_ATK)


def _s255_hook(body, ctx):
    if _s255_prev is not None: body = _s255_prev(body, ctx)
    return _s255_fill(body, ctx)


POST_BODY[S255_ATK] = _s255_hook
