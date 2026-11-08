# sticky4 fix file (round 4, 2026-11-06). Every global name here has the prefix _st4_.
#
# FIX 1  page 148 CrossBomberDragon (Grom skin 006), super 45, type 6000980 Grom006UltiProjectile = the flying cube.
# Owner (11-06): "part of the label just gets cut off ... check every label"; flagged: super 45 tick 345 "too small / cut off".
# Cause: while the cube spawns at the thrower (ticks ~343-353) the cyan swoosh arc (a draw in front of the cube) and the HP bar cover
# half of the cube: the label only has the uncovered half, with an inner line where the swoosh starts (tick 345), a stair step where the HP
# bar hides the corner (tick 347) and a notch where a cyan blob covers the top (tick 307). The cube is a convex rounded block, so the
# covered part is filled in (hidden parts rule): the label becomes the convex hull of its own main pieces (pieces >= 20 % of the largest,
# so specks of the landing burst stay as they are). Only type 6000980 (used by skin 148 only).
import numpy as _st4_np


def _st4_hull_main(body, ctx, rel=0.2, lo=0.05, hi=0.5):
    if not body.any(): return body
    from scipy.ndimage import label as _cc
    L, n = _cc(body, structure=np.ones((3, 3), bool))
    sz = np.bincount(L.ravel()); sz[0] = 0
    main = (sz >= rel * sz.max())[L] & body
    if main.sum() < 30: return body
    a = main.sum()
    P, k = _cc(hull_mask(main) & ~body)
    draws = ctx['R']['draws']
    bg = [d for d, dr in draws.items() if str(dr.get('th', '')).startswith('1d304abe')]
    ok = ~np.isin(ctx['ids_h'], bg) & (ctx['C'] != 2)
    out = body.copy()
    for j in range(1, k + 1):
        m = P == j
        # big pockets of the hull = parts of the cube hidden behind the swoosh / HP bar; small ones are the notches between its corner arrows
        if lo * a <= m.sum() <= hi * a: out |= m & ok
    return out


POST_BODY[6000980] = _st4_hull_main


# ---------------------------------------------------------------------------------------------------------------------------------
# FIX 2 + 3  StickyBomb (Dynamike) super 45, the landing frame t367 of pages 281 292 312 333 403 445 (and 286). Owner 11-06 on super 45
# tick 367: 281 / 286 / 445 "should not be labelled"; 292 / 312 / 403 "holes / ragged"; 333 "too big".
# What happens: the super bomb (skin specific type: 281 Veteran 6000936, 286 Potato 6000939, 292 Ghost 6000942, 312 Rabbit 6000945,
# 333 University 6000948, 403 SandsOfTimes 6000951, 445 Brawlentine 6000954) lands (t365) and spawns 6 child bubbles. ALL skins share the
# child type 6000934 StickyBombUltiProjectile2 (clip sticky_def_atk_projectile, also the default skin 060 StickyBombDefault).
# (a) 292 / 312 / 333 / 403: the bubbles are visible in a ring. The old label is ragged: the bubbles' translucent rim (coverage < 128) is
#     lost (FADE_CUT) and the pixels in the middle of the ring (glow between the bubbles) are claimed by the neighbouring child. Fix:
#     FADE_CUT[6000934] = 32 (rims in) and every child keeps only the whole connected bubble silhouette(s) (clip body pixels of the type,
#     holes filled) that lie mostly on its own predicted body; the middle of the ring is not a bubble sprite and stays out.
# (b) 281 (Veteran), 286 (Potato), 445 (Brawlentine): the landing explosion (cloud / fire / hearts) is drawn over the new bubbles, the old
#     labels are scraps of that effect. The children of these three skins are not labelled in the first frames after the spawn (age <= 2 ticks,
#     i.e. t365 and t367); from t369 on the bubbles are separate and clear and stay labelled. Decided per skin by the main super's type
#     (the three types above) because the children share one type id; the other skins and the default skin keep their children.
from scipy.ndimage import binary_fill_holes as _st4_fill, label as _st4_cc

FADE_CUT[6000934] = 32


def _st4_child(body, ctx):
    pb = ctx['pred_body']
    if not pb.any(): return body
    cand = (ctx['C'] == 1) & (ctx['cv'] >= 32)
    cand = _st4_fill(cand | (_close(cand, 1) & ctx['region']))
    L, n = _st4_cc(cand, structure=np.ones((3, 3), bool))
    if n < 1: return body
    sel = np.zeros(body.shape, bool); tot = float(pb.sum())
    for k in range(1, n + 1):
        m = L == k
        ov = (m & pb).sum()
        if ov >= 0.15 * tot or (ov >= 0.5 * m.sum() and m.sum() >= 30): sel |= m
    if not sel.any(): return body
    return sel | (body & _dilate(sel, 3))


POST_BODY[6000934] = _st4_child

_st4_hidden_mains = {6000936, 6000939, 6000954}   # Squeak002 / 003 / 008 UltiProjectile: Veteran 281, Potato 286, Brawlentine 445
_st4_child_t = 6000934
_st4_seen = {}
_st4_last = [0]
_st4_prev_labels = labels


def labels(objs, cam, R, state=None, ended=(), img=None):
    tick = int(state.get('tick', 0)) if state else 0
    if tick < _st4_last[0]: _st4_seen.clear()
    _st4_last[0] = tick
    for o in objs:
        if o['t'] == _st4_child_t: _st4_seen.setdefault(o['id'], tick)
    mains = {o.get('src') for o in objs if o['t'] in _st4_hidden_mains}
    if mains:
        objs = [o for o in objs if not (o['t'] == _st4_child_t and o.get('src') in mains and tick - _st4_seen[o['id']] <= 2)]
    P = _st4_prev_labels(objs, cam, R, state=state, ended=ended, img=img)
    # owner 11-06 (223): touching bullets are ONE label, no seams: the child bubbles of one shot that touch (<= 1 px) become one label
    import bulletlabels as _BL
    ks = []
    for o in objs:
        if o['t'] != _st4_child_t: continue
        lid = (_BL._lid(o), -1, 'body')
        if lid in P['lids']:
            k = P['lids'].index(lid)
            if (P['ids'] == k).any(): ks.append(k)
    if len(ks) > 1:
        ids = P['ids']; par = {k: k for k in ks}
        def find(a):
            while par[a] != a: a = par[a]
            return a
        D = {k: _dilate(ids == k, 1) for k in ks}
        for a in ks:
            for b in ks:
                if a < b and (D[a] & (ids == b)).any(): par[find(b)] = find(a)
        groups = {}
        for k in ks: groups.setdefault(find(k), []).append(k)
        for k0, g in groups.items():
            if len(g) < 2: continue
            u = np.isin(ids, g)
            u2 = _close(u, 1) & ((ids < 0) | u)
            P['ids'][u2] = k0
            P['share'][u2 & ~u] = _BL.CUT
    return P
