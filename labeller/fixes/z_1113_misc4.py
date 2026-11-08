# misc4 fix (round 4, 2026-11-06): owner-flagged pages 216, 310, 452, 560, 590 (+207, 541, 425 owner answers).
# Exec'd in idlabels' namespace (IDLABELS_OVERRIDE / z_1113_misc4). Unique prefix _m4_.
import numpy as _m4_np

# ---- 452 Mender199Gem (Mender002Projectile 6001758, jet + spray; only this skin uses it) ----
# Owner 11-06: "attacks wrong" -> ONE outline per shot (jet + spray) on BOTH attack rows. z_1104_misc_c only did that for the flat
# attack 0 (its _horizontal() test fails for the vertical attack 90). The beam test itself is the dust-pixel count, orientation-free,
# so _horizontal is made always true (z_1104_misc_c's helpers look it up by global name at call time).
def _horizontal(m):
    return True


# ---- generic per-type smoother (used for 216 Crow003 daggers; others below) ----
from scipy.ndimage import (gaussian_filter as _m4_gauss, binary_fill_holes as _m4_fill, binary_dilation as _m4_dil,
                           binary_erosion as _m4_ero, binary_closing as _m4_close, label as _m4_cc)


def _m4_smooth_mask(m, sigma, grow=1, keep_core=1):
    """round a label's outline: close, fill holes, blur+threshold; never more than `grow` px outside the input, never loses its core"""
    if not m.any(): return m
    c = _m4_fill(_m4_close(_m4_np.pad(m, 3), iterations=1))[3:-3, 3:-3] | m
    s = _m4_gauss(c.astype(float), sigma) > 0.5
    s &= _m4_dil(c, iterations=grow)
    if keep_core: s |= _m4_ero(c, iterations=keep_core)
    return s


def _m4_disk(r):
    y, x = _m4_np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r + 0.5


def _m4_smooth_types(P, objs, types, sigma, grow=1, close_r=0, small_rel=0.0, drop_far=0, merge_r=3, min_px=0, open_r=0):
    """per label of the given types: (1) a label under small_rel x the area of a label of the same types within 3 px is a torn piece of
    that neighbour: its pixels go to it; (2) hidden-part fill: closing with a disc of close_r (new pixels only where nothing is labelled);
    (3) rounding (see _m4_smooth_mask)"""
    import bulletlabels as BL
    ids, lids, share = P['ids'], list(P['lids']), P['share']
    tl = {BL._lid(o) for o in objs if o['t'] in types}
    ks = [k for k, l in enumerate(lids) if l[0] in tl and l[2] == 'body']
    if not ks: return P
    ids = ids.copy(); share = share.copy()
    if small_rel and len(ks) > 1:
        ar = {k: int((ids == k).sum()) for k in ks}
        for k in sorted(ks, key=lambda q: ar[q]):
            if not ar[k]: continue
            near = _m4_dil(ids == k, iterations=merge_r) if merge_r else _m4_np.zeros_like(ids, bool)
            big = [q for q in ks if q != k and ar[q] > 0 and ar[k] < small_rel * ar[q] and (near & (ids == q)).any()]
            if big:
                q = max(big, key=lambda q: ar[q]); ids[ids == k] = q; ar[q] += ar[k]; ar[k] = 0
            elif drop_far:       # a torn fragment a few px away from a much bigger label of the same shot: not a bullet of its own
                near = _m4_dil(ids == k, iterations=drop_far)
                if any(q != k and ar[q] > 0 and ar[k] < small_rel * ar[q] and (near & (ids == q)).any() for q in ks):
                    ids[ids == k] = -1; ar[k] = 0
    for k in ks:
        m = ids == k
        if not m.any(): continue
        if m.sum() < min_px: ids[m] = -1; continue       # a speck, not a recognisable object
        if close_r:
            c = _m4_np.pad(m, close_r + 2); c = _m4_ero(_m4_dil(c, _m4_disk(close_r)), _m4_disk(close_r))[close_r + 2:-close_r - 2, close_r + 2:-close_r - 2]
            oth = _m4_dil((ids >= 0) & (ids != k), iterations=2)     # no bridging towards a neighbouring label
            m = m | (c & (ids < 0) & ~oth); ids[m & (ids < 0)] = k
        if open_r:        # thin necks / tails off the body (a dot joined by a thread) are cut, if the body keeps most of its area
            mo = _m4_np.pad(m, open_r + 2); mo = _m4_dil(_m4_ero(mo, _m4_disk(open_r)), _m4_disk(open_r))[open_r + 2:-open_r - 2, open_r + 2:-open_r - 2] & m
            if mo.sum() >= 0.6 * m.sum():
                L2, n2 = _m4_cc(mo)
                if n2 > 1: mo = L2 == (_m4_np.bincount(L2.ravel())[1:].argmax() + 1)
                m = mo
        s = _m4_smooth_mask(m, sigma, grow)
        add = s & (ids < 0); rem = m & ~s
        ids[rem] = -1
        ids[add] = k; share[add] = 1.0
    # pockets enclosed by the labels of several objects of these types (<= 60 px, nothing else labelled inside) -> neighbour, most border
    U = _m4_np.isin(ids, ks)
    Hh = _m4_fill(U) & ~U
    Lh, nh = _m4_cc(Hh)
    for j in range(1, nh + 1):
        hj = Lh == j
        if hj.sum() > 60: continue
        nb = ids[_m4_dil(hj) & ~hj]
        if nb.size == 0 or (nb < 0).any() or not _m4_np.isin(nb, ks).all(): continue
        k = int(_m4_np.bincount(nb).argmax()); ids[hj] = k; share[hj] = 1.0
    P2 = dict(P); P2['ids'] = ids; P2['share'] = share
    return P2


# ---- 216 CrowWhite (Crow003Projectile 6000317; only this skin) ----
# Owner 11-06: "labels are good, but still a little ragged": outline stair-steps along the blade, bumps at the handle. Rounded with a
# stronger per-label blur than the global step (sigma 1.6, at most 1 px outside the old label, core kept).
_m4_prev_labels_216 = labels


def labels(objs, cam, R, state=None, ended=(), img=None):
    P = _m4_prev_labels_216(objs, cam, R, state=state, ended=ended, img=img)
    P = _m4_smooth_types(P, objs, (6000317,), 1.6, close_r=4, small_rel=0.3)
    # 310 (see the 310 block): the flame's loop above a blade is closed (it left a ring-shaped hook), outline rounded
    return _m4_smooth_types(P, objs, (6000321,), 1.6, close_r=5)


# ---- 310 CrowVulture (Crow005Projectile 6000321, 0 / 90 deg attacks; only this skin) ----
# Owner 11-06 (attack 90, t309): "too big". At t309 the third blade is already spent at the wall: its crystal is drawn as a loose blue
# shard beside the wall, and the green flame of the neighbouring blade reaches over to it, so the blade's label grew a lobe that takes in
# the shard (and, between blade and shard, pieces of flame that belong to nothing). Fix (POST_BODY, after the installed ones): the label is
# opened with a disc of 3 px; only the parts that touch the blade's own predicted body (grown 2 px) stay (the shard is neither
# overlapping nor part of the blade); the original label is kept within 3 px of the kept core (flame fringe and rim stay).
_M4_T310 = 6000321
_m4_prev_310 = POST_BODY.get(_M4_T310)


def _m4_310(body, ctx):
    b = _m4_prev_310(body, ctx) if _m4_prev_310 else body
    pb = ctx['pred_body']
    if not b.any() or not pb.any(): return b
    o3 = _m4_np.pad(b, 4); o3 = _m4_dil(_m4_ero(o3, _m4_disk(3)), _m4_disk(3))[4:-4, 4:-4] & b
    L, n = _m4_cc(o3)
    if n <= 1: return b
    keep = _m4_np.unique(L[_m4_dil(pb, iterations=2) & (L > 0)]); keep = keep[keep > 0]
    if not keep.size: return b
    core = _m4_np.isin(L, keep)
    return b & _m4_dil(core, _m4_disk(3))


POST_BODY[_M4_T310] = _m4_310


# ---- 560 PowerLevelerGenie (Surge006Projectile 6000802 / Surge006Projectile2 6000803; only this skin) ----
# Owner 11-06 (attack 90, t313): "holes / ragged". At t313 the fist has turned into its landing animation: the cyan body is covered by sparkle
# stars and the lamp, so the label has bites on the right, a ragged edge and a torn little fragment beside it (a second Projectile2 object).
# Fix (labels() wrapper, own types only): torn fragments (under 30 % of the biggest label of the shot) within 3 px are merged, within 15 px
# are dropped; the hidden parts (stars drawn over the body) are filled by a 4-px closing (never towards a neighbouring label), holes filled,
# outline rounded (sigma 1.6, at most 1 px outside).
_m4_prev_labels_560 = labels


def labels(objs, cam, R, state=None, ended=(), img=None):
    P = _m4_prev_labels_560(objs, cam, R, state=state, ended=ended, img=img)
    return _m4_smooth_types(P, objs, (6000802, 6000803), 1.6, close_r=4, small_rel=0.3, drop_far=15, merge_r=0)


# ---- 590 PowerLevelerBuzz (Surge008Projectile 6000806 / Surge008Projectile2 6000807; only this skin) ----
# Owner 11-06 (attack 90, t313): "holes / ragged". At t313 the fist has burst into its round bloom (a white disc with a cyan rim, the big
# glow quad is not labelled): all that is labelled is a 13-px speck of the inner swirl (with holes in it), then at t315 / t317 small
# worm-shaped blobs. Fix (same wrapper as 560): specks under 40 px are dropped (nothing a model could learn, and it is only the pattern
# inside the bloom), the other labels get their hidden parts filled, holes filled and the outline rounded.
_m4_prev_labels_590 = labels


def labels(objs, cam, R, state=None, ended=(), img=None):
    P = _m4_prev_labels_590(objs, cam, R, state=state, ended=ended, img=img)
    return _m4_smooth_types(P, objs, (6000806, 6000807), 1.6, close_r=4, small_rel=0.3, drop_far=15, merge_r=0, min_px=40, open_r=2)


# ---- 207 AssaultShotgunDragon (Griff006Projectile 6001031 / ...Sp1Buddy 6001032; only this skin) ----
# Owner 11-06: "the gold thread at the end of the pellet is CUT COMPLETELY" (z_1110_misc AB5M_207 only opened thin spurs of it, a part always
# stayed). The crystal pellet is a cyan crystal (clip shape 5) ending in a gold cap wound like thread (clip shape 21, a narrow cone that
# tapers to a tip, the "spur" at the pellet's end); beside it the bracelet's gold-and-green clasp band (also shape 21) lies against it and the
# re-cut of z_1109_mortis_dragon glued it to the crystal as a hook. Rule (POST_BODY, before that re-cut): every shape-21 pixel is removed from
# the body, with a 1-px edge of it (not the crystal's own pixels); the pellet label is the crystal. The jewellery pellets (ring etc., other
# shapes) are not touched.
_M4_207 = (6001031, 6001032)
_M4_207_F = 'sc/effects_brawler_griff.sc'
_M4_207_C = ('griff_006_atk_projectile',)
_m4_prev_207 = {t: POST_BODY.get(t) for t in _M4_207}


def _m4_207_hook(t):
    def f(body, ctx):
        b = _m4_prev_207[t](body, ctx) if _m4_prev_207[t] else body
        if not b.any(): return b
        SP = _a4_shape_pixels(ctx['R'], _M4_207_F, _M4_207_C)
        g = b & (SP == 22)                       # shape 21 (+1)
        if not g.any(): return b
        cut = _m4_dil(g, iterations=1) & b & (SP != 6)
        return b & ~cut
    return f


# NOT INSTALLED (coordinator 11-06): the 207 hook cut the whole gold cone + gave the middle piece a loose U label -> ask owner.
# for _t in _M4_207: POST_BODY[_t] = _m4_207_hook(_t)


# ---- 541 MechaDudePirate (Mecha005UltiProjectile 6001079, the green shells; only this skin) ----
# Owner answer 11-06: thin glow rims / swoosh arcs that cling to a bullet DO count as part of it ("include them, but the label isnt perfect
# on the 2 bullets"). z_1111_smooth3b opens thin arcs away (S3B_MECHA_OPEN) and z_1027_s541 drops loose pieces, so the shell labels lost the
# lime arc over the bulb (clip shape 2934), and the pale side wings (53425, only where they cling); the wavy bubble rim under the bottle (636) is NOT taken: it made a stem at the bottle's tip. Fix
# (labels() wrapper): those sprite pixels (coverage >= CUT, unlabelled) that chain onto a shell label within 2-px steps (at most 3 steps, and
# not farther than 14 px from the label) are added; debris leaves / chunks further out are not reached. Shells of one swirl whose labels then
# touch (within 2 px) become ONE label (the bullets overlap: one silhouette, owner 223 rule), closed (r 3), holes filled, outline rounded.
_M4_541 = 6001079
_M4_541_F = 'sc/effects_brawler.sc'
_M4_541_C = ('mecha_005_atk2_projectile', 'mecha_005_atk2_projectile_red')
_M4_541_RIM = (2934, 53425)
_M4_541_RIMCV = 48       # rims are translucent: lower coverage cut than the body's 128
_m4_prev_labels_541 = labels


def _m4_541_labels(P, objs, R, cv):
    import bulletlabels as BL
    from scipy.ndimage import distance_transform_edt as edt
    ids, lids, share = P['ids'].copy(), list(P['lids']), P['share'].copy()
    ks = []
    for o in objs:
        if o['t'] != _M4_541: continue
        lid = (BL._lid(o), -1, 'body')
        if lid in lids and (ids == lids.index(lid)).any(): ks.append(lids.index(lid))
    if not ks: return P
    SP = _a4_shape_pixels(R, _M4_541_F, _M4_541_C)
    cand = _m4_np.isin(SP, [s + 1 for s in _M4_541_RIM]) & (cv >= _M4_541_RIMCV) & (ids < 0)
    U = _m4_np.isin(ids, ks)
    reach = U.copy()
    for _ in range(3):
        reach = reach | (_m4_dil(reach, iterations=2) & cand)
    add = reach & cand & (edt(~U) <= 14)
    if add.any():
        d = _m4_np.stack([edt(ids != k) for k in ks], 0)
        ys, xs = _m4_np.nonzero(add); own = d[:, ys, xs].argmin(0)
        for n_, k in enumerate(ks):
            s = own == n_; ids[ys[s], xs[s]] = k; share[ys[s], xs[s]] = 1.0
    # swirls: labels touching within 2 px = one bullet silhouette
    masks = {k: ids == k for k in ks}; grp = {k: k for k in ks}

    def find(a):
        while grp[a] != a: a = grp[a]
        return a
    for a in ks:
        da = _m4_dil(masks[a], iterations=8)
        for b in ks:
            if a < b and (da & masks[b]).any(): grp[find(b)] = find(a)
    groups = {}
    for k in ks: groups.setdefault(find(k), []).append(k)
    for g in groups.values():
        Um = _m4_np.zeros_like(U)
        for k in g: Um |= masks[k]
        keep = max(g, key=lambda k: int(masks[k].sum()))
        c = _m4_np.pad(Um, 10); c = _m4_ero(_m4_dil(c, _m4_disk(7)), _m4_disk(7))[10:-10, 10:-10] | Um
        c = _m4_fill(c)
        free = (ids < 0) | _m4_np.isin(ids, g)
        s = _m4_smooth_mask(c & free | Um, 1.0, grow=1) & free
        for k in g: ids[masks[k]] = -1
        ids[s] = keep; share[s] = 1.0
    P2 = dict(P); P2['ids'] = ids; P2['share'] = share
    return P2


def labels(objs, cam, R, state=None, ended=(), img=None):
    P = _m4_prev_labels_541(objs, cam, R, state=state, ended=ended, img=img)
    return _m4_541_labels(P, objs, R, R['cv'][::2, ::2])


# ---- 425 CrowBrawloween (Crow009Projectile 6000327; only this skin) ----
# Owner answer 11-06: the thin glow rims / green puff rims that cling to a dagger DO count ("include them, but the label isnt perfect on the 2
# bullets"). z_1111_smooth3b kept only a 1-px rim of the puff shapes (clip shapes 0 / 4) and the single daggers (attack 90 t313-319, the
# "2 bullets") had a pinched waist where the guard is thin: two blobs joined by a neck. Fix (labels() wrapper, own type): puff-shape pixels
# (coverage >= 80) within 2 px of a dagger label are added (the clinging rim, not the cloud further out); the label is then closed with a
# 4-px disc (the waist and notches), holes filled and the outline rounded (sigma 1.4). Only unclaimed pixels are added. Clusters of
# overlapping daggers (hulls of z_1103) are only rounded, their hull is kept.
_M4_425 = 6000327
_m4_prev_labels_425 = labels


def labels(objs, cam, R, state=None, ended=(), img=None):
    import bulletlabels as BL
    from scipy.ndimage import distance_transform_edt as edt
    P = _m4_prev_labels_425(objs, cam, R, state=state, ended=ended, img=img)
    ids, lids, share = P['ids'], list(P['lids']), P['share']
    ks = []
    for o in objs:
        if o['t'] != _M4_425: continue
        lid = (BL._lid(o), -1, 'body')
        if lid in lids and (ids == lids.index(lid)).any(): ks.append(lids.index(lid))
    if not ks: return P
    ids = ids.copy(); share = share.copy()
    cv = R['cv'][::2, ::2]
    SP = _a4_shape_pixels(R, 'sc/effects_brawler_crow.sc', ('crow_009_atk_projectile',))
    puff = _m4_np.isin(SP, (0 + 1, 4 + 1)) & (cv >= 80) & (ids < 0)
    U = _m4_np.isin(ids, ks)
    d = _m4_np.stack([edt(ids != k) for k in ks], 0)
    near = puff & (d.min(0) <= 2)
    ys, xs = _m4_np.nonzero(near); own = d[:, ys, xs].argmin(0)
    for n_, k in enumerate(ks):
        s = own == n_; ids[ys[s], xs[s]] = k; share[ys[s], xs[s]] = 1.0
    for k in ks:
        m = ids == k
        if not m.any(): continue
        c = _m4_np.pad(m, 8); c = _m4_ero(_m4_dil(c, _m4_disk(4)), _m4_disk(4))[8:-8, 8:-8] | m
        c = _m4_fill(c) & ((ids < 0) | m)
        s = _m4_smooth_mask(c, 1.2, grow=1) & ((ids < 0) | m)
        ids[m & ~s] = -1; ids[s & (ids < 0)] = k; share[s] = _m4_np.maximum(share[s], 1.0 * (share[s] > 0))
        share[s & (share == 0)] = 1.0
    P2 = dict(P); P2['ids'] = ids; P2['share'] = share
    return P2
