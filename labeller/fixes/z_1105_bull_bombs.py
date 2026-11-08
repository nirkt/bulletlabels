# installed 11-02 from out/sweep2/haikutasks/siteai3/bull_bombs/fix.py (regression round: 238 robots split by anchor when apart, 483 orb spur opening; 515 orb candidate NOT registered)
# Haiku fix, regression-sweep round 2026-11-02, group bull_bombs (skins 213, 238, 483, 515, 531). Exec'd in idlabels'
# namespace (IDLABELS_OVERRIDE) after all installed fixes; every hook chains the installed one for its type.
#
# ---- skin 238 TntDudeMecha, attack 0 (Mike004Projectile, 6000273): the two robot dynamites, ticks 265-281.
# Reviewer (2026-11-02, t273): "one outline wraps two separate blue projectiles". Checked on sw2_238: the two robots are two
# objects (2000006, 2000007) drawn as ONE instanced mesh draw (a0_agent0 _mike004_atk), and the draw has one owner, so at t273
# the whole pair went to 2000007 (1024 px) and 2000006 got 4 px. The robots are separate projectiles (on screen 25-50 px apart
# from t273 on; at t265-t271 they overlap, 7-20 px apart, and keep one label, as the overlap rule says).
# Fix (labels()): when the two robots of a shot are at least _SPLIT238_SEP px apart on screen, the label pixels of both are
# split by nearest predicted screen anchor (the anchor mapping of z_1104_bombs_misc). Checked: at t281 (already split by the
# z_1103 hull) and on attack 90 (t305-t321, already split) the split reproduces the installed pixel counts within 1-2 px.
_SPLIT238_TYPE = 6000273
_SPLIT238_SEP = 22.0      # px (1200 x 540 frame): robots closer than this overlap: one label (unchanged)
_prev_labels_bb = labels


def _split238_anchor(o, cam):
    import numpy as _n
    q = cam @ _n.array([float(o['anchor'][0]), float(o['anchor'][1]), float(o['anchor'][2]), 1.0])
    return ((q[0] / q[3] * 0.5 + 0.5) * 1200, (1 - (q[1] / q[3] * 0.5 + 0.5)) * 540)


def labels(objs, cam, R, state=None, ended=(), img=None):
    import numpy as _n
    import bulletlabels as BL
    P = _prev_labels_bb(objs, cam, R, state=state, ended=ended, img=img)
    rob = [o for o in objs if o['t'] == _SPLIT238_TYPE and o.get('drawn') and o.get('anchor') is not None]
    if len(rob) < 2:
        return P
    A = _n.array([_split238_anchor(o, cam) for o in rob], float)
    sep = min(_n.hypot(*(A[i] - A[j])) for i in range(len(rob)) for j in range(i + 1, len(rob)))
    if sep < _SPLIT238_SEP:
        return P
    ids = P['ids']; lids = list(P['lids'])
    keys = [(BL._lid(o), -1, 'body') for o in rob]
    if not all(k in lids for k in keys):
        return P
    kb = [lids.index(k) for k in keys]
    U = _n.isin(ids, kb)
    if not U.any():
        return P
    yy, xx = _n.nonzero(U)
    D = _n.hypot(xx[:, None] - A[None, :, 0], yy[:, None] - A[None, :, 1])
    pick = D.argmin(1)
    for j, k in enumerate(kb):
        sel = pick == j
        ids[yy[sel], xx[sel]] = k
    P['ids'] = ids
    return P


# ---- skin 483 BullDudeBarbking, attack 90 (Bull002Projectile, 6000043): the orb pellets, t313 (and the attack 0 orbs of the
# same type). Reviewer (2026-11-02, t313): "the blue flame is covered by several tiny separate fragments instead of one hull".
# Checked on sw2_483 ticks 305-321 and 267-273: the four orbs of the t313 cluster are four pellets, each one outline (the owner's
# per-pellet rule, z_1104_bull, accepted), and they are not fragments: the three upper / right / lower orbs are round. The real
# defect is the left orb (pellet partly behind the pink spiked object): its outline has a thin spur running left into the pink
# object (a spur of 2-4 px width). Fix (POST_BODY, the body of each orb object before the pellet split): morphological opening with
# a disc of radius 3 (orbs are about 10 px across: their shape survives, spurs thinner than 6 px go). A piece the opening would
# remove completely is kept (never lose a projectile). Only type 6000043 (single-skin type of this brawler family).
_BULL_SPUR_DISK = (lambda yy, xx: (xx * xx + yy * yy) <= 9)(*np.ogrid[-3:4, -3:4])
_prev_post_483 = POST_BODY.get(6000043)


def _fix_6000043_spur(body, ctx):
    from scipy.ndimage import binary_opening as _bo483, label as _cc483
    if _prev_post_483 is not None:
        body = _prev_post_483(body, ctx)
    if not body.any():
        return body
    op = _bo483(body, structure=_BULL_SPUR_DISK)
    L, n = _cc483(body)
    has = np.bincount(L[op], minlength=n + 1) > 0
    lost = [k for k in range(1, n + 1) if not has[k]]
    if lost:
        op = op | np.isin(L, lost)
    return op


POST_BODY[6000043] = _fix_6000043_spur


# ---- skin 515 MechaDudeRhino, attack 90 (Mecha002UltiProjectile, 6001073), t305: the round bright orb at the lower left of
# the flame (the flame's head): candidate fix, NOT installed (see the note below). Reviewer (2026-11-02): "round blue orb only partly inside its outline; the bright core and most of
# the orb are outside". Checked on sw2_515 t305 (ASCII map of label, sprite codes and frame brightness): the flame body (code 2 /
# code 1) is labelled, the white-cyan core of the orb (min(R,G,B) 200-255, 13 px across, touching the body within 1-2 px) is code 0,
# so the ID pass never gave it to the flame. The orb is part of the flame: it is added. Rule (POST_BODY, after the installed
# installed drip-leg opening z_1104_speedy_mecha_roller): bright pixels (min channel >= 190) of no sprite (code 0) within 4 px of the
# body, inside the object's predicted region; a 2 px closing bridges the 1 px gap, and only pixels with min channel >= 120 or the
# orb itself are used for the bridge (dark background stays out, the glow halo stays out: the halo is min < 120).
# The orb is taken as a whole: the bright connected blob (min >= 190, no sprite) that touches the body (the orb's far side is 10+ px away).
from scipy.ndimage import binary_dilation as _bd515, binary_closing as _bc515, binary_fill_holes as _bfh515, label as _cc515
_prev_515_flame = POST_BODY.get(6001073)
_ORB515_MIN = 190
_ORB515_NEAR = 4
_ORB515_DISK = (lambda yy, xx: (xx * xx + yy * yy) <= 4 + 0.5)(*np.ogrid[-2:3, -2:3])


def _fix_6001073_orb(body, ctx):
    if _prev_515_flame is not None:
        body = _prev_515_flame(body, ctx)
    if not body.any() or ctx.get('img') is None:
        return body
    img = ctx['img']
    mn = img.min(axis=2)
    C = ctx['C']; region = ctx['region']
    near = _bd515(body, structure=np.ones((2 * _ORB515_NEAR + 1, 2 * _ORB515_NEAR + 1), bool))
    # the whole bright blob (the orb is 13-14 px across) that touches the body's neighbourhood
    bright = (mn >= _ORB515_MIN) & (C == 0) & region
    LB, nb = _cc515(bright)
    tid = np.unique(LB[near & bright]); tid = tid[tid > 0]
    keep = []
    for q in tid:
        mq = LB == q; n = int(mq.sum()); r, g, b = img[mq].mean(0)
        # the orb: a cyan-white blob of 60-150 px (its mean colour: green and blue high, red clearly lower). The white sparks of
        # the flame tips (neutral or warm) are not the orb.
        if 60 <= n <= 150 and g >= 240 and b >= 240 and r <= g - 20:
            keep.append(q)
    if not keep:
        return body
    tid = np.array(keep)
    add = np.isin(LB, tid)
    # the orb's anti-aliased rim (1 px, bright but min < 190): still the orb, not the glow (max >= 200 only)
    add = add | (_bd515(add, structure=np.ones((3, 3), bool)) & (C == 0) & region & (img.max(axis=2) >= 200))
    m = body | add
    gap = (mn >= 110) & (C != 1) & near & region
    closed = _bc515(m, structure=_ORB515_DISK) & (m | gap)
    out = _bfh515(closed | m)
    L, n = _cc515(out)
    if n > 1:
        sz = np.bincount(L.ravel()); sz[0] = 0
        out = L == sz.argmax()
    return out


# NOT INSTALLED in this round (needs_user, see report): the orb is static. Its blob sits at the same screen spot (563-574 x
# 309-320) on t305, t309 and t313 while the flame bodies move away, and on t309 it touches no flame; so it is not the flame's
# head. The label of the flame body is left as it is. To enable (if the owner says the orb is part of the flame):
# POST_BODY[6001073] = _fix_6001073_orb
