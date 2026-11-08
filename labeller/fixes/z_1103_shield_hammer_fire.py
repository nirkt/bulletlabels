# AI-review batch 2026-11-01 fix (Haiku fix, reviewed by the coordinator, installed 11-01 LOCAL ONLY; source out/sweep2/haikutasks/siteai/shield_hammer_fire/fix2.py)
# Task shield_hammer_fire, fix2 (coordinator decisions 2026-11-01). Exec'd in idlabels' namespace (IDLABELS_OVERRIDE).
# Skins 418 and 368 only. Types: 6001263 (418 Buster005 Angel), 6000835 (368 amber_004 attack), 6000836 (368 amber_004 super). Other types and skins are not touched.
import numpy as _np2
from scipy.ndimage import binary_fill_holes as _fh2, binary_closing as _clo2, label as _lab2, binary_dilation as _dil2


def _keep_touch2(m, pred, r=4):
    """pieces of m that touch the predicted label (dilated r px); a single piece is kept as is"""
    if not m.any(): return m
    L, n = _lab2(m)
    if n <= 1: return m
    near = (_dil2(pred, iterations=r) & m) if (pred is not None and pred.any()) else m
    keep = _np2.unique(L[near]); keep = keep[keep > 0]
    return _np2.isin(L, keep) if keep.size else m


def _shape_solid2(shapes):
    """POST_BODY factory: the body is the projectile's own atlas shapes (the ids in `shapes`, from the clip and its trail
    clips of the same atlas file), inside the search region; closed by 1 px and filled to one solid outline; pieces that do
    not touch the predicted label are dropped. Shapes not in the list (a beam or streak above the projectile) stay out."""
    def fn(body, ctx):
        T = projtab.table()[ctx['t']]
        names = tuple(n for n in (T.get('blue'), T.get('red')) if n)
        SP = _a4_shape_pixels(ctx['R'], T['file'], names)          # shape id + 1 per pixel, 0 = none
        m = _np2.isin(SP - 1, list(shapes)) & (SP > 0) & ctx['region']
        if not m.any(): return body
        m = _fh2(_clo2(m, structure=_np2.ones((3, 3), bool), iterations=1))
        return _keep_touch2(m, ctx.get('pred_any'))
    return fn


# 368 FireDudeMake, amber_004: the orb cluster of the super (6000836) and the attack's sphere + bubbles (6000835) are drawn with
# the atlas shapes 63 and 142 (the ids of the cyan orb). Survey (work/sv368*.txt): no other shape of the clip or its trail
# clips lies on the orb. One solid outline per projectile.
POST_BODY[6000836] = _shape_solid2({63, 142})
POST_BODY[6000835] = _shape_solid2({63, 142})

# 456 (6000839): not changed. Shape 0 is the clip's generic soft circle (halo), not the heart; no clean heart shape id found.


# 418 ShieldTankAngel, Buster005Projectile (6001263): ONE label per shot (the owner's ShieldTank decision, z_1101_s073
# convention): the members of one shot (same owner, first shown on the same tick) are merged into one convex hull on the
# lowest id; the neighbours' pixels inside it are the shot's, so the holes disappear. Only this type is touched.
_S418_T = 6001263
_fix2_prev_labels = labels


def labels(objs, cam, R, state=None, ended=(), img=None):
    import bulletlabels as BL
    P = _fix2_prev_labels(objs, cam, R, state=state, ended=ended, img=img)
    if not any(o['t'] == _S418_T for o in objs): return P
    ids, lids, share = P['ids'], P['lids'], P['share']
    shots = {}
    for o in objs:
        if o['t'] == _S418_T: shots.setdefault((o.get('owner'), o.get('first')), []).append(o)
    T = projtab.table()[_S418_T]
    names = tuple(n for n in (T.get('blue'), T.get('red')) if n)
    SP = _a4_shape_pixels(R, T['file'], names)
    ids_h = R['id'][::2, ::2]; draws = R['draws']
    own = _np2.isin(ids_h, _np2.unique(ids_h[SP > 0]))
    wdraw = _np2.isin(ids_h, [int(d) for d, r in draws.items() if r.get('blend') == 0 and r.get('dmask') == 1])
    for key, grp in sorted(shots.items(), key=lambda kv: min(o['id'] for o in kv[1])):
        mem = []
        for o in grp:
            lid = (BL._lid(o), -1, 'body')
            if lid not in lids: continue
            k = lids.index(lid); m = ids == k
            if m.any(): mem.append((o['id'], k, m))
        if not mem: continue
        mem = sorted(mem, key=lambda q: q[0]); ks = [k for _, k, _ in mem]; carrier = ks[0]
        U = _np2.zeros(ids.shape, bool)
        for _, _, m in mem: U |= m
        H = hull_mask(U)
        mine = _np2.isin(ids, ks)
        take = H & (mine | ((ids < 0) & own & ~wdraw))     # unlabelled pixels only where this type's own shapes show
        ids[mine & ~take] = -1
        new = take & ~mine
        ids[take] = carrier; share[new] = BL.CUT
    return P
