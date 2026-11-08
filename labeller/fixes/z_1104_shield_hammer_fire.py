# installed 11-01 from out/sweep2/haikutasks/siteai2/shield_hammer_fire/fix.py (2nd AI sweep: 327 leaf particles out + touching sprays one hull, 418 starburst pieces in, 456 heart/head solid; single-skin types)
# AI-review sweep 2 (2026-11-01), task shield_hammer_fire (coordinator folder siteai2/shield_hammer_fire). Exec'd in idlabels'
# namespace after all installed fixes (IDLABELS_OVERRIDE), so it wraps the installed labels() of z_1103_shield_hammer_fire.py.
# Skins 327, 418, 456 only. Types: 6001259 (327 Buster003 spray), 6001263 (418 Buster005 wave), 6000839 (456 FireDude006 flame/heart).
# Skins 315, 266 (no_defect) and the Frank skin 401 (needs_user, Frank's attack template is not changed) are not touched.
#
# 418 ShieldTankAngel, attack 90 t307 (user review: "the outline covers only the lower-right of the wave; the top blue wave and
# the white starburst on the left sit outside"): the shot's one label (z_1103: one hull per shot) cut the white starburst's arms
# (atlas shape 145 of the clip, the same star that sits inside the wave at t305 and t309). The shot's own star pieces (shapes 145,
# 137, 22; not shape 0, the soft halo) that touch its label are added as they are. No hull is drawn over them: a hull reached the
# brawler behind the flash (tested). Other types, skins and ticks without these pieces are unchanged.
#
# 327 ShieldTankCroc, attack 0 t265 (user review: "zigzag, jagged outline with interior lines; it should be one plain hull"):
# the olive leaf particles of the attack's trail (atlas shape 37549, buster_003_atk_particle_leaf) were labelled as part of the
# sprays, which made the interior lines and notches. They are cut out of the body by their shape id (POST_BODY). The three sprays
# of one shot touch at the spawn tick, so their touching labels become one plain hull (the overlapping-projectile rule); sprays
# that are apart keep their own outlines, as at t267 / t269 (the same counts as before there, except the leaf cut).
#
# 456 FireDudeStarrGirls, attack 0 t265-269 and attack 90 t305-315 (user review: "the flame projectile is labelled only as a speck
# or a small loop on its glowing head; the flame body above it is outside the outline"; the heart body is the owner's open 456
# item). The label was the clip's soft circle (shape 0, code 1) only. The body's bright pixels are the clip's body / shared sprite
# codes (1 and 5 in fixdiag's sprite codes; the glow code 2 and the soft halo stay out). They are closed, filled and cut to the
# piece that touches the prediction, so the heart is one solid outline (t305-t315) and the head is labelled. NOT done: the blue
# streak above the head (atlas shape 116). Adding it brought in its soft halo, spurs and the brawler's green ring, so it is left
# out and listed as needs_user.

import numpy as _np3
from scipy.ndimage import binary_fill_holes as _fh3, binary_closing as _clo3, label as _lab3, binary_dilation as _dil3

T327_SPRAY = 6001259     # 327 ShieldTankCroc, Buster003Projectile
T418_WAVE = 6001263      # 418 ShieldTankAngel, Buster005Projectile (one label per shot, see z_1103)
T456_FLAME = 6000839     # 456 FireDudeStarrGirls, FireDude006Projectile (attack 0 flame and attack 90 heart)
S418_STAR = [145, 137, 22]   # the wave's star and streak pieces (shape id), not the soft halo (shape 0)


def _f006_body(body, ctx):
    """456 FireDude006: the body is the clip's body / shared sprite pixels in the search region (codes 1 and 5), closed and
    hole-filled, cut to the piece that touches the prediction. The glow (code 2) and the halo stay out."""
    C = ctx['C']; region = ctx['region']; pred = ctx.get('pred_any')
    m = _np3.isin(C, [1, 5]) & region
    if not m.any(): return body
    m = _fh3(_clo3(m, structure=_np3.ones((3, 3), bool), iterations=1))
    L, n = _lab3(m)
    if n > 1 and pred is not None and pred.any():
        near = _dil3(pred, iterations=4) & m
        keep = _np3.unique(L[near]); keep = keep[keep > 0]
        m = _np3.isin(L, keep) if keep.size else m
    return body | m


POST_BODY[T456_FLAME] = _f006_body


T327_LEAF = ('buster_003_atk_particle_leaf_01', 'buster_003_atk_particle_leaf_02', 'buster_003_atk_particle_leaf_03')


def _spray_body(body, ctx):
    """327 Buster003 spray: the olive leaf particles that drift through the sprays (the type's trail clip buster_003_atk_particle_leaf,
    atlas shape 37549) are an effect, not the spray; they are cut out of the body by their shape id (27 of 28 olive labelled
    pixels at t265 and all 68 at t267 are this shape)."""
    T = projtab.table()[T327_SPRAY]
    SPl = _a4_shape_pixels(ctx['R'], T['file'], T327_LEAF) > 0
    body = body & ~SPl
    if body.any(): body = (_fh3(body) & ~SPl) | body    # small enclosed holes in the sprays (unlabelled bright pixels); never a leaf
    return body


POST_BODY[T327_SPRAY] = _spray_body


_fix3_prev_labels = labels


def labels(objs, cam, R, state=None, ended=(), img=None):
    import bulletlabels as BL
    P = _fix3_prev_labels(objs, cam, R, state=state, ended=ended, img=img)
    if not any(o['t'] in (T327_SPRAY, T418_WAVE) for o in objs): return P
    ids, lids, share = P['ids'], P['lids'], P['share']
    shots = {}
    for o in objs:
        if o['t'] in (T327_SPRAY, T418_WAVE): shots.setdefault((o['t'], o.get('owner'), o.get('first')), []).append(o)
    ids_h = R['id'][::2, ::2]; draws = R['draws']
    wdraw = _np3.isin(ids_h, [int(d) for d, r in draws.items() if r.get('blend') == 0 and r.get('dmask') == 1])
    for (t, _own_id, _first), grp in sorted(shots.items(), key=lambda kv: min(o['id'] for o in kv[1])):
        T = projtab.table()[t]
        names = tuple(n for n in (T.get('blue'), T.get('red')) if n)
        SP = _a4_shape_pixels(R, T['file'], names)
        own = _np3.isin(ids_h, _np3.unique(ids_h[SP > 0]))
        mem = []
        for o in sorted(grp, key=lambda q: q['id']):
            lid = (BL._lid(o), -1, 'body')
            if lid not in lids: continue
            k = lids.index(lid)
            if (ids == k).any(): mem.append(k)
        if not mem: continue
        if t == T418_WAVE:
            # the shot's star / streak pieces that touch its label (the white starburst's arms) join the hull
            mine = _np3.isin(ids, mem)
            E = _np3.isin(SP - 1, S418_STAR) & (SP > 0) & (ids < 0)
            L, n = _lab3(E)
            if n:
                near = _dil3(mine, iterations=3) & E
                keep = _np3.unique(L[near]); keep = keep[keep > 0]
                E = _np3.isin(L, keep) if keep.size else _np3.zeros(ids.shape, bool)
            else:
                E = _np3.zeros(ids.shape, bool)
            # the star's own pixels are added as they are (no hull over them: a hull would reach the brawler behind the flash)
            new = E & ~mine
            if not new.any(): continue
            ids[new] = mem[0]; share[new] = BL.CUT
        else:
            # the sprays of one shot whose labels touch (one shot, one spawn tick) become one hull; apart, they stay apart
            masks = {k: (ids == k) for k in mem}
            cl = []   # clusters of touching members
            for k in mem:
                hit = [c for c in cl if any((_dil3(masks[k], iterations=2) & masks[j]).any() for j in c)]
                cl = [c for c in cl if c not in hit] + [sum(hit, []) + [k]]
            for c in cl:
                if len(c) < 2: continue
                U = _np3.zeros(ids.shape, bool)
                for k in c: U |= masks[k]
                H = hull_mask(U)
                take = H & (U | ((ids < 0) & (SP > 0) & ~wdraw))   # unlabelled pixels only where this type's own shapes show
                carrier = min(c)
                for k in c: ids[ids == k] = -1
                ids[take] = carrier; share[take & ~U] = BL.CUT
    return P
