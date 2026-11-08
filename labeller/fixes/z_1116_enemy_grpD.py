# enemy-label fix 2026-11-07 (task away_1106_enemy, group grpD; will be installed as z_1116_enemy_grpD.py; every global has the prefix _eD_)
# Skins / types of the group (the enemy label of these types must equal the blue rule; own labels stay pixel-identical):
#   364 Shelly009Projectile 6000004 / Shelly009UltiProjectile 6000005, 460 Shelly010Projectile 6000018, 390 Poco009Projectile / UltiProjectile 6000351 / 6000352, ...
import numpy as _eD_np
import functools as _eD_ft
import projtab as _eD_projtab

# ---------------------------------------------------------------------------------------------------------------------------------
# (1) red twin shapes: the red clip of a type is the blue layout with other shape ids (other atlas regions). Rules written for the blue
# shape ids ('body = shape 56893') see nothing in an enemy shot. _a4_shape_pixels() of these types' clips also asks for the red twin clip
# (listed FIRST: on an atlas overlap the blue polygon wins) and reports every red-only shape id as its blue counterpart (data-derived: the
# two clips are walked as display trees, shapes on the same slot path are twins). Blue shape ids are never renamed -> own labels identical.
_EDT = ()   # types (filled below)


def _eD_walk(S, cid, seen=()):
    key = (id(S), cid)
    if key in _EDW: return _EDW[key]
    if cid in seen: return {}
    out = {}; c = S.clips[cid]
    for f in range(max(1, c['nframes'])):
        for slot, child, cm, cc, bl in S.frame(cid, f):
            if child in S.shapes: out.setdefault((slot,), set()).add(child)
            elif child in S.clips:
                for p, v in _eD_walk(S, child, seen + (cid,)).items(): out.setdefault((slot,) + p, set()).update(v)
    out = {p: frozenset(v) for p, v in out.items()}
    _EDW[key] = out
    return out


_EDW = {}


def _eD_box(S, sid):
    a = _eD_np.concatenate([_eD_np.asarray(xy, _eD_np.float64).reshape(-1, 2) for tex, uv, xy in S.shapes[sid]])
    return a.min(0), a.max(0)


def _eD_iou(a, b):
    lo = _eD_np.maximum(a[0], b[0]); hi = _eD_np.minimum(a[1], b[1]); i = _eD_np.prod(_eD_np.maximum(0, hi - lo))
    u = _eD_np.prod(a[1] - a[0]) + _eD_np.prod(b[1] - b[0]) - i
    return float(i / u) if u > 0 else 0.0


@_eD_ft.lru_cache(maxsize=None)
def _eD_twin_map(swf, blue, red):
    S = PD.sc(swf); cb = S.exports.get(blue); cr = S.exports.get(red)
    if cb is None or cr is None: return {}
    wb = _eD_walk(S, cb); wr = _eD_walk(S, cr)
    ids_b = set().union(*wb.values()) if wb else set(); ids_r = set().union(*wr.values()) if wr else set()
    votes = {}
    for p in set(wb) & set(wr):
        B = sorted(wb[p]); Rr = sorted(wr[p])
        if len(B) != len(Rr): continue
        for x, y in zip(B, Rr):
            if y not in ids_b and x != y: votes.setdefault(y, {}).setdefault(x, 0); votes[y][x] += 1
    out = {}
    only_b = sorted(ids_b - ids_r)
    for y in sorted(ids_r - ids_b):
        yb = _eD_box(S, y); sc = sorted(((_eD_iou(yb, _eD_box(S, x)), x) for x in only_b), reverse=True)
        cand = None
        if sc and sc[0][0] >= 0.8 and (len(sc) < 2 or sc[1][0] < 0.8): cand = sc[0][1]
        if cand is None and y in votes:
            x = max(votes[y], key=lambda k: votes[y][k])
            if x in only_b and _eD_iou(yb, _eD_box(S, x)) >= 0.3: cand = x
        if cand is None and sc and sc[0][0] >= 0.5 and (len(sc) < 2 or sc[1][0] < 0.7 * sc[0][0]):
            x = sc[0][1]; ay = _eD_np.prod(yb[1] - yb[0]); ax = _eD_np.prod(_eD_box(S, x)[1] - _eD_box(S, x)[0])
            if min(ay, ax) >= 0.7 * max(ay, ax): cand = x
        if cand is not None: out[y] = cand
    return out


def _eD_pairs(types):
    out = []
    T = _eD_projtab.table()
    for t in types:
        x = T.get(t)
        if x and x['file'] and x['blue'] and x['red'] and x['blue'] != x['red']: out.append((x['file'], x['blue'], x['red']))
    return list(dict.fromkeys(out))


@_eD_ft.lru_cache(maxsize=None)
def _eD_plan(swf, names, types):
    S = PD.sc(swf); ex = S.exports
    pairs = [p for p in _eD_pairs(types) if p[0] == swf and p[1] in ex and p[2] in ex and (p[1] in names or p[2] in names)]
    if not pairs: return names, None
    extra = [r for f, b, r in pairs if r not in names] + [b for f, b, r in pairs if b not in names]
    ask = tuple(dict.fromkeys(tuple(extra) + names))
    blue_ids = set()
    for f, b, r in pairs: blue_ids |= {s for s, k in _clip_shapes(S, ex[b])}
    lut = _eD_np.arange(65536, dtype=_eD_np.int32); n = 0
    for f, b, r in pairs:
        for y, x in _eD_twin_map(swf, b, r).items():
            if y in blue_ids or y + 1 >= 65536: continue
            if lut[y + 1] not in (y + 1, x + 1): continue
            lut[y + 1] = x + 1; n += 1
    return ask, (lut if n else None)


_eD_prev_sp = _a4_shape_pixels


def _a4_shape_pixels(R, swf, names):
    try: ask, lut = _eD_plan(swf, tuple(names), _EDT)
    except Exception: return _eD_prev_sp(R, swf, names)
    out = _eD_prev_sp(R, swf, ask)
    if lut is not None and out.size: out = lut[out]
    return out
_EDT = (6000004, 6000005, 6000352)


# ---------------------------------------------------------------------------------------------------------------------------------
# (2) colour tests tuned on the blue art: for an enemy shot (every object of the type is red-art) the frame handed to the installed
# chain has the red art's pink hues rotated back to the blue art's cyan (hue band + shift per type), so every colour test written for the
# blue art applies unchanged. Own shots: the image is never touched.
_EDH = {}      # type -> (hue lo, hue hi, shift) in degrees (hue band of the red art that corresponds to blue-art hues; shift added, mod 360)


def _eD_recolor(img, lo, hi, shift, vmin=None):
    f = img.astype(_eD_np.float32) / 255.0
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    mx = f.max(-1); mn = f.min(-1); d = mx - mn
    h = _eD_np.zeros_like(mx)
    nz = d > 1e-6
    rc = nz & (mx == r); gc = nz & (mx == g) & ~rc; bc = nz & ~rc & ~gc
    h[rc] = ((g - b)[rc] / d[rc]) % 6.0
    h[gc] = (b - r)[gc] / d[gc] + 2.0
    h[bc] = (r - g)[bc] / d[bc] + 4.0
    h = h * 60.0
    sel = nz & (h >= lo) & (h <= hi) & (d >= 0.04)
    if not sel.any(): return img
    h2 = (h + shift) % 360.0
    # back to rgb for the selected pixels (same V and S)
    v = mx; s = _eD_np.where(mx > 0, d / _eD_np.maximum(mx, 1e-6), 0)
    if vmin is not None: v = _eD_np.where(v >= vmin, 1.0, v)
    hp = h2 / 60.0; c = v * s; x = c * (1 - _eD_np.abs(hp % 2 - 1)); m = v - c
    k = _eD_np.floor(hp).astype(int) % 6
    R_ = _eD_np.choose(k, [c, x, 0 * c, 0 * c, x, c]); G_ = _eD_np.choose(k, [x, c, c, x, 0 * c, 0 * c]); B_ = _eD_np.choose(k, [0 * c, 0 * c, x, c, c, x])
    out = img.copy()
    rgb = _eD_np.stack([R_ + m, G_ + m, B_ + m], -1) * 255.0
    out[sel] = _eD_np.clip(_eD_np.rint(rgb[sel]), 0, 255).astype(img.dtype)
    return out


_eD_prev_labels = labels


import os as _eD_os
import bulletlabels as _eD_BL
from scipy.ndimage import binary_dilation as _eD_dil
_EDM_NEAR = 24      # px (half res): a sprite pixel counts for an enemy object when it lies this close to the object's predicted label
_EDM_PRED = 4       # px: the predicted label of an enemy object, grown (additive glow / burst pixels the ID pass does not show)
_eD_chk = {'n': 0, 'maxout': 0}


def _eD_apply(img, ops):
    for op in ops:
        if op[0] == 'hue': img = _eD_recolor(img, *op[1:])
        elif op[0] == 'neon':       # bright warm pixels of the red art (the hit rings / bolts / rectangles) -> the blue art's neon yellow-green
            im = img.astype(_eD_np.int16); m = (im[..., 0] >= op[1]) & (im[..., 1] >= op[2])
            img = img.copy(); img[m] = op[3]
    return img


def _eD_mask(t, sel, objs, cam, R, shape):
    """pixels (half res) that belong to the enemy objects `sel` of type t: the draws of the type's red-art sprites (codes 1, 3, 4, 5) near the objects'
    predicted labels, and the predicted labels themselves grown a few px"""
    Q = _eD_BL.labels(sel, cam, cut=0.01); qids, qlids = Q['ids'], Q['lids']
    keys = {_eD_BL._lid(o) for o in sel}
    ks = [k for k, l in enumerate(qlids) if l[0] in keys]
    P = _eD_np.isin(qids, ks) if ks else _eD_np.zeros(shape, bool)
    if not P.any(): return P
    C = sprite_pixels(t, False, R)
    spr = _eD_np.isin(C, (1, 3, 4, 5))
    return _eD_dil(P, iterations=_EDM_PRED) | (spr & _eD_dil(P, iterations=_EDM_NEAR))


_eD_prev_labels = labels


def labels(objs, cam, R, state=None, ended=(), img=None):
    sel = {}
    for o in objs:
        if o['t'] in _EDH and not o.get('own'): sel.setdefault(o['t'], []).append(o)
    if img is not None and sel:
        out = img; allm = _eD_np.zeros(img.shape[:2], bool)
        for t, L in sorted(sel.items()):
            try: m = _eD_mask(t, L, objs, cam, R, img.shape[:2])
            except Exception: continue
            if not m.any(): continue
            rec = _eD_apply(img, _EDH[t])
            if out is img: out = img.copy()
            out[m] = rec[m]; allm |= m
        if out is not img and _eD_os.environ.get('ED_ASSERT'):
            ch = (out != img).any(-1); _eD_chk['n'] += int(ch.sum()); _eD_chk['maxout'] = max(_eD_chk['maxout'], int((ch & ~allm).sum()))
            assert not (ch & ~allm).any(), 'recolour changed pixels outside the mask'
            print('ED_ASSERT frame: changed %d px, outside mask %d' % (ch.sum(), (ch & ~allm).sum()), flush=True)
        img = out
    return _eD_prev_labels(objs, cam, R, state=state, ended=ended, img=img)


_EDH[6000004] = [('hue', 285.0, 355.0, -139.0)]
_EDH[6000005] = [('hue', 285.0, 355.0, -139.0)]
# 460: neon (hit effect) = bright warm pixels; bulbs crimson -> navy
_EDH[6000018] = [('neon', 200, 100, (170, 235, 60)), ('hue', 300.0, 360.0, -115.0), ('hue', 0.0, 20.0, -115.0)]
# 390 Poco009 keyboards: blue keys (hue ~220) <-> magenta keys (hue ~300-330)
_EDH[6000351] = [('hue', 270.0, 345.0, -80.0, 0.85)]
_EDH[6000352] = [('hue', 270.0, 345.0, -80.0, 0.85)]


# ---------------------------------------------------------------------------------------------------------------------------------
# (3) POST_BODY hooks that apply to ENEMY shots only (ctx['own'] False): a hook is chained after the hooks already installed for the type.
def _eD_chain_post(t, fn):
    prev = POST_BODY.get(t)
    if prev is None: POST_BODY[t] = fn
    else: POST_BODY[t] = (lambda p, f: (lambda body, ctx: f(p(body, ctx), ctx)))(prev, fn)


def _eD_lowcv(cut):
    """enemy art draws the clip's body sprites more translucent than the blue art (Byron003's bubble rim: coverage 64-127 where the blue one is
    >= 128): its body-class sprite pixels (code 1, after the shared-sprite resolution) with coverage >= cut that touch the body are body too"""
    def f(body, ctx):
        if ctx['own'] or not body.any(): return body
        from scipy.ndimage import label as cc
        cand = (ctx['C'] == 1) & (ctx['cv'] >= cut) & ctx['region'] & ~body
        if not cand.any(): return body
        L, n = cc(cand | body, structure=_eD_np.ones((3, 3)))
        keep = _eD_np.unique(L[body]); keep = keep[keep > 0]
        return body | (_eD_np.isin(L, keep) & cand)
    return f


# 301 Byron003 (SnakeOilVillain) attack 6000866
_eD_chain_post(6000866, _eD_lowcv(72))


def _eD_pre_post(t, fn):
    """fn(body, ctx) applied BEFORE the hooks already installed for t (enemy shots only: fn is given the body first)"""
    prev = POST_BODY.get(t)
    if prev is None: POST_BODY[t] = fn
    else: POST_BODY[t] = (lambda p, f: (lambda body, ctx: p(f(body, ctx), ctx)))(prev, fn)


def _eD_near_pred(k):
    """enemy art: a launch burst drawn with the clip's own body shapes (Byron006: shapes 11 / 92 / 118 at the first ticks, a white-pink cloud
    round the bottle that the blue art does not show) is part of the ID-pass body. Only body pixels within k px of the predicted body count"""
    def f(body, ctx):
        if ctx['own'] or not body.any(): return body
        pb = ctx['pred_body']
        if not pb.any(): return body
        return body & _dilate(pb, k)
    return f


# 463 Byron006 (SnakeOilWindstock) attack 6000872
_eD_pre_post(6000872, _eD_near_pred(3))
