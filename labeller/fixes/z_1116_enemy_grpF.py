# enemy-label fix 2026-11-07, group grpF (13 single skins: 207 298 326 354 375 410 420 450 515 546 610 612 625).
# Installed as py/fixes/z_1116_enemy_grpF.py (after every z_1115_*, before zz_*). Global prefix _eF_ / _EF_. Only the types of these skins are touched
# (scoped by clip name / type id); blue (own) shapes, colours and labels are never changed (every rule below acts only on red-twin shape ids or on
# enemy-flagged objects), proved with pixel diffs of the own captures.
#
# (A) shape-id twins: several installed fixes find a bullet's pixels by the BLUE clip's sprite shape ids. The red clip (<blue>_red) has the same layout
#     with other shape ids. _a4_shape_pixels(R, swf, names) is wrapped for the clips listed in _EF_SCOPE only: it also asks for the red twin clip
#     (listed first, the blue polygon drawn later wins on an overlap) and reports every red-only shape id as its data-derived blue counterpart.
import numpy as _eF_np
import functools as _eF_ft

_EF_RED_SFX = '_red'


def _eF_walk(S, cid, seen=()):
    """{slot path: frozenset of shape ids} of a clip and its sub-clips over all frames (memoised per clip)"""
    key = (id(S), cid)
    if key in _EF_WALK: return _EF_WALK[key]
    if cid in seen: return {}
    out = {}; c = S.clips[cid]
    for f in range(max(1, c['nframes'])):
        for slot, child, cm, cc, bl in S.frame(cid, f):
            if child in S.shapes: out.setdefault((slot,), set()).add(child)
            elif child in S.clips:
                for p, v in _eF_walk(S, child, seen + (cid,)).items(): out.setdefault((slot,) + p, set()).update(v)
    out = {p: frozenset(v) for p, v in out.items()}
    _EF_WALK[key] = out
    return out


_EF_WALK = {}


def _eF_box(S, sid):
    a = _eF_np.concatenate([_eF_np.asarray(xy, _eF_np.float64).reshape(-1, 2) for tex, uv, xy in S.shapes[sid]])
    return a.min(0), a.max(0)


def _eF_iou(a, b):
    lo = _eF_np.maximum(a[0], b[0]); hi = _eF_np.minimum(a[1], b[1]); i = _eF_np.prod(_eF_np.maximum(0, hi - lo))
    u = _eF_np.prod(a[1] - a[0]) + _eF_np.prod(b[1] - b[0]) - i
    return float(i / u) if u > 0 else 0.0


@_eF_ft.lru_cache(maxsize=None)
def _eF_twin_map(swf, blue, red):
    """{red-only shape id: blue shape id} of the clips blue / red of swf (empty when either is missing)"""
    S = PD.sc(swf); cb = S.exports.get(blue); cr = S.exports.get(red)
    if cb is None or cr is None: return {}
    wb = _eF_walk(S, cb); wr = _eF_walk(S, cr)
    ids_b = set().union(*wb.values()) if wb else set(); ids_r = set().union(*wr.values()) if wr else set()
    votes = {}
    for p in set(wb) & set(wr):
        B = sorted(wb[p]); Rr = sorted(wr[p])
        if len(B) != len(Rr): continue
        for x, y in zip(B, Rr):
            if y not in ids_b and x != y: votes.setdefault(y, {}).setdefault(x, 0); votes[y][x] += 1
    out = {}
    only_b = sorted(ids_b - ids_r)           # blue shapes the red clip does not draw: the only possible counterparts
    for y in sorted(ids_r - ids_b):
        yb = _eF_box(S, y); sc = sorted(((_eF_iou(yb, _eF_box(S, x)), x) for x in only_b), reverse=True)
        cand = None
        if sc and sc[0][0] >= 0.8 and (len(sc) < 2 or sc[1][0] < 0.8): cand = sc[0][1]      # same local box: the same sprite part
        if cand is None and y in votes:                                                       # same place in the display tree
            x = max(votes[y], key=lambda k: votes[y][k])
            if x in only_b and _eF_iou(yb, _eF_box(S, x)) >= 0.3: cand = x
        if cand is None and sc and sc[0][0] >= 0.5 and (len(sc) < 2 or sc[1][0] < 0.7 * sc[0][0]):   # a similar box of similar area (Edgar's strip frames)
            x = sc[0][1]; ay = _eF_np.prod(yb[1] - yb[0]); ax = _eF_np.prod(_eF_box(S, x)[1] - _eF_box(S, x)[0])
            if min(ay, ax) >= 0.7 * max(ay, ax): cand = x
        if cand is not None: out[y] = cand
    return out


@_eF_ft.lru_cache(maxsize=None)
def _eF_plan(swf, names):
    """(names to ask for: red twins first, the shape-id LUT or None) for a call of _a4_shape_pixels"""
    S = PD.sc(swf); ex = S.exports; names = tuple(names)
    pairs = []      # (blue, red)
    for n in names:
        if not isinstance(n, str): continue
        if n.endswith(_EF_RED_SFX):
            b = n[:-len(_EF_RED_SFX)]
            if b in ex: pairs.append((b, n))
        elif n + _EF_RED_SFX in ex: pairs.append((n, n + _EF_RED_SFX))
    # types whose red clip is not '<blue>_red' (projectiles_skin table)
    try:
        for t, T in projtab.table().items():
            if T['file'] == swf and T['blue'] and T['red'] and T['blue'] != T['red'] and (T['blue'] in names or T['red'] in names):
                pairs.append((T['blue'], T['red']))
    except Exception: pass
    pairs = list(dict.fromkeys(p for p in pairs if p[0] in ex and p[1] in ex))
    if not pairs: return names, None
    extra = [r for b, r in pairs if r not in names] ; extra += [b for b, r in pairs if b not in names and b not in extra]
    ask = tuple(dict.fromkeys(tuple(extra) + names))
    blue_ids = set(); lut = None
    for b, r in pairs:
        blue_ids |= {s for s, k in _clip_shapes(S, ex[b])}
    lut = _eF_np.arange(65536, dtype=_eF_np.int32); n_ren = 0
    for b, r in pairs:
        for y, x in _eF_twin_map(swf, b, r).items():
            if y in blue_ids or y + 1 >= 65536: continue
            if lut[y + 1] not in (y + 1, x + 1): continue      # two blue targets for one red shape: keep the first
            lut[y + 1] = x + 1; n_ren += 1
    return ask, (lut if n_ren else None)



_EF_SCOPE = {'bo_006_atk_projectile', 'pearl_004_atk_projectile_k2'}     # blue clip names whose calls get the red-twin treatment (grown per skin below)


def _eF_scoped(names):
    for n in names:
        if isinstance(n, str) and (n in _EF_SCOPE or (n.endswith('_red') and n[:-4] in _EF_SCOPE)): return True
    return False


_eF_prev_sp = _a4_shape_pixels


def _a4_shape_pixels(R, swf, names):
    if not _eF_scoped(names): return _eF_prev_sp(R, swf, names)
    try: ask, lut = _eF_plan(swf, tuple(names))
    except Exception: return _eF_prev_sp(R, swf, names)
    out = _eF_prev_sp(R, swf, ask)
    if lut is not None and out.size: out = lut[out]
    return out


# (B) red trail clips: trail-body types label the drawn TRAIL sprites (sprites(t, own)), and _trail_clips(t) lists only the BLUE emitters' clips (the
#     red emitters carry no movie clips in wave._emitters). The red art uses the clips <name with _red before the trailing _NN> / <name>_red
#     (buzz_006_ulti_trail_ice_red_01, buzz_006_atk_end_red_01, ...): sprites(t, own=False) of the listed types also gets the red twins of
#     the blue trail clips, with the class the blue clip has. Own (own=True) sprites are never touched.
import re as _eF_re
_EF_TRAIL_TYPES = {6000992, 6000490}      # 354 Buzz006Projectile (ice cluster), 450 Frank009UltiProjectile (super)


def _eF_red_twin(S, name):
    for c in (_eF_re.sub(r'_(\d+)$', r'_red_\1', name), name + '_red'):
        if c != name and c in S.exports: return c
    return None


_eF_prev_sprites = sprites


def sprites(t, own):
    out = _eF_prev_sprites(t, own)
    if own or t not in _EF_TRAIL_TYPES: return out
    out = {k: list(v) for k, v in out.items()}
    tb = trail_body(t)
    for swf, name in _trail_clips(t):
        S = PD.sc(swf); red = _eF_red_twin(S, name)
        if red is None: continue
        sk = (name.startswith(TRAIL_SKIP[t]) if t in TRAIL_SKIP else False) or (t not in ZONE and any(k in name.lower() for k in TRAIL_FX))
        for sid, k in _clip_shapes(S, S.exports[red]):
            tex, uv, xy = S.shapes[sid][k]
            out.setdefault((swf, tex), []).append((_eF_np.asarray(uv, _eF_np.float64) / 65535.0, 'trail', 'body' if tb and not sk else 'over'))
    return out


# (C) coverage of red sprites: some red twin clips draw their sprites with half the alpha of the blue ones (354: the red ice sparkles peak at cv 127, the blue
#     ones at 255), so every tuned coverage cut (CUT = 128) that the blue sparkles pass is never reached by the red ones (no label at all at the start of the
#     shot). For the listed types ENEMY objects (o['own'] False) only: the coverage of the pixels of their own sprites (listed sprite codes of
#     sprite_pixels) is multiplied by the factor (clipped to 255) before the whole chain runs, the same cut then keeps the same pixels. Own frames: R untouched.
_EF_CVBOOST = {6000992: ((3,), 2.0)}     # type -> (sprite codes, factor)
_eF_prev_labels_cv = labels


def labels(objs, cam, R, state=None, ended=(), img=None):
    ts = {o['t'] for o in objs if not o['own'] and o['t'] in _EF_CVBOOST}
    if ts:
        cv = R['cv'].copy(); hit = False
        for t in ts:
            codes, fac = _EF_CVBOOST[t]
            C = sprite_pixels(t, False, R)
            m = _eF_np.isin(C, codes)
            if m.any():
                M = _eF_np.repeat(_eF_np.repeat(m, 2, 0), 2, 1)[:cv.shape[0], :cv.shape[1]]
                cv[M] = _eF_np.minimum(255, cv[M].astype(_eF_np.float32) * fac).astype(cv.dtype); hit = True
        if hit:
            R = dict(R); R['cv'] = cv
    return _eF_prev_labels_cv(objs, cam, R, state=state, ended=ended, img=img)


# (D) 375 CrowHarvest (Crow007Projectile 6000325), attack-90 row: z_1104_gale_crow._gc_fire_body opens the fire-dagger body with a 5 x 5 square and keeps only
#     the LARGEST connected part (spur removal). At the first ticks of a red shot the three feathers are narrow and separate: the opening deletes
#     most of them and the label is one feather instead of three (blue shots are big merged flames at that phase). ENEMY objects only: when the
#     spur removal would take away more than 40 % of the (flash-free) body, the flash-free body is kept as it is (the same fallback the rule has
#     when the opening is empty).
_eF_prev_375 = POST_BODY.get(6000325)


def _eF_crow_body(body, ctx):
    res = _eF_prev_375(body, ctx) if _eF_prev_375 else body
    if ctx['o'].get('own', True) or not body.any(): return res
    pre = _crow_flash(body, ctx)
    if pre.any() and res.sum() < 0.6 * pre.sum(): return pre
    return res


POST_BODY[6000325] = _eF_crow_body


# (E) 625 LuchadorMHA (Primo014Projectile 6000410), enemy shots: z_1104_barrel_luchador._lu625_flames keeps of the body only the cyan flame (g >= 140,
#     b >= 180, r <= 170) and the fist mesh found by its texture hash 20e53de0a7693f5d (r <= 150); the red fist is a mesh with another hash
#     (afe57a465f471e46, same program) and a pink glow, so for an enemy shot nothing passed, the 30 % guard returned the WHOLE body and the label
#     held the yellow / orange torso streak (twice the blue label's area). ENEMY objects only: the same rule with the red mesh hash and a pink flame test
#     (additive glow: b >= 120, r >= 150, b >= g - 30; the yellow / orange streak has b <= 100). Own objects take the installed rule unchanged.
_EF_625_MESH_RED = 'afe57a465f471e46'
_eF_prev_625 = POST_BODY.get(6000410)


def _eF_primo_body(body, ctx):
    if ctx['o'].get('own', True): return _eF_prev_625(body, ctx)
    from scipy.ndimage import binary_fill_holes, binary_closing, binary_opening, label as _cc
    if _lu625_prev is not None: body = _lu625_prev(body, ctx)
    if not body.any(): return body
    img = ctx['img']; ids_h = ctx['ids_h']; R = ctx['R']
    if img is None: return body
    mesh_ids = [d for d, v in R['draws'].items() if v.get('th') == _EF_625_MESH_RED and v.get('blend') == 0]
    mesh = _eF_np.isin(ids_h, mesh_ids) if mesh_ids else _eF_np.zeros(body.shape, bool)
    r, g, b = img[..., 0].astype(int), img[..., 1].astype(int), img[..., 2].astype(int)
    flame = (b >= 120) & (r >= 150) & (b >= g - 30)
    keep = body & (flame | mesh)
    if keep.sum() < 0.3 * body.sum(): return body
    p = 8
    K = binary_opening(_eF_np.pad(keep, p), structure=_lu625_disk(2))
    L, n = _cc(K)
    if n > 1:
        sz = _eF_np.bincount(L.ravel())[1:]
        K = _eF_np.isin(L, 1 + _eF_np.nonzero(sz >= 0.25 * sz.max())[0])
    K = binary_fill_holes(binary_closing(K, structure=_lu625_disk(2)))[p:-p, p:-p]
    return K & body


POST_BODY[6000410] = _eF_primo_body
