# Skin 453 JetpackGirlMHA, round 3 (2026-10-28), finding pair 8496 (attack 90 deg, tick 309: right crystal's top point outside).
# Type 6001174 (JetpackGirl009Projectile, clip janet_009_atk_projectile in sc/effects_brawler_janet.sc): the projectiles are
# crystals. Each crystal is two of the clip's sprites: a translucent body fill and, in front, a bright rim ring; the soft cyan
# glow (big disc / crescent: shapes 0 / 1 of the same clip) is drawn around / over them, and over parts of the crystal it
# is the front-most draw of the ID pass. Rounds 1-2 (fix.py / fix2.py: draw candidates, colour cut of the glow) tried to
# separate crystal and glow by pixels: the top point stayed out or glow came in.
# Here the crystal's own outline comes from the client's data. The clip has four crystal variants, each a body shape + a rim
# shape (46/47, 48/49, 50/51, 52/53; the rim sprite is a ring = the crystal's silhouette). The ID pass shows, per pixel, the
# atlas uv of the front-most draw. The visible body / rim pixels of one crystal (found by a robust RANSAC affine fit in the
# clip's local xy, error < 0.1 px on a clean instance; body and rim share one placement, measured to 0.4 px) give the screen
# transform of the whole sprite, also where a glow disc or a nearer crystal hides part of it in the ID pass. The label is the
# atlas alpha of the rim sprite rendered through that transform (3 x 3 sub-samples, alpha >= ALPHA53), ring closed, filled solid,
# spurs opened: the crystal's own outline, top point in, glow out. Several fits of one crystal (its lower lip is a second
# fit) are merged; a nearer crystal (higher draw id) cuts a farther one.
# One crystal per object: labels() is called twice for frames with this type (pass 1 records each object's old label, then a
# one-to-one assignment: largest overlap, leftovers to the nearest object still without one; fading trail stones, coverage
# < GHOST53, are never given out by proximity; pass 2 applies it). An object without a crystal keeps its old label (a
# projectile is never lost). The ended rule (drop_ended_parts) still runs after this hook, as before.
# Both teams (2026-10-28 red-shot review): the red clip janet_009_atk_projectile_red has the same shape ids (0, 1, 46-53, 61)
# and the same atlas regions as the blue one (the enemy tint is a colour transform, not other art), so the shape-id / atlas
# based fit finds the enemy crystals exactly as the own ones (checked on W/runs/gc/en_453 against sw2_453; no colour test and no
# own-only filter is used anywhere here). Naming the red clip too in PROJ53 makes that explicit and changes nothing else.
# 2026-10-28, round 4 (fill1028, the user's decisions of 10-28; this file replaces z_1028_s453.py, same output where nothing is hidden):
#  (1) the crystal that stays next to Janet at ticks 307-311 (objects 2000018 / 19, sw2_453): the tracker hides an object whose logic
#      projectile finished with a state other than 1 / 3 (here 4: it stopped at tick 307, its hit flash = the white ball), so
#      BL.labels gives it no region and the body of that object was empty. The crystal is still drawn, from the same sprites. A solid
#      crystal (coverage >= GHOST53) that no object claims goes to the hidden object (finished not in 0 / 1 / 3) whose own drawn path
#      (o['path'], projected) passes within HELD53 px of it (the nearest one): it gets its own crystal outline like the others.
#      Objects with finished 1 / 3 (reached the range: end animation, the user's 10-23 rule) are never given a leftover crystal.
#  (2) hidden parts are filled (the user: "the label should fill in the missing part, so the model learns the exact bullet shape"):
#      - a partly hidden crystal already gets its full silhouette (the affine fit places the whole rim sprite: nothing is cut at a
#        glow disc / flash); where two crystals overlap the nearer one keeps the shared pixels (the label map has one owner per pixel);
#      - an object that has a label but no fitted crystal (its crystal is under a glow disc in the ID pass: only a few pixels of its
#        sprites show, or none) gets the crystal placed from its own sprite geometry (_hidden53): the visible sprite pixels give the
#        variant and pin the placement, a start affine of the clip (a crystal of the same frame or of this object a tick before) is
#        turned / scaled / moved so that the rim ring lies on the rim ridge of the frame (the glow is translucent: the frame shows
#        the crystal); with no visible pixel at all and no clear ridge (under an opaque flash) the crystal is placed at the object's
#        predicted position (anchor) with its last (else a sibling's) geometry (_predicted53; an approximation, ~5 px).
#        Never for an object without a label today: a projectile whose logic ended stays unlabelled.
from scipy.ndimage import (binary_fill_holes as _fh53, label as _cc53, map_coordinates as _mc53)
from PIL import Image as _Im53, ImageDraw as _ID53
VARIANTS53 = ((46, 47), (48, 49), (50, 51), (52, 53))   # (body shape, rim shape) of the four crystal variants
PROJ53 = ('sc/effects_brawler_janet.sc', ('janet_009_atk_projectile', 'janet_009_atk_projectile_red'))
MIN53 = 12                             # fewest rim / body pixels that fit an instance
SPREAD53 = 0.05                        # their spread (convex hull area in local xy) as a share of the rim shape's area: fewer = unstable fit
TOL53 = 0.9                            # px: an inlier of an affine fit
ALPHA53 = 0.5
_CACHE53 = {'R': None, 'out': None}   # the frame's instances, kept for the frame object R itself (never by id(): ids are reused)

def _fit53(loc, xy, rng):
    """robust affine local -> screen (3-point sampling, then refit); returns (G, inlier mask) or None"""
    n = len(loc); A = np.c_[loc, np.ones(n)]; best = None
    for _ in range(100):
        i = rng.choice(n, 3, replace=False)
        try: g = np.linalg.solve(A[i], xy[i])
        except np.linalg.LinAlgError: continue
        inl = np.abs(A @ g - xy).max(1) < TOL53
        if best is None or inl.sum() > best.sum(): best = inl
    if best is None or best.sum() < MIN53: return None
    for _ in range(3):
        g = np.linalg.lstsq(A[best], xy[best], rcond=None)[0]
        best = np.abs(A @ g - xy).max(1) < TOL53
    if best.sum() < MIN53 or abs(np.linalg.det(g[:2])) < 1e-3: return None
    from scipy.spatial import ConvexHull
    try: ha = ConvexHull(loc[best]).volume
    except Exception: return None
    if ha < SPREAD53 * 70 * 62: return None   # the rim shape spans about 77 x 70 local units
    return g, best

_UX53 = {}
def _local53():
    """{shape id: affine uv -> the clip's local xy} of the crystal shapes (the body and the rim of one crystal share one
    placement in local xy: measured, the same screen transform to 0.4 px)"""
    if not _UX53:
        swf, names = PROJ53; S = PD.sc(swf)
        for sid, k in sorted(_clip_shapes(S, S.exports[names[0]])):
            tex, uv, xy = S.shapes[sid][k]; uv = np.asarray(uv, np.float64) / 65535.0
            _UX53[sid] = np.linalg.lstsq(np.c_[uv, np.ones(len(uv))], np.asarray(xy, np.float64), rcond=None)[0]
    return _UX53

_AT53 = {}
def _mask53(g, rim_sh, H, W):
    """the crystal's silhouette on screen: the atlas alpha of its rim sprite rendered through the affine g (local -> screen),
    ring closed, filled solid, spurs opened; None if (almost) nothing"""
    swf, names = PROJ53
    if not _AT53:
        (key, pl), = _a4_polys(swf, names).items(); _AT53['P'] = dict(pl); _AT53['a'] = tex_alpha(swf, key[1]).astype(np.float64) / 255.0
    P = _AT53['P']; alpha = _AT53['a']; ah, aw = alpha.shape; UX = _local53()
    Hm = np.zeros((3, 3)); Hm[:, :2] = UX[rim_sh]; Hm[2, 2] = 1          # [u v 1] Hm = [lx ly 1]
    g3 = np.c_[g, [0, 0, 1]]                                              # [lx ly 1] g3 = [x y 1]
    tot = np.linalg.inv(g3) @ np.linalg.inv(Hm)                           # [x y 1] tot = [u v 1]
    Pu = np.c_[P[rim_sh], np.ones(len(P[rim_sh]))]
    poly = ((Pu @ Hm) @ g3)[:, :2]
    lo = np.floor(poly.min(0)).astype(int) - 2; hi = np.ceil(poly.max(0)).astype(int) + 3
    x0, y0 = max(0, lo[0]), max(0, lo[1]); x1, y1 = min(W, hi[0]), min(H, hi[1])
    if x1 <= x0 or y1 <= y0: return None
    I = _Im53.new('L', (W, H), 0); dr = _ID53.Draw(I)
    for k in range(len(poly) - 2): dr.polygon([tuple(p) for p in poly[k:k + 3]], fill=1)
    tri = np.asarray(I, bool)
    gy, gx = np.mgrid[y0:y1, x0:x1]
    a = np.zeros(gx.shape)   # 3 x 3 sub-samples per pixel, max: a small crystal's thin rim ring must not fall between samples
    for oy in (-1 / 3, 0, 1 / 3):
        for ox in (-1 / 3, 0, 1 / 3):
            q = np.c_[gx.ravel() + ox, gy.ravel() + oy, np.ones(gx.size)] @ tot
            a = np.maximum(a, _mc53(alpha, [q[:, 1] * ah - 0.5, q[:, 0] * aw - 0.5], order=1, mode='nearest').reshape(gx.shape))
    mk = np.zeros((H, W), bool); mk[y0:y1, x0:x1] = a >= ALPHA53
    mk &= tri
    if mk.sum() < MIN53: return None
    mk = _fh53(_close(mk, 1))
    from scipy.ndimage import binary_opening as _bo53
    op = _bo53(mk, iterations=1); Lm, nm = _cc53(op)       # spurs (rim lines of the atlas next to the sprite) opened away
    if nm: mk = Lm == (1 + np.argmax(np.bincount(Lm.ravel())[1:]))
    return mk

def _instances53(ctx):
    """[(draw id, full crystal mask, support px)] of every crystal instance with visible rim or body pixels, back to front"""
    R = ctx['R']
    if _CACHE53['R'] is R: return _CACHE53['out']
    swf, names = PROJ53; ids_h = ctx['ids_h']; uv = R['uv']
    SP = _a4_shape_pixels(R, swf, names); UX = _local53()
    (key, pl), = _a4_polys(swf, names).items(); P = dict(pl); tex = key[1]
    alpha = tex_alpha(swf, tex).astype(np.float64) / 255.0; ah, aw = alpha.shape
    H, W = ids_h.shape; rng = np.random.RandomState(7); out = []
    for body_sh, rim_sh in VARIANTS53:
        m = np.isin(SP, [body_sh + 1, rim_sh + 1])
        ys, xs = np.nonzero(m)
        if len(ys) < MIN53: continue
        sp = SP[ys, xs] - 1; u = uv[ys, xs].astype(np.float64); did = ids_h[ys, xs]
        loc = np.zeros((len(ys), 2))
        for sh in (body_sh, rim_sh):
            q = sp == sh; loc[q] = np.c_[u[q], np.ones(q.sum())] @ UX[sh]
        xy = np.c_[xs, ys].astype(np.float64); alive = np.ones(len(ys), bool)
        for _ in range(8):
            if alive.sum() < MIN53: break
            r = _fit53(loc[alive], xy[alive], rng)
            if r is None: break
            g, inl = r; idx = np.nonzero(alive)[0]; sel = idx[inl]; alive[sel] = False
            if (sp[sel] == rim_sh).sum() + (sp[sel] == body_sh).sum() < MIN53: continue
            mk = _mask53(g, rim_sh, H, W)
            if mk is None: continue
            out.append((int(np.median(did[sel])), mk, len(sel), float(np.median(R['cv'][::2, ::2][ys[sel], xs[sel]])), dict(var=(body_sh, rim_sh), g=g.copy())))
    out.sort(key=lambda t: t[0])
    _CACHE53['R'] = R; _CACHE53['out'] = out
    return out

_ST53 = {'mode': 'off', 'old': {}, 'assign': {}, 'objs': {}, 'ctx': None}
GHOST53 = 100.0   # a crystal drawn with a median coverage under this (a translucent trail stone fading out) is never given to an object by proximity
NEAR53 = 60.0     # px: an unmatched crystal goes to the nearest object still without one if its label's centre is this close
HELD53 = 16.0     # px: a leftover solid crystal belongs to an object the tracker no longer draws if it sits this close to the object's own path
_OG53 = {}        # {object id: (tick, variant index, affine g, g origin - anchor)}: the last crystal of each object (a start for a hidden one)

def _fix_6001174(body, ctx):
    oid = ctx['o']['id']
    if _ST53['mode'] == 'record':
        _ST53['old'][oid] = body.copy(); _ST53['objs'][oid] = ctx['o']; _ST53['ctx'] = ctx; return body
    if _ST53['mode'] == 'apply':
        m = _ST53['assign'].get(oid)
        if m is not None and m.any(): return m       # also for an object the tracker does not draw (a crystal held in place)
    return body

POST_BODY[6001174] = _fix_6001174

def _anchor53(cam, o):
    q = cam @ np.r_[np.asarray(o['anchor'], np.float64), 1.0]
    return np.array([(q[0] / q[3] * 0.5 + 0.5) * 1200, (1 - (q[1] / q[3] * 0.5 + 0.5)) * 540])

def _rot53(a): c, s = np.cos(a), np.sin(a); return np.array([[c, s], [-s, c]])

def _place53(F, tm, EL, ES, starts, t0):
    """a hidden crystal: affine local -> screen = a start affine (a crystal of the same frame / of this object earlier: the clip's own
    geometry) turned / scaled / moved, by the rim-ridge score of the frame, with the visible pixels of its sprites (EL local, ES screen)
    pinned to where they are; (score, g)"""
    best = (-1e9, None)
    for go in starts:
        for flip in (0, 1):
            lin0 = go[:2] @ np.diag([1.0, -1.0]) if flip else go[:2]
            for phi0 in (-0.5, 0.0, 0.5):
                R0 = lin0 @ _rot53(phi0)
                def mk(x): return np.vstack([R0 @ _rot53(x[0]) * x[1], [x[2], x[3]]])
                def cost(x):
                    g = mk(x); pen = 20 * max(0.0, abs(np.log(max(x[1], 1e-3))) - 0.3) + 4 * max(0.0, abs(x[0]) - 0.5)
                    if len(EL): pen += 3.0 * np.maximum(np.abs(np.c_[EL, np.ones(len(EL))] @ g - ES).max(1) - 1.0, 0).mean()
                    return -_score53(g, tm, F) + pen
                tr0 = (ES - EL @ R0).mean(0) if len(EL) else t0 - 0 * R0[0]
                x0 = np.array([0.0, 1.0, tr0[0], tr0[1]])
                sim = np.vstack([x0] + [x0 + np.eye(4)[k] * [0.15, 0.08, 1.5, 1.5][k] for k in range(4)])
                r = _min53(cost, x0, method='Nelder-Mead', options=dict(xatol=0.003, fatol=0.005, maxiter=400, initial_simplex=sim))
                if -r.fun > best[0]: best = (-r.fun, mk(r.x))
    return best

HID_SCORE53 = 12.0   # least rim-ridge score for a hidden crystal found from the frame alone (no visible sprite pixel of it)

def _hidden53(R, img, cam, o, inst, covered, tick):
    """mask of the crystal of object o that the ID pass does not show (glow disc / flash over it) or None"""
    H, W = R['id'][::2, ::2].shape; F = _feat53(img); an = _anchor53(cam, o)
    swf, names = PROJ53; SP = _a4_shape_pixels(R, swf, names); uv = R['uv']; UX = _local53()
    yy, xx = np.mgrid[0:H, 0:W]
    near = np.hypot(xx - an[0], yy - an[1]) < 30
    cand = {}
    for vi, (bs, rs) in enumerate(VARIANTS53):
        m = np.isin(SP, [bs + 1, rs + 1]) & near & ~covered
        if m.sum() >= 4: cand[vi] = m
    starts = [e[4]['g'] for e in inst]
    if o['id'] in _OG53 and 0 < tick - _OG53[o['id']][0] <= 8: starts = [_OG53[o['id']][2]] + starts
    if not starts: return None
    offs = [(e[4]['g'][2] - an) for e in inst if np.hypot(*(e[4]['g'][2] - an)) < 25]
    t0 = an + (np.mean(offs, 0) if offs else np.zeros(2))
    if cand:                                         # the sprites that show: which variant, and where they are
        vi = max(cand, key=lambda v: cand[v].sum()); bs, rs = VARIANTS53[vi]; m = cand[vi]
        ys, xs = np.nonzero(m); sp = SP[ys, xs] - 1; u = uv[ys, xs].astype(np.float64); EL = np.zeros((len(ys), 2))
        for sh in (bs, rs):
            q = sp == sh; EL[q] = np.c_[u[q], np.ones(q.sum())] @ UX[sh]
        sc, g = _place53(F, _tmpl53(rs, bs), EL, np.c_[xs, ys].astype(np.float64), starts[:5], t0)
        res = (sc, g, vi, 0.0)
    else:                                            # nothing of it in the ID pass: the frame's rim ridge decides, over all variants
        out = []
        for vi, (bs, rs) in enumerate(VARIANTS53):
            sc, g = _place53(F, _tmpl53(rs, bs), np.zeros((0, 2)), np.zeros((0, 2)), starts[:4], t0)
            if g is not None and np.hypot(*(g[2] - t0)) < 16: out.append((sc, g, vi))
        if not out: return None
        out.sort(key=lambda r: -r[0]); res = out[0] + (out[0][0] - (out[1][0] if len(out) > 1 else 0.0),)
        if res[0] < HID_SCORE53 + 6 or res[3] < 2.0: return _predicted53(o, an, inst, tick, H, W)
    sc, g, vi = res[:3]
    if g is None or sc < HID_SCORE53: return _predicted53(o, an, inst, tick, H, W)
    mk = _mask53(g, VARIANTS53[vi][1], H, W)
    if mk is None: return None
    _OG53[o['id']] = (tick, vi, g, g[2] - an)
    return mk

def _predicted53(o, an, inst, tick, H, W):
    """nothing of the crystal shows (not in the ID pass, no clear rim ridge in the frame: under an opaque flash): the crystal's own
    geometry at the object's predicted position: its last crystal (<= 8 ticks back) else the same frame's nearest one of the shot,
    moved to the predicted position (anchor + that crystal's offset to its own anchor)"""
    h = _OG53.get(o['id'])
    if h is not None and 0 < tick - h[0] <= 8: vi, g, off = h[1], h[2].copy(), h[3]
    elif inst:
        e = min(inst, key=lambda e: np.hypot(*(e[4]['g'][2] - an))); vi = VARIANTS53.index(e[4]['var']); g = e[4]['g'].copy(); so = [v[3] for v in _OG53.values() if v[0] == tick]; off = np.mean(so, 0) if so else np.zeros(2)
    else: return None
    g[2] = an + off
    return _mask53(g, VARIANTS53[vi][1], H, W)

def _assign53(R, old):
    """{object id: crystal mask}: one crystal per object. Overlap with the label the object had decides (largest first, one to
    one); a crystal left over goes to the nearest object that still has no crystal (its old label's centre within NEAR53); a solid
    leftover crystal on the path of an object the tracker no longer draws (a crystal held in place) is that object's; an object with
    a label but no crystal of the ID pass gets its crystal placed from the sprite geometry (_hidden53)"""
    from scipy.optimize import linear_sum_assignment
    ctx = _ST53['ctx']; objs = _ST53['objs']; cam = ctx['cam'] if ctx else None; img = ctx['img'] if ctx else None
    tick = (ctx['state'] or {}).get('tick', 0) if ctx else 0
    inst = _instances53(dict(R=R, ids_h=R['id'][::2, ::2]))
    if not inst and not any(old[i].any() for i in old): return {}
    # one crystal fitted twice (the leftover pixels of its other sprite layer, e.g. the lower lip of the crystal, fitted again):
    # the fits of which the smaller lies > 85 % inside the other are one crystal: their union, the draw of the better supported
    inst = [list(t) for t in inst]; i = 0
    while i < len(inst):
        merged = False
        for j in range(len(inst)):
            if j != i and (inst[i][1] & inst[j][1]).sum() > 0.85 * min(inst[i][1].sum(), inst[j][1].sum()):
                ka, kb = (i, j) if inst[i][2] >= inst[j][2] else (j, i)
                inst[ka][1] = inst[ka][1] | inst[kb][1]; inst[ka][2] += inst[kb][2]; inst[ka][3] = max(inst[ka][3], inst[kb][3]); del inst[kb]; merged = True; break
        if not merged: i += 1
    inst = [tuple(t) for t in sorted(inst, key=lambda t: t[0])]
    # nearer crystals (drawn later) cut a farther one (one owner per pixel of the label map)
    masks = []
    for i, (d, f, _, _, _) in enumerate(inst):
        f = f.copy()
        for d2, f2, _, _, _ in inst[i + 1:]:
            if d2 > d: f &= ~f2
        masks.append(f)
    keep = [j for j, f in enumerate(masks) if f.sum() >= MIN53]
    cvm = [t[3] for t in inst]
    ids = [i for i in old if old[i].any()]
    out = {}; used = set(); done = set()
    if ids and keep:
        ov = np.array([[(old[i] & masks[j]).sum() for j in keep] for i in ids], np.float64)
        ok = np.array([[ov[a, b] >= 20 and ov[a, b] >= 0.3 * min(old[ids[a]].sum(), masks[keep[b]].sum()) for b in range(len(keep))] for a in range(len(ids))])
        if ok.any():
            r, c = linear_sum_assignment(-(ov * ok))
            for a, b in zip(r, c):
                if ok[a, b]: out[ids[a]] = masks[keep[b]]; used.add(b); done.add(a)
        cen = lambda m: np.array(np.nonzero(m)).mean(1)
        pairs = sorted((np.linalg.norm(cen(old[ids[a]]) - cen(masks[keep[b]])), a, b) for a in range(len(ids)) if a not in done
                       for b in range(len(keep)) if b not in used and cvm[keep[b]] >= GHOST53)
        for dist, a, b in pairs:
            if dist <= NEAR53 and a not in done and b not in used:
                out[ids[a]] = masks[keep[b]]; used.add(b); done.add(a)
    # remember each object's crystal (a start for the frames where it is hidden)
    if cam is not None:
        for oid, m in out.items():
            for j, e in enumerate(inst):
                if masks[j] is m and oid in objs: _OG53[oid] = (tick, VARIANTS53.index(e[4]['var']), e[4]['g'], e[4]['g'][2] - _anchor53(cam, objs[oid]))
    # a solid crystal nobody has: the object the tracker no longer draws whose own path passes there (held in place)
    if cam is not None:
        for b in keep:
            if b in used or cvm[b] < GHOST53: continue
            cm = np.array(np.nonzero(masks[b])).mean(1)[::-1]; bestd = None
            for oid, o in objs.items():
                if oid in out or o.get('drawn', True) or not o.get('path') or o.get('finished', 0) in (0, 1, 3): continue   # finished 1 / 3 = reached its range: end animation, never labelled
                dmin = min(np.hypot(*(np.array(_proj(cam, x, y, 100)) - cm)) for x, y in o['path'])
                if dmin <= HELD53 and (bestd is None or dmin < bestd[0]): bestd = (dmin, oid)
            if bestd: out[bestd[1]] = masks[b]; used.add(b)
    # an object with a label and no crystal: place the crystal from the sprite geometry
    if cam is not None and img is not None and HIDDEN53:
        covered = np.zeros(masks[0].shape if masks else R['id'][::2, ::2].shape, bool)
        for m in masks: covered |= _dilate(m, 1)
        for oid in ids:
            if oid in out or not objs[oid].get('drawn', True): continue
            try: mk = _hidden53(R, img, cam, objs[oid], inst, covered, tick)
            except Exception: mk = None
            if mk is not None:
                for m in masks: mk = mk & ~m          # visible crystals stay in front of it
                if mk.sum() >= MIN53: out[oid] = mk
    return out

HIDDEN53 = True
_prev_labels53 = labels

def labels(objs, cam, R, state=None, ended=(), img=None):
    if not any(o['t'] == 6001174 for o in objs): return _prev_labels53(objs, cam, R, state=state, ended=ended, img=img)
    _ST53.update(mode='record', old={}, assign={}, objs={}, ctx=None)
    try:
        _prev_labels53(objs, cam, R, state=state, ended=ended, img=img)      # pass 1: each object's old label
        _ST53['assign'] = _assign53(R, _ST53['old']); _ST53['mode'] = 'apply'
        return _prev_labels53(objs, cam, R, state=state, ended=ended, img=img)
    finally: _ST53['mode'] = 'off'


# ---------------------------------------------------------------------------------------------------------------------------
# 2026-10-28 (user decision 2): a crystal hidden in the ID pass (glow disc / flash drawn over it) is placed from its own sprite
# geometry: the affine local -> screen of the rim sprite is searched so that the rim ring sits on the bright rim ridge of the frame
# (the glow / flash is translucent: the frame shows the crystal where the ID pass shows only the glow draw).
from scipy.ndimage import gaussian_filter as _gf53, binary_erosion as _be53
from scipy.optimize import minimize as _min53
_TM53 = {}
def _tmpl53(rim_sh, body_sh):
    """local xy of the rim ring pixels / of the crystal's interior (inside the ring, away from it), from the atlas"""
    if (rim_sh, body_sh) in _TM53: return _TM53[(rim_sh, body_sh)]
    swf, names = PROJ53; (key, pl), = _a4_polys(swf, names).items(); P = dict(pl); tex = key[1]
    alpha = tex_alpha(swf, tex).astype(np.float64) / 255.0; ah, aw = alpha.shape
    I = _Im53.new('L', (aw, ah), 0); dr = _ID53.Draw(I); poly = P[rim_sh] * [aw, ah]
    for k in range(len(poly) - 2): dr.polygon([tuple(p) for p in poly[k:k + 3]], fill=1)
    tri = np.asarray(I, bool); rim = (alpha >= ALPHA53) & tri
    L, n = _cc53(rim); 
    if n > 1: rim = L == (1 + np.argmax(np.bincount(L.ravel())[1:]))
    full = _fh53(_close(rim, 1)); inner = _be53(full & ~_close(rim, 2), iterations=2)
    UX = _local53()[rim_sh]
    def loc(m, step):
        ys, xs = np.nonzero(m); sel = np.arange(0, len(ys), step)
        uv = np.c_[(xs[sel] + 0.5) / aw, (ys[sel] + 0.5) / ah, np.ones(len(sel))]; return uv @ UX
    r = max(1, int(rim.sum() // 350)); i = max(1, int(inner.sum() // 350))
    out = (loc(rim, r), loc(inner, i)); _TM53[(rim_sh, body_sh)] = out
    return out

def _feat53(img):
    f = img.min(-1).astype(np.float64); return f - _gf53(f, 3.0)

def _score53(g, tm, F):
    H, W = F.shape
    def samp(pts):
        s = np.c_[pts, np.ones(len(pts))] @ g
        return _mc53(F, [s[:, 1], s[:, 0]], order=1, mode='nearest')
    return samp(tm[0]).mean() - 0.6 * samp(tm[1]).mean()

def _imgfit53(F, tm, starts, anchor=None, maxshift=14.0):
    """best affine local -> screen from the start affines by Nelder-Mead on the rim-ridge score; (score, g)"""
    best = (-1e9, None); steps = np.array([0.03, 0.03, 0.03, 0.03, 1.5, 1.5])
    for g0 in starts:
        lim = np.linalg.svd(g0[:2])[1]
        def cost(x):
            g = x.reshape(3, 2); pen = 0.0
            if anchor is not None: pen += 0.5 * max(0.0, np.hypot(*(g[2] - anchor)) - maxshift)
            s = np.linalg.svd(g[:2])[1]
            pen += 20.0 * max(0.0, abs(np.log(s[0] / lim[0])) - 0.25) + 20.0 * max(0.0, abs(np.log(max(s[1], 1e-6) / lim[1])) - 0.35)
            return -_score53(g, tm, F) + pen
        x0 = g0.reshape(-1); sim = np.vstack([x0] + [x0 + np.eye(6)[k] * steps[k] for k in range(6)])
        r = _min53(cost, x0, method='Nelder-Mead', options=dict(xatol=0.002, fatol=0.005, maxiter=600, initial_simplex=sim))
        if -r.fun > best[0]: best = (-r.fun, r.x.reshape(3, 2))
    return best
