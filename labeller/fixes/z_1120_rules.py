# Rules fix 2026-11-08 (owner-approved general rules, BOTH colours). Prefix _rl_. To be installed as py/fixes/z_1120_rules.py.
#
# RULE 1 ("427 class", LIMITED to 427 Ruffs007 6000900: _RL_C4_TYPES): idlabels.labels() has the term
#       if not trail_body(t): mine |= region & (C == 4) & pb
#   (its own trail drawn over its body: the body under it, from the PREDICTION, no test that anything is drawn there). Where the
#   prediction (the template / aligned prediction) lies where nothing of the object is drawn, the label follows the prediction into
#   empty space (427 Ruffs007 6000900: the bone-chain label sits left of the drawn bone, own and enemy).
#   FIX: the term needs a visible draw: coverage (R['cv'], 0..255) >= _RL_C4_CUT. Done by re-defining the function that holds the
#   line: the source of idlabels.labels is read, the ONE line is replaced, and the new code object is put on the ORIGINAL function
#   object (so every fix wrapper chained around labels() stays in place). idlabels.py itself is not edited.
# RULE 2 ("131 class"): "if something small is far away from the attack, it ignores it". Before a convex hull (hull_mask, used by
#   every ONE_LABEL 'hull' type and by the per-type hull hooks) or a ONE_LABEL closing is built, detached pieces that are small
#   relative to the main mass and far from it are dropped (not labelled at all). The hull itself stays.
#   LIMITED to Ash's attacks (_RL_FAR_TYPES, logic KnightProjectile*): applied to every hull it cut parts of other hull attacks
#   (Mortis super 26 lost a bat of its swarm, Mender 452 a droplet of its wave); the owner's rule: hull attacks keep their hulls.
import os as _rl_os
import numpy as _rl_np

_RL_C4_CUT = int(_rl_os.environ.get('RL_C4', '64'))          # coverage a Rule-1 pixel needs (0 = rule off)
_RL_C4_HI = int(_rl_os.environ.get('RL_C4_HI', '128'))      # a term pixel with this coverage is always kept
_RL_C4_MODE = _rl_os.environ.get('RL_C4_MODE', 'comp')
_RL_C4_MINPX = int(_rl_os.environ.get('RL_C4_MINPX', '3'))
_RL_C4_FRAC = float(_rl_os.environ.get('RL_C4_FRAC', '0.05'))
_RL_HULLKEEP = int(_rl_os.environ.get('RL_HULLKEEP', '2'))   # -1: off
_RL_LM_R = int(_rl_os.environ.get('RL_LM_R', '3'))          # radius of the local mean (px, half resolution)
_RL_FAR_ON = int(_rl_os.environ.get('RL_FAR', '1'))          # Rule 2 on / off
_RL_SMALL = float(_rl_os.environ.get('RL_SMALL', '0.20'))    # a piece is small when its area <= this share of the main (largest) piece
_RL_FARMULT = float(_rl_os.environ.get('RL_FARMULT', '1.5')) # and far when its gap to the main group > this x sqrt(main area)
_RL_LINK = float(_rl_os.environ.get('RL_LINK', '0.35'))       # pieces closer than this x sqrt(total pixels) belong to one cluster
_RL_LINKMIN = 8                                              # px (half resolution)
_RL_FARMIN = float(_rl_os.environ.get('RL_FARMIN', '12'))    # px (half resolution): and at least this far
_RL_STATS = []                                               # (n pieces, [(area, gap)...]) per hull call, for calibration only (RL_LOG=1)
_RL_LOG = _rl_os.environ.get('RL_LOG') == '1'
_RL_REC_ON = _rl_os.environ.get('RL_REC') == '1'
_RL_REC = []                                                 # RL_REC=1: every mask given to hull_mask (calibration of rule 2 only; output unchanged)


# ------------------------------------------------------------------------------------------------------------------ Rule 1
_RL_OLD_LINE = "if not trail_body(t): mine |= region & (C == 4) & pb"


# the owner's hull types (their labels are hulls by decision; rule 1 leaves them alone): Nita attack (not skins 232 / 358 / 393 =
# 6000226 / 6000238 / 6000240), Sandy, Janet, ShieldTank, Frank attacks; Mortis / Frank / Mender supers; and every ONE_LABEL 'hull' type
_RL_HULL_TYPES = ({6000222, 6000224, 6000228, 6000230, 6000232, 6000234, 6000236, 6000242, 6000244, 6000246}
                  | {6000666, 6000668, 6000670, 6000672, 6000674, 6000676, 6000678}
                  | {6001158, 6001160, 6001162, 6001164, 6001166, 6001168, 6001170, 6001172, 6001174}
                  | {6001254, 6001259, 6001261, 6001263, 6001257}
                  | {6000475, 6000477, 6000479, 6000481, 6000483, 6000485, 6000487, 6000489}
                  | {6000476, 6000478, 6000480, 6000482, 6000484, 6000486, 6000488, 6000490, 6001470}
                  | {6000299, 6001497, 6001498, 6000300, 6000301, 6000302, 6000303, 6000304, 6000305, 6000306, 6000307, 6001754}
                  | {6001756, 6001757, 6001759})


# rule 1 is LIMITED to the type the owner flagged (427 Ruffs007, own + enemy): applied to every type it also trimmed faint edges of
# approved labels (Sandy super 42: 20-50 px of the stone's rim per tick; Buzz 282 trail-glow edges).
_RL_C4_TYPES = {6000900}


def _rl_is_hull(t):
    return t in _RL_HULL_TYPES or ONE_LABEL.get(t) == 'hull'


def _rl_c4_term(t, region, C, pb, cv, mine):
    """the pixels the term 'region & (C == 4) & pb' adds: predicted body under its own trail sprite. A pixel needs a visible draw:
    its coverage >= _RL_C4_HI, or the mean coverage of the term's pixels around it (window _RL_LM_R px) >= _RL_C4_CUT. (A faint
    trail sprite over NOTHING has coverage 32-63 throughout: 427's bone chain. A translucent body under a trail sprite has
    stronger coverage somewhere around each pixel: Buzz's ice block.) Hull types keep the term unchanged."""
    base = region & (C == 4) & pb
    if _RL_C4_CUT <= 0 or not base.any() or _rl_is_hull(t) or int(t) not in _RL_C4_TYPES: return base
    if _RL_C4_MODE == 'comp':
        from scipy.ndimage import label as cc
        L, n = cc(base, structure=_rl_np.ones((3, 3), int)); strong = _rl_np.bincount(L[cv >= _RL_C4_CUT], minlength=n + 1)
        size = _rl_np.bincount(L.ravel(), minlength=n + 1)
        keep = (strong >= _rl_np.maximum(_RL_C4_MINPX, _RL_C4_FRAC * size)); keep[0] = False
        out = keep[L]
    else:
        from scipy.ndimage import uniform_filter
        f = base.astype(_rl_np.float32); w = 2 * _RL_LM_R + 1
        lm = uniform_filter(cv.astype(_rl_np.float32) * f, size=w, mode='constant') / _rl_np.maximum(uniform_filter(f, size=w, mode='constant'), 1e-6)
        out = base & ((cv >= _RL_C4_HI) | (lm >= _RL_C4_CUT))
    if _RL_HULLKEEP >= 0:
        # a dropped pixel that lies inside (the hull of the object's remaining pixels, grown by _RL_HULLKEEP px) is an enclosed / edge
        # part of a translucent body (Buzz's rings and ice block), not a protrusion into empty space: it stays
        rem = mine | out
        if rem.any():
            H = _rl_hull_orig(rem)
            if _RL_HULLKEEP: H = _rl_np.asarray(__import__('scipy.ndimage', fromlist=['x']).binary_dilation(H, iterations=_RL_HULLKEEP))
            out = out | (base & H)
    if _RL_LOG: _RL_STATS.append(('c4', t, int(base.sum()), int(out.sum())))
    return out


def _rl_new_line():
    return "if not trail_body(t): mine |= _rl_c4_term(t, region, C, pb, cv, mine)"


def _rl_find_labels_origin():
    """the ORIGINAL idlabels.labels function object (defined in idlabels.py), wherever the fix wrappers stored it"""
    import types
    me = globals().get('__file__') or ''
    seen = set(); stack = [globals().get('labels')]; found = None
    while stack:
        f = stack.pop()
        if not isinstance(f, types.FunctionType) or id(f) in seen: continue
        seen.add(id(f))
        co = f.__code__
        if co.co_name == 'labels' and _rl_os.path.basename(co.co_filename) == 'idlabels.py': found = f; break
        for c in (f.__closure__ or ()):
            try: stack.append(c.cell_contents)
            except ValueError: pass
        stack.extend(f.__defaults__ or ())
        for v in f.__globals__.values():     # the 'previous labels' a wrapper stored in a module global
            if isinstance(v, types.FunctionType) and v.__code__.co_name == 'labels' and id(v) not in seen: stack.append(v)
    return found


def _rl_patch_labels():
    import ast
    if _RL_C4_CUT <= 0: return False
    fn = _rl_find_labels_origin()
    if fn is None or getattr(fn, '_rl_patched', None) == (_RL_C4_CUT, _RL_C4_HI, _RL_LM_R, _RL_C4_MODE, _RL_C4_MINPX, _RL_C4_FRAC, _RL_HULLKEEP): return False
    path = fn.__code__.co_filename
    src = open(path).read()
    node = [n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == 'labels'][0]
    text = '\n'.join(src.split('\n')[node.lineno - 1:node.end_lineno])
    if _RL_OLD_LINE not in text: return False
    text = text.replace(_RL_OLD_LINE, _rl_new_line())
    ns = dict(globals())
    exec(compile('\n' * (node.lineno - 1) + text, path, 'exec'), ns)
    fn.__code__ = ns['labels'].__code__
    fn._rl_patched = (_RL_C4_CUT, _RL_C4_HI, _RL_LM_R, _RL_C4_MODE, _RL_C4_MINPX, _RL_C4_FRAC, _RL_HULLKEEP)
    return True


_rl_patch_labels()


# ------------------------------------------------------------------------------------------------------------------ Rule 2
def _rl_drop_far(m):
    """m without its small far outliers. Pieces = 8-connected components; pieces whose gaps are <= _RL_LINK * sqrt(total area) (at
    least _RL_LINKMIN px) are one cluster (single linkage). The main cluster is the one with most pixels. Another cluster is an
    outlier (dropped) when it is SMALL (<= _RL_SMALL of the main cluster's pixels) and FAR (gap to the main cluster > _RL_FARMULT x
    sqrt(main pixels), and at least _RL_FARMIN px)."""
    from scipy.ndimage import label as cc, distance_transform_edt as edt
    if not _RL_FAR_ON or m is None or not m.any(): return m
    ys, xs = _rl_np.nonzero(m); y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    c = m[y0:y1, x0:x1]
    L, n = cc(c, structure=_rl_np.ones((3, 3), int))
    if n < 2 or n > 150: return m
    area = _rl_np.bincount(L.ravel(), minlength=n + 1)
    link = max(_RL_LINKMIN, _RL_LINK * float(area[1:].sum()) ** 0.5)
    parent = list(range(n + 1))

    def find(x):
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for k in range(1, n + 1):
        near = _rl_np.unique(L[(edt(L != k) <= link) & (L > 0)])
        for j in near:
            if j != k: parent[find(int(j))] = find(k)
    lab = _rl_np.array([find(i) for i in range(n + 1)]); C = lab[L]
    ids = [i for i in _rl_np.unique(C) if i > 0]
    if len(ids) < 2: return m
    mass = {i: int((C == i).sum()) for i in ids}
    main = max(ids, key=lambda i: mass[i]); A = float(mass[main]); dist = edt(C != main); drop = []
    for i in ids:
        if i == main: continue
        gap = float(dist[C == i].min())
        if mass[i] <= _RL_SMALL * A and gap > max(_RL_FARMIN, _RL_FARMULT * A ** 0.5): drop.append(i)
    if _RL_LOG: _RL_STATS.append((int(A), [(mass[i], float(dist[C == i].min())) for i in ids if i != main], drop))
    if not drop: return m
    out = m.copy(); out[y0:y1, x0:x1] &= ~_rl_np.isin(C, drop)
    return out


def _rl_far_types():
    """Ash's attack projectiles (logic KnightProjectile*, every skin): the 131 class. Rule 2 is limited to them: on the owner's
    hull types it dropped parts of the attack (Mortis super 26: a bat of the swarm; Mender 452: a droplet of the wave)."""
    try:
        import projtab
        return {int(t) for t, r in projtab.table().items()
                if str(r.get('logic') or '').startswith('KnightProjectile') or str(r.get('parent') or '').startswith('KnightProjectile')}
    except Exception:
        return {6001040, 6001041, 6001042, 6001043, 6001044, 6001045, 6001046, 6001047, 6001048, 6001049, 6001050, 6001051,
                6001052, 6001053, 6001054, 6001055, 6001056, 6001057, 6001062, 6001063, 6001064}


_RL_FAR_TYPES = _rl_far_types()


def _rl_caller_far_type():
    """True when the object being labelled (found in the calling frames: the labeller's loop variable t, or a hook's ctx['t'])
    is one of _RL_FAR_TYPES."""
    import sys
    f = sys._getframe(2)
    for _ in range(8):
        if f is None: break
        lo = f.f_locals; t = lo.get('t')
        if not isinstance(t, (int, _rl_np.integer)):
            c = lo.get('ctx'); t = c.get('t') if isinstance(c, dict) else None
        if isinstance(t, (int, _rl_np.integer)): return int(t) in _RL_FAR_TYPES
        f = f.f_back
    return False


if not getattr(globals().get('hull_mask'), '_rl_wrapped', False):
    _rl_hull_orig = hull_mask

    def hull_mask(m):
        """the convex hull of a mask's pixels (the original), the mask first stripped of its small far outliers (rule 2)"""
        if _RL_REC_ON and m is not None and m.any(): _RL_REC.append(m.copy())
        if not _rl_caller_far_type(): return _rl_hull_orig(m)     # rule 2 only for Ash's attacks; every other hull stays exactly
        return _rl_hull_orig(_rl_drop_far(m))
    hull_mask._rl_wrapped = True


# ------------------------------------------------------------------------------------------------------------------ Rule 2b (Ash)
# Owner 2026-11-08 on 131 Ash: "the far pieces should get no label" and the label on the ground during the super. Ash's attack
# lingers (its logic projectile stays where it landed, its junk lies on the ground and fades) and two things went wrong:
#   - when an Ash super explodes near it, the explosion's pixels (not labelled themselves: hit effects are off) were handed to the
#     lingering attack, which then got a label on the bright ground (131 super ticks 361 / 363);
#   - its last tiny remnants (a few to ~30 px, far from everything) kept a label of their own (131 super ticks 351 / 353 / 365).
# Fix, Ash attack types only (_RL_FAR_TYPES): no label while an Ash super that ended <= _RL_ASH_EXPL_TICKS ago lies within
# _RL_ASH_EXPL_R logic units, and no label when the attack's drawn pieces total < _RL_ASH_MINPX px (half resolution).
_RL_ASH_ULTI = {6001058, 6001059, 6001060, 6001061, 6001065, 6001066, 6001067, 6001068, 6001069}
_RL_ASH_EXPL_TICKS = int(_rl_os.environ.get('RL_ASH_EXPL_TICKS', '12'))
_RL_ASH_EXPL_R = float(_rl_os.environ.get('RL_ASH_EXPL_R', '1200'))
_RL_ASH_MINPX = int(_rl_os.environ.get('RL_ASH_MINPX', '30'))


def _rl_caller_ended():
    import sys
    f = sys._getframe(2)
    for _ in range(6):
        if f is None: break
        e = f.f_locals.get('ended')
        if isinstance(e, (list, tuple)): return e
        f = f.f_back
    return ()


def _rl_ash_post(prev):
    def fn(body, ctx):
        if _RL_FAR_ON:
            o = ctx.get('o') or {}
            lx, ly = o.get('logic', (None, None))
            if lx is not None:
                for e in _rl_caller_ended():
                    if not isinstance(e, dict) or e.get('t') not in _RL_ASH_ULTI or e.get('age', 99) > _RL_ASH_EXPL_TICKS: continue
                    if ((e['x'] - lx) ** 2 + (e['y'] - ly) ** 2) ** 0.5 <= _RL_ASH_EXPL_R: return _rl_np.zeros_like(body)
            if body is not None and int(body.sum()) < _RL_ASH_MINPX: return _rl_np.zeros_like(body)
        return prev(body, ctx) if prev else body
    fn._rl_ash = True
    return fn


for _rl_t in sorted(_RL_FAR_TYPES):
    if not getattr(POST_BODY.get(_rl_t), '_rl_ash', False): POST_BODY[_rl_t] = _rl_ash_post(POST_BODY.get(_rl_t))
