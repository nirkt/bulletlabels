# PREDICTABLE HULLS 2026-11-03 (group buster_janet_shield): Janet (JetpackGirl) attack and ShieldTank (Buster) attack.
# (Griff / AssaultShotgun was part of this task but the owner moved it back to the plain bullet rule: NO AssaultShotgun type is in this
# file, its labels stay as the installed fixes make them.) Exec'd in idlabels' namespace after all installed fixes (IDLABELS_OVERRIDE); it wraps labels().
#
# WHAT THE HULL IS (one shared model for every type below; per-type / per-skin parameters in the _bj_set(...) lines further down)
#   ONE label per shot = all projectiles of one shot together: members = the frame objects of one type with the same owner and the
#   same first tick ('first'). The label is the convex hull of the members' footprints, rasterised to the frame, minus the solid
#   map blocks drawn on screen (_zs_solid of z_1023_zone_solid.py, the same cut as Frank's template; objects in front such as
#   brawlers / crates / bushes do NOT carve holes in it).
#   A member's footprint is placed at the member's DRAWN position (objs 'anchor': the client's interpolated position, one update
#   behind the logic position) and turned by the member's own aim angle ('angle'). Two footprint models (a type uses one):
#     ground model (Janet, ShieldTank 327 Croc / 400 atoms): superellipse (exponent n; 2 = ellipse) in the ground plane, half-axes La
#       along the aim, Wc across it, centre fo along the aim, lifted to the heights anchor.z + zo -/+ H/2; the footprint on screen
#       = convex hull of that ring at the two heights projected with the frame's camera (a standing sprite has H > 0, a flat one 0);
#     screen model (ShieldTank plates 073 / 315 / 418; keys A, B, FO, DY): superellipse in SCREEN space around the projected drawn
#       position, A px along the aim's screen direction, B px across, FO px shifted along it, DY px down (the plates are big
#       camera-facing / tilted boards that rotate with the aim on screen).
#   Nothing is measured from the frame's pixels: inputs are the objects' drawn positions, aim angles, finished state, path, the shot
#   key (type, owner, first), the type id (-> per-skin parameters), the camera and the static map. Sparks, glow and smoke cannot
#   change it. The shape is mirror-symmetric about each member's aim axis. No label before the skin draws its first frame (the
#   objects exist from that frame on) and none after the shot's projectiles ended.
#
# LIFETIME (per type, measured against the frames the old pixel labels covered: the same frames are labelled on every captured skin)
#   life 'move' (Janet, 327, 400): a member counts while its logic projectile has not finished (state 'finished' 0); a member that
#     finishes at its range also counts on the update its drawn position still moved (Janet; fin1=False for 327 / 400: their finishing
#     frame is not labelled); a member stopped at a wall (finished 4) does not count (its hit animation is not labelled).
#   life 'shot' (ShieldTank plates 073 / 315 / 418): all members of the shot count while at least one still travels (a plate that hit
#     a wall stays drawn until the shot ends); the finishing frame is not labelled.
#   When no member counts there is no label.
#
# REPLACES (for these types only): the pixel-measured labels of the installed fixes
#   Janet: z_1101_s068 (6001158), z_1103_janet (6001160..6001172), z_1027_s399 (6001170), z_1027_s423 (6001172), z_1028_s453 (6001174),
#     a6_agent6 (JetpackGirl part)
#   ShieldTank: z_1101_s073 / z_1107_janet_shield_twins_mender part B (6001254), z_1028_s400 (6001261), z_1103_shield_hammer_fire /
#     z_1104_shield_hammer_fire (6001259, 6001263), a6_agent6 (ShieldTank part, 6001257).
#   Janet's and ShieldTank's SUPERS and every other type are not touched. The labels of these types are taken out of the result of the
#   installed chain and replaced; pixels owned by other labels stay theirs.
import numpy as _bn
from PIL import Image as _bI, ImageDraw as _bD
import bulletlabels as _BBL
from scipy.ndimage import binary_opening as _bop, binary_dilation as _bdil, binary_fill_holes as _bfh, label as _bcc

BJ_RING = 32
BJ_ANG = _bn.linspace(0, 2 * _bn.pi, BJ_RING, endpoint=False)

# per type: La, Wc (logic units, half-axes along / across the aim), fo (centre shift along the aim), zo (height offset of the
# centre above the anchor), H (vertical extent), n (superellipse exponent, 2 = ellipse)
BJ_CFG = {}


def _bj_set(types, **kw):
    for t in types: BJ_CFG[t] = dict(kw)


# ---- ShieldTank / Buster attack: 3 plates per shot (fan at aim -17, 0, +17 deg), flat rectangles turned with the aim; screen model.
# 073 Default fitted at 0 / 90 / 200 deg: plate half-extents 19.5 (along the aim) x 31.5 (across) px, shifted 13.5 px forward (superellipse n 8), life 'shot'
_bj_set((6001254,), A=19.5, B=31.5, FO=13.5, DY=0, n=8, life='shot')

# 327 Croc (three small claws; a claw stopped at a wall is gone at once: life 'move')
_bj_set((6001259,), La=90, Wc=70, H=60, zo=0, fo=-20, n=8, fin1=False)
# 400 SupercellMake (atoms; ground model, life 'move')
_bj_set((6001261,), La=150, Wc=140, H=110, zo=0, fo=10, n=8, fin1=False)
# 418 Angel (wave of plates / wings; screen model, life 'shot')
_bj_set((6001263,), A=21, B=36, FO=-4.5, DY=3, n=4, life='shot')
# 315 Chainsaw (blade)
_bj_set((6001257,), A=19.5, B=12, FO=6, DY=0, n=4, life='shot')

# ---- Janet / JetpackGirl attack: 4 projectiles per shot (a fan at aim -29, -10, +10, +29 deg), one hull over their drawn cards.
# 068 Default, fitted on captures at 0 / 90 / 200 deg (each skin below fitted on its own capture, same aims)
_bj_set((6001158,), La=70, Wc=110, H=20, zo=0, fo=0, n=4)
# 304 Valkyrie (golden gems)
_bj_set((6001160,), La=130, Wc=120, H=100, zo=0, fo=0, n=4)

# 324 Popstar
_bj_set((6001162,), La=160, Wc=140, H=30, zo=0, fo=-20, n=2)

# 331 DarkFairy
_bj_set((6001164,), La=40, Wc=30, H=40, zo=0, fo=10, n=2)

# 337 Moon
_bj_set((6001166,), La=150, Wc=140, H=50, zo=-10, fo=-10, n=2)

# 391 Christmas
_bj_set((6001168,), La=200, Wc=110, H=80, zo=0, fo=-140, n=4)

# 399 SupercellMake (cylinders)
_bj_set((6001170,), La=170, Wc=130, H=90, zo=-10, fo=-70, n=2)

# 423 Surfer (shard fan)
_bj_set((6001172,), La=120, Wc=160, H=170, zo=0, fo=40, n=4)

# 453 MHA (crystals, scattered around the drawn position: partly)
_bj_set((6001174,), La=120, Wc=160, H=260, zo=50, fo=-10, n=4)

# ---- end of Janet


def _bj_proj(cam, X, Y, Z):
    X = _bn.asarray(X, float)
    Q = _bn.stack([X, -_bn.asarray(Y, float) * _bn.ones_like(X), _bn.asarray(Z, float) * _bn.ones_like(X), _bn.ones_like(X)], -1) @ _bn.asarray(cam, float).T
    return _bn.stack([(Q[..., 0] / Q[..., 3] * .5 + .5) * 1200, (1 - (Q[..., 1] / Q[..., 3] * .5 + .5)) * 540], -1)


def _bj_hull(P):
    P = _bn.unique(_bn.round(_bn.asarray(P, float), 3), axis=0)
    if len(P) < 3: return P
    P = P[_bn.lexsort((P[:, 1], P[:, 0]))]
    cr = lambda o, a, b: (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo, up = [], []
    for p in P:
        while len(lo) >= 2 and cr(lo[-2], lo[-1], p) <= 0: lo.pop()
        lo.append(p)
    for p in P[::-1]:
        while len(up) >= 2 and cr(up[-2], up[-1], p) <= 0: up.pop()
        up.append(p)
    return _bn.array(lo[:-1] + up[:-1])


def _bj_present(o, fin1=True):
    """the member is part of the shot's footprint on this frame (see LIFETIME)"""
    f = o.get('finished') or 0
    if o.get('ended'): return False
    if f == 0: return True
    if f == 1 and fin1:
        p = o.get('path') or []
        return len(p) >= 2 and abs(p[-1][0] - p[-2][0]) + abs(p[-1][1] - p[-2][1]) > 1
    return False


def _bj_member_pts(o, cam, c):
    """screen points of one member's footprint (see the header): ground model (La, Wc, H, zo, fo, n) or, when the type has 'A',
    the screen model (a superellipse in screen space, A px along the aim's screen direction, B px across, FO px shift along it,
    DY px down, centred on the projected drawn position)"""
    th = _bn.radians(o['angle']); co, si = _bn.cos(th), _bn.sin(th)
    cs, sn = _bn.cos(BJ_ANG), _bn.sin(BJ_ANG); e = 2.0 / c.get('n', 2.0)
    a = o['anchor']
    if 'A' in c:
        p0 = _bj_proj(cam, a[0], -a[1], a[2]); p1 = _bj_proj(cam, a[0] + 100 * co, -a[1] + 100 * si, a[2])
        d = p1 - p0; d = d / _bn.hypot(*d); nrm = _bn.array([-d[1], d[0]])
        u = c['A'] * _bn.sign(cs) * _bn.abs(cs) ** e + c.get('FO', 0.0); v = c['B'] * _bn.sign(sn) * _bn.abs(sn) ** e
        return p0 + [0, c.get('DY', 0.0)] + _bn.outer(u, d) + _bn.outer(v, nrm)
    u = c['La'] * _bn.sign(cs) * _bn.abs(cs) ** e + c.get('fo', 0.0); v = c['Wc'] * _bn.sign(sn) * _bn.abs(sn) ** e
    X = a[0] + u * co - v * si; Y = -a[1] + u * si + v * co
    z0 = a[2] + c.get('zo', 0.0); h = c.get('H', 0.0) * 0.5
    return _bn.concatenate([_bj_proj(cam, X, Y, _bn.full_like(X, z)) for z in (z0 - h, z0 + h)])


def bj_shot_polygon(mem, cam):
    """screen polygon (convex hull) of one shot: its members' footprints"""
    pts = [_bj_member_pts(o, cam, BJ_CFG[o['t']]) for o in mem]
    return _bj_hull(_bn.concatenate(pts)) if pts else None


def bj_groups(objs):
    """{(type, owner, first): members}. life 'move' (default): a member counts while its logic projectile travels, plus the update
    it finishes at its range; one stopped by a wall does not. life 'shot' (ShieldTank plates): all members of the shot count
    while at least one still travels (a plate stopped at a wall stays drawn until the shot ends)."""
    G = {}
    for o in objs:
        if o['t'] in BJ_CFG: G.setdefault((o['t'], o.get('owner'), o.get('first')), []).append(o)
    out = {}
    for k, m in G.items():
        if BJ_CFG[k[0]].get('life') == 'shot':
            m = [o for o in m if not o.get('ended')]
            if any((o.get('finished') or 0) == 0 for o in m): out[k] = m
        else:
            m = [o for o in m if _bj_present(o, BJ_CFG[k[0]].get('fin1', True))]
            if m: out[k] = m
    return out


def bj_raster(poly, shape=(540, 1200)):
    I = _bI.new('L', (shape[1], shape[0]), 0)
    _bD.Draw(I).polygon([tuple(q) for q in poly], fill=1, outline=1)
    return _bn.asarray(I, bool)


BJ_WALLCUT = True   # coordinator 11-03: False = no cut at walls (owner choice)


def _bj_cut(M, solid):
    """cut the part over solid blocks; clean crumbs left along the cut (as Frank's template)"""
    if not BJ_WALLCUT or solid is None or not (M & solid).any(): return M
    M = M & ~solid
    near = _bdil(solid, iterations=4)
    M = (M & ~near) | (_bop(M, structure=_bn.ones((3, 3), bool), iterations=2) & near)
    L, n = _bcc(M)
    if n > 1:
        sz = _bn.bincount(L.ravel()); sz[0] = 0
        M = _bn.isin(L, _bn.nonzero(sz >= max(30, 0.05 * sz.max()))[0])
    return _bfh(M)   # a hull has no holes: pixels of a block seen through it are taken back


_bj_prev = labels


def labels(objs, cam, R, state=None, ended=(), img=None):
    P = _bj_prev(objs, cam, R, state=state, ended=ended, img=img)
    if not any(o['t'] in BJ_CFG for o in objs): return P
    ids = P['ids'].copy(); share = P['share'].copy(); lids = list(P['lids'])
    byid = {o['id']: o for o in objs}
    old = [k for k, l in enumerate(lids) if l[0] in byid and byid[l[0]]['t'] in BJ_CFG and l[2] == 'body']
    for k in old:
        m = ids == k; ids[m] = -1; share[m] = 0
    solid = None
    try:
        _ZS_IMG[0] = img; _zs_grid(R); solid = _zs_solid(cam, R)
    except Exception:
        solid = None
    cv = R['cv'][::2, ::2]
    for key, mem in sorted(bj_groups(objs).items(), key=lambda kv: min(o['id'] for o in kv[1])):
        poly = bj_shot_polygon(mem, cam)
        if poly is None or len(poly) < 3: continue
        M = _bj_cut(bj_raster(poly, ids.shape), solid) & (ids < 0)
        if not M.any(): continue
        carrier = min(mem, key=lambda o: o['id'])['id']; lid = (carrier, -1, 'body')
        if lid not in lids: lids.append(lid)
        k = lids.index(lid); ids[M] = k; share[M] = _bn.maximum(cv[M] / 255.0, _BBL.CUT)
    P2 = dict(P); P2.update(ids=ids, share=share, lids=lids)
    return P2
