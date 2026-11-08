# Frank attack label template v6 (task t42, Opus agent; v5 + per-skin aim-angle harmonics fitted on 17 angles, leave-one-angle-out validated). Installed by the coordinator 10-28; previous v3 backup: out/sweep2/coord_1024/z_1027_frank_template_v3_installed.py.
# Frank's attack (HammerDudeProjectile, every skin): ONE replayable label template per wave, v6 = v5 with per-skin parameters
# that depend on the aim angle (FT42_HARM: low-order harmonics, kept only where they help on held-out angles; t42/fit_v6.py).
# v5 = the sprite model (below,
# FT42_MODEL 'sprite') with v4's ground model kept as a per-skin fallback ('ground'). v4 = v3 with a tight back end
# (task 42, 2026-10-27: the user on v3, "its still loose on the front part (the brawler part)"): the sideways margin
# shrinks from the newest row (mv) to the oldest labelled row (mvo), refitted with the back and the front half of the label
# scored separately (t42/fit_v4.py). The user: "lets design a label that we would replay on frank's attacks that is equal on both sides and fits
# both the hitbox and the visual attack. each skin can have different small tweaks if needed and if the attack goes over a
# wall/bumps we ignore that part"; answers of 10-27: HUG THE VISUAL (sized to the drawn rows that are still the attack, the
# hitbox may stick out at the front), no label before a skin draws its first row, fitted on many angles.
# Replaces, for the attack types only, the pixel-measured Frank labels of z_1026_frank_attack_only.py, z_1026_frank_a.py,
# z_1023_frank321.py and z_1023_frank242.py (their supers are untouched). v1 (t42/fix.py) also covered the hitbox.
#
# The label is NOT measured from the frame's pixels. It is a shape in the ground plane (logic units) that depends only on
#   the attack's origin O (the shooter's position when it fired: where the wave's four rays meet), its aim direction,
#   the time tau since the attack started (updates since the wave's first update) and the skin (type id),
# placed in the world, raised to the skin's visual height (the spikes / cones / flames stand up from the ground) and
# projected with the frame's camera. In attack-local coordinates (u along the aim, v across) it is mirror-symmetric
# about the u axis.
#
# Logic (identical for every skin and aim; measured on 8 skins x 8 aims, every update, residual < 3 units): 4 projectiles
# on rays from O at aim +-8.1 deg (inner pair, 'slow', state travel0 125) and +-23.2 deg (outer pair, 'fast', travel0 142):
#   distance from O after k updates  s(k) = min(1800, s0 + b k),  slow: s0 486.2, b 218.8;  fast: s0 548.7, b 250.1
# (outer pair at its range at tau 5, inner at tau 6). The client draws one row of spikes / tiles / rocks / flames per update
# where the projectiles are, FT42_LAG of an update behind the logic position; a row is the attack for its first A updates.
#
# Template at tau (attack-local, logic units):
#   rows k = max(first, tau - A) .. tau, drawn at s_draw(k) = s(k - FT42_LAG) on each projectile's ray (both pairs, +-)
#   footprint F(tau) = convex hull of every row point grown by the skin's 'egg' (mb backward, mf forward, mv sideways),
#                      scaled by g0 (newest row) .. 1 (row of age A) and by the start-up ramp r0 (tau = first) .. 1 (from
#                      first + FT42_RAMP on)
#   label = the solid {(u, v, z): (u, v) in F, 0 <= z <= h(u)} seen by the camera (h: hn at the newest row, ho at the oldest
#           labelled row, linear in between, times the ramp), minus
#           - the part of F in a projectile-blocking tile or beyond one as seen from O (rays from O through the capture's
#             tile map: the wall stopped the wave there),
#           - the blocking tiles as boxes FT42_BLOCK_Z high, and (v3, the user 10-27) the bush tiles as boxes FT42_BUSH_Z
#             high, standing on / in front of the label: what they hide is cut (straight edges; characters still do not cut),
#           - the solid blocks drawn on screen (_zs_solid of z_1023_zone_solid.py), the cut's edge band cleaned of crumbs.
#   No hitbox term: the logic hitbox (r 150) may stick out at the front (the user, 10-27).
#   It exists for tau >= first (the skin's first drawn row) while at least one projectile of the wave is still travelling
#   (state 'finished' 0); once all have finished / ended nothing is labelled (tau 6 for an unobstructed wave).
# Placement from the frame objects and the state (no pixels): tau = len(path) - 1; O and the aim are solved from the logic
# positions of the members still travelling or at their range (distance from O via state travel / travel0), their slot
# (from the integer angle) and a weak prior on the integer angles (O moves <= 4 units, the aim <= 0.06 deg over an attack).
import numpy as _np42
from PIL import Image as _Im42, ImageDraw as _Dr42

FT42_ATTACK = {t for t, d in projtab.table().items() if d['logic'] == 'HammerDudeProjectile'}
# kinematics per logic class (measured, see above); cls: travel0 -> (s0, b, first travel)
FT42_KIN = {'HammerDudeProjectile': dict(R=1800.0, rad=150.0, phi=(8.12, 23.23),
                                         cls={125: (486.2, 218.8, 250), 142: (548.7, 250.1, 284)}),
            'HammerDudeUltiProjectile': dict(R=2100.0, rad=250.0, phi=(8.12, 23.29),
                                             cls={125: (560.6, 256.6, 250), 142: (634.4, 293.0, 284)})}
FT42_DEFAULT_INC = (125, 142)   # inner, outer pair when the state has no travel0 (as on every capture)
# the template's visual parameters (logic units): A rows (age limit, updates), mb / mf / mv the egg around each row point
# (backward / forward / sideways; v4: mb / mv of a new row, going to mbo / mvo at age A), hn / ho the visual height (logic z) of the newest row (age 0) and of the oldest
# labelled row (age A), linear in between (spikes sink, rocks grow), g0 the egg scale of the newest row (1 at age A), r0 the
# scale of egg and heights on the first drawn update (ramps to 1 over FT42_RAMP updates), first the first drawn row = the
# first labelled update
FT42_P = dict(A=2, first=0, mb=150.0, mbo=0.0, mv=225.0, mvo=200.0, mf=200.0, hn=450.0, ho=0.0, g0=1.0, r0=1.0)   # = 035, the default for other types
# Per skin: fitted by t42/fit_v4.py on every imaged attack frame (tau >= first) of the 0 / 90 deg captures (sw2_*) and the
# 30 / 60 / 135 / 180 / 225 / 300 deg captures (fa_*), parts at walls, bushes and the wall side of a projectile that hit a
# wall left out; each frame's label split at the middle between its oldest and newest drawn row, the back and the front half
# scored separately (mean of the two IoUs with the current label of the rows of age <= A), keeping >= 97 % of that label,
# >= 95 % in each half and >= 90 % on every frame. mb / mv: a new row's backward / sideways margin, mbo / mvo: a row of age A
# (the back end of the label), mf: forward margin.
FT42_SKIN = {   # v5: the ground model's parameters (v4 model) refitted at the v5 coverage (>= 95 % of the drawn rows, >= 93 % per
                # half, >= 85 % per frame; t42/fit_v4b.py); used where FT42_MODEL is 'ground'; A / first used by both models
    6000475: dict(A=2, first=0, mb=150.0, mbo=0.0, mv=225.0, mvo=125.0, mf=200.0, hn=450.0, ho=0.0, g0=1.0, r0=1.0),   # 035 FrankDefault (+ Gold / Silver / Xmas / Vacation)
    6000485: dict(A=2, first=0, mb=175.0, mbo=125.0, mv=250.0, mvo=0.0, mf=225.0, hn=400.0, ho=300.0, g0=1.0, r0=1.0),   # 220 FrankCaveman (+ CavemanRed)
    6000477: dict(A=3, first=1, mb=100.0, mbo=100.0, mv=125.0, mvo=175.0, mf=50.0, hn=350.0, ho=0.0, g0=1.0, r0=0.8),   # 242 HammerDudeDJ
    6000479: dict(A=2, first=1, mb=350.0, mbo=125.0, mv=375.0, mvo=325.0, mf=325.0, hn=450.0, ho=700.0, g0=0.5, r0=1.0),   # 321 HammerDudeknight
    6000481: dict(A=2, first=2, mb=125.0, mbo=75.0, mv=150.0, mvo=100.0, mf=125.0, hn=200.0, ho=100.0, g0=0.8, r0=0.6),   # 339 HammerDudeHindu
    6000483: dict(A=1, first=1, mb=150.0, mbo=125.0, mv=275.0, mvo=225.0, mf=300.0, hn=100.0, ho=450.0, g0=1.0, r0=0.7),   # 344 HammerDudeClashKingChromaRed
    6000487: dict(A=1, first=1, mb=0.0, mbo=75.0, mv=150.0, mvo=25.0, mf=200.0, hn=450.0, ho=250.0, g0=1.0, r0=0.1),   # 401 HammerDudeSandsOfTimes
    6000489: dict(A=1, first=2, mb=225.0, mbo=175.0, mv=75.0, mvo=25.0, mf=100.0, hn=200.0, ho=150.0, g0=1.2, r0=1.0),   # 450 FrankFairies
}
FT42_TILE = 300.0
FT42_RAY_STEP = 6.0      # logic units along a ray from O (wall test)
FT42_RAY_DA = 0.05       # degrees between rays
FT42_CELL = 8.0          # logic units: local raster of the footprint
FT42_EGG_N = 24
FT42_RAMP = 2            # updates over which the drawing grows to its size (r0)
FT42_LAG = 0.34          # updates: the client draws a row this far behind the logic position (0.26-0.43 measured: it drifts with the frame timing; constant within an attack)
FT42_BLOCK_Z = 600.0     # logic z of a block's top (walls: depth of their top faces 560-610, sw2_242 t319)
FT42_BUSH_Z = 450.0      # logic z of a bush's top (depth of the surfaces drawn in bush tiles: median 440, fa_401 / fa_035 t477)
FT42_CUT_NEAR = 4        # px: band along the solid-block cut that is cleaned (3x3 opening, FT42_CUT_OPEN iterations)
FT42_CUT_OPEN = 2
FT42_SPECK = 30          # px / share of the largest part: smaller separate parts left by the cut are dropped
FT42_SPECK_FRAC = 0.05


# v6: per-skin parameters that depend on the aim (the user 10-28: "make some small tweaks per skin that also count in the
# angle at which the attack is fired"): FT42_HARM[t][name] = coefficients of c0 + c1 cos(aim) + s1 sin(aim) [+ c2 cos(2 aim)
# + s2 sin(2 aim)] (aim: the logic angle of the attack's axis, degrees); it replaces that parameter's constant. The camera
# is fixed, so the aim decides how the drawn sprites stand on screen; the template stays mirror-symmetric about the aim.
FT42_HARM = {   # coefficients [c0, c1, s1, (c2, s2)]; t42/fit_v6.py, fit/fitv6_<skin>.json
    6000475: {'D': [63.83]},   # 035 (sprite model): orders {'D': 'm'}
    6000485: {'D': [105.56, 9.23, -25.05, -50.56, -0.70], 'r0': [0.96, -0.04, -0.04, 0.04, -0.08], 'fo': [6.91]},   # 220 (sprite model): orders {'D': '2', 'r0': '2', 'fo': 'm'}
    6000477: {'mb': [87.91, 11.90, 72.04], 'mf': [59.01, -12.15, 2.25, 49.38, -16.89]},   # 242 (ground model): orders {'mb': '1', 'mf': '2'}
    6000479: {'mb': [316.53, -40.34, 77.20], 'mvo': [250.00]},   # 321 (ground model): orders {'mb': '1', 'mvo': 'm'}
    6000481: {'mb': [189.29], 'mbo': [90.62, 17.62, -15.32], 'g0': [0.76]},   # 339 (ground model): orders {'mb': 'm', 'mbo': '1', 'g0': 'm'}
    6000483: {'mb': [255.59, -18.89, -121.45, -33.31, 13.49], 'mv': [267.60, -2.25, 8.41, 49.82, -42.14], 'mvo': [172.50]},   # 344 (ground model): orders {'mb': '2', 'mv': '2', 'mvo': 'm'}
    6000487: {'mb': [207.68, -34.66, -37.95, -11.76, -98.51], 'mv': [117.47, 41.20, -59.90], 'mvo': [35.55, -14.43, -33.40]},   # 401 (ground model): orders {'mb': '2', 'mv': '1', 'mvo': '1'}
    6000489: {'g0': [1.18, 0.02, 0.01, -0.05, -0.00], 'r0': [0.99, 0.01, -0.03]},   # 450 (ground model): orders {'g0': '2', 'r0': '1'}
}
FT42_BOUNDS = dict(g0=(0.1, 3.0), r0=(0.05, 1.0), aw=(0.0, 3.0), ah=(0.0, 3.0), fo=(-400.0, 400.0))


def _ft42_harm(t, p, aim):
    H = FT42_HARM.get(t)
    if not H or aim is None: return p
    a = _np42.radians(aim); basis = _np42.array([1.0, _np42.cos(a), _np42.sin(a), _np42.cos(2 * a), _np42.sin(2 * a)])
    for k, c in H.items():
        if k not in p: continue
        v = float(_np42.dot(_np42.asarray(c, float), basis[:len(c)]))
        lo, hi = FT42_BOUNDS.get(k, (0.0, 5000.0)); p[k] = min(hi, max(lo, v))
    return p


def _ft42_param(t, aim=None):
    p = dict(FT42_P); p.update(FT42_SKIN.get(t, {})); return _ft42_harm(t, p, aim)


def _ft42_blocking():
    """tile codes (map value - 27000000) whose BlocksProjectiles is true (csv_logic/tiles.csv)"""
    global _FT42_BLOCK
    try: return _FT42_BLOCK
    except NameError: pass
    import csv as _csv, os as _os
    Rw = list(_csv.reader(open(_os.path.join(rc.ASSETS, 'csv_logic', 'tiles.csv'), encoding='utf8')))
    i = Rw[0].index('BlocksProjectiles')
    _FT42_BLOCK = _np42.array(sorted(k for k, r in enumerate(Rw[2:]) if len(r) > i and r[i].strip().lower() == 'true'))
    return _FT42_BLOCK


def _ft42_bushes():
    """tile codes of the bushes (IsForest true in csv_logic/tiles.csv)"""
    global _FT42_BUSH
    try: return _FT42_BUSH
    except NameError: pass
    import csv as _csv, os as _os
    Rw = list(_csv.reader(open(_os.path.join(rc.ASSETS, 'csv_logic', 'tiles.csv'), encoding='utf8')))
    i = Rw[0].index('IsForest')
    _FT42_BUSH = _np42.array(sorted(k for k, r in enumerate(Rw[2:]) if len(r) > i and r[i].strip().lower() == 'true'))
    return _FT42_BUSH


def _ft42_hull(P):
    P = _np42.unique(_np42.round(_np42.asarray(P, float), 3), axis=0)
    if len(P) < 3: return P
    P = P[_np42.lexsort((P[:, 1], P[:, 0]))]
    def cross(o, a, b): return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo, up = [], []
    for p in P:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0: lo.pop()
        lo.append(p)
    for p in P[::-1]:
        while len(up) >= 2 and cross(up[-2], up[-1], p) <= 0: up.pop()
        up.append(p)
    return _np42.array(lo[:-1] + up[:-1])


def _ft42_ramp(tau, prm):
    """the drawing starts small and reaches its size FT42_RAMP updates after the first drawn update: scale r0 at
    tau = first, 1 from first + FT42_RAMP on (egg and heights)"""
    r0 = prm.get('r0', 1.0)
    return r0 + (1.0 - r0) * min(1.0, max(0.0, (tau - prm['first']) / float(FT42_RAMP)))


def _ft42_sdraw(kin, inc, k):
    """distance from O at which the client draws row k of a projectile of class inc: its logic distance FT42_LAG of an
    update earlier (the client's interpolated position)"""
    s0, b, _ = kin['cls'][inc]
    return max(0.0, min(kin['R'], s0 + b * (k - FT42_LAG)))


def _ft42_height(kin, inc, tau, prm):
    """(u_old, u_new, h_old, h_new): the visual height is h_new at the newest row's mean u and h_old at the oldest
    labelled row's (linear in between, constant beyond)"""
    k0 = min(tau, max(int(prm['first']), tau - int(prm['A'])))
    def urow(k):
        return _np42.mean([_ft42_sdraw(kin, inc[c], k) * _np42.cos(_np42.radians(kin['phi'][c])) for c in (0, 1)])
    ho = prm['hn'] + (prm['ho'] - prm['hn']) * (tau - k0) / max(1, int(prm['A']))
    r = _ft42_ramp(tau, prm)
    return urow(k0), urow(tau), ho * r, prm['hn'] * r


def _ft42_hfun(hgt, u):
    u0, u1, h0, h1 = hgt
    if u1 - u0 < 1e-6: return _np42.full_like(_np42.asarray(u, float), h1)
    return h0 + (h1 - h0) * _np42.clip((_np42.asarray(u, float) - u0) / (u1 - u0), 0, 1)


def _ft42_footprint(kin, inc, tau, prm):
    """convex footprint polygon (n, 2), attack-local (u, v), of the template at update tau. inc: travel0 of the inner and
    the outer pair (125 slow / 142 fast)"""
    a = _np42.linspace(0, 2 * _np42.pi, FT42_EGG_N, endpoint=False)
    ca, sa = _np42.cos(a), _np42.sin(a); rp = _ft42_ramp(tau, prm); A = max(1, int(prm['A']))
    pts = []
    for c in (0, 1):
        ph = _np42.radians(kin['phi'][c])
        for k in range(min(tau, max(int(prm['first']), tau - int(prm['A']))), tau + 1):   # rows drawn from update 'first' on
            s = _ft42_sdraw(kin, inc[c], k); age = min(tau - k, A) / float(A)
            # v4: the egg's sideways margin goes from mv (newest row) to mvo (oldest labelled row): the back of the wave,
            # where the four projectiles are still close together, is narrow (the user 10-27: "loose on the brawler part")
            # and the backward margin from mb (a new row: the first update's burst around the hammer) to mbo (a row of age A)
            mvk = prm['mv'] + (prm.get('mvo', prm['mv']) - prm['mv']) * age
            mbk = prm['mb'] + (prm.get('mbo', prm['mb']) - prm['mb']) * age
            sc = prm.get('g0', 1.0) + (1.0 - prm.get('g0', 1.0)) * age
            egg = _np42.c_[ca * _np42.where(ca > 0, prm['mf'], mbk), sa * mvk] * rp * sc
            for sg in (1, -1):
                pts.append(_np42.array([s * _np42.cos(ph), sg * s * _np42.sin(ph)]) + egg)
    return _ft42_hull(_np42.concatenate(pts))


def _ft42_world(O, th, P):
    c, s = _np42.cos(_np42.radians(th)), _np42.sin(_np42.radians(th)); P = _np42.asarray(P, float)
    return _np42.c_[O[0] + P[:, 0] * c - P[:, 1] * s, O[1] + P[:, 0] * s + P[:, 1] * c]


def _ft42_proj(cam, X, Y, Z):
    X = _np42.asarray(X, float); Q = _np42.stack([X, -_np42.asarray(Y, float), 0.6 * _np42.asarray(Z, float) * _np42.ones_like(X),
                                                 _np42.ones_like(X)], -1) @ _np42.asarray(cam, float).T
    return _np42.stack([(Q[..., 0] / Q[..., 3] * 0.5 + 0.5) * 1200, (1 - (Q[..., 1] / Q[..., 3] * 0.5 + 0.5)) * 540], -1)


def _ft42_unproj(cam, sx, sy, z):
    """screen (half resolution) -> logic ground x, y of the point at height z under that pixel (the inverse of _proj)"""
    cam = _np42.asarray(cam, float); ndx = sx / 1200.0 * 2 - 1; ndy = (1 - sy / 540.0) * 2 - 1
    A0 = cam[0][None, :] - ndx[:, None] * cam[3][None, :]; A1 = cam[1][None, :] - ndy[:, None] * cam[3][None, :]
    b0 = -(A0[:, 2] * 0.6 * z + A0[:, 3]); b1 = -(A1[:, 2] * 0.6 * z + A1[:, 3])
    det = A0[:, 0] * A1[:, 1] - A0[:, 1] * A1[:, 0]
    return (b0 * A1[:, 1] - A0[:, 1] * b1) / det, -((A0[:, 0] * b1 - b0 * A1[:, 0]) / det)


def _ft42_stop(grid, O, alphas, rmax):
    """distance from O at which a ray (angle alphas, degrees) first enters a projectile-blocking tile (inf: none)"""
    blk = _ft42_blocking(); r = _np42.arange(0.0, rmax + FT42_RAY_STEP, FT42_RAY_STEP)
    a = _np42.radians(alphas)[:, None]
    X = O[0] + r[None, :] * _np42.cos(a); Y = O[1] + r[None, :] * _np42.sin(a)
    cx = _np42.floor(X / FT42_TILE).astype(int); cy = _np42.floor(Y / FT42_TILE).astype(int)
    out = (cx < 0) | (cx >= grid.shape[1]) | (cy < 0) | (cy >= grid.shape[0])
    code = _np42.where(out, -1, grid[_np42.clip(cy, 0, grid.shape[0] - 1), _np42.clip(cx, 0, grid.shape[1] - 1)])
    hit = out | _np42.isin(code, blk)
    first = _np42.where(hit.any(1), hit.argmax(1), -1)
    return _np42.where(first >= 0, r[_np42.maximum(first, 0)], _np42.inf)


def _ft42_mask(cam, O, th, F, hgt, grid=None, solid=None, shape=(540, 1200)):
    """(H, W) bool: the solid {(u, v, z): (u, v) in F, 0 <= z <= h(u)} (F attack-local, placed at O / th; h from
    hgt = _ft42_height) as drawn, minus the part of F in or beyond a projectile-blocking tile seen from O (grid: the
    60 x 60 tile codes; None: no cut) and minus solid (screen)"""
    H, Wd = shape
    h = max(hgt[2], hgt[3])
    Fw = _ft42_world(O, th, F)
    S = _np42.concatenate([_ft42_proj(cam, Fw[:, 0], Fw[:, 1], 0.0), _ft42_proj(cam, Fw[:, 0], Fw[:, 1], h)])
    Sh = _ft42_hull(S)
    I = _Im42.new('L', (Wd, H), 0); _Dr42.Draw(I).polygon([tuple(p) for p in Sh], fill=1, outline=1)
    C = _np42.asarray(I, bool)
    if not C.any(): return C
    # local raster of the footprint, minus the wall part (in / beyond a blocking tile, rays from O)
    g = FT42_CELL; u0, v0 = F.min(0) - 2 * g; u1, v1 = F.max(0) + 2 * g
    nu, nv = int(_np42.ceil((u1 - u0) / g)) + 1, int(_np42.ceil((v1 - v0) / g)) + 1
    J = _Im42.new('L', (nu, nv), 0); _Dr42.Draw(J).polygon([((p[0] - u0) / g, (p[1] - v0) / g) for p in F], fill=1, outline=1)
    keep = _np42.asarray(J, bool)   # [v index, u index]
    if grid is not None:
        vv, uu = _np42.nonzero(keep)
        Pw = _ft42_world(O, th, _np42.c_[u0 + uu * g, v0 + vv * g])
        al = _np42.degrees(_np42.arctan2(Pw[:, 1] - O[1], Pw[:, 0] - O[0])); rr = _np42.hypot(Pw[:, 0] - O[0], Pw[:, 1] - O[1])
        ref = al[0]; rel = (al - ref + 180) % 360 - 180
        k = _np42.round((rel - rel.min()) / FT42_RAY_DA).astype(int)
        rays = ref + rel.min() + _np42.arange(k.max() + 1) * FT42_RAY_DA
        d = _ft42_stop(grid, O, rays, rr.max())
        keep[vv[rr >= d[k]], uu[rr >= d[k]]] = False
    # each candidate pixel: kept when the column under it (heights 0 .. h) meets the kept footprint; ztop: the highest
    # such height (the label's point nearest the camera along the pixel's ray)
    ys, xs = _np42.nonzero(C)
    c0 = _ft42_proj(cam, *_ft42_world(O, th, F.mean(0, keepdims=True))[0], 0.0); c1 = _ft42_proj(cam, *_ft42_world(O, th, F.mean(0, keepdims=True))[0], h)
    nz = int(_np42.ceil(abs(c1[1] - c0[1]) / 0.7)) + 2
    ztop = _np42.full(len(ys), -1.0); c, s = _np42.cos(_np42.radians(th)), _np42.sin(_np42.radians(th))
    for z in _np42.linspace(0.0, h, nz):
        x, y = _ft42_unproj(cam, xs + 0.5, ys + 0.5, z)
        u = (x - O[0]) * c + (y - O[1]) * s; v = -(x - O[0]) * s + (y - O[1]) * c
        iu = _np42.round((u - u0) / g).astype(int); iv = _np42.round((v - v0) / g).astype(int)
        inb = (iu >= 0) & (iu < nu) & (iv >= 0) & (iv < nv)
        hit = _np42.zeros(len(iu), bool); hit[inb] = keep[iv[inb], iu[inb]] & (z <= _ft42_hfun(hgt, u[inb]) + 1e-6)
        ztop[hit] = z
    ok = ztop >= 0
    if grid is not None and ok.any():
        # the blocking tiles as boxes (tile x [0, FT42_BLOCK_Z]) in front of the label: a pixel whose ray enters a box above
        # the label's highest point there shows the block (or an effect drawn on it: a hit spark), not the attack
        # v3: bushes too (the user, 10-27: the part of the attack hidden under / behind a bush is not labelled): bush tiles
        # (IsForest in tiles.csv) as boxes FT42_BUSH_Z high, the same test; projectiles pass through bushes, so bushes do
        # not cut the footprint itself (no ray-from-O test for them)
        blk = _ft42_blocking(); bsh = _ft42_bushes(); Pb = Fw; pad = FT42_TILE * 2
        tx0, tx1 = int((Pb[:, 0].min() - pad) // FT42_TILE), int((Pb[:, 0].max() + pad) // FT42_TILE)
        ty0, ty1 = int((Pb[:, 1].min() - pad) // FT42_TILE), int((Pb[:, 1].max() + pad) // FT42_TILE)
        tiles = [(tx, ty, FT42_BLOCK_Z if grid[ty, tx] in blk else FT42_BUSH_Z)
                 for ty in range(max(0, ty0), min(grid.shape[0], ty1 + 1)) for tx in range(max(0, tx0), min(grid.shape[1], tx1 + 1))
                 if grid[ty, tx] in blk or grid[ty, tx] in bsh]
        if tiles:
            i = _np42.nonzero(ok)[0]; Hz = max(FT42_BLOCK_Z, FT42_BUSH_Z)
            ax, ay = _ft42_unproj(cam, xs[i] + 0.5, ys[i] + 0.5, 0.0); bx, by = _ft42_unproj(cam, xs[i] + 0.5, ys[i] + 0.5, Hz)
            dx, dy = bx - ax, by - ay; zin = _np42.full(len(i), -1.0)
            for tx, ty, Ht in tiles:
                lo = _np42.zeros(len(i)); hi = _np42.full(len(i), Ht / Hz)
                for a0, d0, e0, e1 in ((ax, dx, tx * FT42_TILE, (tx + 1) * FT42_TILE), (ay, dy, ty * FT42_TILE, (ty + 1) * FT42_TILE)):
                    with _np42.errstate(divide='ignore', invalid='ignore'):
                        t0 = (e0 - a0) / d0; t1 = (e1 - a0) / d0
                    par = _np42.abs(d0) < 1e-9; inside = (a0 >= e0) & (a0 <= e1)
                    t0 = _np42.where(par, _np42.where(inside, -_np42.inf, _np42.inf), t0); t1 = _np42.where(par, _np42.where(inside, _np42.inf, -_np42.inf), t1)
                    lo = _np42.maximum(lo, _np42.minimum(t0, t1)); hi = _np42.minimum(hi, _np42.maximum(t0, t1))
                zin = _np42.where(lo <= hi, _np42.maximum(zin, hi * Hz), zin)
            ok[i[zin > ztop[i] + 1.0]] = False
    M = _np42.zeros(shape, bool); M[ys[ok], xs[ok]] = True
    if solid is not None and (M & solid).any():
        # cut by the blocks drawn on screen; along the cut, effects drawn over a block (debris, sparks: not solid by depth)
        # would leave crumbs and strips: opened away within FT42_CUT_NEAR px of the block, then specks dropped
        from scipy.ndimage import binary_opening as _op42, binary_dilation as _di42, label as _cc42
        M &= ~solid
        near = _di42(solid, iterations=FT42_CUT_NEAR)
        M = (M & ~near) | (_op42(M, structure=_np42.ones((3, 3), bool), iterations=FT42_CUT_OPEN) & near)
        L, n = _cc42(M)
        if n > 1:
            sz = _np42.bincount(L.ravel()); sz[0] = 0
            M = _np42.isin(L, _np42.nonzero(sz >= max(FT42_SPECK, FT42_SPECK_FRAC * sz.max()))[0])
    return M


def _ft42_place(mem, state):
    """(kin, O, aim, tau, inc (inner, outer), alive) of a wave from its members' logic positions / state rows, or None"""
    import track as _tr
    J = _tr.J
    mem = [o for o in mem if not o.get('ended') and o.get('path')]
    if len(mem) < 2: return None
    kin = FT42_KIN[projtab.table()[mem[0]['t']]['logic']]
    rows = {int(p[0]): p for p in (state or {}).get('proj', [])}
    tau = max(len(o['path']) for o in mem) - 1
    a = _np42.radians([o['angle'] for o in mem]); th0 = _np42.degrees(_np42.arctan2(_np42.sin(a).mean(), _np42.cos(a).mean()))
    obs = []; inc = [None, None]; prior = []
    for o in mem:
        rel = (o['angle'] - th0 + 540) % 360 - 180; c = 0 if abs(rel) < 15.5 else 1; sg = 1 if rel > 0 else -1
        prior.append((o['angle'], c, sg))
        p = rows.get(o['id'])
        i0 = int(p[J['travel0']]) if p is not None else FT42_DEFAULT_INC[c]
        i0 = 142 if i0 >= 134 else 125
        inc[c] = max(inc[c] or 0, i0)
        s0, b, tf = kin['cls'][i0]
        fin = int(o.get('finished') or 0)
        if fin not in (0, 1): continue   # stopped by a wall: its travel counts the whole last update, its position does not
        if p is not None: s = s0 + (int(p[J['travel']]) - tf) * b / i0   # distance from O (exact while it travels / at its range)
        elif not fin: s = min(kin['R'], s0 + b * tau)
        else: s = kin['R']
        obs.append((_np42.array(o['logic'], float), s, c, sg))
    inc = tuple(inc[c] or FT42_DEFAULT_INC[c] for c in (0, 1))
    if not obs: return None
    best = None
    for th in th0 + _np42.linspace(-1.5, 1.5, 151):
        Oi = _np42.array([p - s * _np42.array([_np42.cos(_np42.radians(th + sg * kin['phi'][c])), _np42.sin(_np42.radians(th + sg * kin['phi'][c]))])
                          for p, s, c, sg in obs])
        O = Oi.mean(0); r = ((Oi - O) ** 2).sum()
        # weak prior: each member's integer logic angle (+-0.5 deg) against its slot direction (31 units per degree ~ 1800 x
        # 1 deg in radians / 1): decides the aim when few members travel
        r += sum((31.0 * ((th + sg * kin['phi'][c] - a_ + 540) % 360 - 180)) ** 2 for a_, c, sg in prior) * 0.05
        if best is None or r < best[0]: best = (r, th, O)
    alive = any(not o.get('finished') for o in mem)
    return kin, best[2], best[1], tau, inc, alive


# ---- v5: the sprite model (task 42, 2026-10-27; the coordinator's idea after "lets keep trying") ----------------------
# The drawn attack is rows of camera-facing sprites along the 4 projectile lanes. Each row point of each lane (the same
# predicted, symmetric row points as before: rays from O at aim +-phi, drawn at s(k - FT42_LAG), rows max(first, tau-A)..tau)
# carries one billboard, centred FO further along its lane (per skin; < 0: behind): a ground ellipse of half-axes W (along the logic / screen x) and D (along the logic y = screen up /
# down) around the point, raised to height H; its screen shape is the hull of that ring at z 0 and z H, i.e. it faces the
# camera (does not turn with the aim). A row of age a uses W, D scaled by 1 + (aw - 1) a / A and H by 1 + (ah - 1) a / A, all
# times the start-up ramp r0 (as v4). The label is the union, over neighbouring lanes (j, j+1) and consecutive rows (k, k+1),
# of the convex hull of those (up to) 4 sprites (one row: neighbouring lane pairs): one solid piece that follows the lanes,
# no hull over the whole wave. Lanes stop at walls: a sprite whose ground point is in / beyond a projectile-blocking tile
# seen from O is dropped (its cells use the remaining sprites). Then, as v4: the wall and bush boxes standing in front of the
# label (per pixel, the label point is its nearest sprite's billboard at the pixel's height, or the ground under the pixel
# below the sprite's foot) and the solid pixels cut it. Symmetric on the ground (mirrored lanes, the same sprite sizes),
# predictable (origin, aim, tau, skin, the frame's camera), nothing before the first drawn row or after the attack.
# per skin, the model that fits its drawing better at equal coverage (t42/fit_v5.py vs fit_v4b.py, mean of the back / front
# IoU with the current pixel label): the sprite model for the ice spikes (035, 220), the ground model (v4) for the others
# (their tiles / cones / rocks / flames / stars are followed better by the aim-aligned ground egg)
FT42_MODEL = {6000475: 'sprite', 6000485: 'sprite', 6000477: 'ground', 6000479: 'ground', 6000481: 'ground', 6000483: 'ground', 6000487: 'ground', 6000489: 'ground'}
# sprite parameters (logic units; t42/fit_v5.py, hull 1: the old pixel labels are one convex hull per wave, so is this)
FT42_S = {6000475: dict(W=175.0, D=75.0, H=700.0, aw=0.4, ah=0.4, r0=1.0, fo=25.0, hull=1), 6000485: dict(W=200.0, D=100.0, H=700.0, aw=0.7, ah=0.5, r0=1.0, fo=25.0, hull=1)}
FT42_S_DEFAULT = dict(W=200.0, D=150.0, H=400.0, aw=1.0, ah=1.0, r0=1.0, fo=0.0, hull=0)
FT42_S_RING = 20


def _ft42s_param(t, aim=None):
    p = dict(FT42_S_DEFAULT); p.update(FT42_S.get(t, {})); return _ft42_harm(t, p, aim)


def _ft42s_sprites(kin, inc, O, th, tau, prm, sp, grid=None):
    """[(lane 0..3 from -v to +v, row k, ground point (2,), ring (n, 2) world, height)] of the wave's sprites"""
    A = max(1, int(prm['A'])); k0 = min(tau, max(int(prm['first']), tau - A)); rp = _ft42_ramp(tau, dict(prm, r0=sp['r0']))
    a = _np42.linspace(0, 2 * _np42.pi, FT42_S_RING, endpoint=False); ca, sa = _np42.cos(a), _np42.sin(a)
    lanes = [(1, -1), (0, -1), (0, 1), (1, 1)]   # (pair, side): outer -, inner -, inner +, outer +
    out = []; stop = {}
    if grid is not None:
        for j, (c, sg) in enumerate(lanes):
            stop[j] = _ft42_stop(grid, O, _np42.array([th + sg * kin['phi'][c]]), kin['R'] + 400)[0]
    for j, (c, sg) in enumerate(lanes):
        ang = _np42.radians(th + sg * kin['phi'][c])
        for k in range(k0, tau + 1):
            s = _ft42_sdraw(kin, inc[c], k)
            if grid is not None and s >= stop[j]: continue   # the lane stopped at a wall
            s = max(0.0, s + sp.get('fo', 0.0))   # the sprite's centre FO ahead of (behind: < 0) the drawn row point, along the lane
            age = min(tau - k, A) / float(A)
            fw = rp * (1 + (sp['aw'] - 1) * age); fh = rp * (1 + (sp['ah'] - 1) * age)
            p = _np42.array([O[0] + s * _np42.cos(ang), O[1] + s * _np42.sin(ang)])
            out.append((j, k, p, _np42.c_[p[0] + sp['W'] * fw * ca, p[1] + sp['D'] * fw * sa], sp['H'] * fh))
    return out


def _ft42s_shape(cam, spr, shape, hull=False):
    """(H, W) bool: the union of the cell hulls of the sprites (screen); plus per sprite its screen polygon"""
    Hh, Wd = shape
    polys = {}
    for j, k, p, ring, h in spr:
        P = _np42.concatenate([_ft42_proj(cam, ring[:, 0], ring[:, 1], 0.0), _ft42_proj(cam, ring[:, 0], ring[:, 1], h)])
        polys[(j, k)] = P
    I = _Im42.new('L', (Wd, Hh), 0); d = _Dr42.Draw(I)
    if hull and polys:   # per skin (FT42_S hull 1): one convex hull over all its sprites
        d.polygon([tuple(q) for q in _ft42_hull(_np42.concatenate(list(polys.values())))], fill=1, outline=1)
        return _np42.asarray(I, bool), polys
    ks = sorted({k for j, k in polys}); js = range(4)
    cells = []
    if len(ks) == 1: cells = [[(j, ks[0]), (j + 1, ks[0])] for j in range(3)]
    else: cells = [[(j, k), (j + 1, k), (j, k2), (j + 1, k2)] for j in range(3) for k, k2 in zip(ks[:-1], ks[1:])]
    for cell in cells:
        pts = [polys[q] for q in cell if q in polys]
        if pts: d.polygon([tuple(q) for q in _ft42_hull(_np42.concatenate(pts))], fill=1, outline=1)
    for q, P in polys.items():   # a lone sprite (its neighbours dropped at a wall)
        d.polygon([tuple(x) for x in _ft42_hull(P)], fill=1, outline=1)
    return _np42.asarray(I, bool), polys


def _ft42s_mask(cam, O, th, kin, inc, tau, prm, sp, grid=None, solid=None, shape=(540, 1200)):
    spr = _ft42s_sprites(kin, inc, O, th, tau, prm, sp, grid)
    if not spr: return _np42.zeros(shape, bool)
    M, polys = _ft42s_shape(cam, spr, shape, hull=bool(sp.get('hull', 0)))
    if grid is not None and M.any():
        # the wall / bush boxes in front: the label point of a pixel is its nearest sprite's billboard at the pixel's height
        # (or the ground under the pixel below the sprite's foot)
        ys, xs = _np42.nonzero(M)
        anc = _np42.array([p for _, _, p, _, _ in spr]); hs = _np42.array([h for *_, h in spr])
        a0 = _ft42_proj(cam, anc[:, 0], anc[:, 1], 0.0); a1 = _ft42_proj(cam, anc[:, 0], anc[:, 1], 100.0)
        dd = (xs[:, None] - a0[None, :, 0]) ** 2 + (ys[:, None] - a0[None, :, 1]) ** 2; nb = dd.argmin(1)
        zpx = (a0[nb, 1] - a1[nb, 1]) / 100.0   # px per unit of height at the anchor
        zl = _np42.clip((a0[nb, 1] - (ys + 0.5)) / _np42.maximum(zpx, 1e-6), 0.0, hs[nb])
        blk = _ft42_blocking(); bsh = _ft42_bushes(); pad = FT42_TILE * 2
        tx0, tx1 = int((anc[:, 0].min() - pad) // FT42_TILE), int((anc[:, 0].max() + pad) // FT42_TILE)
        ty0, ty1 = int((anc[:, 1].min() - pad) // FT42_TILE), int((anc[:, 1].max() + pad) // FT42_TILE)
        tiles = [(tx, ty, FT42_BLOCK_Z if grid[ty, tx] in blk else FT42_BUSH_Z)
                 for ty in range(max(0, ty0), min(grid.shape[0], ty1 + 1)) for tx in range(max(0, tx0), min(grid.shape[1], tx1 + 1))
                 if grid[ty, tx] in blk or grid[ty, tx] in bsh]
        if tiles:
            Hz = max(FT42_BLOCK_Z, FT42_BUSH_Z)
            ax, ay = _ft42_unproj(cam, xs + 0.5, ys + 0.5, 0.0); bx, by = _ft42_unproj(cam, xs + 0.5, ys + 0.5, Hz)
            dx, dy = bx - ax, by - ay; zin = _np42.full(len(xs), -1.0)
            for tx, ty, Ht in tiles:
                lo = _np42.zeros(len(xs)); hi = _np42.full(len(xs), Ht / Hz)
                for a0_, d0, e0, e1 in ((ax, dx, tx * FT42_TILE, (tx + 1) * FT42_TILE), (ay, dy, ty * FT42_TILE, (ty + 1) * FT42_TILE)):
                    with _np42.errstate(divide='ignore', invalid='ignore'):
                        t0 = (e0 - a0_) / d0; t1 = (e1 - a0_) / d0
                    par = _np42.abs(d0) < 1e-9; inside = (a0_ >= e0) & (a0_ <= e1)
                    t0 = _np42.where(par, _np42.where(inside, -_np42.inf, _np42.inf), t0); t1 = _np42.where(par, _np42.where(inside, _np42.inf, -_np42.inf), t1)
                    lo = _np42.maximum(lo, _np42.minimum(t0, t1)); hi = _np42.minimum(hi, _np42.maximum(t0, t1))
                zin = _np42.where(lo <= hi, _np42.maximum(zin, hi * Hz), zin)
            occ = zin > zl + 1.0
            M[ys[occ], xs[occ]] = False
    if solid is not None and (M & solid).any():
        from scipy.ndimage import binary_opening as _op42, binary_dilation as _di42, label as _cc42
        M &= ~solid
        near = _di42(solid, iterations=FT42_CUT_NEAR)
        M = (M & ~near) | (_op42(M, structure=_np42.ones((3, 3), bool), iterations=FT42_CUT_OPEN) & near)
        L, n = _cc42(M)
        if n > 1:
            sz = _np42.bincount(L.ravel()); sz[0] = 0
            M = _np42.isin(L, _np42.nonzero(sz >= max(FT42_SPECK, FT42_SPECK_FRAC * sz.max()))[0])
    return M


FT42_CUSTOM = globals().get('FT42_CUSTOM', {})   # type id -> per-skin label function (v7 hook; empty = v6 for every skin)


def ft42_wave_mask(mem, cam, R=None, state=None, grid=None, solid=None, shape=(540, 1200)):
    """the template label of one Frank attack wave (its frame objects), or None (not labelled on this frame)"""
    pl = _ft42_place(mem, state)
    if pl is None: return None
    kin, O, th, tau, inc, alive = pl
    t = mem[0]['t']
    if t in FT42_CUSTOM:   # 10-28 (v7): a skin's own label function (py/fixes/z_1028_frank_v7_sNNN.py), same inputs, no pixels
        return FT42_CUSTOM[t](mem=mem, cam=cam, state=state, kin=kin, O=O, th=th, tau=tau, inc=inc, alive=alive,
                              grid=grid, solid=solid, shape=shape)
    prm = _ft42_param(t, th)
    if not alive or tau < prm['first']: return None
    if FT42_MODEL.get(t, 'sprite') == 'sprite':
        return _ft42s_mask(cam, O, th, kin, inc, tau, prm, _ft42s_param(t, th), grid=grid, solid=solid, shape=shape)
    F = _ft42_footprint(kin, inc, tau, prm)
    return _ft42_mask(cam, O, th, F, _ft42_height(kin, inc, tau, prm), grid=grid, solid=solid, shape=shape)


if 'labels' in globals() and 'POST_BODY' in globals():
    _ft42_prev = labels

    def labels(objs, cam, R, state=None, ended=(), img=None):
        import bulletlabels as BL
        P = _ft42_prev(objs, cam, R, state=state, ended=ended, img=img)
        groups = {}
        for o in objs:
            if o['t'] in FT42_ATTACK: groups.setdefault(o.get('label_id', o['id']), []).append(o)
        if not groups: return P
        ids, share, lids = P['ids'], P['share'], P['lids']
        grid = solid = None
        cv = R['cv'][::2, ::2]
        for gid, mem in groups.items():
            if _ft42_place(mem, state) is None and any(not o.get('ended') and not o.get('finished') for o in mem):
                continue   # a travelling wave the template cannot place (fewer than 2 members tracked): its old label stays
            ks = [lids.index(l) for l in {(gid, -1, 'body')} | {(o['id'], -1, 'body') for o in mem} if l in lids]
            if ks:   # the old (pixel) label of the wave goes
                old = _np42.isin(ids, ks); ids[old] = -1; share[old] = 0
            if grid is None:
                _ZS_IMG[0] = img
                grid = _zs_grid(R); solid = _zs_solid(cam, R)
            M = ft42_wave_mask(mem, cam, R, state, grid=grid, solid=solid, shape=ids.shape)
            if M is None or not M.any(): continue
            lid = (gid, -1, 'body')
            if lid not in lids: lids.append(lid)
            k = lids.index(lid); m = M & (ids < 0)
            ids[m] = k; share[m] = _np42.maximum(cv[m] / 255.0, BL.CUT)
        return P
