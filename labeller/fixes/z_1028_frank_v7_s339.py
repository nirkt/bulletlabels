# Frank v7, skin 339 HammerDudeHindu (attack type 6000481, rocks): FT42_CUSTOM label function.  (task t42 round 7, 2026-10-28)
#
# What the skin draws (studied frame by frame on 17 aims; t42/v7/s339/study/): the four lanes (inner pair +-8.1 deg, outer pair
# +-23.2 deg) get rocks. Nothing on rows 0 and 1. Row 2 (the first drawn row) is a PUFF on the first drawn update: four dirt blobs
# of only ~8 x 6 screen px, at the client's drawn lane point (the logic point ~0.36 update behind) and ~6.5 px BELOW its ground
# projection; two updates later row 2 is a small rock, and the front row (row 4, age 0) is already a big rock. Row 3 never gets a rock
# (debris and scuff trails only); dirt scatter and dark scuff trails lie in the ground plane around the rocks. A lane whose projectile a
# wall stopped at update k has rocks only on its rows before k (nothing at row k: fb aim 75, tau 2; the wall-hit dirt is not a row).
# The old (pixel) label of the first drawn update is therefore a thin sliver (160-940 px) through the four puffs, while v6's single
# egg family (r0 start-up ramp) gave a blob 2-4x its area (IoU with the old label 0.16-0.44, mean 0.30). On the rock frames (tau 4)
# v6 is already close to the ceiling (the old label itself is only 0.93 IoU with the hull of the rock pixels), but it hulled the row
# points of a lane beyond the wall that stopped it (empty ground between that lane's last rock and the wall).
#
# Model (a function of the attack origin O, the aim, tau, the skin and the camera only; no pixels; ground-symmetric: the lanes are
# mirrored and every lane carries the same sprite / egg):
#   tau < first (2) or the wave is over: no label (as before).
#   Start-up stage, PUFF sprites (tau = 2, 3): for every lane and every drawn row k in max(2, tau-2) .. tau a camera-facing ellipse
#     (half width w along the screen x, half height h in logic z, centre at height zc, i.e. below the ground projection) at the drawn
#     lane point s(k - L), all hulled together (one convex piece like the old label). Fitted on the 14 tau-2 frames: L 0.355,
#     w 36.2, h 75.4, zc -139.3 (tightest ellipse that covers >= 95 % of the drawn puff pixels and >= 97 % of the old label).
#   Rock stage (tau >= 3): v6's ground model of this skin (egg around the row points of the four lanes, raised to the rock height,
#     hulled; mb / mbo / mv / mvo / mf / g0 / hn / ho, mb-mbo with aim harmonics) with its own refit parameters (below) and the
#     start-up ramp r0 = 0: its size is rho(tau) = clip((tau - first) / FT42_RAMP, 0, 1) times the full one, i.e. absent at tau 2,
#     half at tau 3 (not imaged: interpolated), full from tau 4 on. A lane's row points from the update at which the wall stops its
#     projectile on (FT42V7_339_WALLM below) are dropped BEFORE the hull. The label is the union of both stages (the puffs only while
#     rho < 1: afterwards they lie inside the rock egg). A crumb (< FT42_SPECK px) of the puff hull left by a wall cut is dropped.
#   Walls, bushes, solid blocks: as the template. The puff lanes stop at the wall like the rock lanes; wall and bush boxes in front of
#     a pixel cut it, solid pixels cut it, the cut's edge band is cleaned; the rock stage is cut by _ft42_mask like v6.
import numpy as _np42v7339

# a lane's projectile is stopped by a wall at the first update k whose logic distance s(k) + FT42V7_339_WALLM reaches the first blocking tile along
# its ray from O (measured on 17 aims: stopped lanes have stop - s(k) <= 31, lanes that kept travelling >= 34); its row k and the later ones have no rock
FT42V7_339_WALLM = 32.5
FT42V7_339 = dict(L=0.355, w=36.2, h=75.4, zc=-139.3)
# rock stage: v6's parameters of this skin refitted (t42/v7/s339/work/fit_rock.py, fit_rock2.py) under the dropped-lane footprint on the 15 tau-4 frames of
# the 17 aims: mean IoU with the old label, pooled coverage of the old label >= 0.965, of the rock (brown) pixels >= 0.96, every frame >= 0.88
FT42V7_339_P = dict(A=2, first=2, mb=250.0, mbo=100.0, mv=150.0, mvo=87.5, mf=125.0, hn=225.0, ho=75.0, g0=0.76, r0=0.0)
FT42V7_339_HARM = {'mbo': [100.0, 17.62, -25.32]}   # c0 + c1 cos(aim) + s1 sin(aim), as FT42_HARM


def _ft42v7_339_sprite(cam, P, w, h, zc, n=14):
    """screen polygon (n, 2) of a camera-facing ellipse at ground point P (logic): centre at height zc, half width w (logic x
    units), half height h (logic z units)"""
    P = _np42v7339.asarray(P, float); X = _np42v7339.array([P[0], P[0] + 1.0, P[0]]); Y = _np42v7339.array([P[1], P[1], P[1]])
    Q = _ft42_proj(cam, X, Y, _np42v7339.array([zc, zc, zc + 1.0]))
    ex, ez = Q[1] - Q[0], Q[2] - Q[0]
    a = _np42v7339.linspace(0, 2 * _np42v7339.pi, n, endpoint=False)
    return Q[0] + (w * _np42v7339.cos(a))[:, None] * ex + (h * _np42v7339.sin(a))[:, None] * ez


def _ft42v7_339_puffs(cam, kin, inc, O, th, tau, prm, grid, shape):
    """(H, W) bool: hull of the puff sprites of every lane and drawn row; plus the lane points (n, 2) and their z tops"""
    A = max(1, int(prm['A'])); k0 = min(tau, max(int(prm['first']), tau - A)); P7 = FT42V7_339
    lanes = [(1, -1), (0, -1), (0, 1), (1, 1)]
    stop = {}
    if grid is not None:
        for j, (c, sg) in enumerate(lanes):
            stop[j] = _ft42_stop(grid, O, _np42v7339.array([th + sg * kin['phi'][c]]), kin['R'] + 400)[0]
    polys = []; anc = []
    for j, (c, sg) in enumerate(lanes):
        ang = _np42v7339.radians(th + sg * kin['phi'][c]); s0, b, _ = kin['cls'][inc[c]]
        for k in range(k0, tau + 1):
            s = max(0.0, min(kin['R'], s0 + b * (k - P7['L'])))
            if grid is not None and min(kin['R'], s0 + b * k) + FT42V7_339_WALLM >= stop[j]: continue   # the projectile stopped at the wall at or before update k: no puff
            p = _np42v7339.array([O[0] + s * _np42v7339.cos(ang), O[1] + s * _np42v7339.sin(ang)])
            polys.append(_ft42v7_339_sprite(cam, p, P7['w'], P7['h'], P7['zc'])); anc.append(p)
    H, Wd = shape
    if not polys: return _np42v7339.zeros(shape, bool), _np42v7339.zeros((0, 2))
    I = _Im42.new('L', (Wd, H), 0); _Dr42.Draw(I).polygon([tuple(q) for q in _ft42_hull(_np42v7339.concatenate(polys))], fill=1, outline=1)
    return _np42v7339.asarray(I, bool), _np42v7339.array(anc)


def _ft42v7_339_cut(M, cam, anc, grid, solid):
    """the template's cuts (_ft42s_mask): wall / bush boxes in front of a pixel, solid pixels, crumbs along the cut. The puffs
    lie on the ground (height 0)"""
    if grid is not None and M.any():
        ys, xs = _np42v7339.nonzero(M)
        a0 = _ft42_proj(cam, anc[:, 0], anc[:, 1], 0.0)
        nb = ((xs[:, None] - a0[None, :, 0]) ** 2 + (ys[:, None] - a0[None, :, 1]) ** 2).argmin(1)
        blk = _ft42_blocking(); bsh = _ft42_bushes(); pad = FT42_TILE * 2
        tx0, tx1 = int((anc[:, 0].min() - pad) // FT42_TILE), int((anc[:, 0].max() + pad) // FT42_TILE)
        ty0, ty1 = int((anc[:, 1].min() - pad) // FT42_TILE), int((anc[:, 1].max() + pad) // FT42_TILE)
        tiles = [(tx, ty, FT42_BLOCK_Z if grid[ty, tx] in blk else FT42_BUSH_Z)
                 for ty in range(max(0, ty0), min(grid.shape[0], ty1 + 1)) for tx in range(max(0, tx0), min(grid.shape[1], tx1 + 1))
                 if grid[ty, tx] in blk or grid[ty, tx] in bsh]
        if tiles:
            Hz = max(FT42_BLOCK_Z, FT42_BUSH_Z)
            ax, ay = _ft42_unproj(cam, xs + 0.5, ys + 0.5, 0.0); bx, by = _ft42_unproj(cam, xs + 0.5, ys + 0.5, Hz)
            dx, dy = bx - ax, by - ay; zin = _np42v7339.full(len(xs), -1.0)
            for tx, ty, Ht in tiles:
                lo = _np42v7339.zeros(len(xs)); hi = _np42v7339.full(len(xs), Ht / Hz)
                for a0_, d0, e0, e1 in ((ax, dx, tx * FT42_TILE, (tx + 1) * FT42_TILE), (ay, dy, ty * FT42_TILE, (ty + 1) * FT42_TILE)):
                    with _np42v7339.errstate(divide='ignore', invalid='ignore'):
                        t0 = (e0 - a0_) / d0; t1 = (e1 - a0_) / d0
                    par = _np42v7339.abs(d0) < 1e-9; inside = (a0_ >= e0) & (a0_ <= e1)
                    t0 = _np42v7339.where(par, _np42v7339.where(inside, -_np42v7339.inf, _np42v7339.inf), t0)
                    t1 = _np42v7339.where(par, _np42v7339.where(inside, _np42v7339.inf, -_np42v7339.inf), t1)
                    lo = _np42v7339.maximum(lo, _np42v7339.minimum(t0, t1)); hi = _np42v7339.minimum(hi, _np42v7339.maximum(t0, t1))
                zin = _np42v7339.where(lo <= hi, _np42v7339.maximum(zin, hi * Hz), zin)
            occ = zin > 1.0
            M[ys[occ], xs[occ]] = False
    if solid is not None and (M & solid).any():
        from scipy.ndimage import binary_opening as _op42, binary_dilation as _di42, label as _cc42
        M &= ~solid
        near = _di42(solid, iterations=FT42_CUT_NEAR)
        M = (M & ~near) | (_op42(M, structure=_np42v7339.ones((3, 3), bool), iterations=FT42_CUT_OPEN) & near)
        L, n = _cc42(M)
        if n > 1:
            sz = _np42v7339.bincount(L.ravel()); sz[0] = 0
            M = _np42v7339.isin(L, _np42v7339.nonzero(sz >= max(FT42_SPECK, FT42_SPECK_FRAC * sz.max()))[0])
    return M


def _ft42v7_339_footprint(kin, inc, O, th, tau, prm, grid):
    """v6's footprint (_ft42_footprint: convex hull of the egg around every row point of the four lanes) but a lane that a wall
    stopped has no rocks beyond the wall: its row points at or past the first blocking tile are dropped before the hull (v6 hulled
    them and only cut the result at the wall, which left the empty ground between the stopped lane's last rock and the wall)"""
    stop = {}
    if grid is not None:
        for c in (0, 1):
            for sg in (1, -1):
                stop[(c, sg)] = _ft42_stop(grid, O, _np42v7339.array([th + sg * kin['phi'][c]]), kin['R'] + 400)[0]
    a = _np42v7339.linspace(0, 2 * _np42v7339.pi, FT42_EGG_N, endpoint=False)
    ca, sa = _np42v7339.cos(a), _np42v7339.sin(a); rp = _ft42_ramp(tau, prm); A = max(1, int(prm['A']))
    pts = []
    for c in (0, 1):
        ph = _np42v7339.radians(kin['phi'][c])
        for k in range(min(tau, max(int(prm['first']), tau - int(prm['A']))), tau + 1):
            s = _ft42_sdraw(kin, inc[c], k); age = min(tau - k, A) / float(A)
            mvk = prm['mv'] + (prm.get('mvo', prm['mv']) - prm['mv']) * age
            mbk = prm['mb'] + (prm.get('mbo', prm['mb']) - prm['mb']) * age
            sc = prm.get('g0', 1.0) + (1.0 - prm.get('g0', 1.0)) * age
            egg = _np42v7339.c_[ca * _np42v7339.where(ca > 0, prm['mf'], mbk), sa * mvk] * rp * sc
            sl = min(kin['R'], kin['cls'][inc[c]][0] + kin['cls'][inc[c]][1] * k)   # the projectile's own distance at update k
            for sg in (1, -1):
                if grid is not None and sl + FT42V7_339_WALLM >= stop[(c, sg)]: continue   # the projectile stopped at the wall at or before update k: no rock
                pts.append(_np42v7339.array([s * _np42v7339.cos(ph), sg * s * _np42v7339.sin(ph)]) + egg)
    return _ft42_hull(_np42v7339.concatenate(pts)) if pts else None


def _ft42v7_339(mem=None, cam=None, state=None, kin=None, O=None, th=None, tau=None, inc=None, alive=None, grid=None, solid=None,
                shape=(540, 1200)):
    prm = dict(FT42V7_339_P)   # r0 = 0: the rock stage ramps from nothing (tau = first, the puffs only) to full size at first + FT42_RAMP
    a = _np42v7339.radians(th); basis = _np42v7339.array([1.0, _np42v7339.cos(a), _np42v7339.sin(a)])
    for k, co in FT42V7_339_HARM.items(): prm[k] = float(_np42v7339.dot(_np42v7339.asarray(co, float), basis[:len(co)]))
    if not alive or tau < prm['first']: return None
    M = _np42v7339.zeros(shape, bool)
    if tau - prm['first'] > 0:
        F = _ft42v7_339_footprint(kin, inc, O, th, tau, prm, grid)
        if F is not None and len(F) >= 3:
            M |= _ft42_mask(cam, O, th, F, _ft42_height(kin, inc, tau, prm), grid=grid, solid=solid, shape=shape)
    if tau - prm['first'] < FT42_RAMP:
        Mp, anc = _ft42v7_339_puffs(cam, kin, inc, O, th, tau, prm, grid, shape)
        if len(anc):
            Mp = _ft42v7_339_cut(Mp, cam, anc, grid, solid)
            if Mp.any():   # a crumb of the puff hull left by a wall cut (a wave that hit a wall at once draws no puffs) is dropped
                from scipy.ndimage import label as _cc42
                Lp, n = _cc42(Mp)
                if n:
                    sz = _np42v7339.bincount(Lp.ravel()); sz[0] = 0
                    Mp = _np42v7339.isin(Lp, _np42v7339.nonzero(sz >= FT42_SPECK)[0])
            M |= Mp
    return M


FT42_CUSTOM[6000481] = _ft42v7_339
