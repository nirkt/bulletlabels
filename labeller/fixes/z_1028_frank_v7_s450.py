# Frank 450 (FrankFairies, attack type 6000489) label v7 (task t42 round 7, 2026-10-28). Install as py/fixes/z_1028_frank_v7_s450.py
# (after z_1027_frank_template.py: it only registers FT42_CUSTOM[6000489], the hook of the template's ft42_wave_mask).
#
# THE DRAWING (measured on fa_450 / fb_450, 17 aims, every imaged frame): the attack lays one ROW of flat, glowing blue star outlines
# (additive ground sprites, a small sprout standing in each) per update on the four projectile lanes (rays from the origin O at the
# aim +-8.1 deg inner pair, +-23.2 deg outer pair; same kinematics as every Frank skin: s(k) = s0 + b k, slow 486.2 / 218.8, fast
# 548.7 / 250.1, range 1800). A star is NOT a thing that stands up: it lies on the floor, round in the world (outline radius ~125,
# its glow ~150 logic units), the stars of neighbouring lanes / rows touch tip to tip. Measured, per star: its centre sits at the lane
# point of row k at s(k - 0.40) (0.40 of an update behind the logic position) and lifted ~130 logic units (the sprite is drawn above the
# floor: 5-6 px straight up on screen at every aim, the same at every aim), size the same for age 0 and age 1 rows (fit: 140 / 150),
# first stars on row 2 (nothing at tau 0, 1). Only the rows of age <= 1 are the attack (the older rows grow big sprouts and fade: after-
# effects), as in the earlier rounds.
# THE LABEL (v6 was one hull of aim-aligned 'eggs' with margins fitted to the old pixel label, which itself stretched to the brawler's
# aura / to specks near the hammer, so v6 was loose at the back end, 30-50 % of its pixels on tau 2 frames were bare floor):
# a function of only the origin O, the aim, tau, the skin, the static map (walls and bushes cut it, characters do not):
#   stars = lanes x rows (max(2, tau - 1) .. tau), centre = O + s(k - F450['lag']) along the lane, lanes that stopped at a wall drop
#   their stars from the wall on; each star = a ground DISC of radius F450['R'] (a disc, not the 5-point star: a world-aligned star
#   would not be mirror-symmetric about the aim axis; the disc's radius is the star's tip radius + glow), at the height F450['zl'];
#   label = the convex hull of all discs (one label per wave), projected with the frame's camera, then cut as the template cuts: the
#   footprint beyond a projectile-blocking tile seen from O, the blocking tiles (600 high) and bushes (450 high) standing in front,
#   the solid blocks drawn on screen (crumbs cleaned). Mirror-symmetric in the ground (lanes mirrored, discs round), predictable (no pixels).
# Labelled for tau >= 2 while at least one projectile of the wave still travels (nothing before the first star / after the wave ended).
# Parameters (lag 0.40, zl 130, R 155) fitted on the visual (bright blue pixels nearest to the age <= 1 stars), not on the old pixel
# label (which stretches to the brawler's aura and to specks): see report.md. Knobs: R (looser / tighter), union=1 (the union of the
# discs instead of their hull: ~5 % smaller, not convex, use R 175), ray_cut=0 (no 'beyond the wall as seen from O' cut).
import numpy as _n450
from PIL import Image as _I450, ImageDraw as _D450

F450 = dict(first=2, A=1, lag=0.40, zl=130.0, R=155.0, ring=24, union=0, ray_cut=1)
F450_LANES = [(1, -1), (0, -1), (0, 1), (1, 1)]   # (pair: 0 inner / 1 outer, side): from the -v side to the +v side


def _f450_stars(kin, inc, O, th, tau, grid):
    """[(lane, row k, age, centre (x, y))]: the stars of the wave at update tau (rows of age <= A from the first drawn row; lanes
    that stopped at a wall have no stars from the wall on)"""
    P = F450; k0 = max(P['first'], tau - P['A']); stop = {}
    if grid is not None:
        for j, (c, sg) in enumerate(F450_LANES):
            stop[j] = _ft42_stop(grid, O, _n450.array([th + sg * kin['phi'][c]]), kin['R'] + 400)[0]
    out = []
    for j, (c, sg) in enumerate(F450_LANES):
        ang = _n450.radians(th + sg * kin['phi'][c]); s0, b, _f = kin['cls'][inc[c]]
        for k in range(k0, tau + 1):
            s = max(0.0, min(kin['R'], s0 + b * (k - P['lag'])))
            if grid is not None and s >= stop[j]: continue
            out.append((j, k, tau - k, _n450.array([O[0] + s * _n450.cos(ang), O[1] + s * _n450.sin(ang)])))
    return out


def _f450_mask(cam, O, th, stars, grid=None, solid=None, shape=(540, 1200)):
    P = F450; Hh, Wd = shape
    a = _n450.linspace(0, 2 * _n450.pi, P['ring'], endpoint=False); ca, sa = _n450.cos(a), _n450.sin(a)
    rings = [_n450.c_[c[0] + P['R'] * ca, c[1] + P['R'] * sa] for j, k, age, c in stars]
    if not rings: return _n450.zeros(shape, bool)
    zl = P['zl']
    # screen candidate region: the discs at height zl (convex hull of all, or one polygon per disc: F450['union'])
    I = _I450.new('L', (Wd, Hh), 0); d = _D450.Draw(I)
    scr = [_ft42_proj(cam, r[:, 0], r[:, 1], zl) for r in rings]
    if P['union']:
        for q in scr: d.polygon([tuple(p) for p in q], fill=1, outline=1)
    else:
        d.polygon([tuple(p) for p in _ft42_hull(_n450.concatenate(scr))], fill=1, outline=1)
    C = _n450.asarray(I, bool)
    if not C.any(): return C
    # footprint raster in attack-local (u, v) cells, minus the part in / beyond a blocking tile seen from O (rays from O)
    th_r = _n450.radians(th); c_, s_ = _n450.cos(th_r), _n450.sin(th_r)
    loc = [_n450.c_[(r[:, 0] - O[0]) * c_ + (r[:, 1] - O[1]) * s_, -(r[:, 0] - O[0]) * s_ + (r[:, 1] - O[1]) * c_] for r in rings]
    L = _n450.concatenate(loc); g = FT42_CELL; u0, v0 = L.min(0) - 2 * g; u1, v1 = L.max(0) + 2 * g
    nu, nv = int(_n450.ceil((u1 - u0) / g)) + 1, int(_n450.ceil((v1 - v0) / g)) + 1
    J = _I450.new('L', (nu, nv), 0); dj = _D450.Draw(J)
    polys = loc if P['union'] else [_ft42_hull(L)]
    for q in polys: dj.polygon([((p[0] - u0) / g, (p[1] - v0) / g) for p in q], fill=1, outline=1)
    keep = _n450.asarray(J, bool)
    if grid is not None and P['ray_cut']:
        vv, uu = _n450.nonzero(keep)
        Pw = _ft42_world(O, th, _n450.c_[u0 + uu * g, v0 + vv * g])
        al = _n450.degrees(_n450.arctan2(Pw[:, 1] - O[1], Pw[:, 0] - O[0])); rr = _n450.hypot(Pw[:, 0] - O[0], Pw[:, 1] - O[1])
        ref = al[0]; rel = (al - ref + 180) % 360 - 180
        k = _n450.round((rel - rel.min()) / FT42_RAY_DA).astype(int)
        rays = ref + rel.min() + _n450.arange(k.max() + 1) * FT42_RAY_DA
        dd = _ft42_stop(grid, O, rays, rr.max())
        keep[vv[rr >= dd[k]], uu[rr >= dd[k]]] = False
    ys, xs = _n450.nonzero(C)
    x, y = _ft42_unproj(cam, xs + 0.5, ys + 0.5, zl)
    u = (x - O[0]) * c_ + (y - O[1]) * s_; v = -(x - O[0]) * s_ + (y - O[1]) * c_
    iu = _n450.round((u - u0) / g).astype(int); iv = _n450.round((v - v0) / g).astype(int)
    inb = (iu >= 0) & (iu < nu) & (iv >= 0) & (iv < nv)
    ok = _n450.zeros(len(iu), bool); ok[inb] = keep[iv[inb], iu[inb]]
    if grid is not None and ok.any():
        # blocking tiles (FT42_BLOCK_Z high) and bushes (FT42_BUSH_Z high) standing in front of the flat label: a pixel whose ray
        # enters such a box above the label's height shows the block, not the attack (as the template does)
        blk = _ft42_blocking(); bsh = _ft42_bushes(); Pb = _n450.concatenate(rings); pad = FT42_TILE * 2
        tx0, tx1 = int((Pb[:, 0].min() - pad) // FT42_TILE), int((Pb[:, 0].max() + pad) // FT42_TILE)
        ty0, ty1 = int((Pb[:, 1].min() - pad) // FT42_TILE), int((Pb[:, 1].max() + pad) // FT42_TILE)
        tiles = [(tx, ty, FT42_BLOCK_Z if grid[ty, tx] in blk else FT42_BUSH_Z)
                 for ty in range(max(0, ty0), min(grid.shape[0], ty1 + 1)) for tx in range(max(0, tx0), min(grid.shape[1], tx1 + 1))
                 if grid[ty, tx] in blk or grid[ty, tx] in bsh]
        if tiles:
            i = _n450.nonzero(ok)[0]; Hz = max(FT42_BLOCK_Z, FT42_BUSH_Z)
            ax, ay = _ft42_unproj(cam, xs[i] + 0.5, ys[i] + 0.5, 0.0); bx, by = _ft42_unproj(cam, xs[i] + 0.5, ys[i] + 0.5, Hz)
            dx, dy = bx - ax, by - ay; zin = _n450.full(len(i), -1.0)
            for tx, ty, Ht in tiles:
                lo = _n450.zeros(len(i)); hi = _n450.full(len(i), Ht / Hz)
                for a0, d0, e0, e1 in ((ax, dx, tx * FT42_TILE, (tx + 1) * FT42_TILE), (ay, dy, ty * FT42_TILE, (ty + 1) * FT42_TILE)):
                    with _n450.errstate(divide='ignore', invalid='ignore'):
                        t0 = (e0 - a0) / d0; t1 = (e1 - a0) / d0
                    par = _n450.abs(d0) < 1e-9; inside = (a0 >= e0) & (a0 <= e1)
                    t0 = _n450.where(par, _n450.where(inside, -_n450.inf, _n450.inf), t0); t1 = _n450.where(par, _n450.where(inside, _n450.inf, -_n450.inf), t1)
                    lo = _n450.maximum(lo, _n450.minimum(t0, t1)); hi = _n450.minimum(hi, _n450.maximum(t0, t1))
                zin = _n450.where(lo <= hi, _n450.maximum(zin, hi * Hz), zin)
            ok[i[zin > zl + 1.0]] = False
    M = _n450.zeros(shape, bool); M[ys[ok], xs[ok]] = True
    if solid is not None and (M & solid).any():
        from scipy.ndimage import binary_opening as _op450, binary_dilation as _di450, label as _cc450
        M &= ~solid
        near = _di450(solid, iterations=FT42_CUT_NEAR)
        M = (M & ~near) | (_op450(M, structure=_n450.ones((3, 3), bool), iterations=FT42_CUT_OPEN) & near)
        Lb, n = _cc450(M)
        if n > 1:
            sz = _n450.bincount(Lb.ravel()); sz[0] = 0
            M = _n450.isin(Lb, _n450.nonzero(sz >= max(FT42_SPECK, FT42_SPECK_FRAC * sz.max()))[0])
    return M


def _f450_label(mem=None, cam=None, state=None, kin=None, O=None, th=None, tau=None, inc=None, alive=None, grid=None, solid=None, shape=(540, 1200)):
    """FT42_CUSTOM[6000489]: the wave's label mask (half-resolution screen) or None (not labelled on this frame)"""
    if not alive or tau < F450['first']: return None
    st = _f450_stars(kin, inc, O, th, tau, grid)
    return _f450_mask(cam, O, th, st, grid=grid, solid=solid, shape=shape)


if 'FT42_CUSTOM' not in globals(): raise RuntimeError('z_1028_frank_v7_s450 needs the template z_1027_frank_template.py (FT42_CUSTOM hook) loaded before it')
FT42_CUSTOM[6000489] = _f450_label
