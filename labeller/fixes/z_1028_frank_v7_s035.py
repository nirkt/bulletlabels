# Frank 035 FrankDefault (attack type 6000475; also Gold / Silver / Xmas / Vacation), v7 (task t42 round 7, 2026-10-28).
# A per-skin label function for the installed Frank template (py/fixes/z_1027_frank_template.py, v6): this file only registers
#   FT42_CUSTOM[6000475] = _ft7s035_wave
# and uses the template's helpers (_ft42_stop, _ft42_proj, _ft42_unproj, _ft42s_shape, _ft42_blocking, _ft42_bushes, ...) so walls,
# bushes and solid blocks cut it exactly as in v6.  Same inputs as v6 (origin O, aim, tau, skin, the static map, the frame's
# camera), no pixels, mirror-symmetric about the aim axis in the ground.
#
# What the skin draws (studied on fa_035 / fb_035 + the 0 / 90 data, ID pass per draw): the attack is a spray of ~8 white-blue
# ice-crystal sprites per update (a camera-facing billboard, 20-35 px wide and as tall at half resolution), scattered randomly
# around the four lanes (rays at aim +-8.1 / +-23.2 deg) where the projectiles are; a crystal is the attack for its first three
# updates (age 0, 1, 2), then it turns into a cyan blob (after-effect).  The old pixel label is the convex hull of those crystals.
# The scatter is random per attack (the crystals do not repeat between waves), so only an envelope can be predicted.
#
# Model (v7 = v6's sprite model, re-tied to what each edge of the hull is made of; 8 parameters, all in FT7_P035):
#   every (lane, row k) of the rows tau-2 .. tau carries one billboard: a ground ellipse of half-axes W, D (logic units, camera axes)
#   around its centre, raised to H; the label is the convex hull of all of them (the user's one hull per wave).  The row's centre is
#   FO ahead of the drawn row point s(k - FT42_LAG) along its lane.  The billboard of a row of AGE a = tau - k (0, 1, 2) has width
#   x w(a), height x h(a) and forward offset FO(a), each linear between age 0 (w = h = 1, FO = fo0) and age 2 (w2, h2, fo2); r0 scales
#   width and height on the first update (r0 -> 1 over FT42_RAMP updates).
#   v6 had the same kind of size aging (0.4 / 0.4) but ONE forward offset (25) for every age.  The fit (fit7b.py: IoU with the old
#   label, back and front half scored separately, subject to >= 97 % of the drawn crystal pixels, >= 95 % in each half, >= 90 % of
#   the old label) moves the centres of the oldest labelled row ~115 units ahead (the back end near the brawler was the loose
#   part), widens the billboards (W 175 -> 190, D 64 -> 85, to keep the crystals) and makes the aged billboards less flat
#   (h2 0.4 -> 0.6): the back half of the label gets ~5 IoU points tighter at the same coverage; the front half is unchanged.
# Fitted on the 17 aims (fa_035 / fb_035 + the 0 / 90 deg data); leave-one-aim-out and sw2+fa -> fb held-out checks in logs/run11_loao.log and logs/run11_fafb.log; full-pipeline numbers in logs/verify7.log.
import numpy as _np7
from PIL import Image as _Im7, ImageDraw as _Dr7

FT7_P035 = dict(W=190.0, D=85.0, H=625.0, w2=0.40, h2=0.60, fo0=15.0, fo2=115.0, r0=0.95)
FT7_LANES = [(1, -1), (0, -1), (0, 1), (1, 1)]   # (pair: 0 inner / 1 outer, side) from -v to +v
FT7_RING = 20


def _ft7s035_sprites(kin, inc, O, th, tau, grid=None, P=None):
    """[(lane, row k, ground point, ring (n, 2) world, height)] as the template's _ft42s_sprites"""
    P = P or FT7_P035; A = 2
    a = _np7.linspace(0, 2 * _np7.pi, FT7_RING, endpoint=False); ca, sa = _np7.cos(a), _np7.sin(a)
    ramp = P['r0'] + (1.0 - P['r0']) * min(1.0, tau / float(FT42_RAMP))   # the first update's spray is a little smaller
    k0 = max(0, tau - A); out = []; stop = {}
    if grid is not None:
        for j, (c, sg) in enumerate(FT7_LANES):
            stop[j] = _ft42_stop(grid, O, _np7.array([th + sg * kin['phi'][c]]), kin['R'] + 400)[0]
    for j, (c, sg) in enumerate(FT7_LANES):
        ang = _np7.radians(th + sg * kin['phi'][c])
        for k in range(k0, tau + 1):
            age = tau - k
            s = _ft42_sdraw(kin, inc[c], k)
            if grid is not None and s >= stop[j]: continue   # the lane stopped at a wall
            x = min(age, A) / float(A)   # 0: the newest row .. 1: the oldest labelled row
            fo = P['fo0'] + (P['fo2'] - P['fo0']) * x; fw = (1.0 + (P['w2'] - 1.0) * x) * ramp; fh = (1.0 + (P['h2'] - 1.0) * x) * ramp
            s = max(0.0, s + fo)
            p = _np7.array([O[0] + s * _np7.cos(ang), O[1] + s * _np7.sin(ang)])
            out.append((j, k, p, _np7.c_[p[0] + P['W'] * fw * ca, p[1] + P['D'] * fw * sa], P['H'] * fh))
    return out


def _ft7s035_mask(cam, O, th, kin, inc, tau, grid=None, solid=None, shape=(540, 1200), P=None):
    """the template's _ft42s_mask with this skin's sprites (the same wall / bush box and solid cuts)"""
    spr = _ft7s035_sprites(kin, inc, O, th, tau, grid, P)
    if not spr: return _np7.zeros(shape, bool)
    M, polys = _ft42s_shape(cam, spr, shape, hull=True)
    if grid is not None and M.any():
        ys, xs = _np7.nonzero(M)
        anc = _np7.array([p for _, _, p, _, _ in spr]); hs = _np7.array([h for *_, h in spr])
        a0 = _ft42_proj(cam, anc[:, 0], anc[:, 1], 0.0); a1 = _ft42_proj(cam, anc[:, 0], anc[:, 1], 100.0)
        dd = (xs[:, None] - a0[None, :, 0]) ** 2 + (ys[:, None] - a0[None, :, 1]) ** 2; nb = dd.argmin(1)
        zpx = (a0[nb, 1] - a1[nb, 1]) / 100.0
        zl = _np7.clip((a0[nb, 1] - (ys + 0.5)) / _np7.maximum(zpx, 1e-6), 0.0, hs[nb])
        blk = _ft42_blocking(); bsh = _ft42_bushes(); pad = FT42_TILE * 2
        tx0, tx1 = int((anc[:, 0].min() - pad) // FT42_TILE), int((anc[:, 0].max() + pad) // FT42_TILE)
        ty0, ty1 = int((anc[:, 1].min() - pad) // FT42_TILE), int((anc[:, 1].max() + pad) // FT42_TILE)
        tiles = [(tx, ty, FT42_BLOCK_Z if grid[ty, tx] in blk else FT42_BUSH_Z)
                 for ty in range(max(0, ty0), min(grid.shape[0], ty1 + 1)) for tx in range(max(0, tx0), min(grid.shape[1], tx1 + 1))
                 if grid[ty, tx] in blk or grid[ty, tx] in bsh]
        if tiles:
            Hz = max(FT42_BLOCK_Z, FT42_BUSH_Z)
            ax, ay = _ft42_unproj(cam, xs + 0.5, ys + 0.5, 0.0); bx, by = _ft42_unproj(cam, xs + 0.5, ys + 0.5, Hz)
            dx, dy = bx - ax, by - ay; zin = _np7.full(len(xs), -1.0)
            for tx, ty, Ht in tiles:
                lo = _np7.zeros(len(xs)); hi = _np7.full(len(xs), Ht / Hz)
                for a0_, d0, e0, e1 in ((ax, dx, tx * FT42_TILE, (tx + 1) * FT42_TILE), (ay, dy, ty * FT42_TILE, (ty + 1) * FT42_TILE)):
                    with _np7.errstate(divide='ignore', invalid='ignore'):
                        t0 = (e0 - a0_) / d0; t1 = (e1 - a0_) / d0
                    par = _np7.abs(d0) < 1e-9; inside = (a0_ >= e0) & (a0_ <= e1)
                    t0 = _np7.where(par, _np7.where(inside, -_np7.inf, _np7.inf), t0); t1 = _np7.where(par, _np7.where(inside, _np7.inf, -_np7.inf), t1)
                    lo = _np7.maximum(lo, _np7.minimum(t0, t1)); hi = _np7.minimum(hi, _np7.maximum(t0, t1))
                zin = _np7.where(lo <= hi, _np7.maximum(zin, hi * Hz), zin)
            occ = zin > zl + 1.0
            M[ys[occ], xs[occ]] = False
    if solid is not None and (M & solid).any():
        from scipy.ndimage import binary_opening as _op7, binary_dilation as _di7, label as _cc7
        M &= ~solid
        near = _di7(solid, iterations=FT42_CUT_NEAR)
        M = (M & ~near) | (_op7(M, structure=_np7.ones((3, 3), bool), iterations=FT42_CUT_OPEN) & near)
        L, n = _cc7(M)
        if n > 1:
            sz = _np7.bincount(L.ravel()); sz[0] = 0
            M = _np7.isin(L, _np7.nonzero(sz >= max(FT42_SPECK, FT42_SPECK_FRAC * sz.max()))[0])
    return M


def _ft7s035_wave(mem=None, cam=None, state=None, kin=None, O=None, th=None, tau=None, inc=None, alive=None, grid=None, solid=None, shape=None):
    if not alive or tau < 0: return None   # nothing once every projectile has finished; the first drawn row is update 0
    return _ft7s035_mask(cam, O, th, kin, inc, tau, grid=grid, solid=solid, shape=shape)


FT42_CUSTOM[6000475] = _ft7s035_wave
