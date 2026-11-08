# Frank skin 344 HammerDudeClashKingChromaRed (attack type 6000483): v7 label model "ghost sprites" (task t42 round 7, 2026-10-28,
# Sonnet agent). Exec'd in idlabels' namespace AFTER the template z_1027_frank_template.py (v6); it only registers
# FT42_CUSTOM[6000483], every other Frank skin keeps its v6 model. Install as py/fixes/z_1028_frank_v7_s344.py.
#
# What the skin draws (studied frame by frame on fa_344 / fb_344, 15 aims, ticks tau 2 and 4; t42/v7/s344/study): each of the
# wave's four lanes (rays from the origin O at aim +-8.1 / +-23.2 deg) carries one camera-facing sprite per update ("row") at the
# lane point, FT42_LAG of an update behind the logic position. Row of age 0 (the newest): at tau 2 a white spiky star face with
# a column of ghost wisps above it (about 40 px up), at tau 4 a big white ball (about 42 px wide) with a cyan plume above it and a
# pointed tip below. Row of age 1: cyan fire (almost hidden behind the next row at tau 2, flames about 40 px tall at tau 4).
# Older rows (age >= 2) are dark-blue flames and rings = after-effects, not labelled (the rule since 10-27). Nothing is drawn at
# tau 0. The sprites are billboards: their screen shape does not turn with the aim (checked on 15 aims), so v6's aim-aligned
# ground egg (margins raised to a height, hull) was loose at the sides and at the back end (the wisps and flames above the lane
# points, near the brawler) and cut the tips at the same time.
#
# Model (a function of the attack origin O, the aim, tau = updates since the wave started, the skin and the static map only; no
# pixels): for every lane j (4: outer -, inner -, inner +, outer +; mirror-symmetric about the aim axis in the ground) and every
# row k = max(1, tau - 1) .. tau, the lane point P(j, k) = O + s_draw(k) (cos, sin)(aim +- phi_j) is projected with the frame's
# camera (ground, z 0). Around it one sprite silhouette in screen px (half resolution): the convex hull of an ellipse (half width
# w, up hu, down hd) and two spike tips (up hp, down hq). The silhouette parameters by sprite age (0 newest / 1) are fitted at
# tau 2 and tau 4 (FT42_S344['p']), linear in tau between, constant from tau 4 on, r1 x the tau 2 values at tau 1 (the first
# drawn update; not measured: no capture has an image there). The label is the convex hull of all sprite silhouettes (the user's
# Frank rule: ONE hull per wave), cut as in v3..v6: a lane that hit a wall keeps its sprite at the wall for the update it arrived
# and has none after it (fits the wall frames best), the wall / bush boxes standing in front of the label and the solid blocks
# drawn on screen are cut away (the code of the v5 sprite model), small islands the cut leaves are dropped.
# Exists for tau >= 1 while a projectile of the wave still travels (as v6): nothing at tau 0 or once the wave is over.
FT42_S344 = dict(
    # (tau, age): (w, hu, hd, hp, hq) in px: fitted by t42/v7/s344/scr/fit_final.py on 30 frames (15 aims) of fa_344 / fb_344 + the
    # 0 / 90 deg masks, least-loose label (mean IoU of the back and the front half with the old pixel label) at >= 96 % of its
    # pixels and >= 97.2 % of the drawn rows (pre-hull) covered, >= 94.5 % per half
    p={(2, 0): (24.0, 30.0, 33.0, 42.0, 36.0), (2, 1): (0.0, 15.0, 0.0, 15.0, 0.0),
       (4, 0): (21.0, 42.0, 30.0, 42.0, 30.0), (4, 1): (12.0, 39.0, 20.0, 39.0, 20.0)},
    r1=0.8, first=1, A=1)   # first drawn row (no label before), A: rows of age <= A are still the attack
FT42_S344_LANES = [(1, -1), (0, -1), (0, 1), (1, 1)]   # (pair: 0 inner / 1 outer, side), from -v to +v


def _s344_sil(v):
    """screen-offset points (n, 2), y down, of one sprite silhouette v = (w, hu, hd, hp, hq)"""
    w, hu, hd, hp, hq = v
    a = _np42.linspace(0, 2 * _np42.pi, 24, endpoint=False); ca, sa = _np42.cos(a), _np42.sin(a)
    P = _np42.c_[w * ca, _np42.where(sa > 0, -hu * sa, -hd * sa)]
    return _np42.concatenate([P, [[0.0, -hp], [0.0, hq]]])


def _s344_prm(tau, age):
    p = FT42_S344['p']; a, b = _np42.array(p[(2, age)], float), _np42.array(p[(4, age)], float)
    if tau >= 4: return b
    if tau >= 2: return a + (b - a) * (tau - 2) / 2.0
    return a * FT42_S344['r1']


def _s344_sprites(kin, inc, O, th, tau, grid=None):
    """[(lane j, row k, ground point (2,) world, silhouette offsets (n, 2) px)] of the wave's sprites at update tau"""
    k0 = min(tau, max(FT42_S344['first'], tau - FT42_S344['A'])); out = []
    for j, (c, sg) in enumerate(FT42_S344_LANES):
        ang = _np42.radians(th + sg * kin['phi'][c]); s0, b, _ = kin['cls'][inc[c]]
        stop = _ft42_stop(grid, O, _np42.array([th + sg * kin['phi'][c]]), kin['R'] + 400)[0] if grid is not None else _np42.inf
        for k in range(k0, tau + 1):
            # a lane that hit a wall (its member finished) keeps its sprite at the wall for the update it arrived (drawn at the
            # wall, not beyond it) and has none after it: the wall frames of fa_ / fb_ fit best this way (t42/v7/s344 report)
            if s0 + b * (k - 1) >= stop: continue
            s = min(_ft42_sdraw(kin, inc[c], k), stop - 1.0)
            out.append((j, k, _np42.array([O[0] + s * _np42.cos(ang), O[1] + s * _np42.sin(ang)]), _s344_sil(_s344_prm(tau, tau - k))))
    return out


def _s344_shape(cam, spr, shape):
    """(H, W) bool: the convex hull of all sprite silhouettes placed at their projected ground points"""
    Hh, Wd = shape; pts = []
    for j, k, p, sil in spr:
        a = _ft42_proj(cam, p[0], p[1], 0.0); pts.append(a[None, :] + sil)
    I = _Im42.new('L', (Wd, Hh), 0); _Dr42.Draw(I).polygon([tuple(q) for q in _ft42_hull(_np42.concatenate(pts))], fill=1, outline=1)
    return _np42.asarray(I, bool)


def _s344_mask(cam, O, th, kin, inc, tau, grid=None, solid=None, shape=(540, 1200)):
    spr = _s344_sprites(kin, inc, O, th, tau, grid)
    if not spr: return _np42.zeros(shape, bool)
    M = _s344_shape(cam, spr, shape)
    if grid is not None and M.any():
        # wall / bush boxes in front of the label (as the v5 sprite model): the label point of a pixel is the billboard of its
        # nearest sprite at the pixel's height above that sprite's ground point (the ground under the pixel below it)
        ys, xs = _np42.nonzero(M)
        anc = _np42.array([p for _, _, p, _ in spr]); hs = _np42.array([max(-s[:, 1].min(), 0.0) for *_, s in spr])
        a0 = _ft42_proj(cam, anc[:, 0], anc[:, 1], 0.0); a1 = _ft42_proj(cam, anc[:, 0], anc[:, 1], 100.0)
        dd = (xs[:, None] - a0[None, :, 0]) ** 2 + (ys[:, None] - a0[None, :, 1]) ** 2; nb = dd.argmin(1)
        zpx = (a0[nb, 1] - a1[nb, 1]) / 100.0   # px per unit of height at the anchor
        zl = _np42.clip((a0[nb, 1] - (ys + 0.5)) / _np42.maximum(zpx, 1e-6), 0.0, hs[nb] / _np42.maximum(zpx, 1e-6))
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
            if occ.any():   # what a bush / wall leaves of the label as small separate islands (a plume tip above a bush) is dropped
                from scipy.ndimage import label as _cc42
                L, n = _cc42(M)
                if n > 1:
                    sz = _np42.bincount(L.ravel()); sz[0] = 0
                    M = _np42.isin(L, _np42.nonzero(sz >= max(FT42_SPECK, 0.1 * sz.max()))[0])
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


def _s344_wave(mem=None, cam=None, state=None, kin=None, O=None, th=None, tau=None, inc=None, alive=None, grid=None, solid=None,
               shape=(540, 1200)):
    if not alive or tau < FT42_S344['first']: return None
    return _s344_mask(cam, O, th, kin, inc, tau, grid=grid, solid=solid, shape=shape)


FT42_CUSTOM[6000483] = _s344_wave
