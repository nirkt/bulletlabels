# Frank v7, skin 220 FrankCaveman (+ CavemanRed), attack type 6000485: its own label function, FT42_CUSTOM[6000485]
# (task t42 round 7, 2026-10-28; exec'd in idlabels' namespace after z_1027_frank_template.py; it only registers the function,
# every other type / skin stays v6; the supers are untouched). Install as py/fixes/z_1028_frank_v7_s220.py.
#
# WHAT THE ATTACK LOOKS LIKE (fa_220 / fb_220, 15 aims, every imaged tick; study/anatomy_v7.png): every update the four
# projectiles (lanes at aim +-8.1 deg slow / +-23.2 deg fast, s(k) = min(1800, s0 + b k)) each drop one burst of ice spikes
# (shapes 61 + cyan body 884 / 886): a camera-facing cluster ~ 40 px wide and ~35 px tall that shrinks and sinks with age. The
# FIRST burst (row 0, the muzzle burst by the hammer) is gone by age 2 (84 px of body per lane at tau 2 against 260 px for the
# oldest row of a later tau), the later rows keep ~ 40 % of their size at age 2. What the old pipeline hulls is the BODY pixels
# of the rows of age <= 2 (38 - 60 % of the hull's area); v7 is fitted to those pixels, not to the hull.
#
# THE MODEL (a function of only the origin O, the aim, tau, the skin, the frame's camera and the static map; mirror-symmetric):
#   one billboard per lane (4 lanes, mirrored about the aim axis) and row k = max(0, tau - 2) .. tau, at the drawn row point
#   s(k - FT42_LAG) along its lane: a ground CIRCLE of radius R raised to H (a vertical capsule on screen, the same for every
#   lane: the ground footprint is exactly mirror-symmetric about the aim axis). Size by age a = tau - k (0 .. 2): R x (1 + (AW-1) a / 2),
#   H x (1 + (AH-1) a / 2). Row 0 shrinks faster than the others: factor G0 at age 0 down to G2 at age 2 (G2 = 0: from tau 2 on
#   there is no muzzle burst any more). The label is the UNION of the hulls of neighbouring-lane x consecutive-row cells (not one
#   convex hull over the wave): it follows the arc of the rows and leaves out the empty corners one hull spans. Lanes stop at a
#   wall; walls / bushes / solid blocks cut the label exactly as in v6 (copied from the template's sprite mask).
#   Six parameters (R, H, AW, AH, G0, G2; the lane kinematics are the template's), fitted by t42/v7/s220/work/fit10.py on the
#   body pixels of 41 attack frames (15 aims; tau 0, 2, 4), coverage >= 95 % of them per tau, leave-aims-out validated (report.md).

F220_TYPE = 6000485
F220_A = 2            # rows k = max(0, tau - F220_A) .. tau are the attack (age = tau - k)
F220_RING = 20
F220_MINSC = 0.05     # a row-0 sprite scaled below this is not drawn (G2 = 0: no muzzle burst from age 2 on)
F220_LANES = [(1, -1), (0, -1), (0, 1), (1, 1)]   # (pair, side): outer -, inner -, inner +, outer +
F220_P = dict(R=185.0, H=450.0, AW=0.55, AH=0.40, G0=0.9, G2=0.0)   # logic units / factors; t42/v7/s220/work/fit10.py on 41 frames of 15 aims (3-fold leave-aims-out in report.md)
# (an ellipse footprint W 200 x D 125 as in v5 / v6 fits 0.015 IoU better but is not mirror-symmetric about the aim axis: R is the circle's radius)


def _f220_sprites(kin, inc, O, th, tau, grid=None):
    """[(lane 0..3 from -v to +v, row k, ground point (2,), ring (n, 2) world, height)] of the wave's billboards (the sprite
    format of the template's _ft42s_*), rows k0..tau; lanes stop at a projectile-blocking tile (their sprites beyond it are dropped)"""
    P = F220_P; k0 = max(0, tau - F220_A)
    a_ = _np42.linspace(0, 2 * _np42.pi, F220_RING, endpoint=False); ca, sa = _np42.cos(a_), _np42.sin(a_)
    stop = {}
    if grid is not None:
        for j, (c, sg) in enumerate(F220_LANES):
            stop[j] = _ft42_stop(grid, O, _np42.array([th + sg * kin['phi'][c]]), kin['R'] + 400)[0]
    out = []
    for j, (c, sg) in enumerate(F220_LANES):
        ang = _np42.radians(th + sg * kin['phi'][c])
        for k in range(k0, tau + 1):
            s = _ft42_sdraw(kin, inc[c], k)
            if grid is not None and s >= stop[j]: continue
            age = tau - k
            sc = (P['G0'] + (P['G2'] - P['G0']) * age / 2.0) if k == 0 else 1.0   # the muzzle burst (row 0) ages out faster
            if sc <= F220_MINSC: continue   # ... and is gone
            fw = 1 + (P['AW'] - 1) * age / 2.0; fh = 1 + (P['AH'] - 1) * age / 2.0
            p = _np42.array([O[0] + s * _np42.cos(ang), O[1] + s * _np42.sin(ang)])
            out.append((j, k, p, _np42.c_[p[0] + P['R'] * fw * sc * ca, p[1] + P['R'] * fw * sc * sa], P['H'] * fh * sc))
    return out


def _f220_mask(cam, spr, grid=None, solid=None, shape=(540, 1200)):
    """the union of the cell hulls of the billboards (screen), minus the wall / bush boxes standing in front and the solid
    pixels (the template's cut, v3 / v5)"""
    if not spr: return _np42.zeros(shape, bool)
    M, polys = _ft42s_shape(cam, spr, shape, hull=False)
    if grid is not None and M.any():
        ys, xs = _np42.nonzero(M)
        anc = _np42.array([p for _, _, p, _, _ in spr]); hs = _np42.array([h for *_, h in spr])
        a0 = _ft42_proj(cam, anc[:, 0], anc[:, 1], 0.0); a1 = _ft42_proj(cam, anc[:, 0], anc[:, 1], 100.0)
        dd = (xs[:, None] - a0[None, :, 0]) ** 2 + (ys[:, None] - a0[None, :, 1]) ** 2; nb = dd.argmin(1)
        zpx = (a0[nb, 1] - a1[nb, 1]) / 100.0
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


def _f220_label(mem=None, cam=None, state=None, kin=None, O=None, th=None, tau=None, inc=None, alive=None, grid=None, solid=None, shape=(540, 1200)):
    """the label of one 220 attack wave, or None (not labelled on this frame): from update 0 while a projectile still travels"""
    if not alive or tau < 0: return None
    return _f220_mask(cam, _f220_sprites(kin, inc, O, th, tau, grid), grid=grid, solid=solid, shape=shape)


FT42_CUSTOM[F220_TYPE] = _f220_label
