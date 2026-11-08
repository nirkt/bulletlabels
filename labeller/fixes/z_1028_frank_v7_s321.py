# Frank skin 321 HammerDudeknight (attack type 6000479), v7: the CONE model (task t42 round 7, 2026-10-28). Exec'd in idlabels'
# namespace after the template z_1027_frank_template.py; it only registers FT42_CUSTOM[6000479] (every other type keeps v6).
#
# What 321 draws (studied on fa_321 / fb_321 with the shape-id pixels of the attack clip: shape 47 = the cyan ground ring, shape 61
# = the spike bodies; both camera-facing 2D sprites, constant depth per sprite): one spike per projectile lane and per update
# ("row"): a tall triangle (blue cone) on a flat ring. A spike is born at the lane's drawn position and then SINKS linearly: apex
# heights of the inner lanes at ages 0 / 1 / 2 / 3 updates were about 84 / 61 / 38 / 15 px (0.0405 px per logic z). The old pixel
# label (and v6) is ONE convex hull of the drawn spikes of the rows of age <= 2. v6 rebuilt that hull from a ground "egg" raised to
# at most 700 logic z = 30 px, while the spikes are 60-85 px tall; its fit therefore blew the egg up backwards and sideways, and
# that is the loose back end near the brawler at the aims where the tall tips point at it (aim 45-135: 1.4k-3k px of empty label).
# The tips are the hull's boundary, so v7 builds the hull from the real sprites:
#   cones: lanes (inner pair +-8.1 deg, slow; outer pair +-23.2 deg, fast) x rows k = max(1, tau-2) .. tau (nothing is drawn on the
#     first update), the row's lane point P = O + s(k - LAG) (cos, sin)(aim +- phi) (the client draws a row LAG of an update behind
#     the logic position; a lane stops at a projectile-blocking tile and its later cones are not drawn),
#   per cone: foot = P + (0, RY0) (the sprite anchor sits RY0 logic units below the lane point, screen down), ring = ground ellipse of
#     half axes (RX, RY) around the foot, triangle base = foot +- WC, apex = foot raised to H_pair(age) = (H0_pair - HS_pair * age)
#     * hs(aim), hs = 1 + KUP * min(0, sin aim): the spikes of a wave pointing up the screen are drawn shorter (x0.43 at aim -90),
#   label = convex hull (screen) of all cones' ring points, base corners and apexes of the wave (one hull per wave, as the old
#     labels), minus what the wall / bush boxes (BLOCK_Z / BUSH_Z high) and the solid blocks drawn on screen hide (as v3 .. v6: a
#     label pixel's height is the highest cone under it; the bush canopy hides up to ~BUSH_Z = 1000, fitted on the six bush aims:
#     IoU with the visible old label 0.70 -> 0.76), crumbs of the cut cleaned.
# Fitted by t42/v7/s321/fit6.py (Nelder-Mead on all 32 labelled frames of the 17 aims, E = away from walls / bushes / solid; objective =
# mean IoU with the old pixel label, kept: >= 97.5 % of the drawn cone pixels (shapes 47 / 61) covered, >= 97 % in the back and in
# the front half, >= 90 % per frame, <= 60 px per frame of drawn cone more than 3 px outside the label) and checked leave-one-aim-out.
# Predictable: only O, aim, tau, skin, the frame's camera and the static map. Lane points / rows are mirror-symmetric about the aim axis
# (0.0000 units); the sprites themselves are camera-facing (as drawn), the same on both sides.
# Nothing before the first drawn row (tau 0) and nothing once the wave is over (no projectile travelling), as v6.
FT42V7_321 = dict(first=1, A=2, lag=0.31, ry0=72.0, RX=175.0, RY=68.0, WC=172.0, H0=(2050.0, 1650.0), HS=(520.0, 510.0), KUP=0.57, BUSH_Z=1000.0)


def _ft42v7_cones(kin, inc, O, th, tau, P, grid):
    """[(age, pair, side, lane point (2,))] of the cones drawn at update tau (lanes stopped by a wall drop their later cones)"""
    out = []
    for c in (0, 1):
        for sg in (1, -1):
            ang = th + sg * kin['phi'][c]
            stop = _ft42_stop(grid, O, _np42.array([ang]), kin['R'] + 400)[0] if grid is not None else _np42.inf
            s0, b, _ = kin['cls'][inc[c]]
            for k in range(max(int(P['first']), tau - int(P['A'])), tau + 1):
                s = max(0.0, min(kin['R'], s0 + b * (k - P['lag'])))
                if s >= stop: continue
                a = _np42.radians(ang)
                out.append((tau - k, c, sg, _np42.array([O[0] + s * _np42.cos(a), O[1] + s * _np42.sin(a)])))
    return out


def _ft42v7_polys(cam, cones, P, th):
    """per cone (ring points, triangle points, foot (2,), height) in screen px"""
    a = _np42.linspace(0, 2 * _np42.pi, FT42_S_RING, endpoint=False); ca, sa = _np42.cos(a), _np42.sin(a); out = []
    # the spikes of a wave pointing up the screen are shorter (fitted: KUP; the aim is the only input)
    hs = max(0.2, 1.0 + P['KUP'] * min(0.0, _np42.sin(_np42.radians(th))))
    for age, c, sg, p in cones:
        fx, fy = p[0], p[1] + P['ry0']; h = max(0.0, P['H0'][c] - P['HS'][c] * age) * hs
        ring = _ft42_proj(cam, fx + P['RX'] * ca, fy + P['RY'] * sa, 0.0)
        b = _ft42_proj(cam, _np42.array([fx - P['WC'], fx + P['WC']]), _np42.array([fy, fy]), 0.0)
        ap = _ft42_proj(cam, _np42.array([fx]), _np42.array([fy]), h)[0]
        out.append((ring, _np42.array([b[0], b[1], ap]), (fx, fy), h))
    return out


def _ft42v7_321(mem=None, cam=None, state=None, kin=None, O=None, th=None, tau=None, inc=None, alive=None, grid=None, solid=None,
                shape=(540, 1200), P=None):
    P = P or FT42V7_321
    if not alive or tau < P['first']: return None
    cones = _ft42v7_cones(kin, inc, O, th, tau, P, grid)
    M = _np42.zeros(shape, bool)
    if not cones: return M
    pl = _ft42v7_polys(cam, cones, P, th)
    H_, W_ = shape; I = _Im42.new('L', (W_, H_), 0)
    _Dr42.Draw(I).polygon([tuple(q) for q in _ft42_hull(_np42.concatenate([_np42.concatenate([r, t]) for r, t, _, _ in pl]))], fill=1, outline=1)
    M = _np42.asarray(I, bool)
    if grid is not None and M.any():
        # wall / bush boxes in front of the label (as v3 .. v6): a pixel whose ray enters a box above the label's highest point
        # there (the highest cone standing under it) shows the block (or a hit spark drawn on it), not the attack
        ys, xs = _np42.nonzero(M)
        foot = _np42.array([(f[0], f[1]) for _, _, f, _ in pl]); hs = _np42.array([h for *_, h in pl])
        a0 = _ft42_proj(cam, foot[:, 0], foot[:, 1], 0.0); a1 = _ft42_proj(cam, foot[:, 0], foot[:, 1], 100.0)
        zpx = (a0[:, 1] - a1[:, 1]) / 100.0
        wpx = _np42.abs(_ft42_proj(cam, foot[:, 0] + P['RX'], foot[:, 1], 0.0)[:, 0] - a0[:, 0])   # ring half width in px
        zl = _np42.zeros(len(xs))
        for j in range(len(pl)):   # the label point: the highest cone standing under the pixel (its vertical strip), else the ground
            inb = _np42.abs(xs + 0.5 - a0[j, 0]) <= wpx[j]
            zj = _np42.clip((a0[j, 1] - (ys + 0.5)) / max(zpx[j], 1e-6), 0.0, hs[j])
            zl = _np42.where(inb, _np42.maximum(zl, zj), zl)
        blk = _ft42_blocking(); bsh = _ft42_bushes(); pad = FT42_TILE * 2
        tx0, tx1 = int((foot[:, 0].min() - pad) // FT42_TILE), int((foot[:, 0].max() + pad) // FT42_TILE)
        ty0, ty1 = int((foot[:, 1].min() - pad) // FT42_TILE), int((foot[:, 1].max() + pad) // FT42_TILE)
        tiles = [(tx, ty, FT42_BLOCK_Z if grid[ty, tx] in blk else P.get('BUSH_Z', FT42_BUSH_Z))
                 for ty in range(max(0, ty0), min(grid.shape[0], ty1 + 1)) for tx in range(max(0, tx0), min(grid.shape[1], tx1 + 1))
                 if grid[ty, tx] in blk or grid[ty, tx] in bsh]
        if tiles:
            Hz = max(FT42_BLOCK_Z, P.get('BUSH_Z', FT42_BUSH_Z))
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


FT42_CUSTOM[6000479] = _ft42v7_321
