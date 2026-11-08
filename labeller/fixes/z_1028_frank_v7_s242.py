# Frank v7, skin 242 HammerDudeDJ (attack type 6000477): the label is the hull of the drawn floor tiles, rebuilt from the
# tile's real geometry (measured with the client's ID pass + its depth buffer, 28 frames of fa_242 / fb_242, 15 aims).
# Exec'd after the template (py/fixes/z_1027_frank_template.py, v6 + the FT42_CUSTOM hook); it only registers
#   FT42_CUSTOM[6000477] = _v7s242_label
#
# WHAT THE SKIN DRAWS (study, t42/v7/s242/study/): every update the client drops one row of four glowing rectangular tiles,
# one per projectile lane (the four rays from the origin O at aim +-8.1 / +-23.2 deg), centred on the lane point
# s(k - lag) (lag 0.35 update behind the logic position, the same lag as v2..v6; it drifts 0.27-0.43 between attacks, which
# is +-20 units along the lane: the label does not follow it, it is not predictable from O / aim / tau). A tile is a FLAT
# rectangle in the ground plane, long side across its lane, floating above the floor and sinking: height (logic z, from the
# depth buffer) 206-216 on its first update, ~150-210 on its second (the frame's sub-tick phase decides), 12-38 on its
# third, 6 (on the floor) from its fourth on. Its size depends on its ROW k, not on its age: rows 1, 2 are 270 x 136,
# rows 3, 4 are 345 x 136, row 5 (and 6) 428 x 146 (across x along the lane). There is no row 0 (nothing at tau 0).
# A row is still the attack for A = 3 updates after it was drawn (the user: the first, blue row stays labelled).
# (The drawn tile is turned by up to +-8 deg from 'across the lane' by the client, -8.2 sin(2 lane angle): not mirror
# symmetric about the aim, so the label keeps 'across the lane'; that costs 0.004 IoU.)
#
# THE LABEL (a function of O, the aim, tau, the skin and the static map only; mirror-symmetric about the aim in the ground):
#   tiles of rows max(1, tau - 3) .. tau on the four lanes, each a rectangle L(k) x S(k) (+ margin m) across / along its lane
#   at height z(age); the label is the convex hull of all their corners, seen by the frame's camera (the tiles float, so the
#   hull is not the ground footprint: its front is raised by the newest tiles' height).
#   Walls / bushes / solid blocks cut it the way the v6 template cuts: a lane whose projectile stopped at a wall draws no row
#   whose centre lies >= 115 units beyond the wall (measured: tiles exist up to +100, none from +137), the cells of the
#   other tiles beyond a projectile-blocking tile as seen from O are cut off BEFORE the hull is taken (no wedge up to the
#   wall), the blocking tiles and bushes as boxes in front hide the label (straight cuts), the solid blocks drawn on screen
#   cut it with the cut's edge band cleaned. Characters do not cut.
#   Exists for 1 <= tau while a projectile of the wave still travels (the template's `alive`), nothing before / after.
FT42_V7_242 = dict(A=3, first=1, lag=0.35,
                   L={1: 270.0, 2: 270.0, 3: 345.0, 4: 345.0, 5: 428.0, 6: 428.0}, S={1: 136.0, 2: 136.0, 3: 136.0, 4: 136.0, 5: 146.0, 6: 146.0},
                   Z=(208.0, 195.0, 12.0, 6.0), m=0.0, drop=115.0)


def _v7s242_tiles(kin, inc, O, th, tau, P, grid=None):
    """[(lane, row, points (n, 3) world x, y, z)]: the tiles of rows max(first, tau - A) .. tau, lanes outer -, inner -, inner +,
    outer +: the 4 corners of each rectangle, or (grid given, tile meets a wall) the corners of its cells in front of the wall
    (a cell is behind a wall when its ray from O first enters a projectile-blocking tile before it); a tile wholly behind: gone"""
    k0 = min(tau, max(int(P['first']), tau - int(P['A']))); out = []
    if grid is not None:
        ra = th + _np42.arange(-60.0, 60.0 + 1e-9, FT42_RAY_DA); dstop = _ft42_stop(grid, O, ra, 2600.0)
        lstop = _ft42_stop(grid, O, _np42.array([th + sg * kin['phi'][c] for c, sg in ((1, -1), (0, -1), (0, 1), (1, 1))]), kin['R'] + 600.0)
    for j, (c, sg) in enumerate(((1, -1), (0, -1), (0, 1), (1, 1))):
        ang = _np42.radians(th + sg * kin['phi'][c]); s0, b, _ = kin['cls'][inc[c]]
        eu = _np42.array([_np42.cos(ang), _np42.sin(ang)]); ev = _np42.array([-_np42.sin(ang), _np42.cos(ang)])
        for k in range(k0, tau + 1):
            s = max(0.0, min(kin['R'], s0 + b * (k - P['lag']))); ctr = _np42.array([O[0] + s * eu[0], O[1] + s * eu[1]])
            if grid is not None and s - lstop[j] >= P['drop']: continue   # the lane's projectile stopped at a wall before this row was drawn
            kk = min(k, max(P['L'])); hl = P['L'][kk] / 2 + P['m']; hs = P['S'][kk] / 2 + P['m']
            z = P['Z'][min(tau - k, len(P['Z']) - 1)]
            C = _np42.array([ctr + eu * sx * hs + ev * sy * hl for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
            if grid is not None:
                g = FT42_CELL; ui = _np42.arange(-hs + g / 2, hs, g); vi = _np42.arange(-hl + g / 2, hl, g)
                UU, VV = _np42.meshgrid(ui, vi); Pc = ctr[None, :] + UU.reshape(-1, 1) * eu[None, :] + VV.reshape(-1, 1) * ev[None, :]
                rr = _np42.hypot(Pc[:, 0] - O[0], Pc[:, 1] - O[1]); al = _np42.degrees(_np42.arctan2(Pc[:, 1] - O[1], Pc[:, 0] - O[0]))
                ik = _np42.clip(_np42.round(((al - th + 180) % 360 - 180 + 60.0) / FT42_RAY_DA).astype(int), 0, len(dstop) - 1)
                ok = rr < dstop[ik]
                if not ok.all():
                    if not ok.any(): continue
                    Pk = Pc[ok]; h = g / 2
                    C = _np42.concatenate([Pk + _np42.array([a, b_]) @ _np42.array([eu, ev]) for a, b_ in ((-h, -h), (h, -h), (h, h), (-h, h))])
            out.append((j, k, _np42.c_[C, _np42.full(len(C), z)]))
    return out


def _v7s242_upper(u, z):
    """upper convex envelope (sorted u, z) of the points (u, z)"""
    o = _np42.lexsort((z, u)); u, z = u[o], z[o]; h = []
    for p in zip(u, z):
        while len(h) >= 2 and (h[-1][0] - h[-2][0]) * (p[1] - h[-2][1]) - (h[-1][1] - h[-2][1]) * (p[0] - h[-2][0]) >= 0: h.pop()
        h.append(p)
    h = _np42.array(h); return h[:, 0], h[:, 1]


def _v7s242_mask(cam, O, th, tiles, grid=None, solid=None, shape=(540, 1200)):
    """(H, W) bool: the hull of the tiles (corners at their heights) seen by the camera, minus the footprint beyond walls, the
    wall / bush boxes in front and the solid blocks (as the template's _ft42_mask, with the tile heights instead of a prism)"""
    H, Wd = shape
    if not tiles: return _np42.zeros(shape, bool)
    X = _np42.concatenate([t[2][:, 0] for t in tiles]); Y = _np42.concatenate([t[2][:, 1] for t in tiles]); Zc = _np42.concatenate([t[2][:, 2] for t in tiles])
    Sh = _ft42_hull(_ft42_proj(cam, X, Y, Zc))
    I = _Im42.new('L', (Wd, H), 0); _Dr42.Draw(I).polygon([tuple(p) for p in Sh], fill=1, outline=1)
    C = _np42.asarray(I, bool)
    if not C.any(): return C
    c, s = _np42.cos(_np42.radians(th)), _np42.sin(_np42.radians(th))
    U = (X - O[0]) * c + (Y - O[1]) * s; V = -(X - O[0]) * s + (Y - O[1]) * c
    F = _ft42_hull(_np42.c_[U, V]); ue, ze = _v7s242_upper(U, Zc); h = float(Zc.max())
    g = FT42_CELL; u0, v0 = F.min(0) - 2 * g; u1, v1 = F.max(0) + 2 * g
    nu, nv = int(_np42.ceil((u1 - u0) / g)) + 1, int(_np42.ceil((v1 - v0) / g)) + 1
    J = _Im42.new('L', (nu, nv), 0); _Dr42.Draw(J).polygon([((p[0] - u0) / g, (p[1] - v0) / g) for p in F], fill=1, outline=1)
    keep = _np42.asarray(J, bool)   # [v index, u index]: the tiles' ground hull
    if grid is not None:   # the wall part: footprint in or beyond a projectile-blocking tile seen from O (rays from O)
        vv, uu = _np42.nonzero(keep)
        Pw = _ft42_world(O, th, _np42.c_[u0 + uu * g, v0 + vv * g])
        al = _np42.degrees(_np42.arctan2(Pw[:, 1] - O[1], Pw[:, 0] - O[0])); rr = _np42.hypot(Pw[:, 0] - O[0], Pw[:, 1] - O[1])
        ref = al[0]; rel = (al - ref + 180) % 360 - 180
        k = _np42.round((rel - rel.min()) / FT42_RAY_DA).astype(int)
        rays = ref + rel.min() + _np42.arange(k.max() + 1) * FT42_RAY_DA
        d = _ft42_stop(grid, O, rays, rr.max())
        keep[vv[rr >= d[k]], uu[rr >= d[k]]] = False
    # each pixel of the hull: kept when the column under it (heights 0 .. envelope of the tile corners) meets the kept
    # footprint; ztop: the highest such height (the label's point nearest the camera along the pixel's ray)
    ys, xs = _np42.nonzero(C)
    c0 = _ft42_proj(cam, *_ft42_world(O, th, F.mean(0, keepdims=True))[0], 0.0); c1 = _ft42_proj(cam, *_ft42_world(O, th, F.mean(0, keepdims=True))[0], h)
    nz = int(_np42.ceil(abs(c1[1] - c0[1]) / 0.7)) + 2
    ztop = _np42.full(len(ys), -1.0)
    for z in _np42.linspace(0.0, h, nz):
        x, y = _ft42_unproj(cam, xs + 0.5, ys + 0.5, z)
        u = (x - O[0]) * c + (y - O[1]) * s; v = -(x - O[0]) * s + (y - O[1]) * c
        iu = _np42.round((u - u0) / g).astype(int); iv = _np42.round((v - v0) / g).astype(int)
        inb = (iu >= 0) & (iu < nu) & (iv >= 0) & (iv < nv)
        hit = _np42.zeros(len(iu), bool); hit[inb] = keep[iv[inb], iu[inb]] & (z <= _np42.interp(u[inb], ue, ze) + 1e-6)
        ztop[hit] = z
    ok = ztop >= 0
    if grid is not None and ok.any():   # wall and bush boxes in front of the label (as the template)
        blk = _ft42_blocking(); bsh = _ft42_bushes(); Pb = _np42.c_[X, Y]; pad = FT42_TILE * 2
        tx0, tx1 = int((Pb[:, 0].min() - pad) // FT42_TILE), int((Pb[:, 0].max() + pad) // FT42_TILE)
        ty0, ty1 = int((Pb[:, 1].min() - pad) // FT42_TILE), int((Pb[:, 1].max() + pad) // FT42_TILE)
        tl = [(tx, ty, FT42_BLOCK_Z if grid[ty, tx] in blk else FT42_BUSH_Z)
              for ty in range(max(0, ty0), min(grid.shape[0], ty1 + 1)) for tx in range(max(0, tx0), min(grid.shape[1], tx1 + 1))
              if grid[ty, tx] in blk or grid[ty, tx] in bsh]
        if tl:
            i = _np42.nonzero(ok)[0]; Hz = max(FT42_BLOCK_Z, FT42_BUSH_Z)
            ax, ay = _ft42_unproj(cam, xs[i] + 0.5, ys[i] + 0.5, 0.0); bx, by = _ft42_unproj(cam, xs[i] + 0.5, ys[i] + 0.5, Hz)
            dx, dy = bx - ax, by - ay; zin = _np42.full(len(i), -1.0)
            for tx, ty, Ht in tl:
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
    if solid is not None and (M & solid).any():   # the solid blocks drawn on screen cut it, the cut's edge band cleaned of crumbs
        from scipy.ndimage import binary_opening as _op42, binary_dilation as _di42, label as _cc42
        M &= ~solid
        near = _di42(solid, iterations=FT42_CUT_NEAR)
        M = (M & ~near) | (_op42(M, structure=_np42.ones((3, 3), bool), iterations=FT42_CUT_OPEN) & near)
        L, n = _cc42(M)
        if n > 1:
            sz = _np42.bincount(L.ravel()); sz[0] = 0
            M = _np42.isin(L, _np42.nonzero(sz >= max(FT42_SPECK, FT42_SPECK_FRAC * sz.max()))[0])
    return M


def _v7s242_label(mem=None, cam=None, state=None, kin=None, O=None, th=None, tau=None, inc=None, alive=None, grid=None, solid=None, shape=(540, 1200)):
    P = FT42_V7_242
    if not alive or tau < P['first']: return None
    return _v7s242_mask(cam, O, th, _v7s242_tiles(kin, inc, O, th, tau, P, grid=grid), grid=grid, solid=solid, shape=shape)


globals().setdefault('FT42_CUSTOM', {})[6000477] = _v7s242_label
