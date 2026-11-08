# human review 2026-10-28 fix for skin 120 (Haiku template fix, reviewed by the coordinator, installed 10-29; source out/sweep2/haikutasks/human1028/s120/fix7.py; runs after z_1099_cleanup like the tests)
# fix7 (2026-10-29, skin 120 JesterHarvest, type 6001280). Coordinator review of fix6 (cone tips and horns accepted): on every tick
# the solid bright YELLOW ball at the projectile's base (bottom at 305-315, nose at 265-269) lay outside the label, which stopped at
# the white face. The yellow ball is part of the projectile (the user flagged only holes in the old label). fix7 = fix6 plus: the
# solid yellow pixels (R > 200, G > 180, B < 120; not the soft yellow glow) that are connected to a projectile's label within 4 px
# are added to it (components up to 400 px), then the same one-piece smoothing. Front / back and the other rules are as fix6.
# (fix6 text follows.)
# fix6 (2026-10-29, skin 120 JesterHarvest, type 6001280 = Chester004Projectile). Coordinator review of fix5: at 313 and 315
# the cone tip sticks out above the labels (left projectile at 315 by ~10-15 px) and the bottom of the yellow base is below them
# at 311-315. The sprite stretches along its motion while it flies, so one uniform scale cannot fit. Fix: the same template shape
# (POLY_UV, as fix4 / fix5), placed per projectile per tick with TWO scales, s_u along its motion (0.8 .. 2.0 of the projected
# logic radius) and s_v across it (0.8 .. 1.5), and an along-motion offset t (-25 .. +25 px). The triple with the best IoU
# against the frame's foreground (saturated blue, white, yellow and its clip body; the purple halo and the lavender crate are
# not in it) is searched coarse, then refined; inside an envelope of the candidates grown by 4 px and outside the pixels the
# projectile in front covers. Used only when it has >= 150 foreground pixels in the envelope, IoU >= 0.6 and beats the unscaled
# template by 0.02. The back projectile of an overlap (305, 307) takes the front one's s_u and s_v and fits only t. Halo cut
# (halo-coloured pixels more than 2 px from projectile-coloured pixels leave the labels), one-piece smoothing, front / back one
# owner per pixel and the logic-end cut are as in fix5. Keyed to type 6001280 only.
ENDED_KEEP.add(6001280)   # the template is placed from the logic state only: the leftover-effect cut must not clip it
_S6_POLY = [
    (-1.201, -1.120),
    (-1.160, -1.132),
    (-1.120, -1.135),
    (-1.080, -1.129),
    (-1.048, -1.120),
    (-1.020, -1.109),
    (-0.980, -1.090),
    (-0.940, -1.069),
    (-0.900, -1.046),
    (-0.862, -1.020),
    (-0.836, -1.000),
    (-0.812, -0.980),
    (-0.790, -0.960),
    (-0.760, -0.933),
    (-0.724, -0.900),
    (-0.700, -0.878),
    (-0.660, -0.841),
    (-0.637, -0.820),
    (-0.600, -0.785),
    (-0.573, -0.760),
    (-0.540, -0.731),
    (-0.500, -0.703),
    (-0.460, -0.684),
    (-0.420, -0.672),
    (-0.380, -0.665),
    (-0.347, -0.660),
    (-0.300, -0.653),
    (-0.260, -0.642),
    (-0.220, -0.626),
    (-0.180, -0.601),
    (-0.151, -0.580),
    (-0.123, -0.560),
    (-0.088, -0.540),
    (-0.060, -0.527),
    (-0.020, -0.516),
    (0.020, -0.512),
    (0.060, -0.513),
    (0.100, -0.516),
    (0.140, -0.517),
    (0.180, -0.516),
    (0.220, -0.512),
    (0.260, -0.506),
    (0.290, -0.500),
    (0.320, -0.493),
    (0.360, -0.482),
    (0.400, -0.469),
    (0.440, -0.452),
    (0.480, -0.431),
    (0.520, -0.404),
    (0.548, -0.380),
    (0.580, -0.348),
    (0.606, -0.320),
    (0.640, -0.281),
    (0.660, -0.256),
    (0.680, -0.228),
    (0.697, -0.200),
    (0.716, -0.160),
    (0.729, -0.120),
    (0.738, -0.080),
    (0.743, -0.040),
    (0.746, 0.000),
    (0.747, 0.040),
    (0.745, 0.080),
    (0.740, 0.120),
    (0.733, 0.160),
    (0.721, 0.200),
    (0.703, 0.240),
    (0.680, 0.278),
    (0.660, 0.306),
    (0.640, 0.331),
    (0.620, 0.354),
    (0.594, 0.380),
    (0.570, 0.400),
    (0.541, 0.420),
    (0.505, 0.440),
    (0.460, 0.460),
    (0.420, 0.475),
    (0.380, 0.488),
    (0.340, 0.498),
    (0.300, 0.505),
    (0.260, 0.510),
    (0.220, 0.512),
    (0.180, 0.511),
    (0.140, 0.510),
    (0.100, 0.508),
    (0.060, 0.509),
    (0.020, 0.516),
    (-0.020, 0.532),
    (-0.060, 0.560),
    (-0.086, 0.580),
    (-0.116, 0.600),
    (-0.157, 0.620),
    (-0.200, 0.635),
    (-0.240, 0.647),
    (-0.280, 0.658),
    (-0.320, 0.667),
    (-0.360, 0.675),
    (-0.400, 0.683),
    (-0.440, 0.691),
    (-0.475, 0.700),
    (-0.520, 0.718),
    (-0.557, 0.740),
    (-0.584, 0.760),
    (-0.607, 0.780),
    (-0.630, 0.800),
    (-0.660, 0.827),
    (-0.696, 0.860),
    (-0.719, 0.880),
    (-0.742, 0.900),
    (-0.765, 0.920),
    (-0.789, 0.940),
    (-0.813, 0.960),
    (-0.838, 0.980),
    (-0.862, 1.000),
    (-0.887, 1.020),
    (-0.914, 1.040),
    (-0.940, 1.057),
    (-0.978, 1.080),
    (-1.017, 1.100),
    (-1.060, 1.118),
    (-1.100, 1.129),
    (-1.140, 1.134),
    (-1.180, 1.132),
    (-1.220, 1.123),
    (-1.260, 1.104),
    (-1.291, 1.080),
    (-1.320, 1.042),
    (-1.336, 1.000),
    (-1.344, 0.960),
    (-1.349, 0.920),
    (-1.360, 0.886),
    (-1.379, 0.860),
    (-1.414, 0.840),
    (-1.460, 0.828),
    (-1.498, 0.820),
    (-1.540, 0.811),
    (-1.580, 0.802),
    (-1.620, 0.791),
    (-1.656, 0.780),
    (-1.700, 0.767),
    (-1.740, 0.756),
    (-1.780, 0.750),
    (-1.820, 0.748),
    (-1.860, 0.752),
    (-1.900, 0.759),
    (-1.940, 0.767),
    (-1.980, 0.772),
    (-2.020, 0.775),
    (-2.060, 0.774),
    (-2.100, 0.770),
    (-2.140, 0.764),
    (-2.180, 0.753),
    (-2.218, 0.740),
    (-2.260, 0.723),
    (-2.300, 0.707),
    (-2.340, 0.691),
    (-2.369, 0.680),
    (-2.400, 0.668),
    (-2.440, 0.653),
    (-2.472, 0.640),
    (-2.520, 0.620),
    (-2.560, 0.602),
    (-2.600, 0.582),
    (-2.638, 0.560),
    (-2.668, 0.540),
    (-2.696, 0.520),
    (-2.720, 0.501),
    (-2.745, 0.480),
    (-2.768, 0.460),
    (-2.791, 0.440),
    (-2.814, 0.420),
    (-2.837, 0.400),
    (-2.862, 0.380),
    (-2.887, 0.360),
    (-2.912, 0.340),
    (-2.937, 0.320),
    (-2.962, 0.300),
    (-2.987, 0.280),
    (-3.013, 0.260),
    (-3.037, 0.240),
    (-3.060, 0.219),
    (-3.080, 0.195),
    (-3.100, 0.160),
    (-3.109, 0.120),
    (-3.107, 0.080),
    (-3.096, 0.040),
    (-3.080, 0.010),
    (-3.060, -0.020),
    (-3.040, -0.045),
    (-3.008, -0.080),
    (-2.980, -0.108),
    (-2.946, -0.140),
    (-2.923, -0.160),
    (-2.900, -0.180),
    (-2.875, -0.200),
    (-2.850, -0.220),
    (-2.823, -0.240),
    (-2.795, -0.260),
    (-2.768, -0.280),
    (-2.741, -0.300),
    (-2.714, -0.320),
    (-2.685, -0.340),
    (-2.654, -0.360),
    (-2.621, -0.380),
    (-2.588, -0.400),
    (-2.560, -0.418),
    (-2.529, -0.440),
    (-2.503, -0.460),
    (-2.477, -0.480),
    (-2.452, -0.500),
    (-2.428, -0.520),
    (-2.404, -0.540),
    (-2.378, -0.560),
    (-2.349, -0.580),
    (-2.320, -0.597),
    (-2.280, -0.617),
    (-2.240, -0.634),
    (-2.200, -0.650),
    (-2.160, -0.667),
    (-2.130, -0.680),
    (-2.100, -0.692),
    (-2.068, -0.700),
    (-2.020, -0.704),
    (-1.980, -0.700),
    (-1.940, -0.694),
    (-1.900, -0.687),
    (-1.860, -0.685),
    (-1.820, -0.686),
    (-1.780, -0.691),
    (-1.740, -0.698),
    (-1.700, -0.707),
    (-1.660, -0.716),
    (-1.620, -0.729),
    (-1.580, -0.746),
    (-1.553, -0.760),
    (-1.520, -0.780),
    (-1.488, -0.800),
    (-1.460, -0.819),
    (-1.432, -0.840),
    (-1.409, -0.860),
    (-1.380, -0.891),
    (-1.360, -0.919),
    (-1.340, -0.958),
    (-1.321, -1.000),
    (-1.300, -1.038),
    (-1.280, -1.064),
    (-1.240, -1.099),
    (-1.201, -1.120)]

_S6_U0, _S6_V0, _S6_RES = -4.0, -2.0, 0.01


def _s6_template_grid():
    from PIL import Image as _I6, ImageDraw as _D6
    nu = int(round(6.0 / _S6_RES)) + 1; nv = int(round(4.0 / _S6_RES)) + 1
    I = _I6.new('L', (nu, nv), 0)
    _D6.Draw(I).polygon([((u - _S6_U0) / _S6_RES, (v - _S6_V0) / _S6_RES) for u, v in _S6_POLY], fill=1)
    return np.asarray(I, bool)


_S6_TM = _s6_template_grid()
_S6 = {'tm': {}, 'minus': {}, 'log': []}
S6_SU = np.round(np.arange(0.8, 2.0001, 0.1), 3)
S6_SV = np.round(np.arange(0.8, 1.5001, 0.1), 3)
S6_T = np.arange(-25, 25.001, 3.0)
S6_T_FINE = np.arange(-25, 25.001, 1.0)
S6_MIN_IOU = 0.6; S6_MIN_GAIN = -0.01; S6_MIN_FG = 150   # a fit may lose 0.01 IoU when it brings the extents in (EXT_W)
EXT_W = 0.01   # IoU penalty per px of cone-tip / base-bottom extent mismatch (along the motion)


def _s6_p(cam, p):
    q = cam @ np.array([p[0], p[1], p[2], 1.0])
    return np.array([(q[0] / q[3] * 0.5 + 0.5) * 1200, (1 - (q[1] / q[3] * 0.5 + 0.5)) * 540])


def _s6_geom(cam, o):
    """centre c (projected anchor), nominal radius s0 (mean of its projected logic radius along / across), unit motion e, unit across p"""
    a = np.asarray(o['anchor'], float); th = np.deg2rad(float(o['angle'])); R_ = float(o['radius'])
    c = _s6_p(cam, a)
    F1 = _s6_p(cam, a + R_ * np.array([np.cos(th), -np.sin(th), 0.0])) - c
    L1 = _s6_p(cam, a + R_ * np.array([-np.sin(th), -np.cos(th), 0.0])) - c
    s0 = 0.5 * (np.linalg.norm(F1) + np.linalg.norm(L1)); e = F1 / np.linalg.norm(F1); p = np.array([-e[1], e[0]])
    return c, s0, e, p


def _s6_box(c, H=540, W=1200, r=100):
    return max(0, int(c[0]) - r), min(W, int(c[0]) + r + 1), max(0, int(c[1]) - r), min(H, int(c[1]) + r + 1)


def _s6_coords(c, e, p, box, ox=0.0, oy=0.0):
    x0, x1, y0, y1 = box
    yy, xx = np.mgrid[y0:y1, x0:x1]
    dx = xx + ox - c[0]; dy = yy + oy - c[1]
    return dx * e[0] + dy * e[1], dx * p[0] + dy * p[1]


def _s6_inside(a, b, s0, su, sv, t):
    u = (a - t) / (s0 * su); v = b / (s0 * sv)
    ui = np.round((u - _S6_U0) / _S6_RES).astype(int); vi = np.round((v - _S6_V0) / _S6_RES).astype(int)
    ok = (ui >= 0) & (ui < _S6_TM.shape[1]) & (vi >= 0) & (vi < _S6_TM.shape[0])
    out = np.zeros(a.shape, bool); out[ok] = _S6_TM[vi[ok], ui[ok]]
    return out


def _s6_fg(img, R, o):
    """projectable-coloured frame pixels: saturated blue, white, yellow (as fix5) and the clip body"""
    im = img.astype(int); R_, G_, B_ = im[..., 0], im[..., 1], im[..., 2]
    blue = (R_ <= 105) & (B_ >= 180) & (B_ - R_ >= 90) & (G_ >= 60)   # red <= 105: the cone and wing spikes, not the lavender crate (red about 120-130)
    white = (R_ >= 150) & (G_ >= 190) & (B_ >= 190)
    yellow = (R_ >= 190) & (G_ >= 190) & (B_ <= 130)
    C = sprite_pixels(6001280, o['own'], R); cv = R['cv'][::2, ::2]
    return blue | white | yellow | ((C == 1) & (cv >= 128))


def _s6_fit(c, s0, e, p, box, fg, excl, s_fixed=None):
    """best (su, sv, t): maximal IoU with the foreground in the envelope, minus EXT_W x the along-motion extent mismatch
    (cone tip and base bottom, px: the tip is a thin part that IoU barely sees). A back projectile keeps the front one's
    along-motion scale (s_fixed[0]) and fits its own across scale and offset. Returns su, sv, t, IoU, nominal IoU, n_fg."""
    x0, x1, y0, y1 = box
    a, b = _s6_coords(c, e, p, box)
    fgb = fg[y0:y1, x0:x1] & ~excl[y0:y1, x0:x1]
    # only the foreground connected to the projectile's nominal template (the water pool and sparks are separate parts)
    from scipy.ndimage import label as _lab6
    Lc, nc = _lab6(fgb, structure=np.ones((3, 3)))
    nomb = _s6_inside(a, b, s0, 1.0, 1.0, 0.0)
    keep = np.unique(Lc[nomb & (Lc > 0)]); fgb = np.isin(Lc, keep[keep > 0]) if keep.size else np.zeros_like(fgb)
    ax_y = abs(e[1]) >= abs(e[0])
    axis_idx = np.nonzero(fgb.any(1 if ax_y else 0))[0]
    if axis_idx.size == 0: axis_idx = np.array([0, 0])
    fg_lo, fg_hi = axis_idx.min(), axis_idx.max()
    def pen(m):
        idx = np.nonzero(m.any(1 if ax_y else 0))[0]
        if idx.size == 0: return 99.0
        return abs(fg_lo - idx.min()) + abs(fg_hi - idx.max())
    SU = S6_SU if s_fixed is None else np.array([s_fixed[0]])
    SV, T = S6_SV, S6_T
    union = np.zeros(fgb.shape, bool)
    for su in SU:
        for sv in SV:
            for t in T: union |= _s6_inside(a, b, s0, su, sv, t)
    env = nd_dilate(union, 4); env_fg = fgb & env; nfg = int(env_fg.sum())
    def score(su, sv, t):
        m = _s6_inside(a, b, s0, su, sv, t)
        iou = int((m & env_fg).sum()) / max(1, int(((m | fgb) & env).sum()))
        return iou - EXT_W * pen(m), iou
    best = (-9.0, 1.0, 1.0, 0.0, 0.0)
    for su in SU:
        for sv in SV:
            for t in T:
                J, iou = score(su, sv, t)
                if J > best[0]: best = (J, su, sv, t, iou)
    _, su0, sv0, t0, _ = best
    su_list = [su0] if s_fixed is not None else list(np.round(np.arange(su0 - 0.1, su0 + 0.101, 0.02), 3))
    for su in su_list:
        if s_fixed is None and not (S6_SU[0] <= su <= S6_SU[-1]): continue
        for sv in np.round(np.arange(sv0 - 0.1, sv0 + 0.101, 0.02), 3):
            if not (S6_SV[0] <= sv <= S6_SV[-1]): continue
            for t in S6_T_FINE:
                if abs(t - t0) > 3: continue
                J, iou = score(su, sv, t)
                if J > best[0]: best = (J, su, sv, t, iou)
    _, su, sv, t, iou = best
    nominal = score(1.0 if s_fixed is None else su, 1.0 if s_fixed is None else 1.0, 0.0)[1] if s_fixed is None else score(su, 1.0, 0.0)[1]
    return su, sv, t, iou, nominal, nfg


def nd_dilate(m, r):
    from scipy.ndimage import binary_dilation
    return binary_dilation(m, iterations=int(r))


def _s6_final_mask(c, s0, e, p, su, sv, t, H=540, W=1200, sub=3):
    box = _s6_box(c, H, W, r=100); x0, x1, y0, y1 = box
    acc = np.zeros((y1 - y0, x1 - x0))
    offs = (np.arange(sub) + 0.5) / sub - 0.5
    for oy in offs:
        for ox in offs:
            a, b = _s6_coords(c, e, p, box, ox, oy)
            acc += _s6_inside(a, b, s0, su, sv, t)
    m = np.zeros((H, W), bool); m[y0:y1, x0:x1] = acc * 2 > sub * sub
    return m


def _s6_prepare(objs, cam, R, img, state):
    _S6['tm'] = {}; _S6['minus'] = {}
    live = [o for o in objs if o.get('t') == 6001280 and o.get('finished', 1) == 0 and o.get('drawn', True) and o.get('angle') is not None]
    if not live: return
    tick = int(state.get('tick', -1)) if state else -1
    geom = {o['id']: _s6_geom(cam, o) for o in live}
    nominal = {o['id']: _s6_final_mask(geom[o['id']][0], geom[o['id']][1], geom[o['id']][2], geom[o['id']][3], 1.0, 1.0, 0.0) for o in live}
    ids_h = R['id'][::2, ::2]; front_of = {}
    for i in range(len(live)):
        for j in range(i + 1, len(live)):
            A, B = live[i], live[j]; TA, TB = nominal[A['id']], nominal[B['id']]
            both = TA & TB
            if not both.any(): continue
            oa = TA & ~TB; ob = TB & ~TA; votes = {A['id']: 0, B['id']: 0}
            for dr in np.unique(ids_h[both]):
                if dr <= 0: continue
                m = ids_h == dr; tot = int(m.sum())
                sa = (m & oa).sum() / tot; sb = (m & ob).sum() / tot
                n = int((m & both).sum())
                if sa >= 0.5 and sa > sb: votes[A['id']] += n
                elif sb >= 0.5 and sb > sa: votes[B['id']] += n
            if votes[A['id']] == votes[B['id']]: continue
            front, back = (A, B) if votes[A['id']] > votes[B['id']] else (B, A)
            front_of[back['id']] = front['id']; _S6['minus'].setdefault(back['id'], []).append(front['id'])
    fitted = {}; placed = {}
    order = sorted(live, key=lambda o: 1 if o['id'] in front_of else 0)
    for o in order:
        oid = o['id']; c, s0, e, p = geom[oid]; box = _s6_box(c); x0, x1, y0, y1 = box
        FG = _s6_fg(img, R, o)
        excl = np.zeros(FG.shape, bool)   # the neighbours' nominal parts are not this projectile's
        for other, m in nominal.items():
            if other != oid: excl |= m & ~nominal[oid]
        s_fix = None
        if oid in front_of:               # a back projectile is covered by the front one (its fitted template)
            fo = front_of[oid]; excl |= placed[fo]
            s_fix = (fitted[fo][0], fitted[fo][1])
        su, sv, t, iou, nom, nfg = _s6_fit(c, s0, e, p, box, FG, excl, s_fixed=s_fix)
        ok = (nfg >= S6_MIN_FG) and (iou >= S6_MIN_IOU) and (iou >= nom + S6_MIN_GAIN)
        if not ok:
            su, sv, t = (s_fix[0], s_fix[1], 0.0) if s_fix is not None else (1.0, 1.0, 0.0)
        fitted[oid] = (su, sv, t); placed[oid] = _s6_final_mask(c, s0, e, p, su, sv, t)
        _S6['log'].append(dict(tick=tick, id=oid, su=float(su), sv=float(sv), t=float(t), iou=round(float(iou), 3), nominal=round(float(nom), 3), nfg=nfg, role='back' if oid in front_of else 'front/single', used=ok))
    for oid in placed: _S6['tm'][oid] = placed[oid]
    if img is not None:   # halo cut, one piece, no notches (as fix5)
        from scipy import ndimage as nd
        im = img.astype(int); R_, G_, B_ = im[..., 0], im[..., 1], im[..., 2]
        broad = ((B_ >= 180) & (B_ - R_ >= 90) & (G_ >= 60)) | ((R_ >= 150) & (G_ >= 190) & (B_ >= 190)) | ((R_ >= 190) & (G_ >= 190) & (B_ <= 130))
        halo_far = (G_ < 80) & (B_ > 120) & (R_ > 60) & (nd.distance_transform_edt(~broad) > 2)
        for oid in list(_S6['tm']):
            m = _S6['tm'][oid] & ~halo_far
            m = nd.binary_opening(nd.binary_closing(m, iterations=1), iterations=1)
            m = nd.binary_fill_holes(m)
            L, n = nd.label(m)
            if n > 1:
                sz = np.bincount(L.ravel()); sz[0] = 0; m = L == sz.argmax()
            _S6['tm'][oid] = m
    if img is not None:   # solid yellow ball of the projectile at its base: connected to its label (within 4 px), not the glow
        from scipy import ndimage as nd
        im = img.astype(int); R_, G_, B_ = im[..., 0], im[..., 1], im[..., 2]
        yel = (R_ > 200) & (G_ > 180) & (B_ < 120)
        YL, nyl = nd.label(yel, structure=np.ones((3, 3)))
        sz_y = np.bincount(YL.ravel())
        for oid in list(_S6['tm']):
            m = _S6['tm'][oid]
            near = nd.binary_dilation(m, iterations=4)
            touch = np.unique(YL[near & yel]); touch = touch[(touch > 0) & (sz_y[touch] <= 400)]
            if touch.size == 0: continue
            m2 = m | np.isin(YL, touch)
            m2 = nd.binary_fill_holes(nd.binary_opening(nd.binary_closing(m2, iterations=1), iterations=1))
            L, n = nd.label(m2)
            if n > 1:
                sz = np.bincount(L.ravel()); sz[0] = 0; m2 = L == sz.argmax()
            _S6['tm'][oid] = m2


_s6_labels_prev = labels


def labels(objs, cam, R, state=None, ended=(), img=None):
    _s6_prepare(objs, cam, R, img, state)
    return _s6_labels_prev(objs, cam, R, state=state, ended=ended, img=img)


def _s6_body(body, ctx):
    oid = ctx['o']['id']
    if oid not in _S6['tm']: return np.zeros_like(body)   # logic ended / not drawn: no label
    m = _S6['tm'][oid].copy()
    for fid in _S6['minus'].get(oid, []): m &= ~_S6['tm'][fid]   # the front projectile owns the shared pixels
    return m


POST_BODY[6001280] = _s6_body
