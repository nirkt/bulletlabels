# sweep 10-27 fix for skin 010 (Sonnet revision fix2.py after a rejected Haiku fix, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s010/)
# sweep 2026-10-27, skin 010 RosaBP (pair 209, sw2_010 attack 0 deg, tick 275, spur_or_speck); second attempt.
# Rosa004Projectile (type 6000587, Rosa's attack: a 3D blade of three ice segments, one label per segment). Each segment's
# draw also holds the translucent blue streak fan trailing behind the blade (same draw, same coverage, no uv), so the
# label grew saw-tooth spurs on the segment's trailing edge (the left edge of the 0 deg attack, the top edge of the 90 deg
# one). Fix, per segment (arrays are turned so that the fan is on the left, whatever the shot direction; the fan side comes from the shot's angle):
#  1. CORE = opening of the label with a disc of radius OPEN_R (half-res px): the solid body; fan teeth (thinner than
#     2 x OPEN_R) vanish. 
#  2. Two cues give the segment's true trailing edge per row: (a) the world depth R['wd']: the fan is a flat plane, the second
#     difference of the depth along a row is strongly negative on the segment's first column where the fan touches it
#     (part of a long connected line, within a few px of CORE's end); (b) else CORE's end plus KEEP_D px (label pixels
#     farther than that from CORE on the fan side are dropped). Rows within EDGE px of the segment's ends are left alone
#     (the opening rounds corners), and the whole leading half (crystal tips, bright rims) is untouched.
#  3. A second, small opening (TAB_R) on the fan half removes the thin tabs the fan leaves at the joints between segments.
#  4. Pieces of up to 6 px removed where segments overlap are put back (no pinholes in the union), small detached pieces
#     are dropped, holes filled. The result is a subset of the label: a segment only loses fan pixels. A segment without an
#     opening or without fan mass is left as it is.
def _fix_6000587(body, ctx):
    from scipy.ndimage import label as _cc, binary_dilation, binary_fill_holes, binary_opening
    OPEN_R = 6; KEEP_D = 3; TAB_R = 3; EDGE = 4; PATH_MIN = 3; THIN = 0.85; T = 0.006; MINPX = 8; MINH = 6; WIN = (-4, 5)
    m0 = body.copy()
    if not m0.any(): return body
    R = ctx['R']; wd0 = R['wd']; ids0 = R['id'][::2, ::2]
    if wd0.shape != m0.shape: wd0 = None
    def disc(r):
        yy, xx = np.mgrid[-r:r + 1, -r:r + 1]; return (yy * yy + xx * xx) <= r * r + 1
    def opening(a, r): return binary_opening(np.pad(a, r + 1), structure=disc(r))[r + 1:-(r + 1), r + 1:-(r + 1)]
    def dilation(a, r): return binary_dilation(np.pad(a, r + 1), structure=disc(r))[r + 1:-(r + 1), r + 1:-(r + 1)]
    core0 = opening(m0, OPEN_R)
    # a segment mostly hidden behind its neighbours (edge-on view) is a thin band the opening would eat: left as it is
    if core0.sum() < THIN * m0.sum(): return body
    # direction of the fan: behind the shot, i.e. on screen from a point ahead of the shot (logic start + 100 along its angle)
    # back to the start (o['angle'] is the shot's own direction; the path alone can still be a point just after the spawn)
    try:
        o = ctx['o']; x0, y0 = o['path'][0]; an = np.deg2rad(float(o['angle']))
        a = np.array(_proj(ctx['cam'], x0, y0)); b = np.array(_proj(ctx['cam'], x0 + 100 * np.cos(an), y0 + 100 * np.sin(an)))
        fx, fy = a - b
    except Exception: return body
    if np.hypot(fx, fy) < PATH_MIN: return body
    # canonical orientation: fan to the left (transpose when the fan is more vertical, flip when it is on the right)
    tr = abs(fy) > abs(fx); fl = (fy if tr else fx) > 0
    def fwd(a):
        if a is None: return None
        a = a.T if tr else a
        return a[:, ::-1] if fl else a
    def back(a):
        a = a[:, ::-1] if fl else a
        return a.T if tr else a
    m = fwd(m0); core = fwd(core0); wd = fwd(wd0); ids = fwd(ids0)
    H, W_ = m.shape
    Y, X = np.mgrid[0:H, 0:W_]
    cy, cx = np.nonzero(core)[0].mean(), np.nonzero(core)[1].mean()
    near = dilation(core, KEEP_D)
    zone = X < cx                                                     # fan half
    cl = np.full(H, -1)
    for y in np.nonzero(core.any(1))[0]: cl[y] = np.nonzero(core[y])[0].min()
    cyr = np.nonzero(core.any(1))[0]
    rows = ((np.arange(H) >= cyr.min() + EDGE) & (np.arange(H) <= cyr.max() - EDGE))[:, None]
    out = m & ~(zone & rows & ~near)
    if wd is not None and not tr:       # depth step only for fans trailing along the screen x axis
        dx = np.zeros(wd.shape, np.float32); dx[:, 1:] = wd[:, 1:] - wd[:, :-1]
        d2 = np.zeros(wd.shape, np.float32); d2[:, 1:] = dx[:, 1:] - dx[:, :-1]
        n1 = np.zeros(m.shape, bool); n1[:, 1:] = ids[:, 1:] == ids[:, :-1]
        n2 = np.zeros(m.shape, bool); n2[:, 2:] = (ids[:, 2:] == ids[:, :-2]) & n1[:, 2:]
        E = m & n1 & n2 & (d2 < -T)
        if E.any():
            L, nl = _cc(binary_dilation(E, structure=np.ones((3, 3), bool)))
            keep = np.zeros(m.shape, bool)
            for k in range(1, nl + 1):
                ys, xs = np.nonzero(E & (L == k))
                if len(ys) >= MINPX and ys.max() - ys.min() >= MINH: keep |= E & (L == k)
            for y in np.nonzero((cl >= 0) & rows[:, 0])[0]:
                xs = np.nonzero(keep[y])[0]
                xs = xs[(xs >= cl[y] + WIN[0]) & (xs <= cl[y] + WIN[1])]
                if xs.size: out[y] = m[y] & (np.arange(W_) >= xs.min())
    out = np.where(zone, opening(out, TAB_R), out)
    Lr, nr = _cc(m & ~out)
    for k in range(1, nr + 1):
        c = Lr == k
        if c.sum() <= 6 and (binary_dilation(c) & out).any(): out |= c
    L2, n2_ = _cc(binary_dilation(out, structure=np.ones((3, 3), bool)))
    if n2_ > 1:
        sz = np.bincount(L2[out], minlength=n2_ + 1); out &= (L2 == sz.argmax())
    out = back(binary_fill_holes(out) & m)
    return out if out.any() else body
POST_BODY[6000587] = _fix_6000587
