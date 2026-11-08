# human review 2026-10-28 fix for skin 111 (Haiku fix, reviewed by the coordinator, installed 10-29; source out/sweep2/haikutasks/human1028/s111/fix2.py; tested via IDLABELS_OVERRIDE after all fixes incl. z_1099_cleanup, hence z_1100_)
# human review 2026-10-28, skin 111 (MenderDefault), task s111 (the user's findings, binding; per-skin decisions in task.md).
# fix2.py (supersedes fix.py; the attack part is identical to fix.py, accepted by the coordinator).
#  (1) attack 0 (row "attack 0 deg", ticks 265-269) and attack 90 (row "attack 90 deg", ticks 305-311): "the whole shining
#      circle should be labeled". Type 6001755 (MenderProjectile). The label is the whole bright cyan disc around the white
#      core, not only the core (the installed a2_agent2 _frame_core, min(R,G,B) >= 150, gave a small piece of it).
#      Disc = pixels within DISC_R of the core centre whose green and blue are both >= DISC_T (the disc is 240-255 there;
#      the teal tether stays below ~170, the soft glow falls off below DISC_T). The part the tether hides is filled by the
#      point reflection of the visible part through the centre, and the result is one convex hull, a clean disc.
#      Tick 269 (attack 0): the logic projectile ends on that tick (tracker age 0) and the ball is still drawn on its tether;
#      from tick 271 the drawn ring is the impact (an end effect, not labelled). The disc is labelled on that end tick only,
#      the core searched on the type's own sprite near the projected end point.
#  (2) super 45 (row "super 45 deg", ticks 355-359, "missing label"; the user's decision 2026-10-28, binding): once the
#      shards have landed (any finished shard), the super is ONE label: the outline of the WHOLE visible blue area (its
#      bright blue rim and outer edge), filled solid, smooth, for as long as the area is on screen. No per-shard polygons,
#      no convex hulls. The area is found from the drawn pixels: the super's own sprite shapes (codes 1, 2, 5 of the type,
#      coverage >= AREA_CODE_MIN: the front-most draw is the super's, so walls and crates standing in front are not in it),
#      closed, the holes that are blue (median B - R >= AREA_HOLE_B) filled (the dark interior has no sprite code; walls and
#      crates are not blue, so they stay cut out), the bright blue rim (B >= 190, B - R >= 120) within AREA_RIM px added,
#      closed, and smoothed (Gaussian, threshold 0.5). Components >= AREA_MIN px are kept. The label belongs to the lowest
#      id finished shard; the other shards get no label while the area mode is on (their pixels are inside the area). Before
#      any shard has landed (ticks 349-353) the installed per-shard labels are unchanged (z_1027_s111 cleanup).
#      ENDED_KEEP exempts this type from the ended-leftover drop.
# Installed fixes for these types: a2_agent2.py (6001755, replaced here) and z_1027_s111.py (6001756, called before landing).
# Type 6001756 is used by skin 111 only (checked); type 6001755 by skin 111 (no other Mender page in the sweep uses it).

DISC_T = 200          # min(G, B) of the bright cyan disc (full-res frame pixels)
DISC_R = 11           # px (full res): the disc's radius limit around the white core's centre (the disc is ~10 px)
CORE_T = 150          # min(R, G, B) of the white core (the ball is 160-255; the teal tether and the glow are below)
AREA_CLOSE = 6        # px: gaps in the blue area's sprite pixels closed
AREA_CODE_MIN = 32    # coverage (0..255) of the super's sprite pixels that belong to the area
AREA_HOLE_B = 90      # median B - R of a hole for it to be blue area (filled); walls and crates are not blue
AREA_RIM = 8          # px: bright blue rim pixels this close to the area are added
AREA_SIG = 4          # px: Gaussian smoothing of the outline (threshold 0.5)
AREA_MIN = 200        # px: area components smaller than this are dropped
_mender_prev_ulti = POST_BODY.get(6001756)   # installed z_1027_s111 cleanup, used before any shard has landed
ENDED_KEEP.add(6001756)
_MENDER_END_NOW = set()    # ids of attack-0 type projectiles whose logic ended on this tick (tracker age 0)
_MENDER_AREA = None        # (owner id, full-res mask) of the super's one area label on this frame, or None


def _mender_area_mask(img, R):
    """the outline of the visible blue area of the super (see the header): full-res bool mask, or None"""
    from scipy.ndimage import binary_closing, binary_fill_holes, binary_dilation, gaussian_filter, label as cc
    C = sprite_pixels(6001756, True, R)
    cv = R['cv'][::2, ::2]
    M = np.isin(C, [1, 2, 5]) & (cv >= AREA_CODE_MIN)
    if not M.any(): return None
    im = img.astype(int); b = im[..., 2] - im[..., 0]
    bright = (im[..., 2] >= 190) & (b >= 120)

    def close(m):
        return binary_closing(np.pad(m, AREA_CLOSE), iterations=AREA_CLOSE)[AREA_CLOSE:-AREA_CLOSE, AREA_CLOSE:-AREA_CLOSE]

    def blue_holes(m):
        F = binary_fill_holes(m) & ~m
        if not F.any(): return m
        L, n = cc(F); out = m.copy()
        for k in range(1, n + 1):
            h = L == k
            if np.median(b[h]) >= AREA_HOLE_B: out |= h
        return out

    Mf = blue_holes(close(M))
    Mf = Mf | (bright & binary_dilation(Mf, iterations=AREA_RIM))
    Mf = blue_holes(close(Mf))
    Mf = blue_holes(gaussian_filter(Mf.astype(np.float32), AREA_SIG) > 0.5)
    L2, n2 = cc(Mf); sz = np.bincount(L2.ravel())
    keep = [k for k in range(1, n2 + 1) if sz[k] >= AREA_MIN]
    if not keep: return None
    return np.isin(L2, keep)


def _mender_frame_setup(objs, cam, R, ended, img):
    """per frame: the ids that end on this tick (attack 0), and the super's one area label once a shard has landed"""
    global _MENDER_AREA
    _MENDER_END_NOW.clear(); _MENDER_AREA = None
    _MENDER_END_NOW.update(e['id'] for e in ended if e.get('t') == 6001755 and e.get('age', 1) == 0)
    fin = [o for o in objs if o['t'] == 6001756 and o.get('finished', 0)]
    if not fin or img is None: return
    A = _mender_area_mask(img, R)
    if A is None: return
    _MENDER_AREA = (min(o['id'] for o in fin), A)


def _mender_disc(body, ctx):
    """the bright cyan disc around the white core (see the header)"""
    from scipy.ndimage import label as cc
    img = ctx['img']
    if img is None: return body
    if ctx['pred_any'].any():
        core = ctx['region'] & (img.min(-1) >= CORE_T) & _dilate(ctx['pred_any'], 6)
    else:
        if ctx['o']['id'] not in _MENDER_END_NOW: return np.zeros_like(body)   # later ticks: the impact ring (end effect)
        ex, ey = _proj(ctx['cam'], ctx['o']['logic'][0], ctx['o']['logic'][1], 0)
        H0, W0 = body.shape; yy0, xx0 = np.mgrid[0:H0, 0:W0]
        core = (yy0 - ey) ** 2 + (xx0 - ex) ** 2 <= 40 ** 2
        core &= (img.min(-1) >= CORE_T) & (ctx['C'] > 0)
    if not core.any(): return np.zeros_like(body)
    L, n = cc(core, structure=np.ones((3, 3), bool)); sz = np.bincount(L.ravel())
    main = 1 + int(np.argmax(sz[1:])); ys, xs = np.nonzero(L == main)
    cy, cx = ys.mean(), xs.mean()
    H, W = body.shape; yy, xx = np.mgrid[0:H, 0:W]
    near = (yy - cy) ** 2 + (xx - cx) ** 2 <= DISC_R ** 2
    bright = (img[..., 1] >= DISC_T) & (img[..., 2] >= DISC_T) & near
    L2, n2 = cc(_close(bright, 1), structure=np.ones((3, 3), bool))
    ks = np.unique(L2[(L == main) | (core & ~(L == 0))]); ks = ks[ks > 0]
    if not ks.size: return np.zeros_like(body)
    vis = np.isin(L2, ks)
    py_, px_ = np.nonzero(vis)
    my, mx = np.round(2 * cy - py_).astype(int), np.round(2 * cx - px_).astype(int)
    ok = (my >= 0) & (my < H) & (mx >= 0) & (mx < W)
    full = vis.copy(); full[my[ok], mx[ok]] = True
    full &= near | vis
    return hull_mask(full)


def _mender_ulti(body, ctx):
    """the super: one area label while the area is on (see the header); before any landing: the installed cleanup"""
    if _MENDER_AREA is not None:
        oid, A = _MENDER_AREA
        return A.copy() if ctx['o']['id'] == oid else np.zeros_like(body)
    if _mender_prev_ulti is not None: return _mender_prev_ulti(body, ctx)
    return body


_mender_prev_labels = labels


def labels(objs, cam, R, state=None, ended=(), img=None):
    _mender_frame_setup(objs, cam, R, ended, img)
    return _mender_prev_labels(objs, cam, R, state=state, ended=ended, img=img)


POST_BODY[6001755] = _mender_disc
POST_BODY[6001756] = _mender_ulti
