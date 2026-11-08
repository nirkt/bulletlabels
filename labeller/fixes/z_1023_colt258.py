# Colt Player 258 super: each star + its comet head, one hull per bullet (user Q5, 2026-10-23; Haiku task 2, reviewed)
# task 2 fix (rework 2): skin 258 GunslingerPlayer (Colt004), super 6000027 (Colt004UltiProjectile, the stars).
# The user's decision (binding): each bullet is labelled with its star and the comet HEAD that moves with it (the bright blue
# blob), each bullet a separate label, one clean convex shape per bullet. The streak / tail that runs to the other bullets, the
# trail particles and the hit effects stay out. The coordinator's rework (2026-10-23): the label is one convex hull per bullet
# (ONE_LABEL 'hull'), no necks or tails, neighbours' hulls do not overlap, partly hidden bullets are hulled from what is visible.
#
# Measured on sw2_258 super ticks 343-377 (100 bullets; scratch/map2d.py, measure2.py), in the bullet's own motion frame
# (along = motion direction, from the state's logic position and o['angle'], projected with the camera; scale = the projected
# logic radius o['radius'] = 101, about 13 px):
#  - the head is the DEEP blue blob in FRONT of the star (green 60-80, blue 200-220): fill 0.6-0.8 for 0.3-1.0 radii
#    forward, centre about 0.5 radii forward, about 0.9 radii across;
#  - the streak trails BEHIND, lighter (green 105-120, blue 250): fill 0.1-0.2, never deep blue.
# Limit used: an ellipse in that frame (centre T2_HEAD_C, half-lengths T2_HEAD_A along and T2_HEAD_B across, all in radii of
# the bullet's own projected logic radius) AND the deep-blue colour (green <= T2_HEAD_G). The ellipse is the most robust part
# (game data + camera; no colour dependence); the colour cuts the streak where it crosses the ellipse.
# Voronoi: a pixel belongs to a bullet only when no other super bullet of the frame is nearer.
# Then ONE_LABEL[6000027] = 'hull' takes the convex hull of star + head per bullet (after POST_BODY, per object).

T2_SUPER = 6000027
T2_HEAD_C = 0.5        # radii along the motion direction: centre of the head ellipse
T2_HEAD_A = 0.75       # radii: half-length of the head along the motion
T2_HEAD_B = 0.95       # radii: half-width across the motion
T2_HEAD_G = 95         # green at most (deep blue head; the streak is green 105-120)


def _t2_head(body, ctx):
    o = ctx['o']
    if o['t'] != T2_SUPER or not body.any() or ctx.get('state') is None or ctx['img'] is None:
        return body
    rows = [r for r in ctx['state'].get('proj', []) if int(r[1]) == T2_SUPER]
    cam = ctx['cam']
    cen = {int(r[0]): _proj(cam, float(r[3]), float(r[4]), float(r[5])) for r in rows}
    mine = [r for r in rows if int(r[0]) == o['id']]
    if o['id'] not in cen or not mine: return body
    r0 = mine[0]; x, y, z = float(r0[3]), float(r0[4]), float(r0[5]); cx, cy = cen[o['id']]
    a = np.deg2rad(float(o['angle']))
    p2 = _proj(cam, x + 100 * np.cos(a), y + 100 * np.sin(a), z)
    u = np.array(p2) - np.array([cx, cy]); u = u / np.linalg.norm(u); v = np.array([-u[1], u[0]])
    pr = np.linalg.norm(np.array(_proj(cam, x + float(o['radius']), y, z)) - np.array([cx, cy]))
    if pr < 1: return body
    H, W = body.shape
    R = int((T2_HEAD_C + T2_HEAD_A + 0.5) * pr) + 4
    x0, x1 = max(0, int(cx) - R), min(W, int(cx) + R + 1); y0, y1 = max(0, int(cy) - R), min(H, int(cy) + R + 1)
    if x1 <= x0 or y1 <= y0: return body
    yy, xx = np.mgrid[y0:y1, x0:x1]; dx, dy = xx - cx, yy - cy
    al = (dx * u[0] + dy * u[1]) / pr; pe = (dx * v[0] + dy * v[1]) / pr
    ell = ((al - T2_HEAD_C) / T2_HEAD_A) ** 2 + (pe / T2_HEAD_B) ** 2 <= 1
    d_own = np.hypot(dx, dy); own = np.ones(d_own.shape, bool)
    for i, (px, py) in cen.items():
        if i != o['id']: own &= d_own <= np.hypot(xx - px, yy - py)
    im = ctx['img'][y0:y1, x0:x1].astype(np.int16)
    deep = (im[..., 2] - im[..., 1] >= 90) & (im[..., 2] >= 170) & (im[..., 0] <= 120) & (im[..., 1] <= T2_HEAD_G)
    head = ell & own & deep
    out = body.copy(); out[y0:y1, x0:x1] |= head
    from scipy.ndimage import label as cc
    L, n = cc(out)                                  # specks: pieces much smaller than the largest are dropped
    if n > 1:
        sz = np.bincount(L.ravel()); sz[0] = 0
        out = np.isin(L, np.nonzero(sz >= 0.1 * sz.max())[0]) & out
    return out


# keep agent 1's rules for this type (own-draw filter, white rim), then add the head
_t2_prev_super = POST_BODY[T2_SUPER]
POST_BODY[T2_SUPER] = lambda body, ctx: _t2_head(_t2_prev_super(body, ctx), ctx)
# one clean convex label per bullet: the hull of its star and head (per object, after POST_BODY)
ONE_LABEL[T2_SUPER] = 'hull'
