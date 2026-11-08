# Colt default 016 super (GunslingerUltiProjectile 6000021): capsule + deep-blue comet head, one hull per bullet (Q5, as Colt 258; Haiku task 30, reviewed 10-24)
# Colt default, skin 016 GunSlingerDefault (projtab 016, super type GunslingerUltiProjectile 6000021, the blue comet bullet).
# The user's decision (binding, Q5 of 2026-10-23, as applied to Colt skin 258 in py/fixes/z_1023_colt258.py): each bullet is
# labelled TOGETHER with the effect that moves with it (its comet head), every bullet a separate label, one clean solid label
# per bullet (ONE_LABEL 'hull'); the streak behind the bullet, trail particles, smoke and hit effects stay out.
# Measured on sw2_016 super ticks 343-377: the bullet = a bright cyan capsule (clip shape 6703, the body the ID pass sees) inside
# a deep-blue shell that sticks out in FRONT of it (the head). The shell is not a shape of the bullet's clip (its draws show no
# clip shape), so it is taken by geometry in the bullet's own motion frame (as for 258: an ellipse of its projected logic radius,
# radius 101 like 258), by blue dominance (smoke is cream, the background is dark), and only where it touches the capsule (a
# detached blue patch, e.g. in smoke, would make the hull bridge the smoke as a strip). A bullet with fewer than T16_MIN_CAP
# capsule pixels (hidden by smoke) gets no head.

T16_SUPER = 6000021     # GunslingerUltiProjectile (Colt default)
T16_HEAD_C = 0.5        # radii along the motion direction: centre of the head ellipse
T16_HEAD_A = 0.75       # radii: half-length along the motion
T16_HEAD_B = 0.95       # radii: half-width across the motion
T16_BLUE_MIN = 150     # blue at least (the shell is blue 190-220; the background under it is blue 80)
T16_HEAD_BR = 90       # blue minus red at least (smoke is cream, background is about 30)
T16_MIN_CAP = 40        # px: the bullet's capsule must be visible this much, else no head (partly hidden bullets stay as they are)


def _t16_head(body, ctx):
    o = ctx['o']
    if o['t'] != T16_SUPER or ctx.get('state') is None or ctx['img'] is None or int(body.sum()) < T16_MIN_CAP: return body
    rows = [r for r in ctx['state'].get('proj', []) if int(r[1]) == T16_SUPER]
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
    R = int((T16_HEAD_C + T16_HEAD_A + 0.5) * pr) + 4
    x0, x1 = max(0, int(cx) - R), min(W, int(cx) + R + 1); y0, y1 = max(0, int(cy) - R), min(H, int(cy) + R + 1)
    if x1 <= x0 or y1 <= y0: return body
    yy, xx = np.mgrid[y0:y1, x0:x1]; dx, dy = xx - cx, yy - cy
    al = (dx * u[0] + dy * u[1]) / pr; pe = (dx * v[0] + dy * v[1]) / pr
    ell = ((al - T16_HEAD_C) / T16_HEAD_A) ** 2 + (pe / T16_HEAD_B) ** 2 <= 1
    d_own = np.hypot(dx, dy); own = np.ones(d_own.shape, bool)
    for i, (px, py) in cen.items():
        if i != o['id']: own &= d_own <= np.hypot(xx - px, yy - py)
    im = ctx['img'][y0:y1, x0:x1].astype(np.int16)
    deep = (im[..., 2] >= T16_BLUE_MIN) & (im[..., 2] - im[..., 0] >= T16_HEAD_BR)
    head = ell & own & deep
    from scipy.ndimage import label as cc0
    Lh, nh = cc0(head | body[y0:y1, x0:x1]); keep_h = np.unique(Lh[body[y0:y1, x0:x1]]); keep_h = keep_h[keep_h > 0]
    head = head & np.isin(Lh, keep_h)   # only the head that touches the capsule (a detached blue patch, e.g. in smoke, is not the head)
    out = body.copy(); out[y0:y1, x0:x1] |= head
    from scipy.ndimage import label as cc
    L, n = cc(out)                                  # specks: pieces much smaller than the largest are dropped
    if n > 1:
        sz = np.bincount(L.ravel()); sz[0] = 0
        out = np.isin(L, np.nonzero(sz >= 0.1 * sz.max())[0]) & out
    return out


_t16_prev = POST_BODY.get(T16_SUPER, lambda body, ctx: body)
POST_BODY[T16_SUPER] = lambda body, ctx: _t16_head(_t16_prev(body, ctx), ctx)
# one clean convex label per bullet: the hull of its capsule and head (per object, after POST_BODY)
ONE_LABEL[T16_SUPER] = 'hull'
