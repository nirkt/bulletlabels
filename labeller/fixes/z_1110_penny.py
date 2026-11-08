# Penny landing marker (owner, final review 2026-11-04/05, pages 233 / 265, row "super 45 deg" tick 389, note: "for ALL penny
# skins, the landing marker is too small, more like an elipse rather than a circle").
# Cause: z_1024_penny_classes builds the marker label as an axis-aligned screen ellipse from the LINEAR scale at the centre
# (rx, ry per logic unit), but the camera is a perspective: the drawn ring (a circle on the ground) is a tilted, wider conic
# and the label fell 4-6 px inside it on the sides. Fix: the label is the ground circle itself: every screen pixel is mapped
# back to the ground plane (inverse of the camera's ground homography) and is in the label when it lies within the ring
# radius of the shot's end point. The radius is fitted in true ground units from the marker sprite's own pixels
# (sprite 'pre' of the turret projectile, coverage >= 64, 99th percentile of the distance from the end point).
# Applies to every type in PENNY_TURRET (ArtilleryDude / Penny skins: 6000440, 6000442, 6000446, 6000456, 6000460, 6000464,
# 6000468, 6000472: Dude + Overcharged Dude, Penny 002/005..009, i.e. all skins of Penny that fire a turret bomb).
# Everything else of the original rule is kept: cut by solid map blocks, not over other labels, class 'target'.
_ab4_penny_prev_labels = labels
_AB4_PENNY_Q = 99          # percentile of the ring pixels' ground distance = ring radius (the old fit used 98 on linear scale)


def _ab4_penny_ground(cam):
    """(3x3) inverse of the ground homography: screen (px, py) (half-res, _proj convention) -> (x, -y, 1) * w"""
    A = cam[:, [0, 1, 3]]
    Pm = np.array([[600.0, 0, 600.0], [0, -270.0, 270.0], [0, 0, 1.0]]) @ np.array([A[0], A[1], A[3]])
    return np.linalg.inv(Pm)


def labels(objs, cam, R, state=None, ended=(), img=None):
    import bulletlabels as BL
    P = _ab4_penny_prev_labels(objs, cam, R, state=state, ended=ended, img=img)
    tg = [k for k, l in enumerate(P['lids']) if l[2] == 'target']
    if not tg: return P
    bombs = {o['id']: o for o in objs if o['t'] in PENNY_TURRET and o.get('drawn') and o.get('target')}
    ids, share = P['ids'], P['share']; H, W = ids.shape
    try: solid = _zs_solid(cam, R)
    except Exception: solid = np.zeros((H, W), bool)
    cv = R['cv'][::2, ::2]
    try: Pi = _ab4_penny_ground(cam)
    except Exception: return P
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    q = np.einsum('ij,jhw->ihw', Pi, np.stack([xx, yy, np.ones_like(xx)]))
    with np.errstate(all='ignore'):
        gx = q[0] / q[2]; gy = -q[1] / q[2]
    for k in tg:
        o = bombs.get(P['lids'][k][0] - PENNY_TARGET_ID)
        if o is None: continue
        M = (sprite_pixels(('pre', o['t']), o['own'], R) > 0) & (cv >= PENNY_MARK_CV)
        if M.sum() < PENNY_MARK_MIN: continue
        x, y = o['target']
        r = float(np.percentile(np.hypot(gx[M] - x, gy[M] - y), _AB4_PENNY_Q))
        disc = np.isfinite(gx) & (np.hypot(gx - x, gy - y) <= r)
        old = ids == k; ids[old] = -1
        m = disc & ~solid & (ids < 0)
        if not m.any(): ids[old] = k; continue
        ids[m] = k; share[m] = BL.CUT
    return P
