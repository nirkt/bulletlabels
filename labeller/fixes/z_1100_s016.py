# human review 2026-10-28 fix for skin 016 (Haiku fix, reviewed by the coordinator, installed 10-29; source out/sweep2/haikutasks/human1028/s016/fix2.py; tested via IDLABELS_OVERRIDE after all fixes incl. z_1099_cleanup, hence z_1100_)
# Colt default skin 016 super (GunslingerUltiProjectile 6000021), fix 2 (coordinator review of fix.py, 2026-10-28; user's HIDDEN PARTS RULE).
# Rule (binding): a super bullet partly covered by smoke / clouds / leaves keeps its FULL shape in the label, and a live bullet
# gets its full capsule even when nothing of it is visible at that tick. Fully visible bullets keep the capsule-only label of fix.py
# (the core of the comet head, the earlier user decision: the deep-blue head in front of the capsule is glow and is not labelled).
# Method: the bullet's capsule is a stadium in its own motion frame (along = motion direction from o['angle'], origin = the
# projected logic position of this bullet, units = pr = the projected logic radius o['radius'], the same scale as z_1024's head).
# Measured on the unhidden bullets of sw2_016 (super ticks 345-375; 40+ bullet-ticks with extents 2.07-2.12 along, 0.92-0.94 across):
#   total length along the motion  S16_L = 2.10 pr,  total width across S16_W = 0.93 pr,  centre along  S16_CA = -1.27 pr (the
#   capsule trails the logic point), centre across S16_CP = 0.
# The stadium is unioned with the bullet's visible capsule pixels (the ID-pass body), then specks are dropped as before.
# Live bullets whose logic has ended get no label. The per-bullet hull (ONE_LABEL 'hull') is kept, so each bullet is one convex label.
# This file replaces the POST_BODY of fix.py for 6000021 (self-contained; IDLABELS_OVERRIDE runs it after the installed fixes).
# Other skins: type 6000021 is only seen in skin 016 among the captured Colt states (see report).

S16_SUPER = 6000021
S16_L = 2.10      # total length along the motion, units of the projected logic radius (pr)
S16_W = 0.93      # total width across the motion, same units
S16_CA = -1.27    # centre along the motion relative to the projected logic point, same units
S16_CP = 0.0      # centre across the motion, same units
S16_MIN_PR = 1.0  # px: skip a bullet whose projected logic radius is not measurable
# A bullet with less visible capsule than this gets no full shape (its visible pixels are a hit / impact splash, or nothing is visible):
# the full shape is only for a bullet that is partly covered, i.e. still shows a real part of its capsule (~300 px when whole).
S16_MIN_BODY = 60  # px


def _s016_full_capsule(body, ctx):
    o = ctx['o']
    if o['t'] != S16_SUPER: return body
    if o.get('ended'): return np.zeros_like(body)          # a bullet whose logic has ended stays unlabelled
    if int(body.sum()) < S16_MIN_BODY: return body         # nothing / only a splash visible: no full shape (see S16_MIN_BODY)
    st = ctx.get('state'); cam = ctx['cam']
    if st is None: return body
    rows = [r for r in st.get('proj', []) if int(r[1]) == S16_SUPER]
    cen = {int(r[0]): _proj(cam, float(r[3]), float(r[4]), float(r[5])) for r in rows}
    mine = [r for r in rows if int(r[0]) == o['id']]
    if o['id'] not in cen or not mine: return body
    r0 = mine[0]; x, y, z = float(r0[3]), float(r0[4]), float(r0[5]); cx, cy = cen[o['id']]
    a = np.deg2rad(float(o['angle']))
    p2 = _proj(cam, x + 100 * np.cos(a), y + 100 * np.sin(a), z)
    u = np.array(p2) - np.array([cx, cy]); u = u / np.linalg.norm(u); v = np.array([-u[1], u[0]])
    pr = float(np.linalg.norm(np.array(_proj(cam, x + float(o['radius']), y, z)) - np.array([cx, cy])))
    if pr < S16_MIN_PR: return body
    H, W = body.shape
    Rw = int((abs(S16_CA) + S16_L / 2 + 1.0) * pr) + 4
    x0, x1 = max(0, int(cx) - Rw), min(W, int(cx) + Rw + 1); y0, y1 = max(0, int(cy) - Rw), min(H, int(cy) + Rw + 1)
    if x1 <= x0 or y1 <= y0: return body
    yy, xx = np.mgrid[y0:y1, x0:x1]; dx, dy = xx - cx, yy - cy
    al = (dx * u[0] + dy * u[1]) / pr - S16_CA; pe = (dx * v[0] + dy * v[1]) / pr - S16_CP
    r = S16_W / 2; s = S16_L / 2 - r                         # round ends of radius r on a segment of half-length s
    dal = np.maximum(np.abs(al) - s, 0)
    cap = dal ** 2 + pe ** 2 <= r ** 2
    out = body.copy(); out[y0:y1, x0:x1] |= cap
    from scipy.ndimage import label as _cc
    L, n = _cc(out)                                          # specks: pieces much smaller than the largest are dropped
    if n > 1:
        sz = np.bincount(L.ravel()); sz[0] = 0
        out = np.isin(L, np.nonzero(sz >= 0.1 * sz.max())[0]) & out
    return out


POST_BODY[S16_SUPER] = _s016_full_capsule
ONE_LABEL[S16_SUPER] = 'hull'
