# 2026-11-08 enemy-label fix, round v2, group v2 (skins 016 243 328 081). Global prefix _f2_. Installed as py/fixes/z_1118_enemy_v2.py.
# Every hook below acts on ENEMY objects only (ctx['own'] False); own (blue) labels are never touched.
import numpy as _f2_np
from scipy.ndimage import label as _f2_cc

# ---------------------------------------------------------------------------------------------------------------------------------
# 016 GunSlingerDefault, super GunslingerUltiProjectile 6000021 (checker v2: enemy label has extra parts, 4-5 % of the area, 5-7 ticks).
# CAUSE: z_1100_s016 labels the capsule as the visible capsule pixels plus a stadium placed from the bullet's LOGIC position (centre
# -1.27 pr behind the projected logic point). In the red run the drawn capsule runs AHEAD of the logic position by an amount that grows
# during the shot (0.3 px on the first tick, 3-4 px after 40 ticks, in the direction of travel; the blue run: 0.0-0.3 px), so the stadium
# stuck out 3 px behind the red capsule (label 20-25 % too big, 100 px of extra at the tail). FIX (enemy only): the stadium is slid along
# the direction of travel (-2..+8 px) to the position that covers most of the visible capsule (ties: the smaller slide), then unioned with
# the visible capsule as before. Same size / shape / rule as for the blue bullets, only its position is read from the drawn capsule.
_F2_T16 = 6000021
_F2_L, _F2_W, _F2_CA = 2.10, 0.93, -1.27       # as z_1100_s016 (S16_L, S16_W, S16_CA), units of the projected logic radius pr
_F2_SLIDE = _f2_np.arange(-2.0, 8.01, 0.5)     # px along the direction of travel


def _f2_t16_enemy(body, ctx):
    o = ctx['o']
    if o.get('ended'): return _f2_np.zeros_like(body)
    if int(body.sum()) < S16_MIN_BODY: return body
    st = ctx.get('state'); cam = ctx['cam']
    if st is None: return body
    rows = [r for r in st.get('proj', []) if int(r[1]) == _F2_T16]
    cen = {int(r[0]): _proj(cam, float(r[3]), float(r[4]), float(r[5])) for r in rows}
    mine = [r for r in rows if int(r[0]) == o['id']]
    if o['id'] not in cen or not mine: return body
    r0 = mine[0]; x, y, z = float(r0[3]), float(r0[4]), float(r0[5]); cx, cy = cen[o['id']]
    a = _f2_np.deg2rad(float(o['angle']))
    p2 = _proj(cam, x + 100 * _f2_np.cos(a), y + 100 * _f2_np.sin(a), z)
    u = _f2_np.array(p2) - _f2_np.array([cx, cy]); u = u / _f2_np.linalg.norm(u); v = _f2_np.array([-u[1], u[0]])
    pr = float(_f2_np.linalg.norm(_f2_np.array(_proj(cam, x + float(o['radius']), y, z)) - _f2_np.array([cx, cy])))
    if pr < S16_MIN_PR: return body
    H, W = body.shape
    Rw = int((abs(_F2_CA) + _F2_L / 2 + 1.0) * pr) + 4 + int(_F2_SLIDE.max()) + 2
    x0, x1 = max(0, int(cx) - Rw), min(W, int(cx) + Rw + 1); y0, y1 = max(0, int(cy) - Rw), min(H, int(cy) + Rw + 1)
    if x1 <= x0 or y1 <= y0: return body
    yy, xx = _f2_np.mgrid[y0:y1, x0:x1]; dx, dy = xx - cx, yy - cy
    pe = (dx * v[0] + dy * v[1]) / pr - S16_CP
    r = _F2_W / 2; s_ = _F2_L / 2 - r
    bsub = body[y0:y1, x0:x1]
    best = None
    for sl in _F2_SLIDE:
        al = (dx * u[0] + dy * u[1] - sl) / pr - _F2_CA
        cap = _f2_np.maximum(_f2_np.abs(al) - s_, 0) ** 2 + pe ** 2 <= r ** 2
        sc = (int((cap & bsub).sum()), -abs(sl))
        if best is None or sc > best[0]: best = (sc, cap)
    out = body.copy(); out[y0:y1, x0:x1] |= best[1]
    L, n = _f2_cc(out)
    if n > 1:
        sz = _f2_np.bincount(L.ravel()); sz[0] = 0
        out = _f2_np.isin(L, _f2_np.nonzero(sz >= 0.1 * sz.max())[0]) & out
    return out


_f2_t16_prev = POST_BODY.get(_F2_T16)
def _f2_t16_hook(body, ctx):
    if ctx.get('own') or _f2_t16_prev is None: return _f2_t16_prev(body, ctx) if _f2_t16_prev else body
    return _f2_t16_enemy(body, ctx)
POST_BODY[_F2_T16] = _f2_t16_hook

# ---------------------------------------------------------------------------------------------------------------------------------
# 243 GunslingerPirate, attack Colt003Projectile 6000024 (checker v2: enemy label has extra parts, 18-22 % of the area, 11 ticks, every
# attack row). CAUSE: the RED clip is launched with a big muzzle burst (fire + dark smoke, ~25 separate blended draws at the shooter, 100-1900 px)
# that the blue clip does not draw. Its draws use the same atlas shapes as the orb (sprite code 1), so the first ticks of every shot got
# the burst as body: one 1100-2000 px label sitting on the muzzle (the orb itself 300 px). FIX (enemy only): the body is cut to the 3 px
# dilated convex hull of the tracker's predicted body (ctx['pred_body'], which lines up with the drawn orb + its flame in the blue run
# and in the red run), as _ecc_ does for the Cocooner. Every blue body of this type lies inside that hull (checked on the own capture), and
# enemy-only: own labels are unchanged. Applied before the installed front cut (z_1027_s243).
_F2_T243 = 6000024
_F2_T243_K = 3


def _f2_t243_make(prev):
    def post(body, ctx):
        if not ctx.get('own') and body.any():
            pb = ctx['pred_body']
            if pb.any():
                from scipy.ndimage import binary_dilation as _f2_dil
                body = body & _f2_dil(hull_mask(pb), iterations=_F2_T243_K)
        return prev(body, ctx) if prev is not None else body
    return post


POST_BODY[_F2_T243] = _f2_t243_make(POST_BODY.get(_F2_T243))

# ---------------------------------------------------------------------------------------------------------------------------------
# 081 DuelistDefault, super DuelistUltiProjectile 6001341 (checker v2: the enemy label lacks parts the own label has, super tick 3 = the first
# tick(s) of a shot). CAUSE: as 051 (z_1109_enemy_pl): at the launch tick the RED mushroom is covered by the big white additive muzzle burst that
# the blue clip does not have (blue label 900 px, red body 372 px, of which 330 low-coverage trail-over pixels; the predicted body, 921 px, matches
# the blue one). FIX (enemy only, ctx['own'] False): when less than half of the predicted body is in the body (the shot is hidden under the
# burst) the label is the predicted body (cut to the region) plus the exposed body pixels within 6 px of it (hidden-part rule, as
# z_1116_enemy_grpE; no convex hull: the mushroom is not convex, the hull was 1.5x too big; pieces of the body further away, the trail, are dropped).
_F2_T081 = 6001341
_f2_t081_prev = POST_BODY.get(_F2_T081)


def _f2_t081_post(body, ctx):
    if _f2_t081_prev is not None: body = _f2_t081_prev(body, ctx)
    if ctx.get('own') or not body.any(): return body
    pb = ctx['pred_body']
    if not pb.any() or (body & pb).sum() >= 0.5 * pb.sum(): return body
    from scipy.ndimage import binary_dilation as _f2_dil
    near = body & _f2_dil(pb, iterations=6)     # the exposed body pixels at the predicted body (not the burst's / trail's pieces further away)
    return (pb & ctx['region']) | near


POST_BODY[_F2_T081] = _f2_t081_post
