# enemy-label fix 2026-11-07 (task away_1106_enemy, group grpC; installed as z_1116_enemy_grpC.py, global prefix _eC_ / _EC_)
# Every rule below is team-aware (ctx['own'] False only) or keyed to red-only data: own (blue) labels are untouched.
import numpy as _eC_np

# ---------------------------------------------------------------------------------------------------------------------
# Part 1: Colt 004 (skin 258, super 6000027, the stars): z_1023_colt258 adds each star's comet HEAD by a colour test tuned on the
# blue art (deep blue: B-G >= 90, B >= 170, R <= 120, G <= 95) and a1_agent1 grows the star's white rim only over 'cool' pixels (B >= R).
# The red comet head is crimson (255,55-95,110-160) inside a translucent maroon pill (R 140-150, G 43, B 88), the red rim is pink-white
# (R > B): neither test fires, the enemy label was the star only (area ratio 0.25, recall 0.31). Same ellipse / Voronoi geometry as
# z_1023 with a red colour test, and the rim test with B >= G (cream smoke has G > B); team-aware (ctx['own'] False only).
_EC_T2 = 6000027
_EC_RED_DEEP_R = 125     # red head: R >= this, G <= _EC_RED_DEEP_G (the light streak behind it has G >= 100; the translucent maroon halo of the head is R 140-150, G 43)
_EC_RED_DEEP_G = 95
_EC_RED_DEEP_D = 60


def _eC_t2_red_head(body, ctx):
    o = ctx['o']
    if ctx.get('own', True) or o['t'] != _EC_T2 or ctx.get('state') is None or ctx['img'] is None: return body
    rows = [r for r in ctx['state'].get('proj', []) if int(r[1]) == _EC_T2]
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
    deep = (im[..., 0] >= _EC_RED_DEEP_R) & (im[..., 1] <= _EC_RED_DEEP_G) & (im[..., 0] - im[..., 1] >= _EC_RED_DEEP_D)
    head = ell & own & deep
    out = body.copy(); out[y0:y1, x0:x1] |= head
    from scipy.ndimage import label as cc
    L, n = cc(out)
    if n > 1:
        sz = np.bincount(L.ravel()); sz[0] = 0
        out = np.isin(L, np.nonzero(sz >= 0.1 * sz.max())[0]) & out
    return out


def _eC_t2_red_rim(body, ctx):
    # a1_agent1 grows the star's white rim over frame pixels min(R, G, B) >= 150 that are 'cool' (B >= R) so that the warm white puffs of
    # broken boxes stay out; the red star's rim is pink-white (R > B, e.g. 255,189,235): for the enemy the same pixels with B >= G
    # (cream smoke has G > B)
    img = ctx['img']
    if ctx.get('own', True) or img is None or not body.any(): return body
    bright = ctx['region'] & (img.min(-1) >= 150) & (img[..., 2].astype(int) >= img[..., 1])
    for _ in range(RIM_R): body = body | (_dilate(body, 1) & bright)
    return body


_eC_prev_t2 = POST_BODY[_EC_T2]
POST_BODY[_EC_T2] = lambda body, ctx: _eC_t2_red_rim(_eC_t2_red_head(_eC_prev_t2(body, ctx), ctx), ctx)


# ---------------------------------------------------------------------------------------------------------------------
# Part 2: Gunslinger 397 (skin 397 GunSlingerWoody, attack 6000038 and super 6000039): z_1110_misc (_ab5m_397_hook) finds the solid
# tiers of the shot's cone sprite by a colour test tuned on the blue art: the blue shot's tiers are bright green / cyan (G >= 190,
# B >= 110, 'strong' G >= 240) while its aura-lit beam is dimmer. The red shot's tiers are bright ORANGE (255,140-165,25-65; the beam is
# brown, R <= 190, G <= 105): no pixel passes, the hook then removes the whole cone as 'beam' and the enemy label was only a part of the
# bulb (area ratio 0.47 / 0.61). Fix, enemy only: the installed chain runs on the frame with its channels permuted (R -> G, G -> B,
# B -> R), which turns bright orange into bright green-cyan and the dim brown beam into the dim teal beam; every geometric rule of the
# chain (prediction, tiers near it, hidden parts, closing) is colour independent and unchanged. Own (blue) shots call the chain as before.
_EC_397 = (6000038, 6000039)


def _eC_397_team(prev):
    def fn(body, ctx):
        if prev is None: return body
        if ctx.get('own', True) or ctx.get('img') is None: return prev(body, ctx)
        c2 = dict(ctx); c2['img'] = _eC_np.ascontiguousarray(ctx['img'][..., [2, 0, 1]])
        return prev(body, c2)
    return fn


for _t_eC in _EC_397:
    POST_BODY[_t_eC] = _eC_397_team(POST_BODY.get(_t_eC))
