# 2026-11-08 enemy-label fix, group off2 (offset-scan suspects 625 Primo014 and 130 Grom003). Prefix _o3_. To be installed as py/fixes/z_1120_enemy_off2.py.
# Enemy objects only (ctx['own'] False / red-only data): OWN labels are never touched.
import numpy as _o3_np
from scipy.ndimage import binary_dilation as _o3_dil, binary_opening as _o3_open, binary_fill_holes as _o3_fill, label as _o3_cc


def _o3_disk(k):
    yy, xx = _o3_np.mgrid[-k:k + 1, -k:k + 1]
    return (xx ** 2 + yy ** 2) <= k * k


# ---------------------------------------------------------------------------------------------------------------------------
# (625 LuchadorMHA) Primo014Projectile 6000410 (flaming fist), enemy shots. z_1116_enemy_grpF / z_1118_enemy_v12 find the red fist by a colour
# test (b >= 120, r >= 150, b >= g - 30) and an opening with a 6 px disk. The red fist's shaded rim and its glow-lit edge are salmon / red
# (r 255, g ~100, b 80-115: b < 120), so they fail the test: the label stopped 5-10 px inside the drawn fist (the blue fist's rim is cyan and
# is taken by the blue rule, so the own label covers the whole fist). Offset scan 11-08: the enemy label is 20-30 % smaller than the own
# label at some ticks (a fist third missing on the open side). Fix: the chain's result grows (<= 8 px, geodesic) into the model's own pixels
# (the label body before the colour test) that are red / pink / very light (not the arm's yellow / orange stripes: g - b >= 50, not its
# magenta stripes: hue 300..335), then is opened by a 2 px disk, holes filled, kept inside the object's region.
_o3_PR = 6000410
_o3_PR_GROW = 10
_o3_STREAK_TH = '10ab7fd9283eda54'


def _o3_hue_ok(img):
    r = img[..., 0].astype(_o3_np.int32); g = img[..., 1].astype(_o3_np.int32); b = img[..., 2].astype(_o3_np.int32)
    mx = _o3_np.maximum(_o3_np.maximum(r, g), b); mn = _o3_np.minimum(_o3_np.minimum(r, g), b); d = (mx - mn).astype(_o3_np.float32)
    h = _o3_np.zeros(r.shape, _o3_np.float32); nz = d > 0
    # only the red-dominant case matters (r is the max for salmon / pink / magenta / orange / yellow)
    h[nz & (mx == r)] = (60.0 * ((g - b)[nz & (mx == r)] / d[nz & (mx == r)])) % 360
    reddish = (mx == r) & (r >= 190)
    pink_red = reddish & ((h >= 338) | (h <= 22)) & ((g - b) < 50)
    light = (mn >= 150) & (mx >= 200)                      # white highlights of the fist
    return pink_red | light


def _o3_pr_make(prev):
    def post(body, ctx):
        res = prev(body, ctx) if prev is not None else body
        if ctx.get('own') or not res.any(): return res
        img = ctx['img']
        if img is None: return res
        R = ctx['R']
        streak_ids = [d for d, v in R['draws'].items() if v.get('th') == _o3_STREAK_TH]
        under = _o3_np.isin(ctx['ids_h'], streak_ids) if streak_ids else _o3_np.zeros(body.shape, bool)   # the red speed streaks drawn over the fist
        pb = ctx['pred_body']
        cand = (body | (under & pb)) & _o3_hue_ok(img) & ctx['region']
        out = res.copy()
        for _ in range(_o3_PR_GROW):
            out = out | (_o3_dil(out, structure=_o3_np.ones((3, 3), bool)) & cand)
        p = 6
        B = _o3_np.pad(out, p); B = _o3_open(B, structure=_o3_disk(2))
        L, n = _o3_cc(B)
        if n > 1:
            sz = _o3_np.bincount(L.ravel())[1:]; B = _o3_np.isin(L, 1 + _o3_np.nonzero(sz >= 0.3 * sz.max())[0])
        B = _o3_fill(B)[p:-p, p:-p]
        return (B | res) & ctx['region']
    return post


if not getattr(POST_BODY.get(_o3_PR), '_o3', False):
    _o3_f = _o3_pr_make(POST_BODY.get(_o3_PR)); _o3_f._o3 = True; POST_BODY[_o3_PR] = _o3_f


# ---------------------------------------------------------------------------------------------------------------------------
# (130 CrossBomberFranken) Grom003UltiProjectile2 6000969 (the burst pieces of the landed super), enemy pieces. The red art has a thick dark-red
# swoosh ribbon (8-9 px wide at half resolution, 150-200 px long) that curls around the shard; when it touches the shard (offset scan, super 45
# t371) it is part of the piece's body (the ID pass attributes it to the same clip draw) and z_1115_sticky5's 'keep parts on the predicted body'
# keeps it (it is connected to the head): the label grew a hooked tail over the ribbon (36 % of the label). The own (blue) pieces never touch
# it. Enemy pieces only: a part that an opening with a 5 px disk removes (thin ribbon; the shard is 20+ px) and that is one connected piece of
# >= 80 px is cut off (the rim of the shard within 2 px stays). Smaller removals (a hidden piece under a flash, shard corners) are left alone.
_o3_GR = 6000969
_o3_GR_OPEN = 5
_o3_GR_MIN = 80


def _o3_gr_make(prev):
    def post(body, ctx):
        res = prev(body, ctx) if prev is not None else body
        if ctx.get('own') or not res.any(): return res
        p = 8
        B = _o3_np.pad(res, p); op = _o3_open(B, structure=_o3_disk(_o3_GR_OPEN))
        if op.sum() < 0.5 * B.sum(): return res
        L, n = _o3_cc(op)
        if n > 1:
            sz = _o3_np.bincount(L.ravel())[1:]; op = _o3_np.isin(L, 1 + _o3_np.nonzero(sz >= 0.3 * sz.max())[0])
        out = op | (B & _o3_dil(op, structure=_o3_disk(2)))
        rem = B & ~out
        L2, n2 = _o3_cc(rem)
        if n2 == 0: return res
        sz2 = _o3_np.bincount(L2.ravel())[1:]
        big = _o3_np.isin(L2, 1 + _o3_np.nonzero(sz2 >= _o3_GR_MIN)[0])
        if not big.any(): return res
        return (B & ~big)[p:-p, p:-p]
    return post


if not getattr(POST_BODY.get(_o3_GR), '_o3', False):
    _o3_g = _o3_gr_make(POST_BODY.get(_o3_GR)); _o3_g._o3 = True; POST_BODY[_o3_GR] = _o3_g
