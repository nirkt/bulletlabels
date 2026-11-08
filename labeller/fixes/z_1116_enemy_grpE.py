# enemy-label fix 2026-11-07 (task away_1106_enemy, group grpE; installed as z_1116_enemy_grpE.py; global prefix _eE_)
# Every rule below is wrapped so it only runs for ENEMY shots (ctx['own'] False / o['own'] False); own (blue) labels call the previously
# installed hook unchanged, so they stay pixel-identical.
import numpy as _eE_np


def _eE_team(prev, red_fn):
    """POST_BODY hook: own shots -> the previous hook (or the body unchanged); enemy shots -> red_fn(body, ctx, prev)"""
    def fn(body, ctx):
        if ctx['own']: return prev(body, ctx) if prev is not None else body
        return red_fn(body, ctx, prev)
    return fn


# ---- 316 FireDudeFrost, FireDude003Projectile 6000833 (frost wave): z_1027_s316 takes the wave from blue-tuned colour tests
# (20 <= G-R <= 90, R >= 120); the red wave is pink-white (R >= G). Same pipeline, red colour test.
def _eE_316_red(body, ctx, prev):
    from scipy.ndimage import label as _cc, binary_fill_holes as _bfh, binary_opening as _bop
    img = ctx['img']; region = ctx['region']
    if img is None: return body
    im = img.astype(_eE_np.int16); mx = im.max(-1); mn = im.min(-1); drg = im[..., 0] - im[..., 1]
    m = (mx >= 140) & (im[..., 2] >= 150) & (drg >= 8) & (drg <= 110)
    white = (mn >= 180) & (_eE_np.abs(drg) < 12)
    m |= white & _dilate(m, 2)
    m = _bfh(_close(m, 3))
    m0 = _bop(m, structure=_eE_np.ones((3, 3), bool))
    yy, xx = _eE_np.mgrid[-2:3, -2:3]; disk = (yy * yy + xx * xx) <= 5
    m1 = _bop(m0, structure=disk)
    m = m1 if m1.sum() >= 0.8 * m0.sum() else m0
    L, n = _cc(m)
    if not n: return body
    sz = _eE_np.bincount(L.ravel())[1:]
    touch = _eE_np.bincount(L[region].ravel(), minlength=n + 1)[1:] > 0
    if not touch.any(): return body
    k = 1 + int(_eE_np.argmax(_eE_np.where(touch, sz, 0)))
    m = _bfh(L == k)
    return m if m.any() else body


POST_BODY[6000833] = _eE_team(POST_BODY.get(6000833), _eE_316_red)



# ---- 405 BeamerCarnival, Beamer005Projectile 6001306 (flame beam): z_1115_beam5 builds the label from the beam's BRIGHT body: the teal
# tail band is 'G >= 170, R <= 130'. The red beam's tail is a coral gradient (R 190..250, G 75..115, B ~100) that never passes the teal
# test, so the enemy label stops at the yellow head. Same function (copied), enemy bright test adds the coral band; its draws filter
# (only pixels of the beam's own draws, leaves / petals out) is unchanged.
_EN405_T = 6001306
_EN405_CORAL = dict(r=185, gmin=72, gmax=214, rg=90, bmax=165)


def _eE_405_red(body, ctx, prev):
    from scipy.ndimage import label as _cc, binary_closing as _bc, binary_fill_holes as _bfh
    T = projtab.table()[_EN405_T]
    if _bm5_prev is not None: body = _bm5_prev(body, ctx)
    if not body.any(): return body
    names = tuple(n for n in (T['blue'], T['red']) if n)
    SP = _a4_shape_pixels(ctx['R'], T['file'], names) - 1
    region = ctx['region']; cv = ctx['cv']
    im = ctx['img'].astype(_eE_np.int16); r, g, b = im[..., 0], im[..., 1], im[..., 2]
    Cc = _EN405_CORAL
    coral = (r >= Cc['r']) & (g >= Cc['gmin']) & (g <= Cc['gmax']) & (r - g >= Cc['rg']) & (b <= Cc['bmax'])
    bright = (g >= 215) | coral | ((r >= 235) & (g >= 150) & (b <= 75))
    core = _eE_np.isin(SP, _bm5_CORE) & (cv >= 64)
    clipbody = (ctx['C'] == 1) & (cv >= CUT)
    petal = (SP == 63) & ~(g >= 215)             # the red clip's pink petal burst sprites (shape 63: 30-100 px in the blue run, 250+ in the red one): not the flame
    body = body & ~petal
    V = ((bright & region) | (core & region) | (clipbody & region)) & ~petal
    ih = ctx['ids_h']; leafpx = _eE_np.zeros_like(V)
    cores = _eE_np.isin(SP, _bm5_CORE) | (SP == 135)
    cnt = {int(d): int((V & (ih == d)).sum()) for d in _eE_np.unique(ih[V]) if d > 0}
    top = max(cnt, key=cnt.get) if cnt else None
    for d in cnt:
        if d == top: continue
        m = (ih == d) & region
        if (m & (SP == 133)).sum() >= 10 and not (m & cores).any(): leafpx |= m
    V &= ~leafpx
    own_draw = _eE_np.zeros_like(V)
    for d in _eE_np.unique(ih[region]):
        if d <= 0: continue
        m = (ih == d) & region
        if (m & (SP >= 0)).sum() >= 8: own_draw |= m
    strong = (g >= 215)
    V &= (own_draw | ((ih == 0) & strong))
    V &= strong | clipbody | ((SP >= 0) & (SP != 0))
    seed = body | core
    L, n = _cc(V | body, structure=_eE_np.ones((3, 3), bool))
    keep = _eE_np.unique(L[seed & V]); keep = keep[keep > 0]
    out = _eE_np.isin(L, keep) & V
    if not out.any(): return body
    ys, xs = _eE_np.nonzero(out); pad = _bm5_CLOSE + 2
    y0, y1, x0, x1 = max(0, ys.min() - pad), ys.max() + pad + 1, max(0, xs.min() - pad), xs.max() + pad + 1
    sl = (slice(y0, y1), slice(x0, x1))
    sub = _eE_np.pad(out[sl], _bm5_CLOSE)
    cl = _bc(sub, structure=_bm5_disc(_bm5_CLOSE))[_bm5_CLOSE:-_bm5_CLOSE, _bm5_CLOSE:-_bm5_CLOSE]
    res = out.copy(); res[sl] |= cl & (g[sl] >= 60) & ~leafpx[sl]
    res[sl] = _bfh(res[sl])
    pb = ctx.get('pred_body')
    if pb is not None and pb.any(): res |= body & pb
    res &= ~petal
    # petals drawn over the flame leave holes in it: the flame is solid (closed by 5 px, only over petal pixels, then holes filled)
    res |= _bc(_eE_np.pad(res, 6), structure=_bm5_disc(5))[6:-6, 6:-6] & petal
    res = _bfh(res)
    # the red shot is covered at its first tick(s) by a big white additive muzzle burst (draws of no beam sprite): when under half of the
    # predicted body is in the label, the hidden part is added (hull of the label and the predicted body, cut to the region; z_1109_enemy_pl)
    if pb is not None and pb.any() and (res & pb).sum() < 0.5 * pb.sum():
        near = res & _dilate(pb, 12)
        res = near | (hull_mask(near | pb) & region)
    return res


POST_BODY[_EN405_T] = _eE_team(POST_BODY.get(_EN405_T), _eE_405_red)


# ---- 433 BeamerMecha, Beamer007Projectile 6001310 (chevron): z_1027_s433 makes the label the solid hull of shape 267 near the ID-pass body. The red
# shot's muzzle effect (pink oval + plates, an enemy-only effect drawn by other draws with the same atlas polygon 267) lies right next to the
# chevron and, drawn over / beside it, is part of that body and of the 267 pixels: the hull swallowed the muzzle (2-3x the own area). In the
# blue run the same muzzle sprites are drawn too but fall outside the search region. Enemy: the body handed to the hook and its result are
# kept within 3-4 px of the PREDICTED label (the chevron + arms, game state), which is where the own label always lies.
_EN433_T = 6001310


def _eE_433_red(body, ctx, prev):
    pa = ctx['pred_any']
    if pa is None or not pa.any(): return body
    near = _dilate(pa, 3)
    # the 267 pixels of draws that do not touch the predicted body are the muzzle's (the chevron's arms are in the head's own draw)
    ih = ctx['ids_h']; pb = ctx['pred_body']
    SP = _a4_shape_pixels(ctx['R'], B7_FILE, B7_NAMES)
    chev = _eE_np.unique(ih[pb & (ih > 0)]) if pb.any() else _eE_np.zeros(0, int)
    muz = (SP == B7_EDGE + 1) & ~_eE_np.isin(ih, chev) & ~pb
    out = prev(body & near & ~muz, ctx) if prev is not None else body & near
    return out & _dilate(pa, 4) & ~muz


POST_BODY[_EN433_T] = _eE_team(POST_BODY.get(_EN433_T), _eE_433_red)


# ---- 466 BullGuyHair 6000048 (cyan shards of Bull's spray) / 603 BullDudeDemon 6000051: at the first drawn tick the RED shot is covered by a big white
# additive muzzle burst (draws of no shard sprite); the ID-pass body keeps a few exposed specks. Same hidden-part rule as z_1109_enemy_pl (enemy only):
# when under 30% of this shard's predicted body is in its label, the predicted body is added (cut to the region).
def _eE_hidden_pred(prev, body, ctx):
    if prev is not None: body = prev(body, ctx)
    pb = ctx['pred_body']
    if pb is None or not pb.any() or (body & pb).sum() >= 0.3 * pb.sum(): return body
    return body | (pb & ctx['region'])


def _eE_hidden_hook(t):
    return _eE_team(POST_BODY.get(t), lambda body, ctx, prev: _eE_hidden_pred(prev, body, ctx))


POST_BODY[6000048] = _eE_hidden_hook(6000048)


# 603 BullDudeDemon, Bull011Projectile 6000051: z_1027_s603 cuts the impact-star flash from the body with 'min(R, G, B) >= 130' (gated by >= 40
# body pixels with min >= 150): true for the blue bullets' dark blue / cyan only. The red bullets are pink with highlights of min(R, G, B) 130..225
# (the star itself is pure white, min >= 235), so the cut ate the bullets' bright faces and every enemy label was a dark-part fragment.
# Enemy: the installed chain of hooks is called with the three thresholds (module globals BULL011_FLASH_MIN / _CORE_MIN / _CORE_N, read at call time)
# set for the white star only (core min >= 235, gate >= 240 on >= 30 px), then restored; then the hidden-part rule (muzzle burst, see 466).
_EN603 = dict(BULL011_FLASH_MIN=235, BULL011_CORE_MIN=240, BULL011_CORE_N=30)


def _eE_603_red(body, ctx, prev):
    g = globals(); old = {k: g[k] for k in _EN603}
    g.update(_EN603)
    try: body = prev(body, ctx) if prev is not None else body
    finally: g.update(old)
    return _eE_hidden_pred(None, body, ctx)


POST_BODY[6000051] = _eE_team(POST_BODY.get(6000051), _eE_603_red)


# ---- 495 EnragerDuel, Enrager002Projectile 6000876 (cyan diamond band): z_1024_enrager495_624 takes the band from the frame with
# 'G >= 180, B >= 235, R <= 200' (cyan). The red band is pink-white (R 245..255, G 85..170, B 140..245): the blue test finds almost nothing
# of it, so the enemy label shrank to the ID-pass shapes. Enemy: the same pipeline with the red band colours (the magenta smoke trail is G < 60).
_EN495 = dict(r=225, gmin=80, bmin=140, rg=70)


def _eE_495_red(body, ctx, prev):
    from scipy.ndimage import label as _cc, binary_fill_holes as _fh
    im = ctx['img'].astype(_eE_np.int16)
    C = _EN495
    bright = (im[..., 0] >= C['r']) & (im[..., 1] >= C['gmin']) & (im[..., 2] >= C['bmin']) & ((im[..., 1] >= 125) | (im[..., 2] <= 238))      # magenta smoke: B >= 240 with G < 125
    SP = _a4_shape_pixels(ctx['R'], 'sc/effects_brawler.sc', ('edgar_002_atk', 'edgar_002_atk_red'))
    own = _eE_np.isin(SP, [s + 1 for s in E002_KEEP]) & (ctx['cv'] >= CUT) & ctx['region']
    m = (bright & ctx['region']) | own
    if not m.any(): return m
    m = _close(m, 2)
    m = _fh(m) & ctx['region']
    L, n = _cc(m)
    if n == 0: return m
    anchor = _eE_np.unique(L[(own | ctx['pred_body']) & m]); anchor = anchor[anchor > 0]
    return _eE_np.isin(L, anchor) if anchor.size else m


POST_BODY[6000876] = _eE_team(POST_BODY.get(6000876), _eE_495_red)


# ---- 199 ChronomancerKaiju, Chronomancer002ProjectileSecondary 6001611 (side shards): z_1102_s199 takes the shard from its own sprite pixels (shape 235). At
# the first ticks the RED shot's shard is covered by the red muzzle burst (draws of no shard shape): only a sliver of shape 235 is left (98 px vs 650 in
# the blue run). Same hidden-part rule as 466 (enemy only; < 30 % of the predicted body in the label -> add the predicted body).
POST_BODY[6001611] = _eE_hidden_hook(6001611)
