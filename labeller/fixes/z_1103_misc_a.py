# AI-review batch 2026-11-01 fix (Haiku fix, reviewed by the coordinator, installed 11-01 LOCAL ONLY; source out/sweep2/haikutasks/siteai/misc_a/fix.py)
# misc_a fix (AI-reviewer findings 2026-11-01; skins 210, 215, 238, 565; sheets 208 / 220 / 599 checked, no change).
# Exec'd in idlabels' namespace after the installed fixes (IDLABELS_OVERRIDE): every hook chains the installed one for its type.
from scipy.ndimage import label as _cc_m, binary_fill_holes as _bfh_m, binary_closing as _bc_m, binary_dilation as _bd_m
from scipy.spatial import ConvexHull as _CH_m
from PIL import Image as _Im_m, ImageDraw as _ImD_m
import numpy as _np_m


def _disk_m(r):
    yy, xx = _np_m.mgrid[-r:r + 1, -r:r + 1]
    return (xx * xx + yy * yy) <= r * r + 0.5


# ---- skin 210 CactusStarrGirls, attack 0 (Spike010Projectile, 6000138): the spiked ball's label had notches, an interior
# line cutting into the star and a hole (reviewer: "notched and ragged", ticks 271 and 273). Closing with a disk of radius 5
# px (half resolution) follows the ring and keeps the spike bases; holes filled; one piece. Checked on ticks 265-275: spikes kept.
_prev_210_ball = POST_BODY.get(6000138)


def _fix_6000138_ball(body, ctx):
    if _prev_210_ball is not None: body = _prev_210_ball(body, ctx)
    if not body.any(): return body
    R_ = 5
    p = _np_m.pad(body, R_ + 1)
    m = _bc_m(p, structure=_disk_m(R_))[R_ + 1:-R_ - 1, R_ + 1:-R_ - 1]
    m = _bfh_m(m)
    L, n = _cc_m(m)
    if n > 1:
        sz = _np_m.bincount(L.ravel()); sz[0] = 0; m = L == sz.argmax()
    return m


POST_BODY[6000138] = _fix_6000138_ball


# ---- skin 215 WallyWindstock, super 45 (Sprout007UltiProjectile, 6000765): the spinning fan's magenta label had an interior
# loop between the blades and a ragged edge (reviewer, tick 349). Closing (disk 3 px, half resolution) and hole fill make the
# fan one solid piece; the separate arc on the bowl's blue rim below it is dropped (not part of the fan's blades and hub).
_prev_215_fan = POST_BODY.get(6000765)


def _fix_6000765_fan(body, ctx):
    if _prev_215_fan is not None: body = _prev_215_fan(body, ctx)
    if not body.any(): return body
    R_ = 3
    p = _np_m.pad(body, R_ + 1)
    m = _bc_m(p, structure=_disk_m(R_))[R_ + 1:-R_ - 1, R_ + 1:-R_ - 1]
    m = _bfh_m(m | body)
    L, n = _cc_m(m)
    if n > 1:
        sz = _np_m.bincount(L.ravel()); sz[0] = 0; m = L == sz.argmax()
    return m


POST_BODY[6000765] = _fix_6000765_fan


# ---- skin 238 TntDudeMecha, attack 0 at tick 281 (Mike004Projectile, 6000273): the robot flame breaks up into splinters;
# each object's main piece and its small crescents were separate outlines (reviewer: "one hull or no label"). Splintered =
# one object with several pieces, a main piece of at least 60 px and a small total (< 300 px): ONE convex hull over that
# object's main piece and the pieces within 12 px of it (the other robot's pieces farther off are not taken in).
# Measured on ticks 263-325 of sw2_238: only tick 281 changes (368 px / 387 px, one piece each).
_prev_238_mike = POST_BODY.get(6000273)


def _fix_6000273_splinters(body, ctx):
    if _prev_238_mike is not None: body = _prev_238_mike(body, ctx)
    if not body.any(): return body
    L, n = _cc_m(body)
    if n < 2: return body
    sz = _np_m.bincount(L.ravel())[1:]
    if body.sum() >= 300 or sz.max() >= 260 or sz.max() < 60: return body
    main = L == (sz.argmax() + 1)
    near = body & _bd_m(main, structure=_np_m.ones((3, 3)), iterations=12)
    pts = _np_m.argwhere(near)
    if len(pts) < 3: return body
    try:
        h = _CH_m(pts[:, ::-1].astype(float))
    except Exception:
        return body
    poly = [tuple(map(float, pts[:, ::-1][v])) for v in h.vertices]
    im = _Im_m.new('L', (body.shape[1], body.shape[0]), 0)
    _ImD_m.Draw(im).polygon(poly, fill=1)
    return _np_m.array(im).astype(bool)


POST_BODY[6000273] = _fix_6000273_splinters


# ---- skin 565 CannonGirlDeepsea, attack 0 (Bonnie003Projectile, 6001182): at tick 265 the big green splat behind the bright
# ball is drawn with the trail-over code (sprite code 4) and was not labelled (reviewer: "label covers only the front ball").
# The splat is the attack's body (at tick 267 it is the streak). Trail-over pixels (opaque) that touch the ball's own body,
# within 60 px of it, are added; a separate added piece smaller than 80 px (a loose particle beside the streak at 269) is not.
_prev_565_splat = POST_BODY.get(6001182)


def _fix_6001182_splat(body, ctx):
    if _prev_565_splat is not None: body = _prev_565_splat(body, ctx)
    if not body.any(): return body
    C = ctx['C']; cv = ctx['cv']
    near = _bd_m(body, iterations=60)
    sh = (C == 4) & (cv >= CUT) & near
    if not sh.any(): return body
    L, n = _cc_m(sh | body)
    add = _np_m.zeros_like(body)
    for k in _np_m.unique(L[body]):
        if k == 0: continue
        part = (L == k) & sh
        if part.sum() >= 80: add |= part
    return body | add


POST_BODY[6001182] = _fix_6001182_splat
