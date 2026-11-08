# installed 11-01 from out/sweep2/haikutasks/siteai2/bombs_misc/fix.py (2nd AI sweep: 289 shard hull per projectile, 490/491 Tick ball round, 238 flame clouds solid; 215 fan part DISABLED pending owner question)
# Haiku fix, AI-review sweep 2026-11-01, task bombs_misc (skins 215 and 289; the other flagged skins are in the report).
# Exec'd in idlabels' namespace after the installed fixes (IDLABELS_OVERRIDE): every hook chains the installed one for its type.
#
# ---- skin 215 WallyWindstock, super 45: Sprout007UltiProjectile (6000765), the spinning propeller.
# Reviewer (2026-11-01): "super45 propeller outlines are jagged with interior lines across the blades (t345-t359)".
# Checked on sw2_215 ticks 345-359: the label had a lobe running down from the blades into the green bowl under the fan
# (glow pixels, sprite code 2, taken in by the aligned prediction) and loops/notches between the blades. The propeller's
# own opaque body is its clip body (code 1). Its outline = the clip-body pixels, closed by 1 px, holes filled, largest piece.
# The installed z_1103_misc_a fix (closing 3 px + hole fill of the whole body) is replaced for this type only.
# Compared at ticks 345-359: blade tips and hub kept, the bowl lobe and loops gone, one piece per frame.
import numpy as _np_fx
from scipy.ndimage import label as _cc_fx, binary_fill_holes as _bfh_fx, binary_closing as _bc_fx

FAN_TYPE = 6000765


def _fix_6000765_clip_fan(body, ctx):
    C = ctx['C']; cv = ctx['cv']
    m = body & (C == 1) & (cv >= CUT)
    if not m.any(): return body & False
    m = _bc_fx(_np_fx.pad(m, 2), structure=_np_fx.ones((3, 3), bool))[2:-2, 2:-2] & (body | m)
    m = _bfh_fx(m)
    L, n = _cc_fx(m)
    if n > 1:
        sz = _np_fx.bincount(L.ravel()); sz[0] = 0; m = L == sz.argmax()
    return m


# NOT INSTALLED (coordinator 11-01): the bowl under the fan may be part of the object (owner question)
# POST_BODY[FAN_TYPE] = _fix_6000765_clip_fan


# ---- skin 289 TntDudeToxic: Mike007Projectile (6000279), the cheese-shard attack (attack 0 and attack 90 rows).
# Reviewer (2026-11-01, tick 273): "cheese shard shot at attack 0 is drawn as three separate outlines; it should be one hull".
# Checked on sw2_289 at ticks 263-283: each shard is its own object (2000006 / 2000007 at t273, anchors on the two shard
# centres), and each shard is a convex wedge (blue top + green stripe). The labels were ragged: the upper shard's label held
# only its blue top, its green stripe went to the lower shard's label, which then zig-zagged across the gap.
# Ruling used: separate projectiles keep separate outlines (owner rule), so each shard keeps its own label. Its body is the
# convex hull of the clip-body pixels (code 1: blue top and green stripe) that lie nearest to its own predicted anchor among the
# drawn shards of this type (within 40 px), so the two wedges no longer swap pixels. Trail-over pixels (code 4: glow) are not used.
_MIKE_CHEESE = 6000279
_MIKE_ANCH = {}   # object id -> (x, y) on screen (1200 x 540 half-res frame) of each drawn Mike007 object of the frame
_prev_labels_289 = labels


def labels(objs, cam, R, state=None, ended=(), img=None):
    _MIKE_ANCH.clear()
    for o in objs:
        if o['t'] == _MIKE_CHEESE and o.get('drawn') and o.get('anchor') is not None:
            q = cam @ _np_fx.array([float(o['anchor'][0]), float(o['anchor'][1]), float(o['anchor'][2]), 1.0])
            _MIKE_ANCH[o['id']] = ((q[0] / q[3] * 0.5 + 0.5) * 1200, (1 - (q[1] / q[3] * 0.5 + 0.5)) * 540)
    return _prev_labels_289(objs, cam, R, state=state, ended=ended, img=img)


_prev_289_post = POST_BODY.get(_MIKE_CHEESE)


def _fix_6000279_shard(body, ctx):
    if _prev_289_post is not None: body = _prev_289_post(body, ctx)
    o = ctx['o']
    if o['id'] not in _MIKE_ANCH or len(_MIKE_ANCH) < 1: return body
    C = ctx['C']; cv = ctx['cv']
    ys, xs = _np_fx.nonzero((C == 1) & (cv >= CUT))
    if xs.size == 0: return body & False
    keys = list(_MIKE_ANCH.keys()); P = _np_fx.array([_MIKE_ANCH[k] for k in keys], float)
    d = _np_fx.hypot(xs[:, None] - P[None, :, 0], ys[:, None] - P[None, :, 1])
    mine = (d.argmin(1) == keys.index(o['id'])) & (d.min(1) <= 40)
    m = _np_fx.zeros(body.shape, bool)
    m[ys[mine], xs[mine]] = True
    if m.sum() < 3: return body & False
    return hull_mask(m)


POST_BODY[_MIKE_CHEESE] = _fix_6000279_shard


# ---- skins 490 ClusterBombDudeGold and 491 ClusterBombDudeSilver, super 45: the Tick ball, TickGoldUltiProjectile (6000550) and
# TickSilverUltiProjectile (6000549). Reviewer (2026-11-01, super t345): "the magenta outline on the ball has a line running through
# its inside". Checked on sw2_490 ticks 345-353 (and sw2_491 at 345): at t345 the label took in the blue burst cloud drawn over the
# ball's upper-left, a blue lobe joined to the ball by a neck (the line). The burst is an effect: it is not labelled (blue-dominant
# pixels are removed). The ball is round, so its label is the convex hull of its remaining pixels: the part of the ball under the
# burst is hidden and is filled in (hidden-parts rule). From t349 on the ball's label is already a clean circle; the hull keeps it.
TICK_BALL_TYPES = (6000549, 6000550)
_prev_tick_post = {_t: POST_BODY.get(_t) for _t in TICK_BALL_TYPES}


def _fix_tick_ball(body, ctx):
    prev = _prev_tick_post[ctx['t']]
    if prev is not None: body = prev(body, ctx)
    if not body.any() or ctx.get('img') is None: return body
    im = ctx['img'].astype(_np_fx.int16)
    blue = (im[..., 2] - im[..., 0] > 40) & (im[..., 2] > 120)   # the burst cloud: blue-dominant pixels
    m = body & ~blue
    if not m.any(): return body & False
    return hull_mask(m)


for _t in TICK_BALL_TYPES: POST_BODY[_t] = _fix_tick_ball


# ---- skin 238 TntDudeMecha, attack 90 (Mike004Projectile, 6000273): the two flame clouds at ticks 305-309. Reviewer (2026-11-01,
# t307): "ragged outline with an inward notch at the top". Checked on sw2_238 ticks 305-309: the label of the two clouds had an
# inner loop across their top (the bright flame tips drawn over the cloud) and a notch between the tips. A closing of 3 px and a
# hole fill make one solid outline per cloud; the outer shape is kept (closing only fills gaps narrower than 6 px). The splinter
# hull of z_1103_misc_a (tick 281) runs first and is not touched by this step.
_fx238_disk = lambda r: (lambda yy, xx: (xx * xx + yy * yy) <= r * r + 0.5)(*_np_fx.mgrid[-r:r + 1, -r:r + 1])
_prev_238_mike4 = POST_BODY.get(6000273)


def _fix_6000273_close(body, ctx):
    if _prev_238_mike4 is not None: body = _prev_238_mike4(body, ctx)
    if not body.any(): return body
    R_ = 3
    p = _np_fx.pad(body, R_ + 1)
    m = _bc_fx(p, structure=_fx238_disk(R_))[R_ + 1:-R_ - 1, R_ + 1:-R_ - 1] | body
    return _bfh_fx(m)


POST_BODY[6000273] = _fix_6000273_close
