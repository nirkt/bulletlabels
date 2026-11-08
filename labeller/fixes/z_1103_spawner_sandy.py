# AI-review batch 2026-11-01 fix (Haiku fix, reviewed by the coordinator, installed 11-01 LOCAL ONLY; source out/sweep2/haikutasks/siteai/spawner_sandy/fix.py)
# Haiku fix, task spawner_sandy (AI-reviewer findings 2026-11-01): skins 291 SpawnerDudeDracula, 369 SpawnerDudeMake,
# 439 SpawnerDudeGym (brawler SpawnerDude) and 296 SandstormKoya, 359 SandstormMake, 414 SandstormGreekGod (brawler Sandstorm).
# Exec'd in idlabels' namespace after the installed fixes (IDLABELS_OVERRIDE). Keyed to explicit type ids only.
#
# (2) 414 attack 0 t265-t271 and attack 90 t305 (Sandy007 6000678; reviewer: "should get a hull" / "holes / ragged"): the
#     cloud's label is ragged: notches and bites between its puffs, interior loops, and a thin spur (t305 top left). The
#     owner's Sandy rule is one solid zone over the cloud, filled, not a hull (the cloud outline is the wanted shape). Fix: the
#     cloud's label is closed (disc radius S_CLOSE: bays and bites narrower than 2 * S_CLOSE filled), holes filled, and opened
#     (disc radius S_OPEN: spurs narrower than about 2 * S_OPEN + 1 removed), limited to the predicted region. Puffs stay.
#
# (3) 369 attack 90 t315 (SpawnerDude005 6000707; reviewer: "too small / cut off" -- the horizontal arms of the white star
#     stay outside): at t315 the projectile is a translucent four-point star with a bright core and the white crystal at its
#     base (zoomed raw, diag/zoom369_t315_raw.png). Its arms are its own draw (the star's atlas texture, the same texture as
#     the trail dots and flame tongues), but lie beyond the predicted region, so only the lower spike is labelled. Fix: the
#     pixels of the body's own atlas texture (2560x4064, the star's and its trail's texture) that are at least half opaque,
#     lie within S_STAR_NEAR px of the predicted label and connect to the labelled body, join the label. Trail dots and streaks
#     are separate parts (kept out by z_1027_s369 and by the connection test); the flame tongues already touch the body.
#
# (4) 439 super t345 (SpawnerDude006 Ulti 6000712; reviewer: "holes / ragged" -- a thin zigzag line on the dark barbell stem):
#     the super is hidden under the cyan blast at t345; the label is only a thin zigzag strip and a blob. Rule: thin / splintered
#     parts of one object = one convex hull (ONE label). Fix: a body whose area is under S_THIN of its convex hull is replaced by
#     that hull. Frames with a solid barbell (t347 to t353) are not thin and stay as they are.
#
# (5) 359 super t349 (Sandy005 Ulti 6000675; reviewer: "too big" -- takes in the chain loop next to the ball): the super is the
#     dark ball with a chain of links trailing from it. The chain is not the projectile (the owner's Spike Mecha decision: ball
#     only, no chain). Fix: the body is opened (disc radius S_BALL_OPEN: the links are thinner than the ball), the largest part
#     is the ball, and the ball's own edge (up to 2 px) is restored from the body. Chain links stay out.
#
# Other approaches considered: (1) a size threshold on the largest part (the leaves at 296 t271 are one clump of 660 px, so it
# fails on 359 t271's 2020 px split cloud); the body/prediction ratio is the game's own data and separates both skins.
# (2) a convex hull of the cloud: rejected, the owner wants the cloud shape. (3) a hull of the star from the lower spike alone:
# the arms are not in the body, so the hull would miss them; the texture-connected pixels are taken instead. (5) cutting the chain
# by colour: the links and the ball are the same blue-grey; opening by thickness uses the shape.
#
# NOT changed here (needs_user): 296 attack 0 t271/t273 and 359 attack 0 t273 are Sandy end debris (leaves, flakes); a burst rule on the
# body/predicted-body ratio also dropped accepted frames on the attack-90 rows (296 t313, 359 t313/t315), so it is not used.
#
# Types touched: 6000678 (414), 6000698 (291 coffin, both rows), 6000707 (369), 6000712 (439), 6000675 (359).

import numpy as _spn_np  # noqa: F401  (idlabels' np is used below)
from scipy.ndimage import (binary_closing as _spn_close, binary_fill_holes as _spn_fill, binary_opening as _spn_open,
                           binary_dilation as _spn_dil, label as _spn_cc)

S_CLOSE = 5            # half-res px: bays and bites narrower than 2 * S_CLOSE filled
S_OPEN = 3             # half-res px: spurs narrower than 2 * S_OPEN + 1 removed
S_STAR_NEAR = 80       # half-res px around the predicted label in which the star's own pixels are searched
S_STAR_TEX = (2560, 4064)   # the atlas texture of the star / trail (tex_size of the draw)
S_THIN = 0.65          # body area / convex hull area under which the parts count as thin (one hull): 439 t345 is 0.54, its solid barbell ticks 0.75-0.80
S_BALL_OPEN = 3        # half-res px: the chain links (thinner than this) are opened away from the ball
S_BALL_EDGE = 2        # half-res px: the ball's edge restored from the body after the opening


def _spn_disc(r):
    yy, xx = _spn_np.mgrid[-r:r + 1, -r:r + 1]
    return (xx * xx + yy * yy) <= r * r


def _spn_chain(t, fn):
    """runs fn after the hook that was installed before for type t (kept, so earlier fixes stay)"""
    prev = POST_BODY.get(t)

    def hook(body, ctx):
        if prev is not None: body = prev(body, ctx)
        return fn(body, ctx)
    POST_BODY[t] = hook


def _spn_smooth(body, ctx):
    if not body.any(): return body
    P = S_CLOSE + 1
    cl = _spn_close(_spn_np.pad(body, P), structure=_spn_disc(S_CLOSE))[P:-P, P:-P] | body
    f = _spn_fill(cl) & ctx['region']
    Q = S_OPEN + 1
    op = _spn_open(_spn_np.pad(f, Q), structure=_spn_disc(S_OPEN))[Q:-Q, Q:-Q]
    return op if op.any() else body


def _spn_star(body, ctx):
    if not body.any(): return body
    R = ctx['R']; ids_h = ctx['ids_h']; cv = ctx['cv']
    tex = set()
    for d in _spn_np.unique(ids_h[body]):
        dr = R['draws'].get(int(d)) if int(d) > 0 else None
        if dr is not None and (dr.get('tw'), dr.get('tht')) == S_STAR_TEX: tex.add(int(d))
    if not tex: return body
    near = _spn_dil(ctx['pred_any'], iterations=S_STAR_NEAR)
    cand = _spn_np.isin(ids_h, sorted(tex)) & (cv >= CUT) & near
    L, n = _spn_cc(cand | body)
    keep = _spn_np.unique(L[body]); keep = keep[keep > 0]
    if not keep.size: return body
    return body | (_spn_np.isin(L, keep) & cand)


def _spn_thin(body, ctx):
    n = int(body.sum())
    if n < 40: return body
    h = hull_mask(body)
    return h if n < S_THIN * int(h.sum()) else body


def _spn_ball(body, ctx):
    if not body.any(): return body
    P = S_BALL_OPEN + 1
    op = _spn_open(_spn_np.pad(body, P), structure=_spn_disc(S_BALL_OPEN))[P:-P, P:-P]
    L, n = _spn_cc(op)
    if n == 0: return body
    sz = _spn_np.bincount(L.ravel())[1:]
    ball = L == (int(_spn_np.argmax(sz)) + 1)
    out = body & _spn_dil(ball, iterations=S_BALL_EDGE)
    return out if out.any() else body


# (2) 414 Sandy007 attack (both rows): one solid cloud outline, bites and spurs removed
_spn_chain(6000678, _spn_smooth)

# (3) 369 SpawnerDude005 attack 90 t315: the white star's own arms (its atlas texture, connected to the body)
_spn_chain(6000707, _spn_star)

# (4) 439 SpawnerDude006 super t345: thin zigzag of the hidden barbell stem -> one hull
_spn_chain(6000712, _spn_thin)

# (5) 359 Sandy005 super: the ball, not the chain
_spn_chain(6000675, _spn_ball)

# (6) 291 SpawnerDude002 attack 0 (coffin with its additive arcs): one hull per shot (multi-piece attack of one object)
ONE_LABEL[6000698] = 'hull'
