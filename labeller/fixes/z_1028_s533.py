# sweep 10-28 round 3 fix for skin 533 Luchador Butler (Primo009), type 6000405 Primo009Projectile, attack 0 deg / 90 deg rows.
# The fist is a 3D mesh. Grey smoke puffs (primo_009_atk_smoke_01/02) are drawn in front of it; a4_agent4.py drops the puff pixels
# from the label (smoke is never the attack), which leaves holes (a second closed contour around each puff, ticks 265-275, 305-313).
# Question for the user (open): fill the fist's hidden part under the puff, or leave the hole?
#   S533_MODE = 'fill' (default, the general rule "the object's OWN outline filled solid"): the fist's hidden part is added.
#   S533_MODE = 'hole' : unchanged (a4 rule, hole stays).
# How the 'fill' outline is obtained: NOT by closing / hulling the visible fist (that guesses; the rejected fix2 took smoke pixels by a
# big closing and also pulled smoke from outside the fist), but from the fist's OWN geometry: ctx['pred_body'] is the projectile's
# predicted body from the game data and already lies on the drawn fist (checked on all ticks of both rows: 98-99 % of the visible fist is
# inside it, best integer shift 0..1 px). Under a puff the ID pass shows only the puff (its depth is the puff's own), so the prediction is
# the only record of what is under it. The label becomes: visible fist  U  (smoke pixels INSIDE pred_body)  U  holes filled.
# Smoke outside pred_body (beside the fist) is never taken. A fist object whose prediction does not agree with its visible pixels
# (fraction of the prediction's non-smoke pixels on the drawn body < S533_AGREE) or with almost nothing visible is left as before.
# Keyed to 6000405 only (skin 533 so far; chain: the a4 rule runs first).
import os as _os
S533_MODE = _os.environ.get('S533_MODE', 'fill')
S533_AGREE = 0.75
S533_CLOSE = 3          # px (half resolution): notch / cap-hole closing radius
S533_MINBODY = 120      # px (half resolution) of visible fist needed to trust the prediction

_s533_prev_6000405 = POST_BODY.get(6000405)


def _s533_fix_6000405(body, ctx):
    from scipy.ndimage import binary_fill_holes, label as cc
    if _s533_prev_6000405 is not None: body = _s533_prev_6000405(body, ctx)
    if S533_MODE != 'fill' or body.sum() < S533_MINBODY: return body
    sm = _a4_shape_pixels(ctx['R'], 'sc/effects_brawler.sc', ('primo_009_atk_smoke_01', 'primo_009_atk_smoke_02')) > 0
    pb = ctx['pred_body'] & ctx['region']
    vis = pb & ~sm
    if not vis.any() or (vis & body).sum() < S533_AGREE * vis.sum(): return body
    out = body | (pb & sm)
    out = binary_fill_holes(out) & ctx['region']
    # the cuff's cyan cap (a lamp on the fist's cuff, its own draw, not labelled) leaves a hole / narrow notch: close gaps <= S533_CLOSE px
    # (round element) and fill what is enclosed; only pixels within S533_CLOSE of the label are added, so no new spurs
    yy, xx = np.mgrid[-S533_CLOSE:S533_CLOSE + 1, -S533_CLOSE:S533_CLOSE + 1]; st = (yy * yy + xx * xx) <= S533_CLOSE ** 2
    from scipy.ndimage import binary_closing
    r = S533_CLOSE + 1; cl = binary_closing(np.pad(out, r), structure=st)[r:-r, r:-r]
    out = (binary_fill_holes(cl) | out) & ctx['region']
    L, n = cc(out)                       # keep the fist piece(s): drop specks far smaller than the main piece
    if n > 1:
        sz = np.bincount(L.ravel())[1:]; out = np.isin(L, 1 + np.nonzero(sz >= 0.15 * sz.max())[0]) | (out & body)
    return out


POST_BODY[6000405] = _s533_fix_6000405
