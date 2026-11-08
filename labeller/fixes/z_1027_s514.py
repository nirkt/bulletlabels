# sweep 10-27 fix for skin 514 (Haiku fix, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s514/)
# sweep 10-27 fix for skin 514 (EnragerVine, sw2_514; task out/sweep2/haikutasks/sweep1027/s514/, pairs 9346 / 9347 / 9350).
# Enrager004Projectile 6000878 (logic EnragerProjectile, the Enrager's beam attack): attack 0 deg ticks 265-267, attack 90 deg
# ticks 305-307. The beam is drawn as thin strips of its own clip (edgar_004_atk: shape 35925 blue, shape 35926 yellow core).
# Its label was ragged: spikes and notches along both edges (9346, 9347), a jog and bites on the edges (9350). Rules 1 and 6
# (one clean convex hull for a streak).
# Fix: the body is cut to the beam's own clip shapes (the trail puffs and the glow around the strips are out), crumbs far
# from the beam are dropped, then ONE convex hull per beam object (ONE_LABEL 'hull'). The two beam objects of a shot keep
# separate labels (rule 5: one object = one label; the overlap rule of Skater 402 / Emz is not listed for this type).
import numpy as _np
E514_ATK = 6000878
E514_FILE = 'sc/effects_brawler.sc'
E514_CLIP = ('edgar_004_atk', 'edgar_004_atk_red')
E514_SHAPES = (35925, 35926)     # the beam's blue strips and its yellow core (clip shape ids)


def _e514_atk(body, ctx):
    if not body.any(): return body
    from scipy.ndimage import label as cc
    SP = _a4_shape_pixels(ctx['R'], E514_FILE, E514_CLIP)   # shape id + 1 per pixel, 0 = no shape of the beam's clips
    keep = body & _np.isin(SP, [s + 1 for s in E514_SHAPES])
    if not keep.any(): return body
    L, n = cc(_dilate(keep, 2))
    sz = _np.bincount(L[keep], minlength=n + 1)[1:]          # the beam's own pixels per bridged part
    if n > 1:
        drop = 1 + _np.nonzero((sz < 25) & (sz < 0.1 * sz.max()))[0]   # crumbs / specks, never the main part
        if drop.size: keep = keep & ~_np.isin(L, drop)
    return keep


POST_BODY[E514_ATK] = _e514_atk
ONE_LABEL[E514_ATK] = 'hull'
