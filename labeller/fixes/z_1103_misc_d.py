# AI-review batch 2026-11-01 fix (Haiku fix, reviewed by the coordinator, installed 11-01 LOCAL ONLY; source out/sweep2/haikutasks/siteai/misc_d/fix.py)
# misc_d fix (AI-reviewer findings 2026-11-01, skins 549, 560, 587, 604, 617). exec'd in idlabels' namespace after all
# installed fixes (IDLABELS_OVERRIDE). Keyed to explicit type ids only.
import numpy as np
from scipy.ndimage import label as _cc, binary_fill_holes as _fill

# --- 560 PowerLevelerGenie (Surge006Projectile, 6000802): owner rule "Surge = the whole cyan blob incl. touching splash,
# detached dots out". The cyan cloud lobes behind / above the gold ring touch the body but were not labelled (only the
# gold ring and the front blob). Same mechanism as Surge's default attack types: the trail pixels touching the body
# (connected, opaque) join the label; detached dots stay out.
SURGE_GENIE = {6000802: dict(file='sc/effects_brawler.sc',
                             clips=('surge_006_atk1_projectile', 'surge_006_atk1_projectile_red', 'surge_006_atk1_trail_smoke'),
                             gas=(63,))}   # shape 63 = the cyan splash puffs of the trail clip surge_006_atk1_trail_smoke


def _surge_genie_splash(body, ctx):
    """cyan splash puffs (shape 63 of the trail clip) that touch the body, opaque, inside its trail region join the label"""
    if not body.any(): return body
    from scipy.ndimage import label as cc
    p = SURGE_GENIE[ctx['t']]
    SP = _a4_shape_pixels(ctx['R'], p['file'], p['clips'])
    reg = _dilate(ctx['pred_any'], TRAIL_NEAR)   # the trail reaches past the NEAR region of a projectile (as trail-body types)
    gas = np.isin(SP, [s + 1 for s in p['gas']]) & (ctx['cv'] >= CUT) & reg
    L, n = cc(gas | body); keep = np.unique(L[body]); keep = keep[keep > 0]
    return body | (np.isin(L, keep) & gas)


POST_BODY[6000802] = _surge_genie_splash

# --- 587 DoorManStarrForce (DoorMan003Projectile, 6001327): the attack is thin lines (the fan of chevrons at 269, the
# claw at 309): owner rule "thin lines = ONE convex hull over the whole attack".
ONE_LABEL[6001327] = 'hull'

# --- 617 GeishaSteamPunk (Geisha002UltiProjectile, 6001632): the super is a thin green wedge with pink / yellow edges;
# the label was two small pieces with a gap over the wedge. The wedge (trail pixels touching the body) joins the label and
# the whole super is one hull (owner rule: thin lines / slashes = one convex hull).
TRAIL_ATTACH.add(6001632)
ONE_LABEL[6001632] = 'hull'

# --- 604 ClusterBombDudeRare (ClusterBombProjectile, 6000521): a 3D grey bomb disc seen from above; its label (the
# prediction, the drawn draw) took in the brawler's orange skirt and red body under it at 307 (AI flag 'too big').
# The bomb is a plain disc: its diameter is the label's width at the top, its centre is one radius below the top
# (the top of the bomb is not touched by the skirt); the label keeps only the disc (the skirt below it is cut).
def _cluster_bomb_disc(body, ctx):
    if not body.any(): return body
    ys, xs = np.nonzero(body)
    r = (xs.max() - xs.min() + 1) / 2.0
    cx = (xs.max() + xs.min()) / 2.0
    cy = ys.min() + r - 0.5
    Y, X = np.mgrid[0:body.shape[0], 0:body.shape[1]]
    return body & ((X - cx) ** 2 + (Y - cy) ** 2 <= (r + 0.5) ** 2)


POST_BODY[6000521] = _cluster_bomb_disc
