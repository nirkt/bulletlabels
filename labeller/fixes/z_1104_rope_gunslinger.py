# installed 11-01 from out/sweep2/haikutasks/siteai2/rope_gunslinger/fix.py (2nd AI sweep: 306 cloud solid, 328 banana head, 394 ring smooth, 397 tiers in; single-skin types)
# rope_gunslinger fix (AI-review second sweep, 2026-11-01): skins 306, 328, 394, 397 (task folder rope_gunslinger/).
# Exec'd in idlabels' namespace (IDLABELS_OVERRIDE). Keyed to explicit type ids only. Findings and decisions: report.md.
#
# 1. 306 GunslingerPaintball, Colt005Projectile 6000028 (attack 90, tick 323; reviewer: 'interior lines, top of the cloud
#    outside'). The bullet's translucent outer layer (shape 427 of the blue clip) at the top of the cloud is taken in when
#    its coverage is >= 32 (inside the region); the water splash (shape 7441) that the bullet drags into the water is cut out
#    again (z_1027_s306 removes it, the hole filling of z_1103_misc_b let some back in); the notch it leaves is closed next to
#    the splash only; holes filled; pieces under 20 % of the largest dropped. The halo (shape 63) stays out.
#
# 2. 328 GunslingerMonkey, Colt007Projectile 6000032 (attack 0, tick 277; reviewer: 'notch cut into the lower right of the
#    banana'). The banana's bright head layer lies at coverage 32..127 (translucent) and the 50 % cut left it out: FADE_CUT
#    lowers the cut for this type only (the same mechanism as FADE_CUT[6001299]); then a disk closing (radius 3, inside the
#    region) closes the small notch on the concave side and the loops of the translucent layers; holes filled; glow stays out.
#
# 3. 397 GunSlingerWoody, Colt010Projectile 6000038 (attack 90, tick 323; reviewer: 'the outline covers only the lower part
#    of the tall green shot, the upper tiers stay outside'). The shot's clip body (its tiers) reaches above the predicted
#    region (radius 50): clip-body pixels of the object's OWN draws that touch the label, within 12 half-res px of its
#    prediction, are taken in (connected growth only: neighbours' draws, glow and trail stay out).
#
# 4. 394 RopeDudeGoodRandoms, Buzz009Projectile 6001000 (attack 90, tick 317; reviewer: 'the outline is notched with a spur
#    at the lower right, not one smooth shape'). The cylinder's label is made of loops and thin spurs: gaps <= 3 half-res px
#    closed, holes filled, spurs thinner than 2 half-res px opened away, pieces under 20 % of the largest dropped.
#
# Not changed here (see report.md): 317 attack 0 t269 (two projectiles alive: needs_user); 354 attack 90 t317 (an ended
# shard's cluster inside the wave-group label: needs_user).

import numpy as np
from scipy.ndimage import label as _cc_rg, binary_fill_holes as _fh_rg, binary_closing as _bc_rg, binary_opening as _bo_rg, binary_dilation as _bd_rg

RG_COLT005 = (6000028,)          # Colt005Projectile (GunslingerPaintball attack, skin 306)
RG_COLT007 = (6000032,)          # Colt007Projectile (GunslingerMonkey banana, skin 328)
RG_COLT010 = (6000038,)          # Colt010Projectile (GunSlingerWoody green shot, skin 397)
RG_BUZZ009 = (6001000,)          # Buzz009Projectile (RopeDudeGoodRandoms cylinder, skin 394)

RG306_CLIPS = ('colt_005_atk_projectile', 'colt_005_ulti_projectile')
RG306_CLIPS_RED = ('colt_005_atk_projectile_red', 'colt_005_ulti_projectile_red')
RG306_OUTER = 427 + 1            # code of shape 427 (the bullet's translucent outer layer; codes are shape id + 1)
RG306_SPLASH = 7441 + 1          # shape 7441 of the blue clips: the water splash
RG306_SPLASH_RED = (7440 + 1, 7444 + 1)
RG306_CV = 32                    # coverage from which the outer layer is body
RG306_NEAR = 4                   # half-res px: the notch the splash leaves is closed only this close to it
RG_SPECK = 0.2                   # pieces under this share of the largest one are dropped
RG397_NEAR = 12                  # half-res px around the prediction in which the clip body may be grown
RG394_CLOSE = 3
RG394_OPEN = 2

FADE_CUT[6000032] = 32           # banana head layer (translucent, 32..127): body for Colt007 only


def _rg_largest(m, frac):
    """the connected pieces of m that are at least frac of the largest one (8-connected)"""
    L, n = _cc_rg(m, structure=np.ones((3, 3), bool))
    if n <= 1: return m
    sz = np.bincount(L.ravel())[1:]
    return np.isin(L, 1 + np.nonzero(sz >= frac * sz.max())[0]) & m


def _rg_keep_touching(body, cand):
    """body plus the candidate pixels connected to it"""
    if not body.any(): return body
    L, n = _cc_rg(cand | body)
    keep = np.unique(L[body]); keep = keep[keep > 0]
    return body | (np.isin(L, keep) & cand)


def _rg_306(body, ctx):
    if not body.any(): return body
    own = bool(ctx['own'])
    SP = _a4_shape_pixels(ctx['R'], 'sc/effects_brawler.sc', RG306_CLIPS if own else RG306_CLIPS_RED)
    reg = ctx['region']
    spl = (SP == RG306_SPLASH) if own else np.isin(SP, RG306_SPLASH_RED)
    outer = reg & (SP == RG306_OUTER) & (ctx['cv'] >= RG306_CV)
    out = (body | outer) & ~spl
    if not out.any(): return body
    pad = RG306_NEAR + 2
    cl = _bc_rg(np.pad(out, pad), iterations=2)[pad:-pad, pad:-pad]
    out = out | (cl & _bd_rg(spl, iterations=RG306_NEAR) & reg)
    out = _fh_rg(out) & (reg | body)
    return _rg_largest(out, RG_SPECK)


def _rg_397(body, ctx):
    if not body.any(): return body
    C = ctx['C']; cv = ctx['cv']; ids_h = ctx['ids_h']
    own_d = np.unique(ids_h[body & (C == 1)]); own_d = own_d[own_d > 0]
    if not own_d.size: return body
    near = _bd_rg(ctx['pred_any'], iterations=RG397_NEAR)
    cand = (C == 1) & (cv >= 128) & near & np.isin(ids_h, own_d)
    return _rg_keep_touching(body, cand)


def _rg_394(body, ctx):
    if not body.any(): return body
    pad = RG394_CLOSE + 2
    b = _bc_rg(np.pad(body, pad), iterations=RG394_CLOSE)[pad:-pad, pad:-pad] | body
    b = _fh_rg(b)
    p4 = 4
    b = _bo_rg(np.pad(b, p4), iterations=RG394_OPEN)[p4:-p4, p4:-p4]
    b = _fh_rg(b) & (ctx['region'] | body)
    if not b.any(): return body
    return _rg_largest(b, RG_SPECK)


for _t_rg in RG_COLT005:
    _rg_prev_005 = POST_BODY.get(_t_rg)
    POST_BODY[_t_rg] = (lambda prev: (lambda body, ctx: _rg_306(prev(body, ctx) if prev else body, ctx)))(_rg_prev_005)
RG328_R = 3                      # px: closing radius (the banana's concave side and the loops of its translucent layers)


def _rg_disk(r):
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r + r // 2


def _rg_328(body, ctx):
    # banana: one solid outline (the notch on the concave side and the loops of the translucent layers are closed)
    if not body.any(): return body
    pad = RG328_R + 1
    from scipy.ndimage import binary_closing as _bcs
    cl = _bcs(np.pad(body, pad), structure=_rg_disk(RG328_R))[pad:-pad, pad:-pad]
    b = (cl & ctx['region']) | body
    b = _fh_rg(b) & (ctx['region'] | body)
    return _rg_largest(b, RG_SPECK)


for _t_rg in RG_COLT007:
    POST_BODY[_t_rg] = _rg_328
for _t_rg in RG_COLT010:
    POST_BODY[_t_rg] = _rg_397
for _t_rg in RG_BUZZ009:
    POST_BODY[_t_rg] = _rg_394
