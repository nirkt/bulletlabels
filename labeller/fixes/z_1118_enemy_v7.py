# enemy-label fix, round v2, group v7 (2026-11-08): skins 083 / 548 (Cooker brawler: Pearl balls / sauce), 379 / 407 (Pam shots of MinigunDude).
# Installed as py/fixes/z_1118_enemy_v7.py. Global prefix _f7_. Every rule is enemy-only (ctx['own'] False): own (blue) labels are never touched
# (the hooks return the body unchanged for own objects). All four types have the same cause: the RED shot has a muzzle burst at the shooter (the
# blue shot has none) whose sprites share atlas shapes with the shot's own clip (the ID pass cannot tell them apart by shape alone), so the body
# rules (sprite code 1 near the prediction) took the burst with the shot; where the burst covers a young shot the shot itself is hidden.
import numpy as _f7_np
from scipy.ndimage import binary_dilation as _f7_dil, label as _f7_cc

_f7_8 = _f7_np.ones((3, 3), bool)


def _f7_chain(t, make):
    """POST_BODY[t] = enemy-only rule `make` (body, ctx) -> body, run after the installed chain; own objects pass unchanged"""
    prev = POST_BODY.get(t)
    if getattr(prev, '_f7', False): return
    def post(body, ctx):
        if prev is not None: body = prev(body, ctx)
        if ctx.get('own', True) or not body.any(): return body
        return make(body, ctx)
    post._f7 = True; POST_BODY[t] = post


# ---------------------------------------------------------------------------------------------------------------------------------
# 083 CookerDefault, CookerProjectile 6001365 (the fire ball). The red shot's muzzle burst (an orange fire blob behind the first balls of every shot)
# shows atlas shapes of the ball's clip: the enemy label was the ball plus the blob (23 % extra area). FIX: the label is cut to the ball, the 1 px
# dilated convex hull of the predicted ball (ctx['pred_body'] lines up with the drawn ball within 1 px in both runs).
def _f7_ball(body, ctx):
    pb = ctx['pred_body']
    if not pb.any(): return body
    return body & _f7_dil(hull_mask(pb), iterations=1)


_f7_chain(6001365, _f7_ball)

# ---------------------------------------------------------------------------------------------------------------------------------
# 548 CookerDeepsea, Pearl002Projectile 6001366 (the sauce streaks). The red burst is a star of spikes at the cooker, drawn with clip shape 61, which
# on the shot itself is only the small white glint at its head (blue run: 6 % of the body; red run: 30 %, the star). FIX: shape-61 pixels (and
# no-shape sprite-edge pixels) further than 2 px from the shot's other shapes are cut, parts without such a shape dropped; the streak the burst hides
# (the predicted body under the dropped star pixels) is put back (hidden-part rule); when no other shape is left the label is the predicted body's hull.
_f7_P_FILE = 'sc/effects_brawler.sc'
_f7_P_NAMES = ('pearl_002_atk_projectile', 'pearl_002_atk_projectile_red')
_f7_P_BURST = 61


def _f7_pearl(body, ctx):
    SP = _a4_shape_pixels(ctx['R'], _f7_P_FILE, _f7_P_NAMES); pb = ctx['pred_body']
    bu = SP == _f7_P_BURST + 1
    if not (body & bu).any(): return body
    core = body & (SP > 0) & ~bu
    if not core.any():
        return body & _f7_dil(hull_mask(pb), iterations=1) if pb.any() else body
    near = _f7_dil(core, iterations=2)
    m = body & (~bu & ((SP > 0) | near) | near)
    L, n = _f7_cc(m, structure=_f7_8); keep = _f7_np.unique(L[core]); keep = keep[keep > 0]
    out = _f7_np.isin(L, keep) & m
    return out | (pb & body & ~out & bu)


_f7_chain(6001366, _f7_pearl)

# ---------------------------------------------------------------------------------------------------------------------------------
# 379 MinigunDudeZombie, Pam007Projectile 6000394 (bone / candy / eyeball shots). The red burst is a flame cloud at the gun (clip shape 7022, which the
# blue shot never shows: 11 px in the whole blue capture, 12000 in the red one) with orange petals drawn with the shot's own shapes (63 / 1 ...) beside it.
# FIX: shape-7022 pixels cut; of what is left, parts that lie mostly (>= 70 %) outside the 3 px dilated predicted body (the petals, detached from the
# shot) are dropped, and in the shots the cloud touches what lies beyond 5 px of the predicted body (puffs drawn with the shot's own shapes) is cut;
# the shot hidden under the cloud (predicted body under shape-7022 pixels of the body) is put back.
_f7_K_FILE = 'sc/effects_brawler.sc'
_f7_K_NAMES = ('pam_007_atk_projectile', 'pam_007_atk_projectile_red')
_f7_K_BURST = 7022


def _f7_pam7(body, ctx):
    SP = _a4_shape_pixels(ctx['R'], _f7_K_FILE, _f7_K_NAMES); pb = ctx['pred_body']
    bu = SP == _f7_K_BURST + 1
    if not (body & bu).any(): return body
    m = body & ~bu
    out = _f7_np.zeros_like(body)
    if m.any() and pb.any():
        near = _f7_dil(pb, iterations=3)
        L, n = _f7_cc(m, structure=_f7_8)
        for j in range(1, n + 1):
            c = L == j
            if (c & near).sum() >= 0.3 * c.sum(): out |= c
    elif m.any(): out = m
    if pb.any(): out = out & _f7_dil(pb, iterations=5)      # the puffs the burst leaves next to the shot (same shapes as the shot's): beyond 5 px of it
    return out | (pb & body & bu)


_f7_chain(6000394, _f7_pam7)

# ---------------------------------------------------------------------------------------------------------------------------------
# 407 MinigunDudeAprilFools, Pam008Projectile 6000396 (egg / toast / bacon). The red muzzle smoke (pink puffs at the gun and the crescent petals beside
# it) is drawn with clip shape 5 / glow shape 127 / sprite edges only, and the label took it (and, through the smoke drawn over it, part of the fist
# area). The own run never has a part made (>= 60 %) of those shapes alone. FIX: such parts are dropped; the young shot hidden under the smoke
# (predicted body next to a dropped part, when less than a quarter of it is left) is put back, limited to the smoke it is hidden in.
_f7_E_FILE = 'sc/effects_brawler2.sc'
_f7_E_NAMES = ('pam_008_atk_projectile', 'pam_008_atk_projectile_red')
_f7_E_SMOKE = (5, 127)


def _f7_pam8(body, ctx):
    SP = _a4_shape_pixels(ctx['R'], _f7_E_FILE, _f7_E_NAMES); pb = ctx['pred_body']
    sm = _f7_np.isin(SP, [s + 1 for s in _f7_E_SMOKE]) | (SP == 0)
    L, n = _f7_cc(body, structure=_f7_8)
    near = _f7_dil(pb, iterations=3) if pb.any() else None
    kept = _f7_np.zeros_like(body); dropped = _f7_np.zeros_like(body); touch = False
    for j in range(1, n + 1):
        c = L == j
        if sm[c].mean() >= 0.6:
            dropped |= c
            if near is not None and (c & near).any(): touch = True
        else: kept |= c
    if not dropped.any(): return body
    if touch and pb.any() and kept.sum() < 0.25 * pb.sum(): kept = kept | (pb & _f7_dil(dropped, iterations=2))
    return kept


_f7_chain(6000396, _f7_pam8)
