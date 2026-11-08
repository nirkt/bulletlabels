# AI-review batch 2026-11-01 fix (Haiku fix, reviewed by the coordinator, installed 11-01 LOCAL ONLY; source out/sweep2/haikutasks/siteai/misc_b/fix.py)
# misc_b fix (AI-reviewer findings 2026-11-01, sweep2 skins 306, 307, 309, 320, 326; task out/sweep2/haikutasks/siteai/misc_b/).
# Exec'd in idlabels' namespace after all installed fixes (IDLABELS_OVERRIDE). Each change is keyed to explicit type ids.
#
# 306 GunslingerPaintball, attack 90 deg, tick 323 (reviewer: 'holes / ragged'). Type 6000028 Colt005Projectile (used by
#   skin 306 only). Cause: the bullet's head has a translucent inner layer (shape 143 of the blue clip, coverage about 115),
#   which was below CUT 128, so the head's middle was not body; the impact splash (shape 7441, removed by z_1027_s306) lies on
#   the head's bottom and left a notch. Fix: the inner layer's pixels (shape 143, coverage >= 32, inside the region) are body
#   (not the outer halo: shape 63 stays out); the notch the splash cut leaves is closed (radius 6, next to the splash only); holes
#   filled. Effect: the label covers the bullet's whole blob, also on attack 0 ticks (265-277), where labels grow (see report).
#
# 307 SilencerPharaoh, attack 90 deg, ticks 315 and 319 (reviewer: 'tiny outline specks above the flames'). Type 6001199
#   Otto002Projectile. Cause: the flame's dark tongue tips sit a few px apart from the flame body; each is its own piece of
#   26 to 41 px. Fix: a body's pieces that are not connected to its largest piece and have fewer than 60 px are dropped
#   (specks). Connected parts and bigger detached parts are unchanged.
#
# 309 MinigunDudeBroker, attack 90 deg, tick 317 (reviewer: 'green arrow stream outlined as several separate pieces instead of
#   one hull per shot'). Type 6000392 Pam006Projectile: each arrow is its own projectile (its own start tick and label), and each
#   arrow's label breaks into 2 or 3 pieces (tip bits 40 to 50 px apart from the chevron). Rule: thin / splintered /
#   multi-piece attacks of ONE object = one convex hull per object (ONE_LABEL 'hull'). Separate arrows keep separate outlines.
#   Other skins that use type 6000392 get the same hull per object (type-level rule; see report).
#
# 320 SoulCollectorworldfinals, super 45 deg, tick 353 (reviewer: 'large triangular hole inside the crown label'). Type 6001243
#   Gus002UltiProjectile (the yellow crown). The crown's middle triangle is open in the drawing (the gap between its arms); the
#   label had it as a hole. Fix: holes filled (the existing FILL_HOLES mechanism: a solid label, as Bibi's bubble).
#
# 326 ControllerAres, attack 90 deg, tick 305 (reviewer: 'label covers only a lower part of the axe cluster; the yellow axes above
#   it are outside'). Type 6000778 Nani005Projectile: three axe projectiles (one per object); each axe's label is splintered into
#   3 to 4 pieces with specks (e.g. obj 2000012 at tick 305: 14, 59, 1, 31 px). Rule: one convex hull per splintered object
#   (ONE_LABEL 'hull'). PARTLY: the hull removes the splinters and covers the lower cluster (tick 307 covers the three axes), but at
#   tick 305 the cyan star above the cluster and a dark-red piece on the left stay outside (needs_user).
#
# Not changed (see report): 309 / 361 / 353 / other per-card and per-blade findings where the labels already follow the rules.

import numpy as np
import bulletlabels as _bl_misc

# ---- 306: Colt005Projectile (6000028): the head's translucent inner layer (shape 143) is body, notch filled


MB_INNER = 143 + 1           # shape 143 of the bullet's clips (the head's translucent inner layer, coverage about 115)
MB_INNER_CV = 32             # px: its translucent pixels from this coverage are body (the outer halo, shape 63, stays out)
MB_SPLASH_CLIPS = ('colt_005_atk_projectile', 'colt_005_ulti_projectile')
MB_SPLASH_CLIPS_RED = ('colt_005_atk_projectile_red', 'colt_005_ulti_projectile_red')
MB_SPLASH_BLUE = 7441 + 1    # shape 7441 of the blue clips: the impact splash (z_1027_s306)
MB_SPLASH_RED = (7440 + 1, 7444 + 1)
MB_NOTCH = 6                 # px: a notch the splash cut leaves at the head's bottom is closed within this radius


def _mb_disk(r):
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r + r // 2


def _mb_fill(body, ctx):
    from scipy.ndimage import binary_fill_holes, binary_closing, binary_dilation
    if not body.any(): return body
    own = bool(ctx['own'])
    SP = _a4_shape_pixels(ctx['R'], 'sc/effects_brawler.sc', MB_SPLASH_CLIPS if own else MB_SPLASH_CLIPS_RED)
    spl = ctx['region'] & ((SP == MB_SPLASH_BLUE) if own else np.isin(SP, MB_SPLASH_RED))
    inner = ctx['region'] & (SP == MB_INNER) & (ctx['cv'] >= MB_INNER_CV)
    out = body | inner
    if spl.any():
        # the splash cut's notch at the head's bottom: closed, only next to the removed splash
        cl = binary_closing(np.pad(body, MB_NOTCH + 1), structure=_mb_disk(MB_NOTCH))[MB_NOTCH + 1:-MB_NOTCH - 1, MB_NOTCH + 1:-MB_NOTCH - 1]
        out = out | (cl & binary_dilation(spl, iterations=MB_NOTCH) & ctx['region'])
    return binary_fill_holes(out) & (ctx['region'] | body)


_mb_prev_306 = POST_BODY.get(6000028)
POST_BODY[6000028] = (lambda prev: (lambda body, ctx: _mb_fill(prev(body, ctx) if prev else body, ctx)))(_mb_prev_306)


# ---- 307: Otto002Projectile (6001199): detached specks (pieces < 60 px, not connected to the largest piece) dropped
def _mb_specks_307(body, ctx):
    from scipy.ndimage import label as cc
    if not body.any(): return body
    L, n = cc(body, structure=np.ones((3, 3), bool))
    if n <= 1: return body
    sz = np.bincount(L.ravel())
    main = int(np.argmax(sz[1:])) + 1
    drop = [k for k in range(1, n + 1) if k != main and sz[k] < 60]
    if not drop: return body
    return body & ~np.isin(L, drop)


_mb_prev_307 = POST_BODY.get(6001199)
POST_BODY[6001199] = (lambda prev: (lambda body, ctx: _mb_specks_307(prev(body, ctx) if prev else body, ctx)))(_mb_prev_307)


# ---- 309: Pam006Projectile (6000392): one hull per object (its splintered arrow); separate arrows keep separate outlines
ONE_LABEL[6000392] = 'hull'

# ---- 326: Nani005Projectile (6000778): one hull per object (its splintered axe)
ONE_LABEL[6000778] = 'hull'

# ---- 320: Gus002UltiProjectile (6001243): solid label (holes filled)
_bl_misc.FILL_HOLES.add(6001243)
