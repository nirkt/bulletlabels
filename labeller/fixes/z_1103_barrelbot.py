# AI-review batch 2026-11-01 fix (Haiku fix, reviewed by the coordinator, installed 11-01 LOCAL ONLY; source out/sweep2/haikutasks/siteai/barrelbot/fix.py)
# AI-review fix, 2026-11-01, task barrelbot (skins 493 BarrelBotLaser, 520 BarrelBotCrash, 527 BarrelBotOmegabox, 600 BarrelBotGreekGod).
# Reviewer findings: 493 attack 0 t265/t271 (holes / ragged), attack 90 t311 (holes / ragged), attack 90 t313 (should get a hull);
# 520 attack 0 t269 (split into several small outlines, should get a hull); 527 attack 0 t267 (ragged, torn, interior lines);
# 600 attack 0 t271 (stack of puffs as many small outlines, should get a hull), attack 0 t275 (holes / ragged).
#
# What they are: each skin's Darryl projectile type (493: 6000430 Darryl004, 520: 6000431 Darryl005, 527: 6000432 Darryl006,
# 600: 6000435 Darryl009; each is a BarrelBotProjectile, one per skin) is fired as a volley of five projectiles (one tick, same
# spawn frame, five different targets: a fan). Each projectile's visible body is several bright cyan orbs / fins / puffs, so
# its body is splintered into 3-14 pieces and the five lanes interleave (the labels had holes and interior lines where the
# lanes meet). The owner's rule for a splintered or multi-piece attack of one object: ONE convex hull per object (per shot);
# separate projectiles keep separate outlines. So each projectile gets one hull of its own body pixels.
#
# Fix (per type, the four types above only): (1) the body is cleaned first: the muzzle / hit flash is removed, i.e. a large
# patch of pale white / pale yellow pixels (R >= 130, G >= 170, B >= 150, patch >= FLASH_MIN px: skin 493 tick 305 white star;
# skin 527 ticks 275-307 pale star whose pixels carry the clip's code), and the two pixels around it are not body (never
# labelled); (2) ONE_LABEL 'hull' over what is left (convex hull of the projectile's own body only, never the glow or trail).
# Checked with the labels of every attack tick (263-317) of the four captures: no projectile loses its label.
# Installed fixes for these types: none (z_1027_s529 keys 6000433 only, Darryl007 of skin 529). Other skins that might use
# these four types were not checked (no capture here).

import numpy as np
from scipy.ndimage import binary_dilation as _darryl_dil

FLASH_MIN = 150     # px: a pale patch this large is the muzzle / hit flash (the orb highlights are under 20 px)

DARRYL_HULL = {6000430,   # Darryl004Projectile (skin 493, BarrelBotLaser)
               6000431,   # Darryl005Projectile (skin 520, BarrelBotCrash)
               6000432,   # Darryl006Projectile (skin 527, BarrelBotOmegabox)
               6000435}   # Darryl009Projectile (skin 600, BarrelBotGreekGod)


def _darryl_flash(body, ctx):
    from scipy.ndimage import label as _cc
    img = ctx['img']
    if img is None or not body.any(): return body
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    pale = (r >= 130) & (g >= 170) & (b >= 150)          # pale white / pale yellow (the flash); cyan and green body is not
    L, n = _cc(pale)
    if n:
        sz = np.bincount(L.ravel())
        big = np.isin(L, 1 + np.nonzero(sz[1:] >= FLASH_MIN)[0]) & pale   # flash = one large pale patch; small highlights are body
        if big.any(): body = body & ~_darryl_dil(big, iterations=2)       # the flash's edge pixels are not the projectile
    return body


for _t in DARRYL_HULL:
    POST_BODY[_t] = _darryl_flash
    ONE_LABEL[_t] = 'hull'
