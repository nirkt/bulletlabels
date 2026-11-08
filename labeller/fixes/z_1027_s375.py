# sweep 10-27 fix for skin 375 (Haiku fix, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s375/)
# Crow 007 (skin 375 CrowHarvest, sweep 2026-10-27, task s375, pair 7053 at tick 273): the attack 0 deg projectile of Crow
# (Crow007Projectile, type 6000325, clip crow_007_atk_projectile in sc/effects_brawler.sc) hit a crate. Its impact flash (a
# yellow-white starburst with a cyan blob beside it) is drawn with shape 63 of that clip, and the label took it in together with
# the crescent blades. Rule 4: hit / impact flashes are never labelled. The crescent blades stay labelled (the task's verdict).
# Shape 63 is the flash only: it is absent from the blades' body on ticks 265-269 and 275-279 (0-3 px of stray pixels), and is
# 263 px at tick 273 and 488 px at tick 271 (the flash). So the flash shape is dropped from the body; the parts of the body that
# have no blade pixel left (leftovers of the flash's sprite edge) are dropped. Nothing else changes.
CROW_FLASH = {6000325: dict(file='sc/effects_brawler.sc',
                            clips=('crow_006_atk_projectile_red', 'crow_007_atk_projectile', 'crow_007_lobby_fire_01',
                                   'fire_spark_blue_1', 'fire_spark_red_3'),
                            flash=(63,))}


FLASH_MIN = 20   # px: flash-shape pieces smaller than this (1-3 px specks on the blades' edges) are not the flash; they stay


def _crow_flash(body, ctx):
    if not body.any(): return body
    from scipy.ndimage import label as cc
    p = CROW_FLASH[ctx['t']]
    SP = _a4_shape_pixels(ctx['R'], p['file'], p['clips'])   # shape id + 1 per pixel, 0: none of the clips' shapes
    fl = body & np.isin(SP, [s + 1 for s in p['flash']])
    Lf, nf = cc(fl)
    sz = np.bincount(Lf.ravel()) if nf else np.zeros(1, int)
    big = np.nonzero(sz >= FLASH_MIN)[0]; big = big[big > 0]
    flash = fl & np.isin(Lf, big)
    if not flash.any(): return body
    m = body & ~flash
    blade = m & (SP > 0)                                       # the blades' own shapes (any other shape of the clips)
    L, n = cc(m)
    near = _dilate(flash, 3)                                   # leftovers of the flash's sprite edge touch it
    drop = np.zeros(n + 1, bool)
    if n:
        has_blade = np.zeros(n + 1, bool); has_blade[np.unique(L[blade])] = True
        touch = np.zeros(n + 1, bool); touch[np.unique(L[near & m])] = True
        drop = touch & ~has_blade; drop[0] = False
    return m & ~drop[L]


for _t in CROW_FLASH: POST_BODY[_t] = _crow_flash
