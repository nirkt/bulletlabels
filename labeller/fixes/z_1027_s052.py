# sweep 10-27 fix for skin 052 (Haiku fix, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s052/)
# Sweep 2026-10-27, skin 052 PercenterDefault (task out/sweep2/haikutasks/sweep1027/s052/, pair 950, attack 90 deg, tick 305):
# extra_label. The Percenter attack (heart projectile, type 6000815 PercenterProjectile, clip colette_def_atk_projectile)
# has a launch flash at the brawler's feet on its first tick: a flat cyan and yellow oval, about 25 x 14 px, drawn with
# the projectile's own clip sprites. The label took it as a separate part (52 px at x 587-598, y 277-283), 15 px above the
# heart's predicted footprint (the heart is 1042 px at y 299-337). Rule 4 (muzzle / launch flashes are never labelled).
# The flash is on tick 305 only (ticks 307-313 have one part each); no other tick of the 0 deg or 90 deg rows changes.
#
# Approaches considered:
#   1. size threshold on separate parts (drop parts under about 60 px): rejected, a fixed pixel count, and it depends on
#      the projectile's size, which differs between skins and zoom levels.
#   2. shape ids of the clip (the flash shares shapes 63 and 64178 with the heart): rejected, not separable by shape.
#   3. draw ids (the flash is in draws 88 / 89, the heart in 85 / 86): rejected, draw ids are not stable between captures.
#   4. position against the prediction (chosen): a separate part of the label that does not touch the projectile's
#      predicted footprint (plus 3 px) is not the projectile. The rule is keyed to this type id only.
#   5. a time rule (drop the first tick's parts): rejected, the object carries no age, and a real first-tick part is body.
# Rule (keyed to type 6000815 only):
#   a. the largest connected part of the body is always kept (a projectile never loses its label);
#   b. any other part is dropped only if it does not touch the predicted footprint (pred_any, dilated by 3 px) AND it is
#      smaller than half the largest part. Parts touching the prediction, or large parts, stay as they are.
# Other Percenter skins (487, 494, 505, 528, 554, 562, 597, 626) may use this type id: the same rule applies to them,
# not verified (no capture of theirs is kept). Only the 052 capture was checked (all frames: see report.md).

PERCENTER_ATK_TYPE = 6000815
PERCENTER_FLASH_NEAR = 3      # px (full resolution): a part this close to the predicted footprint is the projectile


def _fix_percenter_flash(body, ctx):
    from scipy.ndimage import label as _pf_cc
    if not body.any():
        return body
    L, n = _pf_cc(body, structure=np.ones((3, 3), bool))
    if n < 2:
        return body
    sz = np.bincount(L.ravel())
    main = 1 + int(np.argmax(sz[1:]))
    near = _dilate(ctx['pred_any'], PERCENTER_FLASH_NEAR)
    keep = np.zeros(n + 1, bool)
    keep[main] = True
    for k in range(1, n + 1):
        if k == main:
            continue
        touches = bool((near & (L == k)).any())
        keep[k] = touches or sz[k] >= 0.5 * sz[main]
    return body & keep[L]


POST_BODY[PERCENTER_ATK_TYPE] = _fix_percenter_flash
