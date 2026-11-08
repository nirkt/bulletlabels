# sweep 10-27 fix for skin 377 (Haiku fix, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s377/)
# Jessie 008 bubble (MechanicSB, skin 377; type 6000186 Jessie008Projectile1, clip jessie_008_atk_projectile): one solid
# label in the bubble's own outline (sweep 2026-10-27, task s377, pairs 7083-7093, attack 0 deg and attack 90 deg).
# Cause: the bubble is a translucent blue sphere with a bright rim and an orange object inside. The label took only the
# rim (a thin ring) and the orange object; the blue interior (no sprite of the clip: the pipeline's body keeps only drawn
# clip pixels) was not labelled, so the label looked like an outer ring with an inner outline around the orange object.
# Rule (keyed to this type id only; the same clip is used by 6000187 / 6000188, not seen in this capture, left alone):
#  1. close gaps of up to 2 px in the rim (ring drawn in pieces), then fill everything enclosed by it: the blue interior,
#     the orange object and the white lobe, the bubble's own glow reflection inside its outline;
#  2. only inside the projectile's search region; nothing outside the rim is added (the outer glow, the purple shared
#     sprite under the bubble and the crate or turret beside it stay out);
#  3. when the rim is not closed (no enclosed area) the body is kept as it is.
# Approaches considered:
#   1. convex hull (ONE_LABEL 'hull'): rejected, the bubble is round but the hull would take in stray pixels beside it.
#   2. fill holes of the current body: rejected alone, the rim plus object is not closed at every tick (tick 307 and 309
#      the inner outline reaches the rim), so a 2 px closing comes first.
#   3. a radius around the prediction: rejected, it would also take in the glow.
#   4. the rim's colour (bright white-blue) plus fill: rejected, the rim colour changes with the scene (white or blue).
#   5. closing + enclosed fill (chosen): uses only the drawn rim as the outline, no colour threshold.
JESSIE008_BUBBLE = 6000186
JESSIE008_CLOSE = 2     # px: gaps in the rim closed


def _fix_jessie008_bubble(body, ctx):
    from scipy.ndimage import binary_fill_holes
    if not body.any(): return body
    closed = _close(body, JESSIE008_CLOSE)
    filled = binary_fill_holes(closed) & ctx['region']
    if not filled.any(): return body
    return filled | body


POST_BODY[JESSIE008_BUBBLE] = _fix_jessie008_bubble
