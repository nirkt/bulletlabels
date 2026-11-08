# sweep 10-27 fix for skin 125 (Haiku fix, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s125/)
# Task s125 (sweep 2026-10-27), skin 125 KnightNinja = brawler Ash, Ash 002 (Ninja). Attack type 6001040 (Ash002Projectile1,
# trail ash_002_atk_trail, one wave hull). Finding: pair 1999 (row 'sw2_125 attack 90 deg', tick 317, extra_label): the
# white slashes are gone, but the wave's label still sits on the group member 2000003 (logic finished, not drawn, not
# 'ended' in the object list, so the group filter in idlabels.labels keeps it). Its body is then only a few leftover pixels
# of the slash sprite drawn brown (the ground marks: shape ash_002_ground_slash, median brightness 100-150, above DARK_V,
# so DARK_DROP keeps them), and ONE_LABEL 'hull' drew a ~435 px polygon over the leaves and pink dust.
# The user's rules: the attack is its white slash; leftovers of an ended attack (ground marks, dust, debris) are never
# labelled (rules 2, 3, 4). Rule here: the body of 6001040 is kept only while it holds bright pixels of its own
# (max(R, G, B) >= S125_BRIGHT, at least S125_MIN of them); otherwise the attack is gone and nothing is labelled.
# Measured on sw2_125 at every frame with a 6001040 body (bright pixels of the body): ticks 269 (83), 271 (728), 273 (682),
# 275 (191), 309 (95), 311 (683), 313 (864), 315 (183) keep their label unchanged; ticks 277 (0 of 2 px) and 317 (0 of 5 px),
# both with no visible slash, lose the leftover label. Only type 6001040 is changed: 6001041 / 6001042 (Ash 002 lv2 / lv3)
# share the same clip layout and were not captured here.
S125_BRIGHT = 200   # max(R, G, B) from which a body pixel is the bright slash
S125_MIN = 12       # bright body pixels that show the attack is still there

_s125_prev = POST_BODY.get(6001040)


def _s125_slash(body, ctx):
    if _s125_prev is not None: body = _s125_prev(body, ctx)
    if not body.any(): return body
    bright = body & (ctx['img'].max(-1) >= S125_BRIGHT)
    if int(bright.sum()) >= S125_MIN: return body
    return np.zeros_like(body)   # no white slash left: the body is the leftovers of an ended attack


POST_BODY[6001040] = _s125_slash
