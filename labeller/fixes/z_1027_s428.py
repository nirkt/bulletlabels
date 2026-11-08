# sweep 10-27 fix for skin 428 (Haiku fix, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s428/)
# RopeDude 428 (RopeDudeSpc, Buzz010Projectile, type 6001002): the brawler's bubbles are not the attack (sweep 2026-10-27,
# pairs 8046 = attack 0 deg tick 269, and 8053 = attack 90 deg tick 309: extra_label).
# Cause (measured on the capture W/runs/gc/sw2_428, all ticks with a 6001002 object): each shot starts with a small bubble
# pair beside the brawler's feet (or above its head). The bubbles are drawn with the shot's shared clip sprites, so for the
# new shot object they are its only body: a blob of 193-202 px (full resolution), outlined on its own. The shot's real
# splash body is one piece of 1072 px or more in every frame (at tick 275 the splash of the previous shot is 2668 px and
# the new shot's bubble blob 193 px, at tick 315 the blob is the only label until the splash appears at tick 317).
# Rule: a body part of this type smaller than BUBBLE_FIX_MIN px (8-connected, full resolution) is dropped. The bubbles
# are the only parts that small, so the splash bodies are unchanged; a small part touching a larger part stays
# (it is one part). Only this type: the super (another type) and every other brawler are untouched.
# Other skins: type 6001002 is the Buzz010 shot's own type (sc/effects_brawler_buzz.sc), shared by every skin that fires it.
# Only this capture (skin 428) was checked; the rule is keyed to the type, so any other skin of this type gets the same
# change (its bubbles are the same effect). No other type is changed.
BUBBLE_FIX_TYPE = 6001002
BUBBLE_FIX_MIN = 400      # px (full resolution): the splash bodies are >= 1072 px, the bubble blobs 193-202 px


def _fix_6001002_bubbles(body, ctx):
    from scipy.ndimage import label as _cc
    if not body.any(): return body
    L, n = _cc(body, structure=np.ones((3, 3), bool))
    if not n: return body
    sz = np.bincount(L.ravel()); small = sz < BUBBLE_FIX_MIN; small[0] = False
    return body & ~small[L]


POST_BODY[BUBBLE_FIX_TYPE] = _fix_6001002_bubbles
