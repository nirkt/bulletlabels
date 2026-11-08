# sweep 10-27 fix for skin 122 (Haiku fix, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s122/)
# Skin 122 JesterUno (sweep 2026-10-27, task s122): Chester005Projectile (type 6001285), the card with its flame streak.
# Defects (pairs 1926 tick 265, 1927 tick 267; tick 269 the same): the label followed the individual flame strands, a notched
# / sawtooth contour with a detached fragment at the streak's tip. The streak is several thin lines = one attack: ONE
# convex hull of the whole attack (RULES 6; as Belle 382's flame and Yeti's claws). The hull is built from the label's own
# pixels only (the card and its strands); the four-point sparkles and trail-over blobs beside the streak are not labelled
# and stay out (checked on ticks 265-269).
# Scope: the attack aimed at 0 deg (one card, object angle 0). The 90 deg attack is a fan of two cards of this same type
# (angles 85 and 95, one shot): a hull over the fan merges the two cards and fills the gap between them, and a hull per card
# is taken from a body that still holds the neighbour's pixels, so the fan keeps its labels as they were. How the fan should
# be labelled (one hull of the fan, or one per card while they overlap) is a decision for the user, not made here.
# Only this type. The angle test uses the projectile's own aim (object 'angle', degrees), not an object id.
S122_STRAIGHT = 3.0   # degrees from 0 deg (the 85 / 95 fan cards are far outside)


def _s122_hull(body, ctx):
    a = ctx['o'].get('angle')
    if a is None or not body.any(): return body
    if abs((float(a) + 180.0) % 360.0 - 180.0) < S122_STRAIGHT:
        return hull_mask(body)   # ONE convex hull of the card and its streak (the label's own pixels only)
    return body


POST_BODY[6001285] = _s122_hull
