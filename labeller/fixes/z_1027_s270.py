# sweep 10-27 fix for skin 270 (Haiku fix, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s270/)
# Task s270 (sweep 2026-10-27, Haiku): skin 270 SandstormLantern, attack 90 deg, Sandy003Projectile (type 6000670, the
# lantern attack; sprites sandy_003_atk_projectile / sandy_003_atk_trail; a ZONE type, SANDY_ATK).
# Defects (pairs 4983 tick 305, 4986 tick 311: spur_or_speck; 4984 tick 307: too_big):
# the zone label takes the sand trail (trail body, code 3) that streams off the lantern cloud: thin strands reach up-left
# toward the separate faded lantern (305), a narrow neck and a lobe of the sparkle haze stand above the left lantern (307),
# a thin streak rises from the top of the left lantern (311). That trail is not the attack body (rule 3: trails, dust,
# sparkle glow are not labelled); the cloud itself (the two lanterns with their glow and the trail hugging them) stays
# one solid zone label, as the user decided for Sandy (ZONE).
# Fix (this type only, wrapped after the existing Sandy end hook _sandy_end): per object body, trail parts thinner than
# S270_OPEN px (half resolution) are cut off by a disk opening, except within S270_NEAR px of the lantern's own clip body
# (red, code 1), which stays exactly as it was; then only parts connected to the lantern body are kept (detached sparkle
# haze out). The lantern's own pixels are never removed.
S270_NEAR = 3     # px (half resolution): the lantern's own clip body and its immediate surroundings are kept as they are
S270_OPEN = 5     # px (half resolution): disk radius of the opening; trail strands and necks thinner than about 2 x this go
_s270_prev = POST_BODY.get(6000670)
def _s270_disk(r):
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    return (xx * xx + yy * yy) <= r * r
def _s270_spur(body, ctx):
    from scipy.ndimage import binary_opening, label as cc
    if _s270_prev is not None: body = _s270_prev(body, ctx)
    if not body.any(): return body
    core = ctx['region'] & (ctx['C'] == 1)
    if not core.any(): return body
    near = _dilate(core, S270_NEAR) & body
    if not near.any(): return body
    op = binary_opening(body, structure=_s270_disk(S270_OPEN))
    new = (body & _dilate(core, S270_NEAR)) | op
    L, n = cc(new)
    if n > 1:
        keep = np.unique(L[near & new]); keep = keep[keep > 0]
        new = np.isin(L, keep)
    return new
POST_BODY[6000670] = _s270_spur
