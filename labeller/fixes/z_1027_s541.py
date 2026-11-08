# sweep 10-27 fix for skin 541 (Sonnet revision, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s541/)
# sweep 2026-10-27, skin 541 MechaDudePirate (pairs 9684 and 9695; sw2_541 attack 0 deg tick 269 spur_or_speck, attack 90 deg tick
# 325 extra_label).  Second attempt (the first, fix.py, kept the largest part of each label and so never dropped debris that is
# a label of its own).
# Mecha005UltiProjectile (type 6001079, MechaDudeBigProjectile, the green shells).  Its clip (effects_brawler.sc
# mecha_005_atk2_projectile[_red]) draws, besides the shell (shape 53431 = the green bulb, 63 = core, 0/1013/2934 = lime crescent
# and crown, 50859 = neck) the sprite 53425: grey-green leaf / white swoosh / purple chunk particles that fly off the shell and
# lie around it (debris, rule 3: never labelled).  The label took them as separate pieces: next to the shells (tick 269 by the
# crate: leaf, swoosh, purple cube, specks; tick 325: the leaf at the bottom edge, the loop on the smoke at the left, the stray
# between the two shells).  Fix (POST_BODY, before the shared clean-up): split the body into 8-connected parts and drop every part
# that is not a shell: it has fewer than MECHA_CORE_PX (40) pixels of the bulb (shape 53431), fewer than MECHA_PRED_PX (8)
# pixels inside the shell's predicted body, and contains pixels of the debris sprite 53425 or of a loose core sprite (63, e.g. the white fragments at tick 321 of attack 90).  Parts that touch the shell (the
# crown, the crescent, the swoosh joined to it at launch) are in the same part and stay unchanged; other shells' parts (they
# have the bulb) are untouched here and are claimed per pixel as before.  Only type 6001079 is changed.
MECHA_CORE_PX = 40
MECHA_PRED_PX = 8
MECHA_CLIPS = ('mecha_005_atk2_projectile', 'mecha_005_atk2_projectile_red')


def _fix_6001079(body, ctx):
    from scipy.ndimage import label as _cc
    if not body.any(): return body
    L, n = _cc(body, structure=np.ones((3, 3), bool))
    if n <= 1: return body
    SP = _a4_shape_pixels(ctx['R'], 'sc/effects_brawler.sc', MECHA_CLIPS)
    pb = ctx['pred_body']; drop = []
    for j in range(1, n + 1):
        m = L == j
        if (SP[m] == 53431 + 1).sum() >= MECHA_CORE_PX: continue
        if (m & pb).sum() >= MECHA_PRED_PX: continue
        if np.isin(SP[m], (53425 + 1, 63 + 1)).any(): drop.append(j)
    if drop: body = body & ~np.isin(L, drop)
    return body


POST_BODY[6001079] = _fix_6001079
