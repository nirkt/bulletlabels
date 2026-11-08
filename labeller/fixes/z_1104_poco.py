# installed 11-01 from out/sweep2/haikutasks/siteai2/poco/fix.py (2nd AI sweep: 244 attack cell rule, 274 super one zone, 373 barrel out)
# Fix (AI review 2026-11-01, task poco, second sweep): Poco clouds as ONE solid zone, stopping when each member's logic ends.
# Findings (skin, row, tick): 244 attack 0 t275 (+ attack 90 t313/t315, note "straight diagonal edge"): the attack type
#   6000339 (Poco003Projectile) still used the old swept-region cut (z_1028_poco), which cuts straight strips through the
#   running cloud and leaves specks (t275: two specks of 95 and 52 px next to the 2743 px body).
#   -> the Poco cell rule of z_1103_poco (6000337 / 6000343 attacks) is applied to 6000339 too (opening 3 px, no bridging).
# 274 super 45 t349 (+ t351): the super type 6000342 (Poco004UltiProjectile) was cut into two or three outlines and the
#   green puffs at the upper left were left out (the cloud is one zone per the Poco rule). The super of the same cloud
#   family (244 super 6000340) uses the cell rule with bridging of its puffs (3 px open, 8 px bridge): the same here.
# Mechanism: the cell rule of z_1103_poco.py (installed, unchanged): screen Voronoi cells of the group's drawn members,
#   running members only (finished == 0, not ended), solid map blocks cut, closing / hole fill / opening, parts < 15 %
#   of the largest dropped. Keyed to the type ids below only; other types untouched.
#
# Not changed here: 231 attack 0 t275, 256 attack 0 t273, 274 attack 0 t275, 311 attack 0 t275 and 373 attack 0 t275
#   (the pieces / notches at the tick where two members end are the cut of ended members' clouds, owner decision 10-28),
#   311 super t345 (already one outline), 346 and 390 (open owner question Q3).

PZ_TYPES[6000339] = (3, 0)   # skin 244 PocoPirate attack (Poco003Projectile)
PZ_TYPES[6000342] = (3, 8)   # skin 274 DeadMariachiPunk super (Poco004UltiProjectile): several puffs of one shot

for _tp_f in (6000339, 6000342):
    POST_BODY[_tp_f] = (lambda t: (lambda body, ctx: _pz_body(t, body, ctx)))(_tp_f)

# 373 attack 0 t275 (+ the same cloud on the other ticks): the label took in the orange barrel beside the cloud (about 45
#   px of the barrel's top at t275; the map's solid cut does not reach these pixels, and the barrel's draws are batched with
#   the cloud's draws, so a draw / shape cut is not clean). Type 6000349 (Poco008Projectile, skin 373 DeadMariachiOnceupon)
#   only: the warm orange pixels of the frame are removed from its label after the Poco rule. The cloud of this attack has no
#   orange (blue / cyan puffs and skulls), so only the barrel's pixels are taken out.
_fix373_prev_post = POST_BODY.get(6000349)


def _fix373_barrel(body, ctx):
    if _fix373_prev_post is not None:
        body = _fix373_prev_post(body, ctx)
    img = ctx.get('img')
    if img is None or not body.any(): return body
    r = img[..., 0].astype(int); g = img[..., 1].astype(int); b = img[..., 2].astype(int)
    warm = (r > 170) & (g > 80) & (g < 175) & (b < 80) & (r - g > 50)
    return body & ~warm


POST_BODY[6000349] = _fix373_barrel
