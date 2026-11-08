# Janet (JetpackGirl) super: NO label (the user, 2026-10-23, Q18: "dont label the super for janet").
# Every skin's super projectile type (logic JetpackGirlUltiProjectile), incl. the overcharged one.
JANET_SUPER = (6001156, 6001157, 6001159, 6001161, 6001163, 6001165, 6001167, 6001169, 6001171, 6001173)


def _janet_super_off(body, ctx):
    return np.zeros_like(body)


for _t in JANET_SUPER: POST_BODY[_t] = _janet_super_off
