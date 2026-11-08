# sweep 10-27 fix for skin 238 (Haiku fix, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s238/)
# s238 (skin 238 TntDudeMecha, Mike004), sweep 2026-10-27, pair 4306 (sw2_238 attack 90 deg, tick 321): the attack
# 6000273 (Mike004Projectile, the two blue robot flames) had two small detached clip-body blobs above each flame (12-17 px
# each, the flame's own draw pieces, not touching the flame body). They were labelled as red specks ("no specks" rule).
# The shared clean-up (z_1099) bridges parts within 3 px of the flame, so it keeps them.
# Fix, for 6000273 only: after the existing per-type hook (a0_agent0's _mike004_atk), drop body parts smaller than 25 px and
# smaller than a tenth of the largest part. Each flame's own body (the largest part) and every larger part stay.
_prev_s238_mike_atk = POST_BODY.get(6000273)


def _s238_mike_specks(body, ctx):
    from scipy.ndimage import label as _cc
    if _prev_s238_mike_atk is not None:
        body = _prev_s238_mike_atk(body, ctx)
    if not body.any(): return body
    L, n = _cc(body)
    if n < 2: return body
    sz = np.bincount(L.ravel())[1:]
    drop = [k + 1 for k in range(n) if sz[k] < 25 and sz[k] < 0.1 * sz.max()]
    if drop: body = body & ~np.isin(L, drop)
    return body


POST_BODY[6000273] = _s238_mike_specks
