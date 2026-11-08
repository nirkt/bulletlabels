# sweep 10-27 fix for skin 060 (Haiku fix, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s060/)
# sweep 10-27 fix for skin 060 StickyBombDefault (Haiku task s060, sweep 2026-10-27; findings pair 1081 = attack 90 deg, tick 305,
# spur_or_speck; pair 1084 = attack 90 deg, tick 311, spur_or_speck).
# The default sticky bomb's attack, StickyBombProjectile 6000932 (clip sticky_def_atk_projectile, trail particle_stickyBubble), is a
# blob with bubbles and spatter drops flying off it. Those drops are drawn with the same atlas shapes as the blob's body clip (shared
# draws), so they fall inside the body's prediction and get labelled as separate droplets above the blob: tick 305 two (38 px and
# 19 px, a red loop with a stub), tick 311 one droplet (22 px) while its equally small neighbour is unlabelled. The same kind of
# loose bits beside the blob are also labelled: tick 305 a 26 px sliver on the left edge and a 4 px speck, tick 311 an 8 px and a
# 2 px speck, tick 309 a 1 px speck. The verifier's note: such droplets are a speck outside the attack outline.
# Fix (keyed to 6000932 only): in the body the largest connected part is the attack. Every other part that does not touch it (8-connected,
# no bridging) and is smaller than S060_DETACH px (full resolution) is dropped. Parts that touch the blob (the nub on its top at ticks
# 307 and 309, its rim, the bright glint) and the blob itself are unchanged, and nothing is added. The shared clean-up (z_1099)
# runs after this and is untouched. The super (6000933) and the other sticky types are not affected.
# Runs as a POST_BODY hook (before the hull / claim stages); chains any earlier POST_BODY for the same type.

S060_TYPE = 6000932
S060_DETACH = 60   # px (full resolution): detached parts of the body below this size are droplets, not the attack
S060_CLOSE = 2     # px: notches of the blob's outline up to about 4 px wide are closed (tick 311: the slit under its top spike)
S060_CLOSE_CV = 32 # coverage from which the closed pixels count: only the blob's own translucent rim (its own draws), never the glow


def _s060_disk(r):
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r


def _s060_detached(body, ctx):
    from scipy.ndimage import binary_closing, label as cc
    if not body.any(): return body
    L, n = cc(body, structure=np.ones((3, 3)))   # 8-connected: parts touching the blob stay with it
    sz = np.bincount(L.ravel())
    main = int(np.argmax(sz[1:])) + 1            # the blob
    if n > 1:
        drop = [j for j in range(1, n + 1) if j != main and sz[j] < S060_DETACH]
        if drop: body = body & ~np.isin(L, drop)
    # the blob's outline solid: closing of its main part, adding only the blob's own draws at coverage >= S060_CLOSE_CV
    # (measured: 2-23 px per tick on ticks 305-311, all within 2 px of the blob)
    m = L == main
    pad = S060_CLOSE + 2
    c = binary_closing(np.pad(m, pad), structure=_s060_disk(S060_CLOSE))[pad:-pad, pad:-pad]
    own = np.unique(ctx['ids_h'][m]); own = own[own > 0]
    add = c & ~body & ctx['region'] & (ctx['cv'] >= S060_CLOSE_CV) & np.isin(ctx['ids_h'], own)
    return body | add


_s060_prev = POST_BODY.get(S060_TYPE)


def _s060_body(body, ctx):
    if _s060_prev is not None: body = _s060_prev(body, ctx)
    return _s060_detached(body, ctx)


POST_BODY[S060_TYPE] = _s060_body
