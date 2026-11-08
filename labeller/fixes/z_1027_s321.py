# sweep 10-27 fix for skin 321 (Sonnet revision fix2.py after a rejected Haiku fix, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s321/)
# Task s321 (sweep 2026-10-27, attempt 2), skin 321 HammerDudeknight = Frank: super 6000480 (Frank004UltiProjectile), pair 6012,
# tick 367, too_small. The magenta super hull left out bright cyan parts of the attack: the curled cyan blades above the
# hull's top vertex and on the upper left, the tip of the tall cone on the right, and the cyan base rings at the lower
# right. Shapes of sc/effects_brawler.sc (per pixel, _a4_shape_pixels; checked on sw2_321 ticks 365-371):
#   frank_004_ulti_slash_02 / _04 (the super's own slash clips) = 47 (base rings), 61 (spike bodies), 7368 (cone bodies),
#   7743, 7747, 7749, 7750 (the curled cyan blades). These four are the attack's own sprites, not the particle clips.
#   884 is the cyan flame wisp shape of particle_fire_02..06 and is also used by slash_03 / slash_05: it stays out, as in
#   z_1023_frank321.py (wisps and flecks after the spikes fall are not the attack).
# Cause: the body of a trail-type attack is clipped to the prediction's region (idlabels: region = dilated predicted
# region), and the cones / blades lie 18-60 px beyond it, so the kept shapes beyond the region never reached the hull.
# Fix (attempt 2, after review1 rejected the unfiltered grow, which let tiny detached cyan flecks stretch the tick-369 hull
# from 26108 to 32369 px): the blade shapes (7743, 7747, 7749, 7750) count only as parts of a spike group: a piece of kept
# shapes (8-connected, drawn in the body's own draw calls, also beyond the region) is part of the super only if it
# contains the core spike shapes (47 bases, 61 spike bodies, 7368 cone bodies) of the body. Blade-only pieces (the loose
# flecks / wisp curls: 12-213 px at tick 369, some inside the region too) are dropped. At tick 367 every blade touches
# the cones, so all of them join.
# ONE_LABEL 'hull' is unchanged. Only ticks 367 and 369 carry a super label in this capture.
FRANK321_KEEP[6000480] = (47, 61, 7368, 7743, 7747, 7749, 7750)   # super: + the slash blades (7743, 7747, 7749, 7750)
FRANK321_CORE = (47, 61, 7368)                                    # spike bases, spike bodies, cone bodies
FRANK321_PIECE_CORE_MIN = 20  # px of core shapes a piece must hold to be a spike group (not a loose blade / fleck)


def _frank321_super_body(body, ctx):
    """super: kept-shape pieces (own draw calls) that hold core spike pixels of the body, blades included"""
    from scipy.ndimage import label as cc
    body = _frank321_clean(body, ctx)
    if not body.any(): return body
    t = ctx['t']; T = projtab.table()[t]
    names = tuple(sorted({n for f, n in _trail_clips(t) if f == FRANK321_FILE} | {n for n in (T['blue'], T['red']) if n}))
    SP = _a4_shape_pixels(ctx['R'], FRANK321_FILE, names)   # shape id + 1 per pixel, 0: none
    keep = np.isin(SP, [s + 1 for s in FRANK321_KEEP[t]])
    core = np.isin(SP, [s + 1 for s in FRANK321_CORE])
    ids_h = ctx['ids_h']
    own = np.unique(ids_h[body & keep]); own = own[own > 0]
    M = body | (keep & np.isin(ids_h, own))
    L, n = cc(M, structure=np.ones((3, 3), bool))
    cnt = np.bincount(L[body & core], minlength=n + 1)
    ok = cnt >= FRANK321_PIECE_CORE_MIN; ok[0] = False
    return M & ok[L]


POST_BODY[6000480] = _frank321_super_body
