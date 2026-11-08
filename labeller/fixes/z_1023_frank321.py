# Frank Knight 321: one hull per wave, attack and super (user Q16/Q17, 2026-10-23; Haiku task 1, reviewed)
# Task 1: skin 321 HammerDudeknight (Frank, brawler Frank, skin 004). The user's decision (2026-10-23): ONE label per wave
# while the wave's logic lives: the convex hull of the whole attack (standing ice spikes and the flat jagged streaks
# together), no per-spike labels, no holes. The same for the super (one hull over all its spikes). Particles that are not
# the spikes (pebbles, sparks, smoke, ground cracks, the cyan flame wisps of the super) must not stretch the hull: the body
# is cleaned to the spike shapes first (POST_BODY), then ONE_LABEL 'hull'.
#
# Shapes of sc/effects_brawler.sc (per pixel, from _a4_shape_pixels: the atlas shape the front-most draw shows, verified
# with shapediag / panel renders on sw2_321, ticks 277-287, 367-377):
#   attack 6000479 (Frank004Projectile):     61 = the ice spike bodies and the flat jagged streaks on the ground,
#                                            47 = the cyan base ring under each spike
#                                            (crack shapes 1 and the spike_006 / groundFrag / particle_dummy shapes are
#                                            never body here: they are drawn 'over' or are not in the label)
#   super 6000480 (Frank004UltiProjectile):   61 and 47 as above (spikes, bases, ground streaks), 7368 = the cone bodies
#                                            (the super's slash clips). Dropped: 884 = the cyan flame wisps (particle_fire
#                                            clips, drawn over the spikes and as flecks after the spikes fall), and 7743,
#                                            7747, 7749, 7750 = the curled tips of the flame wisps over the cones.
#                                            No-shape edge pixels are kept only within 2 px of a kept shape.
# Measured: attack label at ticks 277-287 = shapes 47 + 61 + edges only (no particle shapes), so nothing to drop there;
# the hull then covers the spikes and the streaks. The super's cleaned body (keep 61, 47, 7368) is the cones, bases
# and ground streaks; the wisp shapes are the ones that pulled the hull out before.
FRANK321_KEEP = {6000479: (47, 61),          # attack: spike bodies, bases, streaks
                 6000480: (47, 61, 7368)}    # super: spike cones, bases, streaks (wisps out)
FRANK321_FILE = 'sc/effects_brawler.sc'


# attack only: parts of the cleaned body further than FRANK321_NEAR px (label resolution) from the wave's OWN predicted label
# (ctx['pred_any'], the group's logic path region, game data) are not the wave. At attack ticks 277 and 279 a 12-35 px part of
# the thin blue-white flare at Frank's hand (kept shape 61 drawn by an effect sprite, draw 105 at 279; 27-29 px from the
# prediction) stretched the hull to the left. Every other attack spike and streak lies within 0 px of the prediction (checked
# on all odd ticks 275-329). Not applied to the super: its outer cones lie 18-26 px outside its prediction (ticks 367, 371),
# they are spikes and are kept.
FRANK321_NEAR = 8
FRANK321_FAR_RULE = {6000479}


def _frank321_clean(body, ctx):
    """body pixels of the spike shapes only (the kept shapes and the no-shape edge pixels that touch them, 2 px), in the
    parts that lie within FRANK321_NEAR px of the wave's own predicted label"""
    if not body.any(): return body
    from scipy.ndimage import label as cc, distance_transform_edt
    t = ctx['t']; T = projtab.table()[t]
    names = tuple(sorted({n for f, n in _trail_clips(t) if f == FRANK321_FILE} | {n for n in (T['blue'], T['red']) if n}))
    SP = _a4_shape_pixels(ctx['R'], FRANK321_FILE, names)   # shape id + 1 per pixel, 0: none
    keep = np.isin(SP, [s + 1 for s in FRANK321_KEEP[t]]) & body
    edge = body & (SP == 0) & _dilate(keep, 2)
    m = keep | edge
    if t not in FRANK321_FAR_RULE: return m
    pa = ctx['pred_any']
    if not pa.any(): return m & False
    dist = distance_transform_edt(~pa)
    L, n = cc(m, structure=np.ones((3, 3), bool))
    if n == 0: return m
    near = np.zeros(n + 1, bool)
    near[1:] = np.array([dist[L == k].min() <= FRANK321_NEAR for k in range(1, n + 1)])
    return m & near[L]


for _t in FRANK321_KEEP:
    POST_BODY[_t] = _frank321_clean
    ONE_LABEL[_t] = 'hull'
