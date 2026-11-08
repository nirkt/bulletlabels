# Ash Knight Toon 138: one hull per wave over portals + swords; super cube only (Haiku task 17, reviewed)
# Task 17: skin 138 KnightToon (Ash 006, brawler Ash, skin 006). The user's rules (BRIEF + addendum) and decision (2026-10-23):
# the attack's own body only; no explosion / impact, no smoke / dust / ground decals, no gadget effects, no loot particles; for a
# multi-part wave attack ONE hull per wave while it lives (the swords of one wave = one label); nothing once the attack has ended
# (labels() drops ended trail-body objects). Coordinator's review (2026-10-23, pass 2): the wave body is built from (a) the
# wave's portal tiles (each sword stands on one; drawn every frame while the wave lives), (b) the sword flames at coverage >= 32,
# (c) generic quads 0/1 connected to (a) or (b); then ONE hull.
#
# Shapes of sc/effects_brawler.sc (per pixel, from shapediag / _a4_shape_pixels on sw2_138; clip names from the type's trail
# clips and own clips):
#   attack 6001049 (Ash006Projectile1, trail ash_006_atk_trail; one object per wave):
#     27842, 27843, 27844   ash_006_atk_1_sword: the sword flames (white tip, hilt, blue flame)            KEPT (b)
#     27811, 27812          ash_006_atk_portal_lv1_back / portal_lv1: the portal tile under each sword     KEPT (a)
#                           (the wave's own, within T17_NEAR of its prediction: the base of every sword is inside the hull)
#     0, 1                  generic quads, in the sword clip and in the pixel particles: KEPT only connected to (a) or (b)
#     61                    ash_def_atk_impact: the jagged crystal burst (impact animation, rule 2)            dropped
#     381, 27795, 27797, 27807  ash_006_bone / potion / key anim: loot items flying out of the portals          dropped
#     709                   pixel_ani_type2-4: sparks / pixel particles                                       dropped
#   super 6001067 (Ash006UltiProjectile, own clip ash_006_ulti_projectile):
#     27745                 the cyan cube: the projectile itself                                              KEPT
#     63                    the small yellow cubes beside it (ticks 345-349): dropped as separate particles (question)
#     0                     generic quad: dropped (no-shape pixels touching the cube stay: 2 px)
T17_FILE = 'sc/effects_brawler.sc'
T17_SWORD = (27842, 27843, 27844)   # attack (b): the sword flames
T17_PORTAL = (27811, 27812)         # attack (a): the wave's portal tiles
T17_QUAD = (0, 1)                   # attack (c): generic quads, connected only
T17_NEAR = 80                       # px (half res): the wave's shapes within this of its prediction (the lowest portal at 313 lies
                                    # outside the 40 px region of labels(), so the region alone would drop its base)
T17_CV = {6001049: 32, 6001067: 128}   # coverage cut: the attack's sword and portal are translucent (29-255); the super's cube 128
T17_SPECK = 12                      # px: pieces of the body smaller than this are specks / spurs
T17_SUPER_KEEP = (27745,)


def _t17_clean(body, ctx):
    """the attack: the wave's body = its portal tiles (a) + sword flames (b) + generic quads connected to (a) or (b), within
    T17_NEAR px of its prediction; the super: the cyan cube (nothing else). Pieces smaller than T17_SPECK px are dropped."""
    if not body.any(): return body
    t = ctx['t']; T = projtab.table()[t]
    names = tuple(sorted({n for f, n in _trail_clips(t) if f == T17_FILE} | {n for n in (T['blue'], T['red']) if n}))
    SP = _a4_shape_pixels(ctx['R'], T17_FILE, names)   # shape id + 1 per pixel, 0: none
    cv = ctx['cv']; ct = T17_CV[t]
    from scipy.ndimage import distance_transform_edt as _edt, label as _cc
    near = _edt(~ctx['pred_any']) <= T17_NEAR if ctx['pred_any'].any() else np.zeros(body.shape, bool)
    if t == 6001049:
        portal = np.isin(SP, [s + 1 for s in T17_PORTAL]) & near & (cv >= ct)
        sword = np.isin(SP, [s + 1 for s in T17_SWORD]) & near & (cv >= ct)
        seed = portal | sword
        quad = np.isin(SP, [s + 1 for s in T17_QUAD]) & near & (cv >= ct)
        LQ, nq = _cc(quad | seed, structure=np.ones((3, 3), bool))   # quads 8-connected to the seeds (through each other)
        tq = np.unique(LQ[seed]); tq = tq[tq > 0]
        m = seed | (np.isin(LQ, tq) & quad)
        # no-shape pixels touching the wave's shapes (sprite edges): 2 px
        m |= body & (SP == 0) & _dilate(m, 2)
    else:
        keep = np.isin(SP, [s + 1 for s in T17_SUPER_KEEP]) & near & (cv >= ct)
        m = keep | (body & (SP == 0) & _dilate(keep, 2))
    # specks and spurs: pieces of the body smaller than T17_SPECK px are dropped
    L, n = _cc(m, structure=np.ones((3, 3), bool))
    if n > 1:
        sz = np.bincount(L.ravel())
        small = np.nonzero((sz < T17_SPECK) & (np.arange(len(sz)) > 0))[0]
        m &= ~np.isin(L, small)
    return m


for _t in (6001049, 6001067): POST_BODY[_t] = _t17_clean
# the attack: ONE label per wave, the convex hull of the body (portal tiles and swords; no holes); the wave's label is dropped
# once its logic has ended (labels(): ended trail-body objects)
ONE_LABEL[6001049] = 'hull'
FADE_CUT[6001049] = T17_CV[6001049]   # the body takes the translucent sword and portal pixels too (labels(): FADE_CUT)
