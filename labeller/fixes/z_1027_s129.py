# sweep 10-27 fix for skin 129 (Haiku fix, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s129/)
# sweep 10-27 fix for skin 129 KnightPizza (Ash skin 003; Haiku fix task s129, out/sweep2/haikutasks/sweep1027/s129/).
# Findings on the attack rows (sw2_129): attack 6001043 Ash003Projectile1 (a wave; trail clips ash_003_atk_*), attack 0 deg:
#   pair 2069 tick 269 'ragged', pair 2070 tick 271 'merge_split', pair 2071 tick 273 'missing_label';
#   attack 90 deg (object 2000003, same type): pairs 2076 (tick 309 too_small), 2077 (311 merge_split), 2078 (313 ragged).
# Cause: the wave's body is a cloud of blue flames (clip ash_003_atk_fire) and rotating pizza toppings (clips
# ash_003_atk_<cheese|mushroom|onion|tomato|pimento>_01_roate_01/02). Most of the flame mass is drawn at coverage 32-127,
# under the 128 cut of idlabels.CUT, so only the opaque bits were outlined: loose red loops and a central mass with no
# outline. Ash 006's wave (z_1024_ash138) has the same translucent body.
# Fix: the body is cut from coverage 32 (FADE_CUT, as z_1024_ash138) and kept only where the front-most draw shows a shape of
# those named clips (not shape numbers). Smoke (explode_smoke), soot, ground fire, sparks (fire_spark_blue) and the particle_gas
# puffs are effects of the attack, not the attack (rule 3): not kept. Then closed (2 * S3_CLOSE px gaps: the topping blobs join
# the cloud), hole-filled and clipped to the attack's region; detached specks under S3_SPECK px (half resolution) dropped.
# Keyed to type 6001043 only. Shared types: 6001043 is the Ash skin 003 attack; other skins using it are checked in report.md.
# Runs after z_1099_cleanup.py (override files are exec'd last), so it sees the final body of this type.

S3_TYPE = 6001043
S3_FILE = 'sc/effects_brawler.sc'
S3_KEEP = ('ash_003_atk_fire',                                                 # the blue flame body
           'ash_003_atk_cheese_roate_01', 'ash_003_atk_cheese_roate_02',       # the rotating toppings of the attack
           'ash_003_atk_mushroom_01_roate_01', 'ash_003_atk_mushroom_01_roate_02',
           'ash_003_atk_onion_01_roate_01', 'ash_003_atk_onion_01_roate_02',
           'ash_003_atk_tomato_01_roate_01', 'ash_003_atk_tomato_01_roate_02',
           'ash_003_atk_pimento_01_roate_01', 'ash_003_atk_pimento_01_roate_02')
S3_CV = 32                            # coverage from which the translucent flame body is body (as FADE_CUT of z_1024_ash138)
S3_CLOSE = 5                          # px (half resolution): gaps up to 2 x this closed (the topping blobs join the cloud)
S3_SPECK = 40                         # px (half resolution): detached pieces smaller than this are specks (not the cloud's flame tongues)

FADE_CUT[S3_TYPE] = S3_CV


def _s129_body(body, ctx):
    """the wave's body = the pixels its flame and topping clips show (coverage >= S3_CV) and the no-shape pixels touching
    them (2 px), closed and hole-filled into one solid outline inside the attack's region, detached specks dropped"""
    if not body.any(): return body
    from scipy.ndimage import binary_fill_holes, label as cc
    SP = _a4_shape_pixels(ctx['R'], S3_FILE, S3_KEEP)   # shape id + 1 per pixel of the kept clips, 0: none of them
    shapes = (SP > 1) & (ctx['cv'] >= S3_CV)            # shape id 0 (a generic quad, also in the smoke clips) is not kept
    m = body & shapes
    m |= body & (SP == 0) & _dilate(m, 2)              # sprite edges touching the body's shapes
    m = _close(m, S3_CLOSE) & ctx['region'] & (body | _dilate(body, S3_CLOSE))
    m = binary_fill_holes(m) & ctx['region']
    L, n = cc(m, structure=np.ones((3, 3)))
    if n > 1:   # specks: detached pieces under S3_SPECK px (the largest piece always stays)
        sz = np.bincount(L.ravel())
        m &= ~np.isin(L, np.nonzero((sz < S3_SPECK) & (np.arange(len(sz)) > 0))[0])
    return m


POST_BODY[S3_TYPE] = _s129_body


# ---- finding 2088 (super 45 deg, tick 411, 'extra_label'): a cyan / blue outline over the bush at the top centre.
# Cause: spawn_labels takes every runtime-texture model draw (MODEL_TEX) to the nearest spawnable within CHAR_NEAR = 80 px.
# A translucent bush draw (fs 4132660303, draw 69: 959 px, centroid 78 px from the pet's ground-top segment) is nearest to
# the KnightPet 1000027 and becomes its label. The pet's own draws lie at its base, 2-25 px from its segment (tick 411:
# draws 78, 79, 81 at 2.5, 2.2, 24 px). Fix: wrap spawn_labels for the KnightPet type (characters.csv did 16000197): a draw
# claimed by a pet of that type is kept only when at least half of its pixels lie within S129_NEAR px of that pet's
# ground-top segment. Other spawnables and all projectiles are untouched.
S129_PET = 16000197     # KnightPet (characters.csv id): spawnable of skin 129 and the other Knight pet skins
S129_NEAR = 30          # px (half resolution): the pet's model lies within this of its ground-top segment
if '_s129_spawn_base' not in globals(): _s129_spawn_base = spawn_labels   # the original, also when this file is exec'd twice


def spawn_labels(state, cam, R, taken):
    out = _s129_spawn_base(state, cam, R, taken)
    if not out: return out
    from scipy.ndimage import label as _cc
    ids_h = R['id'][::2, ::2]
    chars = {int(c[0]): c for c in state['chars']}
    for cid in list(out):
        c = chars.get(cid)
        if c is None or int(c[1]) != S129_PET: continue
        a = np.array(_proj(cam, int(c[3]), int(c[4]), int(c[5]))); b = np.array(_proj(cam, int(c[3]), int(c[4]), int(c[5]) + CHAR_TOP))
        ab = b - a; m = out[cid]
        ys, xs = np.nonzero(m); p = np.c_[xs, ys].astype(float)
        tt = np.clip(((p - a) @ ab) / max(1e-6, float(ab @ ab)), 0, 1)
        dist = np.linalg.norm(a + tt[:, None] * ab - p, axis=1)
        dd = ids_h[ys, xs]
        for d in np.unique(dd):
            sel = dd == d
            if (dist[sel] <= S129_NEAR).mean() < 0.5:   # a draw beside the pet (bush, effect), not its model
                out[cid][ys[sel], xs[sel]] = False
        if not out[cid].any(): del out[cid]
    return out
