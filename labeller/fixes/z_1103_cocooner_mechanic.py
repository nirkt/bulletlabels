# AI-review batch 2026-11-01 fix (Haiku fix, reviewed by the coordinator, installed 11-01 LOCAL ONLY; source out/sweep2/haikutasks/siteai/cocooner_mechanic/fix.py)
# AI-review fix, cocooner_mechanic task (2026-11-01): skins 343 CocoonerRosie, 357 CocoonerVirus, 396 MechanicJessie.
# exec'd in idlabels' namespace (IDLABELS_OVERRIDE), after all installed fixes. Keyed to explicit type ids only.
#
# 1. Skin 343 CocoonerRosie, super 'super 45' ticks 345-347 (AI finding: ragged multi-piece outline with interior loops).
#    Type 6001381 Cocooner002UltiProjectile (clip charlie_002_ulti_projectile): the super is drawn from several sprites
#    (the crystal head, the striped vine loops, the leaves, and the bright burst at the start). The label had 3 parts at
#    tick 345 (a detached 179 px piece and an 8 px speck), enclosed loops at 347, and a notched outline at 349.
#    Fix (POST_BODY): close gaps of up to 3 px in the union of its sprite pixels, then fill the enclosed holes, inside the
#    projectile's search region, and keep the drawn body. Result: one solid piece at every tick 345-351. Compared with
#    a convex hull, this keeps the object's own outline (the owner's rule for multi-sprite objects: fill and smooth the
#    union). 343's attack 0 is type 6001379 (Cocooner002Projectile), not changed: label counts identical with and without
#    this file at ticks 263-271. The type name carries skin 002, so no other skin is expected to use 6001381 (not verified
#    against other skins' captures).
COCOONER_ROSIE_SUPER = 6001381
COCOONER_ROSIE_CLOSE = 3   # px (frame resolution): gaps in the union closed


def _fix_cocooner_rosie_super(body, ctx):
    from scipy.ndimage import binary_fill_holes
    if not body.any(): return body
    filled = binary_fill_holes(_close(body, COCOONER_ROSIE_CLOSE)) & ctx['region']
    return filled | body


POST_BODY[COCOONER_ROSIE_SUPER] = _fix_cocooner_rosie_super


# 2. Skin 357 CocoonerVirus, attack 0 'attack 0' ticks 265, 267, 269 (AI finding: outlined only by small red fragments;
#    'should get a hull'). Type 6001382 Cocooner003Projectile (clip charlie_003_atk_projectile_01): the green swirl disc is
#    translucent (most of its pixels have coverage 32-127, below the body cut of 128), and its sprite is shared with its
#    trail, so only a few disc parts passed the body test (31, 628 and 150 px in 20, 2 and 7 parts).
#    Fix: lower the coverage cut of this type to 32 (translucent disc body), then the owner's splintered-attack rule: one
#    convex hull over the whole disc (ONE_LABEL 'hull'). The 16x16 blended beam crossing the disc has no sprite code and
#    is not body, so the hull fills across it (hidden-part rule). Ticks 263 and 271 stay unlabelled (no body at all).
FADE_CUT[6001382] = 32
ONE_LABEL[6001382] = 'hull'


# 3. Skin 396 MechanicJessie, attack 90 deg ticks 305-313 (AI finding: the beam label covers only the lower half, the
#    bright upper part is outside). Type 6000194 Jessie009Projectile1 (clip jessie_009_atk_projectile, its streak). The same
#    type is the attack 0 deg row (ticks 265-271), which has the same cut-off streak; its label grows the same way (it was
#    not in the findings, changed by this type-level fix, see the report). The upper part of the streak has coverage
#    40-110 (translucent), below the body cut of 128, so it was left out; the installed fix (z_1027_s396, ONE_LABEL
#    'hull') then hulled only the lower part. Fix: coverage cut 64 for this type, so the whole visible streak from its
#    start under the caster's cap down to the tip is body, and the existing hull covers it. The faint overlay on the cap
#    at the very top (coverage 30-56) stays out; the cut of 32 let more of it in. Ticks 303 and 319 have no label before
#    or after; 315 and 317 gain their top arc.
FADE_CUT[6000194] = 64
