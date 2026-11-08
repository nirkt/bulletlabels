# installed 11-01 from out/sweep2/haikutasks/siteai2/misc_f/fix.py (2nd AI sweep: 379 Pam007 bubble cloud droplets/splash out; 357 part DISABLED)
# misc_f fix (AI-reviewer findings 2026-11-01, second sweep; skins 357 CocoonerVirus, 379 MinigunDudeZombie).
# Exec'd in idlabels' namespace after all installed fixes (IDLABELS_OVERRIDE). Every change is keyed to an explicit type id.
# Checked and NOT changed (see report.md): 293 super t369 (zone is a plain circle; the loops are separate wisp projectiles),
# 423 attack 0 t267 / attack 90 t309 (Janet per-shot hull: members whose pieces spread apart are unlabelled, open question Q4),
# 430 attack 0 t269 beam tail (needs_user), 430 attack 90 crown notches (no defect: the gaps are between the crown's tails).
#
# 357 CocoonerVirus, super 45 deg, ticks 345-349 (finding: "the label covers only the lower-right half of the green capsule;
#   the upper-left body is outside"). Type 6001384 Cocooner003UltiProjectile (the 003 skin's super; other skins' use of the type not checked).
#   Cause: the capsule's translucent back end (coverage 32-127, below the body cut of 128) is not body, and the bright white
#   burst covers the head at tick 345. Lowering the cut for the whole type (FADE_CUT 32 or 64) also took in the caster's green
#   shell arcs at the upper-left (checked: a loop around the arcs at tick 345), so it is not used.
#   Fix (POST_BODY): the body plus the translucent clip pixels (coverage >= 64; 32 also took the halo at 349-351) within 3 px
#   (half resolution) of the body, plus
#   the bright clip pixels (coverage >= 200) within 6 px; a 3x3 opening removes spurs of one or two pixels; holes are filled;
#   one piece (largest). Only the projectile's own clip pixels (shape code 1) are added, never the glow (code 2) or the shell.
#   Checked on ticks 343-351: the capsule's back end and the upper-left cap are inside the outline at 345 and 347; 349 and 351
#   gain their rim; tick 343 has no label before or after.
_P357_TYPE = 6001384
_P357_NEAR = 3      # half-resolution px: translucent clip pixels this close to the body are body
_P357_BRIGHT = 6    # half-resolution px: bright clip pixels this close to the body are body
_P357_BRIGHT_V = 200


def _fix_6001384_capsule(body, ctx):
    import numpy as _np
    from scipy.ndimage import binary_fill_holes as _bfh, label as _cc, binary_dilation as _bd, binary_opening as _bo
    if not body.any(): return body
    C = ctx['C']; cv = ctx['cv']; region = ctx['region']
    st = _np.ones((3, 3), bool)
    near = _bd(body, structure=st, iterations=_P357_NEAR) & region
    bright = _bd(body, structure=st, iterations=_P357_BRIGHT) & (cv >= _P357_BRIGHT_V) & (C == 1)
    m = body | (near & (C == 1) & (cv >= 64)) | bright
    m = _bo(m, structure=st) | body
    m = _bfh(m) & (m | region)
    L, n = _cc(m)
    if n > 1:
        sz = _np.bincount(L.ravel()); sz[0] = 0
        m = L == sz.argmax()
    return m


# NOT INSTALLED (coordinator 11-01): outline got ragged / double lines at t347-351
# POST_BODY[_P357_TYPE] = _fix_6001384_capsule


# 379 MinigunDudeZombie, attack 90 deg, tick 329 (finding: "a tiny square speck and a ragged outline on the green splash after
#   the hit; should not be labelled"). Type 6000394 Pam007Projectile (all skins that use it). Object 2000044 is a lilac bubble
#   cloud that flies down to the water: its clip body (shape 63) is 791 of the 880 labelled px at tick 329, and the label also
#   took two separate lilac balls (one touching by a neck, one below), which are droplets of the splash, not the cloud.
#   Fix (POST_BODY, chained after the installed bone fix z_1027_s379): only where the body is mostly shape 63 (>= 60% of the
#   body, body >= 300 px, i.e. the bubble cloud, not the bone), an opening with a disk of radius 6 removes parts that the cloud's
#   own width cannot hold (the balls and the neck), the main part is kept and grown back 2 px inside the body.
#   Checked: tick 329 keeps the cloud only; ticks 321-327 unchanged; attack 0 (263-289) and super (343-411) renders identical
#   except the title text (no other label of skin 379 changes). Bone candies (ticks 305-317) are not shape 63: unchanged.
_P379_TYPE = 6000394
_P379_FILE = 'sc/effects_brawler.sc'
_P379_BONE = ('pam_007_atk_projectile', 'pam_007_atk_projectile_red')
_P379_CLOUD = 63          # clip shape of the bubble cloud (effects_brawler.sc)
_P379_OPEN = 6            # px (full resolution): a part narrower than this disk is a ball or neck of the splash, not the cloud
_prev_379_pam = POST_BODY.get(_P379_TYPE)


def _fix_6000394_cloud(body, ctx):
    import numpy as _np
    from scipy.ndimage import binary_opening as _bo, label as _cc, binary_dilation as _bd
    if _prev_379_pam is not None: body = _prev_379_pam(body, ctx)
    if not body.any(): return body
    SP = _a4_shape_pixels(ctx['R'], _P379_FILE, _P379_BONE)
    cloud = body & (SP == _P379_CLOUD + 1)
    if body.sum() < 300 or cloud.sum() < 0.6 * body.sum(): return body
    yy, xx = _np.mgrid[-_P379_OPEN:_P379_OPEN + 1, -_P379_OPEN:_P379_OPEN + 1]
    op = _bo(body, structure=(xx * xx + yy * yy) <= _P379_OPEN * _P379_OPEN)
    L, n = _cc(op)
    if n == 0: return body
    sz = _np.bincount(L.ravel())[1:]
    main = L == (sz.argmax() + 1)
    return body & _bd(main, structure=_np.ones((3, 3), bool), iterations=2)


POST_BODY[_P379_TYPE] = _fix_6000394_cloud
