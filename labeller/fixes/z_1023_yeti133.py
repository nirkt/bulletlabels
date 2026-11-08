# Yeti 133 attack: one hull over claw lines + head flash, frost out (user, 2026-10-20 / 10-23; Haiku task 10, reviewed)
# task 10 fix (Haiku task 10, 2026-10-23): WeaponThrower Yeti, skin 133 (Bronson003, type 6001224).
# The user's decision (binding): the label is the WHOLE claw attack: one clean convex hull over the claw lines AND the
# bright flash at the swoosh head, without the snow puff, snowflakes, twinkles, glow dots or the frost below. Frames whose
# logic has ended stay unlabelled (no lines touching the object's region -> empty, as before).
#
# Parts (see t10/report.txt for the approaches and the numbers):
#   lines  = shape 666 of bronson_003_atk_trail whose line touches the object's region (agent 4's rule, unchanged)
#   flash  = shape 7294 (the same atlas sprite as the ground-ice clip bronson_003_ulti_trail_iceground: no shape cut separates
#            the flash from the frost) AND frame max(R, G, B) >= 230 (the bright core, not the dim halo) AND within FLASH_NEAR
#            px of the lines' convex hull (the head, not the sprites further out) AND on screen above the midway point between
#            the projectile's airborne anchor and its ground point (the ground frost lies under the head; the airborne flash
#            sits above the ground point). The anchor is projected the way labels() projects it (render coords, cam).
#   body   = lines | flash, then ONE_LABEL 'hull' (already set for 6001224 by the agent-4 fix file).
FLASH_NEAR = 4          # px (frame) from the lines' hull
FLASH_BRIGHT = 230      # min(max RGB) of the flash core
_T10_FILE = 'sc/effects_brawler.sc'
_T10_CLIPS = ('bronson_003_atk_trail',)


def _t10_scr(cam, P):
    """screen position (frame px, 1200 x 540) of a render-coords point P (the anchor convention of labels())"""
    q = cam @ np.array([P[0], P[1], P[2], 1.0])
    return (q[0] / q[3] * 0.5 + 0.5) * 1200, (1 - (q[1] / q[3] * 0.5 + 0.5)) * 540


def _fix_6001224(body, ctx):
    from scipy.ndimage import distance_transform_edt
    ln = _a4_lines(ctx, _T10_FILE, _T10_CLIPS, (666,))
    if not ln.any(): return ln                      # logic ended / no line in the object's region: unlabelled
    o, img = ctx['o'], ctx['img']
    if img is None or o.get('anchor') is None: return ln
    H = hull_mask(ln)                               # the lines' convex hull
    near = distance_transform_edt(~H) <= FLASH_NEAR
    ax, ay, az = o['anchor']
    air = _t10_scr(ctx['cam'], (ax, ay, az)); gnd = _t10_scr(ctx['cam'], (ax, ay, 0.0))
    yy = np.arange(ln.shape[0])[:, None]
    above = yy <= (air[1] + gnd[1]) / 2.0
    SP = _a4_shape_pixels(ctx['R'], _T10_FILE, _T10_CLIPS)
    flash = (SP == 7294 + 1) & (img.max(-1) >= FLASH_BRIGHT) & near & above & ctx['region']
    if A4_DEBUG: print('T10 6001224 obj', o['id'], 'lines', int(ln.sum()), 'flash', int(flash.sum()), 'cut y', round((air[1] + gnd[1]) / 2, 1))
    return ln | flash
POST_BODY[6001224] = _fix_6001224
ONE_LABEL[6001224] = 'hull'
