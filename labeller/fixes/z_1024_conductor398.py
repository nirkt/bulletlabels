# Conductor Supercell 398: clean attack orbs, super one hull, arcs / darts out (Haiku task 19, reviewed)
# task 19 fix: skin 398 ConductorSupercellMake (brawler Conductor). Two projectile types of this skin:
#   6001374 Conductor003Projectile (attack, rows attack 0 deg and attack 90 deg)
#   6001375 Conductor003UltiProjectile (super)
# The user's decision (binding): every live projectile labelled as its visible body, one piece, no trails / sparks /
# particles, nothing once the attack has ended. Keyed to these two type ids only.

_T19_FILE = 'sc/effects_brawler.sc'
_T19_SUPER_CLIPS = ('chuck_003_ulti_projectile', 'chuck_003_ulti_projectile_red')
# shape ids + 1 (as _a4_shape_pixels gives them) of the super's clip that are NOT the body:
#   7067  electric arcs round the orb (a vector colour fill: its pixels flicker in and out of the label frame to frame)
#   0     the soft blue glow round the orb (a halo: not labelled)
#   7384  magenta twinkles (glints drawn over the body)
#   11988 cyan darts (a small arrow shape; loose darts float apart from the orb, and the same shape sits on the attack's
#         orb edge: they are cut off as the spurs the attack shows)
_T19_SUPER_OFF = (7067 + 1, 0 + 1, 7384 + 1, 11988 + 1)
_T19_ATTACK_CLIPS = ('chuck_003_atk_projectile', 'chuck_003_atk_projectile_red')

# attack (6001374, skin 398): the cyan darts on the orb's edge are drawn thin and come and go, so the label kept spurs and
# a second piece (rows attack 0 and 90, ticks 279, 319-327). Thin parts sticking out are opened away (3 px at half
# resolution, as THIN_OFF does for other types); the round orbs keep their outline. Pieces smaller than a sliver stay.
THIN_OFF[6001374] = 3


def _t19_attack_body(body, ctx):
    """attack 6001374: one piece per orb: its cyan darts (shape 11988) are not body; the pieces left that are under a
    fifth of the largest are dropped; enclosed gaps filled"""
    if not body.any(): return body
    from scipy.ndimage import label as cc, binary_fill_holes
    SP = _a4_shape_pixels(ctx['R'], _T19_FILE, _T19_ATTACK_CLIPS)
    darts = body & np.isin(SP, [11988 + 1])
    body = body & ~darts
    if not body.any(): return body
    # the darts' pixels on or inside the orb leave notches: closed again (2 px), only inside the orb or 2 px of a dart
    body = binary_fill_holes((_close(body, 2) & (body | _dilate(darts, 2))) | body)
    L, n = cc(body); sz = np.bincount(L.ravel())[1:]
    keep = [k + 1 for k in range(n) if sz[k] >= 0.2 * sz.max()]
    return binary_fill_holes(np.isin(L, keep))


POST_BODY[6001374] = _t19_attack_body


def _t19_super_body(body, ctx):
    """super 6001375: the orb's body without its electric arcs, halo and twinkles; pieces under a tenth of the largest
    dropped; the label is then one hull (ONE_LABEL below)"""
    if not body.any(): return body
    from scipy.ndimage import label as cc
    SP = _a4_shape_pixels(ctx['R'], _T19_FILE, _T19_SUPER_CLIPS)
    m = body & ~np.isin(SP, list(_T19_SUPER_OFF))
    if not m.any(): return m
    L, n = cc(m); sz = np.bincount(L.ravel())[1:]
    keep = [k + 1 for k in range(n) if sz[k] >= 0.1 * sz.max()]
    return np.isin(L, keep)


POST_BODY[6001375] = _t19_super_body
ONE_LABEL[6001375] = 'hull'   # multi-part body (hub, blade, coil, ball): one hull of the body (the user's multi-part rule)
