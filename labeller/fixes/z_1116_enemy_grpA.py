# grpA enemy fix (2026-11-07): to be installed as py/fixes/z_1116_enemy_grpA.py. Prefix _eA_.
# Jessie 008 bubble (MechanicSB, skin 377, type 6000186 Jessie008Projectile1): the own (blue) bubble is one solid label (z_1027_s377 closes
# the rim by 2 px and fills what is enclosed). The ENEMY (red) bubble's rim is drawn with wider gaps on the first ticks of a shot
# (3 px gap at the top in sw 377 enemy tick 305; at tick 265 / 267 the glow class covers the rim at the bottom, gaps up to ~10 px), so
# the 2 px closing never closes the ring, the fill finds nothing and the label stays a thin rim + the orange object (enemycmp 377:
# first tick of attack 0 / attack 90, area 583 against 1287 own).
# Fix: only when the installed rule gained (almost) nothing, close the rim with a growing radius (3 .. 7 px) until the fill encloses a
# ring-sized, compact area (0.65 .. 1.0 of the body's bounding box: a closed ellipse fills 0.785; solidity >= 0.96 against its convex hull);
# else the convex hull of the rim (the bubble is round) when that is ring-sized too; then the installed rule runs on that body.
# Own (blue) shots: the installed rule always fills (gain >> 20 px) on every own tick of the 377 capture, so this never fires there.
_eA_prev = POST_BODY.get(6000186)
_eA_GAIN = 20       # px: the installed rule counts as "filled" when it added at least this many pixels
_eA_RADII = (3, 4, 5, 6, 7)
_eA_SOLID = 0.96


def _eA_bubble(body, ctx):
    from scipy.ndimage import binary_fill_holes
    res = _eA_prev(body, ctx)
    if not body.any() or (res & ~body).sum() >= _eA_GAIN: return res
    ys, xs = np.nonzero(body); cap = (ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)
    new = None
    for r in _eA_RADII:
        filled = binary_fill_holes(_close(body, r)) & ctx['region']
        if (filled & ~body).sum() >= _eA_GAIN and 0.65 * cap <= filled.sum() <= cap and filled.sum() >= _eA_SOLID * hull_mask(filled).sum():
            new = filled; break
    if new is None:
        hl = hull_mask(body) & ctx['region']
        if (hl & ~body).sum() >= _eA_GAIN and 0.65 * cap <= hl.sum() <= cap: new = hl
    if new is None: return res
    return _eA_prev(new | body, ctx) | new


if _eA_prev is not None: POST_BODY[6000186] = _eA_bubble
