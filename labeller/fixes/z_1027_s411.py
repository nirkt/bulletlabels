# sweep 10-27 fix for skin 411 (Sonnet revision, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s411/)
# sweep 10-27 fix (2nd version) for skin 411 BeamerSCStore (task out/sweep2/haikutasks/sweep1027/s411/, finding pair 7746 tick 269, too_big)
# Beamer006Projectile 6001308 (the green ball of attack 0 deg; the same type is the attack 90 deg shot): its clip sprite at some
# frames carries a pointed tail of the flame / smoke next to the ball (clip shape 0 pixels, the same sprite as the ball), and the
# label followed it (tick 269: a lobe about 17 px long and 10 px tall into the teal smoke wisp; attack 90 deg ticks 309-315: the
# flame spike at the top). The first version (opening radius fixed at 5 px) cut only the thin part of the tail; the lobe is wider.
# Now the opening radius follows the ball's size: r = 0.6 x the largest inscribed radius of the label (r 4..10 px), so a part
# narrower than about 1.2 x the ball's own half-width (the tail / spike) is cut, and the small ball frames (265/267, 305/307,
# whose label is a clean ellipse) change by 1-2 %. The opened shape is restored by S411_KEEP px so the ball's own outline (rim, tips)
# stays as drawn. Applied only when the ball keeps at least 60 % of its label (a projectile never loses its label).
import numpy as _s411_np
S411_BEAMER006 = 6001308
S411_FRAC = 0.6          # opening disk radius = this x the label's largest inscribed radius
S411_RMIN, S411_RMAX = 4, 10
S411_KEEP = 1            # px: the opened shape is restored by this much along the ball's own outline


def _s411_beamer006(body, ctx):
    from scipy.ndimage import binary_opening, binary_dilation, distance_transform_edt
    if not body.any(): return body
    rho = distance_transform_edt(_s411_np.pad(body, 1)).max()
    r = int(min(S411_RMAX, max(S411_RMIN, round(S411_FRAC * rho))))
    yy, xx = _s411_np.ogrid[-r:r + 1, -r:r + 1]
    disk = (xx * xx + yy * yy) <= r * r
    op = binary_opening(body, structure=disk)
    if op.sum() < 0.6 * body.sum(): return body                        # never lose the projectile's label
    keep = binary_dilation(op, structure=_s411_np.ones((2 * S411_KEEP + 1, 2 * S411_KEEP + 1), bool))
    return body & keep


POST_BODY[S411_BEAMER006] = _s411_beamer006
