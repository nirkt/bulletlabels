# Fire Dude Fantasy 416: flame + super orb, full solid body (Haiku task 15, reviewed)
# Task 15 (Haiku), skin 416 FireDudeFantasy (brawler Amber): fix for the blue flame attack and the blue orb super.
# Types: 6000837 FireDude005Projectile (attack, clip amber_005_atk_projectile), 6000838 FireDude005UltiProjectile (super,
# clip amber_005_ulti_projectile). User decision: the visible attack body, one clean solid label per projectile; trails,
# the ground fire cloud, the super's ground shadow, particles and the landing burst stay out.
# Cause: both are translucent (blended draws, coverage about 30-130) and made of several clip shapes; the old label kept
# only the full-coverage parts of the 'body' class: the flame's left half and the orb's dark base and sphere were missing,
# and the outline was ragged with specks.
# Fix: the attack's own clip shapes (shape ids of its clip export, not its trail's), coverage >= 16, as connected parts;
# only the parts that hold a bright body pixel (coverage >= 128) are kept (the super's ground shadow shares shape 0 with the
# orb but is faint and separate); then one solid part (closing, holes filled, specks under 20% of the main part dropped).
from scipy.ndimage import binary_fill_holes, label as _cc

_T15_CLIPS = {6000837: ('amber_005_atk_projectile', 'amber_005_atk_projectile_red'),
              6000838: ('amber_005_ulti_projectile', 'amber_005_ulti_projectile_red')}


def _t15_body(body, ctx):
    t, R, region, cv = ctx['t'], ctx['R'], ctx['region'], ctx['cv']
    SP = _a4_shape_pixels(R, 'sc/effects_brawler_amber.sc', _T15_CLIPS[t]) > 0   # its clip's own shapes (not its trail's)
    M = region & SP & (cv >= 16)
    if M.any():
        L, n = _cc(M); keep = np.unique(L[M & (cv >= 128)]); keep = keep[keep > 0]
        m = np.isin(L, keep)
    else:
        m = M
    m = _close(m, 1) & region if m.any() else m
    m = binary_fill_holes(m) & region
    if m.any():
        L, n = _cc(m)
        if n > 1:
            sz = np.bincount(L.ravel())[1:]; m = np.isin(L, 1 + np.nonzero(sz >= 0.2 * sz.max())[0])
    return m


for _t in _T15_CLIPS: POST_BODY[_t] = _t15_body
