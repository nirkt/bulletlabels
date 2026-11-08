# hosted review 2026-10-29 fix (Haiku fix, reviewed by the coordinator, installed 10-29 LOCAL ONLY; source out/sweep2/haikutasks/site1029/s048/fix3.py; tested via IDLABELS_OVERRIDE after all fixes incl. z_1100_*, hence z_1101_)
# Skin 048 BlowerDefault, super 45 deg (type 6000732 = BlowerUltiProjectile, clip gale_def_ulti_projectile), coordinator review
# of fix2 (2026-10-29): fix2's hull of the bright core left out the lower-left half of the wall (ticks 345-349: a diagonal edge
# through the band) and its lower-right edge was stepped. Fix3:
#  1. the band = ALL pixels of the wall: opaque (coverage >= CUT) shape-63 pixels (the wall's puffs), including the darker cyan
#     rim (part of the wall, not a soft glow). The pieces are joined to the bright near-white core (min(R, G, B) >= F3_CORE)
#     across small gaps, and then generously (F3_JOIN px each side), so the lower-left end is in. Detached loose round puffs and
#     the translucent blue clouds above-left of the band are not joined to the core and stay out;
#  2. ONE convex hull over the band;
#  3. the hull is cut ONLY where a bush / crate is in front (draws that write depth, dmask 1, not blended) and where olive /
#     green glow lies (G - B >= F3_GREEN). Gale's disc and the yellow rays are outside the band, so they stay out.
#  4. the largest remaining piece is kept (no speckles).
# Single owner: the first super object of the tick that has a body carries the label, the others are empty.
# Only type 6000732 (fix.py and fix2.py are kept in the folder, not used).

import numpy as np
from scipy.ndimage import label as _f3_cc, binary_dilation as _f3_dil

F3_TYPE = 6000732
F3_SWF = 'sc/effects_brawler.sc'
F3_CLIPS = ('gale_def_ulti_projectile', 'gale_def_ulti_projectile_red')
F3_SHAPE = 64            # shape 63 + 1: the wall's puffs
F3_CORE = 170            # min(R, G, B) of the bright near-white core of the wall
F3_JOIN = 6              # px (each side): wall pieces with gaps up to 2 x this are one band (the lower-left end included)
F3_GREEN = 60            # G - B from which a pixel is green / olive glow
F3_PROP_PX = 15000       # px (frame resolution): a depth-writing draw this big or less is a prop (bush, crate), not the floor

_f3_prev = POST_BODY.get(F3_TYPE)
_f3_state = {'R': None, 'owner': None, 'mask': None}


def _f3_band_mask(ctx):
    R = ctx['R']
    if _f3_state['R'] is R:
        return _f3_state['mask']
    cv = R['cv'][::2, ::2]; img = ctx['img']
    SP = _a4_shape_pixels(R, F3_SWF, F3_CLIPS)
    opaque = (SP == F3_SHAPE) & (cv >= CUT)
    core = opaque & (img.min(-1) >= F3_CORE)
    mask = np.zeros(opaque.shape, bool); wall_px = np.zeros(opaque.shape, bool)
    if core.any():
        # the wall's opaque puffs joined to the bright core across small gaps (the darker rim is joined through them)
        L, n = _f3_cc(_f3_dil(opaque, iterations=2), structure=np.ones((3, 3)))
        cnt = np.bincount(L[core], minlength=n + 1); cnt[0] = 0
        A = opaque & np.isin(L, np.nonzero(cnt)[0][1:])
        # the pieces of the band joined generously (the lower-left end, pieces apart by a few px)
        L2, n2 = _f3_cc(_f3_dil(A, iterations=F3_JOIN), structure=np.ones((3, 3)))
        cnt2 = np.bincount(L2[core], minlength=n2 + 1); cnt2[0] = 0
        band = A & (L2 == int(np.argmax(cnt2)))
        wall_px = band
        ys, xs = np.nonzero(band)
        if len(ys) >= 3:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(np.stack([xs, ys], 1).astype(np.float64))
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            gy, gx = np.mgrid[y0:y1, x0:x1]
            inside = np.ones(gx.shape, bool)
            for a, b, c in hull.equations:          # outside a facet: a*x + b*y + c > 0
                inside &= (a * gx + b * gy + c) <= 1e-9
            mask[y0:y1, x0:x1] = inside
    # cut ONLY where a bush / crate is IN FRONT of the wall: the front-most draw is world geometry (depth-writing, not blended)
    # and small (a prop, not the floor: the big floor draws are behind the wall and stay)
    ids = R['id'][::2, ::2]; cntd = np.bincount(ids.ravel())
    occ = [d for d, dr in R['draws'].items() if dr.get('dmask') and not dr.get('blend') and d < len(cntd) and 0 < cntd[d] <= F3_PROP_PX]
    if occ and mask.any():
        mask &= ~np.isin(ids, occ)
    # ... and where olive / green glow lies (not the wall)
    im = img.astype(np.int16)
    mask &= ~((im[..., 1] - im[..., 2] >= F3_GREEN) & (im[..., 1] >= im[..., 0] - 10))
    if mask.any():
        L, n = _f3_cc(mask, structure=np.ones((3, 3)))
        cnt = np.bincount(L.ravel()); cnt[0] = 0
        mask = L == int(np.argmax(cnt))
    _f3_state.update(R=R, owner=None, mask=mask)
    return mask


def _fix_blower_super_wall3(body, ctx):
    import bulletlabels as BL
    if _f3_prev is not None:
        body = _f3_prev(body, ctx)
    if not body.any():
        return np.zeros_like(body)
    if _f3_state['owner'] is None or _f3_state['R'] is not ctx['R']:
        _f3_band_mask(ctx)
    if _f3_state['owner'] is None:
        _f3_state['owner'] = BL._lid(ctx['o'])          # the first object of this tick with a body owns the whole wall
    if BL._lid(ctx['o']) != _f3_state['owner']:
        return np.zeros_like(body)
    return _f3_state['mask'].copy()


POST_BODY[F3_TYPE] = _fix_blower_super_wall3
