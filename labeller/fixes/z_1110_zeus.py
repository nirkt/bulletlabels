# 175 RocketGirlZeus super (type 6000066 Brock007UltiProjectile; skin 175 only): owner 2026-11-05 "they're all ragged and some extra spark
# effects also get labeled even though they're not bullets". Each super rocket is drawn as: the cyan crystal (no clip shape: 'none'), a
# pale-yellow jagged patch on it and the big lightning bolt that follows it (both the SAME sprite shape of the rocket's trail clip, which
# is also the shape of the THIN JAGGED CYAN SQUIGGLES / sparks that fly beside the rocket; these squiggles are not bullets). The label
# (z_1102_s175brock_b / z_1107_brock section 2 / z_1109_brock section 4: crystal + patch + bolt body, bolt kept: the owner asked for it)
# also took the squiggles that touch the patch (t347, t363, t377 ...). Fix: of the label's trail-shape pixels only those are kept that
# lie within S175_NEAR px of the crystal (the patch) or of the thick part of the trail sprites (opening by a disc of S175_CORE px: the
# bolt is 8-15 px wide, a squiggle 2-4 px), i.e. thin jagged pieces away from the crystal and the bolt are dropped. The crystal ('none'
# pixels) is untouched, holes are filled. The global smoothing step (fix.py) then smooths the outline.
import numpy as _s175
from scipy.ndimage import binary_fill_holes as _s175_fill, label as _s175_cc, binary_opening as _s175_open, binary_dilation as _s175_dil
import projtab as _s175_pt

S175_TYPE = 6000066
S175_NEAR = 2          # px: trail pixels this close to the bolt core stay
S175_CLOSE = 2         # the crystal is closed by this many px (the patch on it is part of the rocket)
S175_CORE = 3.2        # radius of the disc whose opening gives the thick part (bolt) of the trail sprite
_s175_prev = POST_BODY.get(S175_TYPE)


def _s175_trail_px(ctx):
    t = ctx['t']; T = _s175_pt.table()[t]; clips = {}
    for k in ('blue', 'red'):
        if T.get(k): clips.setdefault(T['file'], []).append(T[k])
    for f, n in _trail_clips(t): clips.setdefault(f, []).append(n)
    out = _s175.zeros(ctx['region'].shape, bool)
    for f, cl in clips.items(): out |= _a4_shape_pixels(ctx['R'], f, tuple(sorted(set(cl)))) > 0
    return out


def _s175_rocket(body, ctx):
    if _s175_prev is not None: body = _s175_prev(body, ctx)
    if not body.any(): return body
    tr = body & _s175_trail_px(ctx); crystal = body & ~tr
    if not tr.any() or not crystal.any(): return body
    k = int(_s175.ceil(S175_CORE)); y, x = _s175.ogrid[-k:k + 1, -k:k + 1]; disc = (x * x + y * y) <= S175_CORE ** 2 + 1e-6
    core = _s175_open(_s175.pad(tr, k + 1), structure=disc)[k + 1:-(k + 1), k + 1:-(k + 1)]       # the thick trail sprite: the bolt
    cr = _s175_fill(_close(crystal, S175_CLOSE))                                                   # the crystal incl. the patch drawn on it
    out = crystal | (tr & (cr | _dilate(core, S175_NEAR)))
    L, n = _s175_cc(out, structure=_s175.ones((3, 3)))        # drop pieces that lost the crystal
    if n > 1:
        keep = _s175.unique(L[crystal & (L > 0)]); out = _s175.isin(L, keep[keep > 0])
    out = _s175_fill(_close(out, 1)) & (out | _dilate(out, 1))
    return out if out.sum() >= 0.3 * body.sum() else body


POST_BODY[S175_TYPE] = _s175_rocket
