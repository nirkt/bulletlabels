# sweep 10-27 fix for skin 094 (Sonnet revision, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s094/)
# Skin 094 CrabDefault, Crab super (CrabUltiProjectile, type 6001482), fix2 (replaces fix.py, which was rejected).
# Two causes, both only for this type:
# 1) The bubble is drawn translucent (coverage under CUT=128 on the lavender / cyan body sprites), so its label came out as
#    broken fragments with spurs (super 45 deg tick 365). Lower coverage cut, as Willow's attack / Beamer's super: FADE_CUT.
#    A lower cut also lets in faint pixels of OTHER bubbles' burst puddles and rings that the ID pass hands to this object
#    (tick 353: a speck on the lavender puddle; tick 365: the teal ring on the green puddle, puddle tail, left speck). Those
#    are end animations / leftovers of ended bubbles (the user's rule 2 and 4: never labelled).
# 2) POST_BODY: keep only the connected parts that lie on the bubble's own predicted body (the logic projectile's position);
#    then close the gaps and fill holes of what is left, so the visible part of the bubble (the lavender orb is mostly hidden
#    behind the burst puddle at tick 365) is ONE solid piece in its own outline. Nothing kept (offset prediction) = unchanged
#    body, so a bubble never loses its label.
import numpy as _np
FADE_CUT[6001482] = 32
_CR_NEAR = 3      # half-res px: a part counts as the bubble's own when mostly within this distance of the predicted body
_CR_FRAC = 0.5
_CR_GONE = 0.4   # drawn fraction of the predicted body under which a finished bubble counts as burst
def _fix_6001482(body, ctx):
    from scipy.ndimage import label as _cc, binary_fill_holes as _bfh
    if not body.any(): return body
    pb = ctx['pred_body']
    if not pb.any(): return body
    near = _dilate(pb, _CR_NEAR)
    L, n = _cc(body, structure=_np.ones((3, 3)))
    keep = _np.zeros_like(body); nk = 0
    for k in range(1, n + 1):
        m = L == k
        if (m & near).sum() >= _CR_FRAC * m.sum(): keep |= m; nk += 1
    if not keep.any(): return body
    # a bubble whose logic has finished and of which less than 40% of the predicted body is still drawn has burst: what is
    # left is a sliver of the burst fade (the bubble hidden behind its own puddle, tick 365) = an end animation (rules 2, 4):
    # no label. Intact finished bubbles (ticks 353, 359) keep theirs.
    if ctx['o'].get('finished') and keep.sum() < _CR_GONE * pb.sum(): return _np.zeros_like(body)
    out = keep
    if nk > 1:   # fragments of one translucent bubble: join them (gaps <= 2 px), never beyond the predicted surroundings
        out = out | (_close(keep, 2) & _dilate(near, 1))
    out = _bfh(out) & (_dilate(near, 2) | out)
    # thin spurs and the little flame / sparkle dots on the bubble's tail (not body): opened away. The opening is the
    # strongest one that keeps most of the label, so a thin visible crescent (bubble hidden behind a burst puddle) survives;
    # edge pixels the opening nibbled stay (within 1 px of the opened body)
    from scipy.ndimage import binary_opening as _bo
    for it, fr in ((2, 0.8), (1, 0.6)):
        op = _bo(out, structure=_np.ones((3, 3), bool), iterations=it)
        if op.sum() >= fr * out.sum():
            L2, n2 = _cc(op, structure=_np.ones((3, 3)))
            if n2 > 1:
                sz = _np.bincount(L2.ravel())[1:]; op = _np.isin(L2, 1 + _np.nonzero(sz >= 0.2 * sz.max())[0])
            out = out & _dilate(op, 1)
            break
    return out
POST_BODY[6001482] = _fix_6001482
