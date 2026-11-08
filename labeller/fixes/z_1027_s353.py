# sweep 10-27 fix for skin 353 (Sonnet revision, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s353/)
# sweep 10-27 fix2 for skin 353 ConductorScorpion, attack 0 deg and attack 90 deg (task s353: pairs 6623 hole at tick 283,
# 6631 hole at tick 327; second attempt after the review of fix.py).
# Type 6001372 Conductor002Projectile (attack, sc/effects_brawler.sc chuck_002_atk_projectile, trail chuck_002_atk_trail):
# the tan dust cloud is ONE solid puff group, but its ID-pass pixels are patchy: the puffs are stacked translucent layers
# (low coverage inside), blue trail sparks are drawn over it, the bright inner puff and the cloud at a wall (ticks 279-283) or
# at its end (ticks 323-329) are drawn by other, partly translucent draws. So the per-pixel body has notches, inner loops and
# whole missing patches (tick 325: the lower half of the cloud). The label is the cloud's own silhouette, solid:
#  1. the body, plus the tan cloud-coloured pixels of the frame inside the predicted body (grown 2 px): this brings in the
#     bright inner puff and the translucent patches (the dark background, the blue sparks and walls are not tan);
#  2. enclosed holes filled, parts under a tenth of the largest dropped, the outline closed with a disk (radius 12 px: it bridges
#     the notches and the wells around the sparks, a puff outline has no real concavity that deep);
#  3. opened with a disk (radius 6) to cut off spurs and hooks (the thin trail wisps next to the cloud), the body's own pixels
#     touching the result kept (edge), holes filled again.
# The blue sparks and the dust wisps are not labelled (rule 3): only the cloud under / around them is. Keyed to this one type id.

_T353_TYPE = 6001372
_T353_CLOSE = 12       # px: closing disk radius for the silhouette
_T353_OPEN = 6         # px: opening disk radius (spurs, hooks, thin wisps)


def _t353_disk(r):
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r + r // 2


def _t353_big(S, f):
    from scipy.ndimage import label as cc
    L, n = cc(S)
    if n > 1:
        sz = np.bincount(L.ravel())[1:]
        S = np.isin(L, 1 + np.nonzero(sz >= f * sz.max())[0])
    return S


def _t353_body(body, ctx):
    if not body.any(): return body
    from scipy.ndimage import binary_fill_holes, binary_opening, binary_closing, binary_dilation
    r = _T353_CLOSE; img = ctx['img'].astype(np.int16)
    tan = (img[..., 0] >= 105) & (img[..., 0] - img[..., 2] >= 35)
    S = body | (tan & binary_dilation(ctx['pred_body'], iterations=2) & ctx['region'])
    S = binary_fill_holes(_t353_big(S, 0.1))
    S = binary_closing(np.pad(S, r + 1), structure=_t353_disk(r))[r + 1:-r - 1, r + 1:-r - 1]
    S = binary_fill_holes(S)
    O = _t353_big(binary_opening(S, structure=_t353_disk(_T353_OPEN)), 0.3)
    if not O.any(): O = S
    O = O | (body & binary_dilation(O, iterations=1))
    return binary_fill_holes(O)


POST_BODY[_T353_TYPE] = _t353_body
