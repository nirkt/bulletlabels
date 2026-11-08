# sweep 10-27 fix for skin 141 (Haiku fix, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s141/)
# WeaponThrower SupercellMake (Bronson005, skin 141), sweep 2026-10-27 task s141: pair 2308 (attack 0 deg, tick 269) and
# pair 2311 (attack 90 deg, tick 309). The thrown ring's label (type 6001232; its hook in py/fixes/a4_agent4.py is kept as
# it is) was a hoop with closed red loops inside it: the hoop's hollow centre (the white sparkle over it at 269, the blue
# inner wall and the brawler's cape seen through it at 309) is not in the ID-pass body, so the label had holes. Rule 1
# (filled solid, one piece) and the user's shape decision (the object's own outline filled, holes filled): the closed holes
# of the ring's label are filled, inside the ring's predicted region only. Also drops the 1-3 px crumbs left beside the
# main part (specks: parts under 25 px and under a tenth of the largest part). Only type 6001232 is touched.
from scipy.ndimage import binary_fill_holes as _s141_fill, label as _s141_cc
_s141_prev_6001232 = POST_BODY[6001232]
def _s141_fix_6001232(body, ctx):
    b = _s141_prev_6001232(body, ctx)
    if not b.any(): return b
    b = _s141_fill(b) & ctx['region']
    L, n = _s141_cc(b)
    if n > 1:
        sz = np.bincount(L.ravel())[1:]
        b = np.isin(L, 1 + np.nonzero((sz >= 25) | (sz >= 0.1 * sz.max()))[0])
    return b
POST_BODY[6001232] = _s141_fix_6001232
