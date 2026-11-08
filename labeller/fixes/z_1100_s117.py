# human review 2026-10-28 fix for skin 117 (Haiku fix, reviewed by the coordinator, installed 10-29; source out/sweep2/haikutasks/human1028/s117/fix2.py; tested via IDLABELS_OVERRIDE after all fixes incl. z_1099_cleanup, hence z_1100_)
# Skin 117 (CrowRankedProgrsesionSkinLevel9_01), human review 2026-10-28, FLAGGED "holes / ragged" on row "attack 90 deg"
# at ticks 307 and 309. Coordinator review (2026-10-28, after fix.py): the LEFT blade at 307 (head only, body under the cyan
# flash missing, a fragment at the flash edge) and at 309 (crown / top half cut off) are not accepted. Binding rule: a projectile
# partly covered by a flash keeps its FULL shape.
#
# Type 6000329 = Crow010Projectile (CrowProjectile, clip crow_010_atk_projectile / _red, trail crow_010_atk_trail). The attack 0 deg
# and attack 90 deg rows of skin 117 both use it (and the other Crow ranked skins that name it: see report).
#
# Step 1, cleaning (as fix.py): effect colours out of the body (B-G > 120 and B > 150: the deep-blue arrow-head and the violet
#   trail beam), and spurs more than 2 px from the thick body out (the diagonal streak, the flash-edge arm).
# Step 2, full silhouette from the state (the template): the object's predicted body (ctx['pred_body']) is the blade placed by its
#   state position and direction, not by the ID pass, so it is not hidden by the flash. Checked on the clean ticks 305-317 of
#   skin 117: the label covers 95-99 % of the prediction there (the prediction is the blade's own silhouette, e.g. 2000012 at 313:
#   513 predicted px, 511 labelled, the same outline). Where the label covers < 80 % of the prediction (the blade is partly hidden),
#   the label becomes the prediction plus the cleaned body within 2 px of it. Pixels of the body outside that 2 px band are
#   dropped (the flash arm, the crescent tip, the streak remnants). Clean blades (coverage >= 80 %) are not touched.
# Measured on skin 117: only the left blade at 307 (coverage 26 %) and at 309 (56 %) are filled. Every other object keeps its
# fix.py outline.
import numpy as np
from scipy.ndimage import binary_opening, distance_transform_edt, binary_dilation

FILL_COVER = 0.8   # apply the prediction fill where the label covers less of the predicted body than this


def _crow010_clean(body, ctx):
    img = ctx['img']
    if img is not None:   # (a) effect colours: deep blue arrow-heads and the violet trail beam
        im = img.astype(np.int16)
        body = body & ~((im[..., 2] - im[..., 1] > 120) & (im[..., 2] > 150))
    core = binary_opening(body, structure=np.ones((3, 3), bool))   # the thick body
    if not core.any(): return body
    dist = distance_transform_edt(~core)
    return body & ~((dist > 2) & body & ~core)                    # (b) spurs more than 2 px from the body


def _crow010_fill(body, ctx):
    body = _crow010_clean(body, ctx)
    pb = ctx['pred_body']
    if body is None or pb is None or pb.sum() < 50: return body
    cov = (body & pb).sum() / float(pb.sum())
    if cov >= FILL_COVER: return body                               # clean blade: unchanged
    band = binary_dilation(pb, iterations=2)
    return (body & band) | pb                                       # the full state-placed silhouette, no stray pixels


POST_BODY[6000329] = _crow010_fill
