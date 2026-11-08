# AI-review batch 2026-11-01 fix (Haiku fix, reviewed by the coordinator, installed 11-01 LOCAL ONLY; source out/sweep2/haikutasks/siteai/luchador/fix.py)
# Fix task luchador (AI-reviewer findings 2026-11-01, skins 475 500 510 533 553 577 609). Written by the Haiku fix agent.
#
# Real defects fixed here (solid label, one clean outline, per the owner rules "holes / ragged" and "hidden parts filled in"):
#  - 6000401 Primo005Projectile (skin 500, attack 0 deg): the blob's translucent left side is cut at coverage 128, so the
#    label had notches, a zigzag left edge and interior contour lines (ticks 265-277; the flag was t275). Fix: close the body
#    (disk radius 3 px, half resolution), fill its holes, and add only pixels of the body's own draws (no smoke, no other draw).
#  - 6000408 Primo012Projectile (skin 577): the red / pink beam is many thin streaks, and the label has zigzag notches between
#    them (attack 0 deg ticks 265-277, attack 90 deg ticks 303-317). Owner rule: thin lines / streaks = ONE convex hull of the
#    whole attack (ONE_LABEL 'hull'), the same mechanism as Yeti's claw streaks in idlabels.py.
#
# Not changed: skin 609 (6000409) fist: the closing that fixes the 500 notches also pulls the yellow spark bar lying over the
# fist into the label (ticks 275, 309, 313), a regression, so 609 is left as it is (see report.md, not_fixed).
# Not changed (see report.md): spawn-tick objects at the brawler's hand / ring (6000399 at t269, 6000402 at t269 and t309,
# 6000405 at t269, 6000406 at t309): the object is the same projectile as the blast one or two ticks later; whether its
# first-tick sprite is a flash is a user decision.
import numpy as np


def _luch_fill_own(body, ctx, r=3):
    """the body closed (disk radius r px, half resolution), its holes filled; added pixels only inside the body's own draws
    and the object's region (no smoke, no other draws). Concavities and notches of the solid outline are filled."""
    from scipy.ndimage import binary_closing, binary_fill_holes
    if not body.any(): return body
    ids = ctx['ids_h']; reg = ctx['region']; p = r + 2
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]; st = (yy * yy + xx * xx) <= r * r
    cl = binary_closing(np.pad(body, p), structure=st)[p:-p, p:-p]
    cl = binary_fill_holes(cl)
    dr = np.unique(ids[body]); dr = dr[dr > 0]
    # only concavities of the outline: pixels inside the body's convex hull, so the closing adds no outward spurs (the
    # 609 spark-bar case was not cured by this restriction, so 609 has no fix; see the header)
    add = cl & ~body & reg & np.isin(ids, dr) & hull_mask(body)
    return body | add


_luch_prev_6000401 = POST_BODY.get(6000401)


def _luch_fix_6000401(body, ctx):
    if _luch_prev_6000401 is not None: body = _luch_prev_6000401(body, ctx)
    return _luch_fill_own(body, ctx, r=3)


POST_BODY[6000401] = _luch_fix_6000401
# skin 577 beam (6000408): one solid convex hull over the whole attack (owner rule: thin lines / streaks)
ONE_LABEL[6000408] = 'hull'
