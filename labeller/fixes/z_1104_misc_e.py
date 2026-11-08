# installed 11-01 from out/sweep2/haikutasks/siteai2/misc_e/fix.py (2nd AI sweep: 587 arrowhead/crystal whole; 549 ellipse part DISABLED)
# misc_e fix (AI-review sweep 2, 2026-11-01). exec'd in idlabels' namespace after all installed fixes (IDLABELS_OVERRIDE).
# Keyed to explicit type ids only. Skins: 549 KickerDudeMake (Fang004Projectile 6001109), 587 DoorManStarrForce
# (DoorMan003Projectile 6001327; also changes its attack-0 ticks 265 / 267, which carry the same type: arrowhead covered whole).

# --- 549 KickerDudeMake, Fang004Projectile (6001109), attack 0 tick 265 and attack 90 tick 305 (AI flag 'holes / ragged',
# note: "outline is lumpy with notches and covers the brawler's smoke next to the blade instead of the blade itself").
# Owner rule (2026-10-28 brief): the kicked disc is the projectile, the brawler's teal mist / smoke is never labelled.
# The disc and the mist share one draw and one sprite shape, and their colours overlap, so no draw / shape / colour
# split separates them. The label is therefore the disc itself: a filled ellipse fitted to the visible disc on the two
# ticks that were wrong (265: disc standing on its edge, 305: disc lying flat on the ground). Other ticks are unchanged.
# Parameters: centre x, centre y, semi-major, semi-minor (frame px), major-axis angle (deg, image coordinates).
FANG004_DISC = {265: (634.5, 249.5, 21.0, 7.5, 112.7),
                305: (593.0, 276.0, 25.0, 10.0, 14.0)}


def _fang004_disc(body, ctx):
    if not body.any(): return body
    p = FANG004_DISC.get(ctx['state']['tick'])
    if p is None: return body
    cx, cy, a, b, phi = p
    u = np.array([np.cos(np.radians(phi)), np.sin(np.radians(phi))]); n = np.array([-u[1], u[0]])
    Y, X = np.mgrid[0:body.shape[0], 0:body.shape[1]]; dx = X - cx; dy = Y - cy
    return (dx * u[0] + dy * u[1]) ** 2 / a ** 2 + (dx * n[0] + dy * n[1]) ** 2 / b ** 2 <= 1


# NOT INSTALLED (coordinator 11-01): per-tick fitted ellipses of one capture, not a rule from the data
# POST_BODY[6001109] = _fang004_disc


# --- 587 DoorManStarrForce, DoorMan003Projectile (6001327), attack 90 tick 305 (AI flag 'too small / cut off', note: "the
# label covers only the lower part of the small green arrowhead, its wings stay outside"). Also 309-315 (crystal top and
# bottom were short) and attack-0 265 / 267 (same type). Verified on the capture: the
# arrowhead's upper rows (y 330-332 of the 305 frame, bright green, 4-6 px wide) were not in the body (the body was 16 px,
# the lower tip only). Owner rule: the whole visible projectile inside the label. The bright green pixels of the arrowhead
# that touch the body (within 3 px) join it; the existing hull (ONE_LABEL 'hull', misc_d) then covers the whole arrowhead.
# Dim halo pixels (G < 170) stay out.
from scipy.ndimage import binary_dilation as _bdil


def _starr_arrow_wings(body, ctx):
    if not body.any() or ctx.get('img') is None: return body
    im = ctx['img'].astype(int)
    bright = (im[..., 1] > 170) & (im[..., 1] - im[..., 0] > 60)
    near = _bdil(body, structure=np.ones((7, 7), bool))
    return body | (bright & near)


POST_BODY[6001327] = _starr_arrow_wings
