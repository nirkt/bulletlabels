# sweep 10-27 fix for skin 374 (Sonnet revision, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s374/)
# Skin 374 ControllerOnceupon (sweep 2026-10-27, pair 7042: attack 90 deg, tick 305, too_small; second attempt after the first
# fix.py was rejected for new spurs / specks / nested loops). Type 6000780 Nani006Projectile (ControllerProjectile, the attack
# projectile of every Controller skin, so the change is shared by all of them; the rules are colour-adaptive, see below).
# Cause: the puffs are drawn as bright clip body (code 1, opaque) plus bright translucent glow lobes (code 2, coverage 30-65) and
# stacked puffs overdraw each other, so the body rule left parts out (too small) and, where puffs overlap, produced nested loops
# and thin crumbs (the first owner claims the whole stack).
# Fix (POST_BODY, own outline filled solid, never loses the projectile):
#  - the puff's silhouette = pixels of its OWN prediction (pred_any, the game's per-projectile partition of the stack, so
#    neighbouring puffs get clean separate pieces; grown 1 px) that are puff pixels: opaque clip body (cv >= 200), or clip
#    body / glow of coverage >= 24 that is bright (>= 75 % of the puff's own brightness) and has the puff's own colour
#    (blue/green ratio within 0.15 of the opaque body's: drops the water-tinted halo and the dark teal halo);
#  - close 2 px and fill holes (eye, highlights), then open with a disk of radius 5 (rounds off the flame tips / tail stems that
#    reached into the detached flame tail above the puff), falling back to 4..1 when that would take away more than 10 % of the
#    area (thin slices of a stack), keep parts of >= 25 % of the largest (and 40 px), fill holes;
#  - nothing usable -> the old body is kept.
# Verified on the 90 deg row (ticks 305 to 315) and the 0 deg row (265 to 275), same type.

from scipy.ndimage import label as _s374_label, binary_opening as _s374_open
from scipy.ndimage import binary_fill_holes as _s374_fill, binary_dilation as _s374_dil, binary_closing as _s374_clo


def _s374_disk(r):
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return x * x + y * y <= r * r + r // 2


def _s374_nani006(body, ctx):
    pa = ctx['pred_any']
    if not body.any() or not pa.any():
        return body
    img = ctx['img']; C = ctx['C']; cv = ctx['cv']; reg = ctx['region']
    im = img.astype(np.int16); v = im.max(-1)
    solid = (C == 1) & (cv >= 200)
    p90 = np.percentile(v[pa], 90)
    rs = solid & pa
    if rs.sum() < 20:   # a puff drawn almost only as glow (stacked puffs): its own brightest pixels give the colour
        rs = pa & (v >= 0.95 * p90)
        if rs.sum() < 20:
            return body
    gg = np.maximum(im[..., 1], 1).astype(np.float32)
    ratio = im[..., 2] / gg
    chroma = np.abs(ratio - np.median(ratio[rs])) <= 0.15
    thr = 0.75 * p90
    puff = solid | ((((C == 1) & (cv >= 24)) | (((C == 2) | (C == 4)) & (v >= thr))) & chroma)
    m = (_s374_dil(pa, iterations=1) & puff) | (body & pa)
    m = _s374_fill(_s374_clo(m, iterations=2))
    n0 = int(m.sum())
    if n0 < 40:
        return body
    out = m
    for r in range(5, 0, -1):
        o2 = _s374_open(m, structure=_s374_disk(r))
        if o2.sum() >= 0.9 * n0:
            out = o2
            break
    L, n = _s374_label(out)
    if n > 1:
        sz = np.bincount(L.ravel())[1:]
        out = np.isin(L, 1 + np.nonzero(sz >= max(40, 0.25 * sz.max()))[0])
    out = _s374_fill(out) & reg
    return out if out.sum() >= 40 else body


POST_BODY[6000780] = _s374_nani006
