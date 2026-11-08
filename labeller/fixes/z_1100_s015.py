# human review 2026-10-28 fix for skin 015 (Haiku fix, reviewed by the coordinator, installed 10-29; source out/sweep2/haikutasks/human1028/s015/fix3.py; tested via IDLABELS_OVERRIDE after all fixes incl. z_1099_cleanup, hence z_1100_)
# Fix 3 for skin 015 (BanditGirlDefault, ShotgunGirl, human review 2026-10-28), keyed to type 6000001 only
# (ShotgunGirlUltiProjectile: Shelly's default super; the skins that use the default super projectile take it too).
# Replaces fix2.py (install this file only; it replaces fix.py too). Findings: row 'super 45 deg', ticks 345 and 347
# (frames 297, 298) 'holes / ragged'; coordinator review of fix2: (1) discs too small, they cover only the dark core
# of each ball; (2) at 353 smoke-covered pellets get no label; (3) 349: touching discs of one object outlined as one
# chain of lobes is acceptable (touching = one label).
#
# Rule:
#  1. Ball radius. The visible ball = dark core + light-blue / white rim, out to the outer edge of the rim. Measured on
#     clear balls (radial profile, 9 balls at 347 and the same at 345, 349, 351): the core is dark out to about 4 px, the
#     rim peaks at 5-6 px and falls to the soft glow at its outer edge, half-maximum at 6.9 px (347), 7.0-7.8 px
#     (345-351). S017_R = 7.0 px (1.4 x the slot disc's radius of about 4.9 px, the old fix's size).
#  2. Balls of an object: its predicted pellet slots (unaligned, cut 0.5; this object alone). A slot is a ball when
#     (a) it is slot 3 (the main pellet, always), or (b) its core is dark blue (red channel of the mean colour within half a
#     radius below S017_CORE_R), or (c) it is hidden: less than S017_OWN of its pixels show the object's own sprite in
#     the ID pass (smoke, the explosion or the splash lies over it; 353: 2000027 slot 2, own 0.03). The bright glow slot of
#     every object at 347 (own sprite, coverage 90-160, pale core) is not a ball. Slots less than 1.5 radii apart are one
#     ball (the slots of one disc at 345; slot 1 next to slot 3 at 347).
#  3. Each ball -> one full disc of radius S017_R at its slot's centroid. Discs of one object may touch or overlap: one
#     label per object (touching = one label, the coordinator's rule for 349). No radius cap, no hull.
#  4. Ended pellets stay out: an object with logic 'finished' (reached its range or hit) gets no label (353: 2000026 and
#     2000029, finished 1, were drawn; their discs are dropped). Live pellets (finished 0) always keep their discs, also
#     under smoke. The depth test of idlabels (world geometry in front) stays, with tolerance S017_DEPTH.
#  5. An object with live balls never loses its label: if no ball survives, its slot-3 disc is kept.
#
# Checked: sw2_015 ticks 343-355 (the super row), attack rows (type 6000000) untouched. See report.md for what is left.

import numpy as _s017np
import bulletlabels as _s017_BL

S017_TYPES = {6000001}
S017_R = 7.0         # px: outer edge of the ball (core + rim), measured
S017_CORE_R = 88     # red channel of a ball's dark blue core (about 57-80 measured); glow 92-131
S017_OWN = 0.5       # share of a slot's pixels whose ID-pass sprite is the object's own (below: hidden by something drawn over)
S017_MERGE = 1.5     # centres closer than this x radius are one ball
S017_MIN = 20        # px: slots smaller than this are not balls
S017_DEPTH = 0.15    # depth tolerance of the world-occlusion test for this type (see fix2 notes)


def _s017_body(body, ctx):
    o = ctx['o']; R = ctx['R']; cam = ctx['cam']; img = ctx.get('img'); C = ctx['C']
    if o.get('finished', 0):   # its logic ended (reached range / hit): an end animation, not a live pellet
        return _s017np.zeros(body.shape, bool)
    if not ctx['pred_body'].any(): return body
    Q = _s017_BL.labels([o], cam, cut=0.5)          # this object's own prediction, per pellet slot
    qids, qlids = Q['ids'], Q['lids']
    H, W = body.shape
    yy, xx = _s017np.mgrid[0:H, 0:W]
    cand = []
    for k, l in enumerate(qlids):
        if l[0] != o['id'] or l[2] != 'body' or l[1] == -1: continue
        M = qids == k; a = int(M.sum())
        if a < S017_MIN: continue
        ys, xs = _s017np.nonzero(M); cx, cy = xs.mean(), ys.mean(); r = float(_s017np.sqrt(a / _s017np.pi))
        core = (xx - cx) ** 2 + (yy - cy) ** 2 <= (0.5 * r) ** 2
        red = float(img[core][:, 0].mean()) if (img is not None and core.any()) else 0.0
        own = float((C[M] > 0).mean())
        if not (l[1] == 3 or red < S017_CORE_R or own < S017_OWN): continue   # glow slot
        cand.append((l[1], a, cx, cy, r))
    if not cand: return body
    cand.sort(key=lambda b: (b[0] != 3, -b[1]))      # slot 3 first, then the largest
    kept = []
    for b in cand:
        if any(((b[2] - c[2]) ** 2 + (b[3] - c[3]) ** 2) ** 0.5 < S017_MERGE * max(b[4], c[4]) for c in kept): continue
        kept.append(b)
    out = _s017np.zeros((H, W), bool)
    for slot, a, cx, cy, r in kept:
        out |= (xx - cx) ** 2 + (yy - cy) ** 2 <= S017_R ** 2
    wd = R.get('wd')
    if wd is not None and o.get('anchor') is not None:
        dz = _eye_depth(cam, o['anchor'], R['meta'].get('wref', 0.0))[0]
        out &= wd >= dz - S017_DEPTH
    return out


for _t in S017_TYPES:   # replaces fix2 / fix for this type; no installed fix sets the body of 6000001
    POST_BODY[_t] = _s017_body
