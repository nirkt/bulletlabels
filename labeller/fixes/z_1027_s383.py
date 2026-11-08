# sweep 10-27 fix for skin 383 (Sonnet revision, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s383/)
# Skin 383 IceDudeToon, super 45 deg (sweep 2026-10-27, task s383, second attempt: pairs 7212, 7213, 7214; the same defect on every
# tick of the row, 361-411). Type 17000397 = IceDude006UltiArea (Ice Dude's super: a lingering zone, logic radius 1100).
#
# Defect: zone_labels takes the pixels where a zone sprite is the front-most draw, so every other draw standing in front of the
# ground cut a notch out of the zone: the bush block at the right (the outline ran along the top of the bushes), the snow mound and
# the pine at the left, loops round white dots. The user's zone rule (RULES.md, rule 7): the zone is the full drawn circle, cut ONLY
# by solid map blocks (walls, crates, barrels). Bushes, trees, snow, characters, UI and effects never cut it.
#
# Why the first attempt (fix.py) still cut the bush: it added the LOGIC disc (radius 1100) only. The zone is DRAWN larger than its
# logic disc (the sprite's edge lies at 1.12 - 1.24 x the logic radius, measured on the old label's outer edge in every tick), so
# the bush's part between the logic circle and the drawn edge (the triangle at the lower right, between the straight cut and the
# bush's right side) stayed out of the label.
#
# Fix, in three parts, all keyed to IV383_ZONES:
#  1. the logic disc minus solid blocks and other labels is added (as fix.py, as the other ZONE_DISC zones);
#  2. the ring between the logic disc and the drawn edge is added where the old label does NOT reach the drawn edge, i.e. where
#     something non-solid stands in front of the zone sprite: the old label's outermost radius per 4 degree angular bin is
#     measured; a bin is "open" (the sprite shows to the edge) when the old label covers >= IV383_DENSE of the non-solid pixels of
#     the ring just inside the edge; the other bins are covered. The edge of a covered bin is the radius interpolated (in angle)
#     between the nearest open bins on both sides (clamped to the typical edge +- IV383_CLAMP), so the label's edge continues the
#     drawn edge smoothly across the bush. Open bins get nothing new (their drawn edge is the old label's edge, unchanged).
#     (a 3x3 closing, 2 iterations, joins the 1-2 px gaps between the old label and the added ring, not over solid blocks)
#  3. old specks: pieces of the old sprite label that lie on solid blocks (white dots drawn over the walls) are removed (the blocks
#     cut the zone), then pieces under IV383_SPECK px separate from the main label are dropped; holes up to IV383_HOLE px filled.
# The label stays one solid piece in the zone's own (drawn) outline; walls and crates stay cut out.

from scipy.ndimage import binary_fill_holes as _s383_fill, label as _s383_cc, binary_closing as _s383_close

IV383_ZONES = {17000397}   # IceDude006UltiArea
IV383_HOLE = 80            # px: holes inside the label filled (not solid, not another label; the largest seen is 55 px)
IV383_SPECK = 200          # px: loose bits of the label smaller than this are dropped (white dots 30-60 px; the real strip under the blocks is 1500 px)
IV383_BIN = 4.0            # degrees: angular bins for the drawn edge
IV383_DENSE = 0.80         # old label covers at least this share of the ring inside the edge: the sprite shows to the edge
IV383_CLAMP = 0.06         # the edge of a covered bin stays within the typical edge +- this (x logic radius)
IV383_OPENTOL = 0.03       # an open bin reaches at least the typical edge minus this
IV383_RING = 0.10          # width of the ring inside the edge used for the coverage test (x logic radius)
IV383_MAXQ = 1.35          # old pixels beyond this radius (x logic radius) are not part of the ground edge (dome, effects)

_s383_zone_labels_prev = zone_labels


def _s383_edge(old, q, th, solid, taken):
    """per pixel: radius (x logic radius) of the zone's drawn edge where a bin is covered, else 0"""
    nb = int(round(360.0 / IV383_BIN))
    b = np.minimum((th / IV383_BIN).astype(int), nb - 1)
    near = old & (q <= IV383_MAXQ)
    own = np.zeros(nb)
    for i in range(nb):
        m = near & (b == i)
        if m.any(): own[i] = q[m].max()
    low = np.arange(nb) * IV383_BIN
    lowerhalf = (low >= 0) & (low < 180)                 # image y grows down: angles 0..180 are below the centre (no dome there)
    vals = own[lowerhalf & (own > 0.5)]
    if len(vals) < 5: return None
    rc = float(np.median(vals))
    ok = ~solid & ~taken
    open_ = np.zeros(nb, bool); unknown = np.zeros(nb, bool)
    for i in range(nb):
        ring = (b == i) & (q >= rc - IV383_RING) & (q <= rc - 0.02) & ok
        if ring.sum() < 12: unknown[i] = True; continue      # hidden by blocks: nothing to extend, blocks cut anyway
        open_[i] = (old & ring).sum() >= IV383_DENSE * ring.sum() and own[i] >= rc - IV383_OPENTOL
    good = open_ | unknown
    if good.all() or not good.any(): return None
    # edge radius per bin: own radius of open bins, interpolated in angle for the covered ones
    R = np.where(open_, own, np.nan)
    idx = np.nonzero(open_)[0]
    if len(idx) == 0: return None
    cen = (np.arange(nb) + 0.5)
    out = np.empty(nb)
    for i in range(nb):
        if open_[i]: out[i] = own[i]; continue
        # nearest open bins going both ways (circular)
        lo = next(k for k in range(1, nb) if open_[(i - k) % nb]); hi = next(k for k in range(1, nb) if open_[(i + k) % nb])
        r0 = own[(i - lo) % nb]; r1 = own[(i + hi) % nb]
        out[i] = r0 + (r1 - r0) * lo / float(lo + hi)
    out = np.clip(out, rc - IV383_CLAMP, rc + IV383_CLAMP)
    cov = ~open_ & ~unknown
    return out, cov, b


def zone_labels(state, cam, R, taken, own_team=None):
    """idlabels.zone_labels, with the ground disc and the drawn ring (where something stands in front of the sprite) of the
    IV383_ZONES zones added, minus solid blocks and other labels; old specks on the blocks dropped"""
    import owners
    out = _s383_zone_labels_prev(state, cam, R, taken, own_team)
    if R.get('wd') is None: return out
    items = []
    for it in state.get('items', []):
        iid, did = int(it[0]), int(it[1])
        if did not in IV383_ZONES: continue
        Z = owners.zone(did)
        if not Z or owners.area(did)['kind'] != 'zone': continue
        rz = float(Z['logic_row'].get('Radius') or 300)
        x, y = int(it[3]), int(it[4])
        cx, cy = _proj(cam, x, y); dx = abs(_proj(cam, x + rz, y)[0] - cx); dy = abs(_proj(cam, x, y + rz)[1] - cy)
        if dx < 1 or dy < 1: continue
        items.append((iid, did, int(it[2]), cx, cy, dx, dy))
    if not items: return out
    solid = _zs_solid(cam, R)
    H, W = R['wd'].shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    Q = [np.sqrt(((xx - cx) / dx) ** 2 + ((yy - cy) / dy) ** 2) for (_, _, _, cx, cy, dx, dy) in items]
    D = np.stack([np.where(q <= 1, q, np.inf) for q in Q])
    best = np.argmin(D, axis=0); anyin = np.isfinite(D.min(axis=0))
    for k, (iid, did, own, cx, cy, dx, dy) in enumerate(items):
        q = Q[k]
        owned = anyin & (best == k) & np.isfinite(D[k])          # one owner per pixel: the nearer circle
        old0 = out[iid][0] if iid in out else np.zeros((H, W), bool)
        old = old0 & ~solid                                      # the blocks cut the zone (specks on walls go)
        ring = np.zeros((H, W), bool)
        if len(items) == 1:
            th = np.degrees(np.arctan2((yy - cy) / dy, (xx - cx) / dx)) % 360.0
            e = _s383_edge(old, q, th, solid, taken)
            if e is not None:
                er, cov, b = e
                ring = (q > 1) & (q <= er[b]) & cov[b]
        part = (owned | ring) & ~solid & ~taken
        new = old | part
        if new.any():                                            # 1-2 px gaps between the old label and the added ring
            new = new | (_s383_close(new, structure=np.ones((3, 3), bool), iterations=2) & ~solid & ~taken)
        if new.any():                                            # loose bits under IV383_SPECK px are dropped
            L, n = _s383_cc(new)
            if n > 1:
                sz = np.bincount(L.ravel()); keep = sz >= IV383_SPECK; keep[0] = False
                if keep.any(): new = keep[L]
        if new.any():                                            # small holes, not solid and not another label
            f = _s383_fill(new) & ~new
            if f.any():
                L, n = _s383_cc(f); sz = np.bincount(L.ravel()); small = sz < IV383_HOLE; small[0] = False
                new = new | (small[L] & ~solid & ~taken)
        if new.any(): out[iid] = (new, did, own)
        else: out.pop(iid, None)
    return out
