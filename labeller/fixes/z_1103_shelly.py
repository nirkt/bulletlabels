# AI-review batch 2026-11-01 fix (Haiku fix, reviewed by the coordinator, installed 11-01 LOCAL ONLY; source out/sweep2/haikutasks/siteai/shelly/fix2.py)
# Shelly skins 223, 240, 280, 341, 349, 364 (AI-reviewer findings 2026-11-01, task shelly). Keyed to Shelly's own projectile type ids.
#
# Owner rule for Shelly (binding, 2026-11-01): NO hull for her bullets; each bullet (object = one pellet) in its own clean,
# filled outline. Outlines torn into fragments, ragged, self-crossing or merged over several pellets are real defects: one
# solid piece per pellet. Her super is judged by the same rules.
#
# What this file does (per type id, all six skins listed in shelly/report.md):
#  1. Per-pellet closing (SH_TYPES = 223 attack 6000002, 364 attack 6000004, 240 super 6000007, 280 super 6000011,
#     349 attack 6000014): each pellet's own body is closed across the gaps between its own sprites (SH_CLOSE px), holes
#     and interior arcs are filled, and specks are dropped. Additions must lie within SH_ADD px of the pellet's own
#     prediction (so an explosion or smoke beside it is not taken), and never outside its neighbourhood (SH_GROW px). A
#     pellet keeps its largest part. No convex hull anywhere.
#  2. Ended pellets are unlabelled (SH_DROP = the six types): an object with logic 'finished' gets no label. The pixels are
#     cleared after ownership is decided (in a labels() wrapper), so live neighbours do not take the explosion beside them.
#     This is the same rule as z_1100_s015.py; for 341 pods it is the only change (the ended pods of tick 317).
#  3. The installed stack merge of z_1027_s364 (one label over touching pellets of type 6000004, which also filled the
#     purple orb at t307) is switched off (SH364_TYPE = -1). Every pellet keeps its own label.
#
# Not changed here: the shape of the 341 pods (type 6000012). Their drawn pixels do not separate per pod in the clusters
# (ticks 267, 305), so no closing gives a clean per-pod outline: see shelly/report.md (needs_user).
#
# Other skins that use the same type ids are re-checked by the coordinator (not verified here).

import numpy as _shnp
from scipy.ndimage import binary_fill_holes as _shfill, label as _shcc

SH_TYPES = {6000002, 6000004, 6000007, 6000011, 6000014}          # per-pellet closing (shape)
SH_DROP = {6000002, 6000004, 6000007, 6000011, 6000012, 6000014}   # ended pellets unlabelled (all six types)
SH_CLOSE = {6000002: 4, 6000004: 3, 6000007: 3, 6000011: 3, 6000014: 3}   # px: iterations of closing (gap bridge)
SH_GROW = {6000002: 4, 6000004: 4, 6000007: 5, 6000011: 5, 6000014: 4}   # px: its own neighbourhood around its prediction
SH_ADD = 3           # px: pixels added by the closing / hole fill lie within this of the pellet's own prediction
SH_SPECK = 0.15      # a part under this share of the object's largest part is a speck
SH_SPECK_PX = 12     # px: and smaller than this

SH364_TYPE = -1      # switch off the installed stack merge of z_1027_s364 (it gives one label to several pellets)


def _sh_pellet(body, ctx):
    if not body.any():
        return body
    # this pellet's own neighbourhood: its predicted label grown by SH_GROW px (the wide search region takes the neighbouring
    # pellets and the loose sprites beside them); the pellet's own drawn pixels are always kept
    reg = (_dilate(ctx['pred_any'], SH_GROW.get(ctx['t'], 4)) & ctx['region']) | body
    t = ctx['t']
    cl = _close(body, SH_CLOSE.get(t, 2)) & reg              # bridge the gaps between this pellet's own sprites
    m = _shfill(body | cl)                                    # holes and interior arcs: solid
    # what is added must be this pellet's own: a pixel of one of its own draws, or inside its prediction. The fill of an
    # enclosed hole takes in an explosion or smoke drawn among the pellets otherwise (280, tick 353)
    near_pred = _dilate(ctx['pred_any'], SH_ADD)              # added pixels: within SH_ADD px of this pellet's prediction
    m = (m & near_pred) | body
    L, n = _shcc(m)
    if n > 1:
        sz = _shnp.bincount(L.ravel())[1:]
        big = sz.max()
        keep = [k + 1 for k in range(n) if sz[k] == big or (sz[k] >= SH_SPECK * big and sz[k] >= SH_SPECK_PX)]
        m = _shnp.isin(L, keep)
    return m


for _t in SH_TYPES:
    POST_BODY[_t] = _sh_pellet


# ended pellets: their pixels are cleared AFTER the ownership is decided. Dropping them inside POST_BODY would free their
# pixels, and the live neighbours would then take the explosion / smoke beside them (280 and 240 super, tick 353)
_sh_prev_labels = labels


def labels(objs, cam, R, state=None, ended=(), img=None):
    P = _sh_prev_labels(objs, cam, R, state=state, ended=ended, img=img)
    fin = {BL._lid(o) for o in objs if o['t'] in SH_DROP and o.get('finished', 0)}
    if not fin: return P
    ids, lids, share = P['ids'], P['lids'], P['share']
    drop = [k for k, l in enumerate(lids) if l[0] in fin and l[2] == 'body']
    if drop:
        m = _shnp.isin(ids, drop); ids[m] = -1; share[m] = 0
    return P

# ===================== fix2 (2026-11-01, coordinator review of shelly): one solid disc per pellet =====================
# The coordinator's decision: the fragmented pellets of the dense clusters (341 attack, 349 attack, 364 attack residual,
# 240 super object 2000032) are ragged; each pellet gets ONE solid shape built from the pellet itself, the default-skin style
# (z_1100_s015.py: one disc per pellet). A convex hull was the coordinator's first option, but the owner's binding Shelly rule
# is 'NO hull for her bullets', so the shape here is a disc, not a hull. The predictions do not separate the pellets of
# one shot (364: every pellet of a shot has the same prediction, about 500 px; 240: each object's prediction is the whole
# spray), so the centre and the size come from the pellet's own drawn label (after fix.py): centre = centroid of its
# drawn pixels, radius = the radius of a disc with the same area. Overlaps: the front pellet (smaller eye depth) owns the
# shared pixels, one owner per pixel, no merged outline. A disc is cut by world geometry nearer than the pellet (depth test,
# as in idlabels). A pellet with fewer than FX2_MIN drawn pixels keeps its fix.py label (never lost).
# Types (final): 6000012 (341 attack) and 6000014 (349 attack). Rejected by eye: 6000004 (364 attack: the discs do not fit the
# spread-out pellets) and 6000007 (240 super: the discs overshoot the leaf groups and leave 3x3 marks); both keep fix.py.

FX2_TYPES = {6000012, 6000014}      # final: 341 pods and 349 pellets only (364 and 240 rejected: see report)
FX2_MIN = 20         # px: fewer drawn pixels than this: the pellet keeps its fix.py label
FX2_R_SCALE = 1.0    # the disc has the area of the pellet's drawn label; scale if a radius check says otherwise

_fx2_prev_labels = labels


def labels(objs, cam, R, state=None, ended=(), img=None):
    P = _fx2_prev_labels(objs, cam, R, state=state, ended=ended, img=img)
    live = [o for o in objs if o['t'] in FX2_TYPES and not o.get('finished', 0)]
    if not live: return P
    ids, lids, share = P['ids'], P['lids'], P['share']
    H, W = ids.shape
    yy, xx = np.mgrid[0:H, 0:W]
    wd = R.get('wd'); wref = R['meta'].get('wref', 0.0)
    cand = []   # (eye depth, label index, disc mask)
    for o in live:
        lid = (BL._lid(o), -1, 'body')
        if lid not in lids: continue
        kb = lids.index(lid)
        M = ids == kb
        a = int(M.sum())
        if a < FX2_MIN: continue
        ys, xs = np.nonzero(M); cx, cy = xs.mean(), ys.mean()
        r = float(np.sqrt(a / np.pi)) * FX2_R_SCALE
        disc = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
        dz = 0.0
        if wd is not None and o.get('anchor') is not None:
            dz = _eye_depth(cam, o['anchor'], wref)[0]
            disc &= wd >= dz - DEPTH_TOL              # world geometry nearer than the pellet hides it
        cand.append((dz, kb, disc))
    if not cand: return P
    own = np.full((H, W), -1, np.int32); zb = np.full((H, W), np.inf)
    for dz, kb, disc in sorted(cand, key=lambda c: c[0]):      # front pellets first: a pixel keeps the first (front) owner
        upd = disc & (zb > dz)
        zb[upd] = dz; own[upd] = kb
    old = {kb: ids == kb for _, kb, _ in cand}                 # the pellets' fix.py shapes
    for dz, kb, disc in cand:
        if not (own == kb).any():                              # never lose a pellet: fully covered by front pellets (no
            ys, xs = np.nonzero(old[kb])                       # visible own pixel), it keeps a 3x3 mark at its centre
            cy, cx = int(round(ys.mean())), int(round(xs.mean()))
            fb = np.zeros((H, W), bool); fb[max(cy - 1, 0):cy + 2, max(cx - 1, 0):cx + 2] = True
            own[fb] = kb
    mine = np.isin(ids, [c[1] for c in cand])
    ids[mine] = -1; share[mine] = 0                            # the pellets' old shapes are replaced by their discs
    w = (own >= 0) & (ids < 0)                                 # other labels' pixels stay with them
    ids[w] = own[w]; share[w] = BL.CUT
    return P
