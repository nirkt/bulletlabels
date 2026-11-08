# AI-review batch 2026-11-01 fix (Haiku fix, reviewed by the coordinator, installed 11-01 LOCAL ONLY; source out/sweep2/haikutasks/siteai/icedude_ropedude/fix.py)
# icedude_ropedude fix (AI-review findings 2026-11-01): skins 273, 282, 317, 354, 383, 457 (task folder icedude_ropedude/).
# Exec'd in idlabels' namespace (IDLABELS_OVERRIDE). Keyed to explicit type ids only. Findings and decisions: report.md.
#
# 1. STREAK HULL, 282 Buzz003Projectile 6000988 (both rows: attack 0 = cyan slab, attack 90 = cyan slab moving down) and
#    317 Buzz005Projectile 6000990 (both rows: rainbow streak). A streak split into pieces = ONE convex hull over the
#    object's own draws (shape codes 0..2), plus the over-sprite pixels that touch them (attack 0 / 317) or all of them
#    inside its predicted body (282 attack 90, flight angle 0..180). Bubbles beside the streak stay out. A finished
#    object (range / hit flag set) keeps its label as before. Same idea as z_1102_s182.py (Buzz 182).
# 2. ICE CLUSTER HULL, 354 Buzz006Projectile 6000992 (wave-kind: all shards of one shot share one body label): shards that
#    touch (gap <= 3 half-res px) are one piece: ONE convex hull per touching group (labels() wrapper, after all per-type
#    rules). Groups under 30 px (sparkle specks) are left unchanged.
# 3. SOLID CONE + BALL, 457 Lou007Projectile 6000859 (both rows): one projectile = cone + ball: seam closed (gaps <= 3 px),
#    holes filled, specks under 20% of the largest piece dropped. (A two-piece hull was tried and rejected: it merged
#    neighbouring projectiles.)
# 4. CLEAN BODY, 273 Lou004Projectile 6000853 (attack 0): the lower projectile (2000011) is alive at t273 (it hits at t275);
#    its label took in the white hit flash of the two projectiles that hit the wall before it (spikes, a speck). Keep
#    its own shape codes (body and cheese), close and fill (hidden parts under the flash are filled), drop specks.
# 5. PREDICTED BODY CLIP, 383 Lou006Projectile 6000857 (attack 0): a strip of the already-hit cylinder was in the lower
#    cylinder's label (AI 383 att0 t273). The label is kept within the object's predicted body (3 px).

from scipy.ndimage import label as _cc_rd, binary_fill_holes as _fh_rd, binary_closing as _bc_rd, binary_dilation as _bd_rd
import bulletlabels as _BL_rd

RD_STREAK = (6000988, 6000990)
RD_CLUSTER = (6000992,)
RD_CONE_BALL = (6000859,)
RD_LOU004 = (6000853,)
RD_LOU006 = (6000857,)


def _rd_components(m, frac):
    """the connected pieces of m at least frac of the largest one"""
    L, n = _cc_rd(m)
    if n == 0: return m
    sz = np.bincount(L.ravel())[1:]
    return np.isin(L, 1 + np.nonzero(sz >= frac * sz.max())[0]) & m


def _rd_streak_hull(body, ctx):
    o = ctx['o']
    if not body.any() or o.get('finished'): return body   # a finished shot is not a streak: its label is left as it is
    C = ctx['C']; region = ctx['region']; pb = ctx['pred_body']
    own = body & (C <= 2)
    if not own.any(): return body
    ov = body & (C == 4) & pb
    if o['t'] == 6000988 and 0 <= float(o.get('angle', 0)) % 360 < 180:
        # Buzz003 attack 90 (flight angle 0..180): the slab's upper part is drawn with over-sprites that do not touch
        # its own draws; they are part of the slab here (the AI finding 282 att90 was not checked before this rule)
        core = own | ov
    else:
        core = own | (ov & _bd_rd(own, iterations=3))      # over-sprites only where they touch the object's own draws
    core = _bc_rd(core, iterations=3) | core                 # close the gaps between the pieces
    return hull_mask(_rd_components(core, 0.05))             # the hull of the object's own drawn pixels (no prediction clip: it notched the slab)


def _rd_solid(body, ctx):
    # one projectile = its cone and its ball: the seam between them is closed (gaps <= 3 px), holes filled, specks dropped
    if not body.any(): return body
    b = np.pad(body, 4)
    b = _bc_rd(b, iterations=3)[4:-4, 4:-4] | body
    b = _fh_rd(b)
    return _rd_components(b, 0.2)


def _rd_lou004(body, ctx):
    if not body.any(): return body
    C = ctx['C']; region = ctx['region']
    core = body & (C <= 2)
    if not core.any(): return body
    core = _bc_rd(np.pad(core, 3), iterations=2)[3:-3, 3:-3] | core
    core = _fh_rd(core) & region
    return _rd_components(core, 0.05)


def _rd_clip_pred(body, ctx):
    if not body.any(): return body
    near = _bd_rd(ctx['pred_any'], iterations=3)
    return body & near


for _t_rd in RD_STREAK:
    POST_BODY[_t_rd] = _rd_streak_hull
for _t_rd in RD_CONE_BALL:
    POST_BODY[_t_rd] = _rd_solid
for _t_rd in RD_LOU004:
    POST_BODY[_t_rd] = _rd_lou004
for _t_rd in RD_LOU006:
    POST_BODY[_t_rd] = _rd_clip_pred


# ---- 2. ice cluster: one hull per touching group of shards (labels() wrapper, runs after the per-type rules)
# Type 6000992 is a wave-kind type: all shards of one shot share ONE body label (the wave group's), so the groups are
# found on the pixels of that label, by proximity (RD_CLUSTER_GAP), and each group gets its own hull.
_rd_labels_prev = labels
RD_CLUSTER_GAP = 3       # half-res px: shards closer than this are one cluster
RD_CLUSTER_MIN = 30      # half-res px: smaller groups (sparkle specks) are left as they are


def labels(objs, cam, R, state=None, ended=(), img=None):
    P = _rd_labels_prev(objs, cam, R, state=state, ended=ended, img=img)
    if not any(o['t'] in RD_CLUSTER for o in objs): return P
    lids = P['lids']; ids = P['ids'].copy(); share = P['share'].copy()
    ks = []
    for o in objs:
        if o['t'] not in RD_CLUSTER or o.get('ended'): continue
        lid = (_BL_rd._lid(o), -1, 'body')
        if lid in lids: ks.append(lids.index(lid))
    ks = sorted(set(ks))
    if not ks: return P
    U = np.isin(ids, ks)
    if not U.any(): return P
    Lu, nu = _cc_rd(_bd_rd(U, iterations=RD_CLUSTER_GAP))   # groups: the union grown by RD_CLUSTER_GAP
    for g in range(1, nu + 1):
        own = U & (Lu == g)
        if own.sum() < RD_CLUSTER_MIN: continue
        lab = int(ids[own][0])
        H = hull_mask(own)
        sel = H & ((ids == -1) | np.isin(ids, ks))
        ids[sel] = lab
        share[H] = np.maximum(share[H], _BL_rd.CUT)
    P = dict(P); P['ids'] = ids; P['share'] = share
    return P
