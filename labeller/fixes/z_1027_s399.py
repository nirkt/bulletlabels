# sweep 10-27 fix for skin 399 (Sonnet revision, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s399/)
# sweep 10-27 fix (2nd attempt) for skin 399 (JetpackGirlSupercellMake), task out/sweep2/haikutasks/sweep1027/s399/
# Finding pair 7529 (attack 0 deg, tick 265, ragged). Janet's 007 attack: type 6001170 (JetpackGirl007Projectile, logic
# JetpackGirlProjectile), four rounded cylinders per shot. Replaces fix.py (rejected: one merged ragged outline at 265, merged
# cylinders at 267, spurs from the splash fringes at the left, notches where cylinders touch).
# What the ID pass shows per projectile (clip janet_007_atk_projectile, sc/effects_brawler.sc), three consecutive draws:
#   draw A: the cylinder = shape 0 (translucent oval ring, coverage 20-64), shape 381 (bright core), shape 3688 (rim);
#   draw B: shape 54747, the teal / green body at the cylinder's left (frayed edge, it breaks into wisps further left);
#   draw C: shape 133, the soft white-blue wash (glow layer: rule 2, not labelled).
# The old labels came from the clip codes mixed over the neighbours' draws, hence the loops, holes and strips.
# Fix (a labels() wrapper, after the normal labelling and the clean-up): per projectile
#   - the draws are assigned to the projectile by position (the cores' centroids against the projectile's projected anchor,
#     one-to-one, Hungarian), never by the polluted baseline body;
#   - label = draw A pixels of shapes 0 / 381 / 3688 with coverage >= JG_THR (the soft fall-off of the oval's edge is halo)
#     plus the draw-B pixels (shape 54747) that belong to the same cylinder (nearest core), connected to the core, closed 2 px,
#     holes filled, then opened with a disk of radius JG_OPEN (spurs, wisps and teeth of the frayed edge are removed);
#   - the wash (shape 133) and everything else of the old label are dropped;
#   - projectiles whose centres are closer than JG_MERGE x radius (logic positions, as Skater 402 / Enrager 495: the user's rule
#     'one label while they overlap, separate once apart') share ONE label: the union of their pieces, filled and opened, on
#     the lowest object id. At tick 265 the cylinders are stacked on each other (logic distances 141-160 < 170) and the front
#     ones hide the lower parts of the others; from tick 267 on (distances >= 265) each has its own label.
#   - a projectile with no recognised cylinder (hidden, no core) keeps its old label.
#   - only the largest connected part of each label is kept (detached wisps / specks).
# Only type 6001170; only the clip names above. Other skins of this type: none seen in projtab / pages (the projectile is
# Janet's own 007 attack).
JG_T = 6001170
JG_FILE = 'sc/effects_brawler.sc'
JG_CLIPS = ('janet_007_atk_projectile', 'janet_007_atk_projectile_red')
JG_CYL = (0, 381, 3688)     # draw A shapes
JG_CORE = 381
JG_BODY = 54747             # draw B shape
JG_THR = 40                 # min coverage of the oval's translucent ring
JG_OPEN = 5                 # px, radius of the opening that removes spurs / wisps
JG_MERGE = 1.7              # x radius: overlapping projectiles = one label (same factor as Skater 402)
JG_STACK_CLOSE = 8   # px: closing of a stack of overlapping projectiles (the parts hidden under the front ones)
JG_MAXD = 30                # px: max distance projected anchor - core centroid for a match

_jg_prev_labels = labels


def _jg_disk(r):
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r + r * 0.5


def _jg_finish(m, rc=0):
    from scipy.ndimage import binary_fill_holes, binary_opening, label as cc
    if rc: m = _close(m, rc)
    m = binary_fill_holes(m)
    q = JG_OPEN + 2
    op = binary_opening(np.pad(m, q), structure=_jg_disk(JG_OPEN))[q:-q, q:-q]
    if not op.any(): return m
    L, n = cc(op)       # only the largest part: detached bits (a wisp beside the cylinder) are not the cylinder
    if n > 1: op = L == (1 + int(np.argmax(np.bincount(L.ravel())[1:])))
    return op


def labels(objs, cam, R, state=None, ended=(), img=None):
    import bulletlabels as BL, masks
    from scipy.ndimage import distance_transform_edt, label as cc
    from scipy.optimize import linear_sum_assignment
    P = _jg_prev_labels(objs, cam, R, state=state, ended=ended, img=img)
    on = [o for o in objs if o['t'] == JG_T and not o.get('ended') and o.get('anchor') is not None]
    if not on: return P
    ids, lids = P['ids'], P['lids']
    mem = []
    for o in on:
        lid = (BL._lid(o), -1, 'body')
        if lid in lids: mem.append((o, lids.index(lid)))
    if not mem: return P
    ids_h = R['id'][::2, ::2]; cv = R['cv'][::2, ::2]
    SP = _a4_shape_pixels(R, JG_FILE, JG_CLIPS)
    cyl = np.isin(SP, [s + 1 for s in JG_CYL]); teal = SP == JG_BODY + 1
    core_all = (SP == JG_CORE + 1) & (cv >= CUT)
    A = [int(d) for d in np.unique(ids_h[core_all]) if d > 0]
    if not A: return P
    cores = {d: core_all & (ids_h == d) for d in A}
    cen = {d: np.array(np.nonzero(cores[d])).mean(1)[::-1] for d in A}
    dist = {d: distance_transform_edt(~cores[d]) for d in A}
    bt = {}
    for b in [int(d) for d in np.unique(ids_h[teal]) if d > 0]:
        mb = teal & (ids_h == b); bt[b] = min(A, key=lambda d: dist[d][mb].min())
    pos = np.array([masks.project(cam, o['anchor'][None])[0][0] / 2.0 for o, _ in mem])
    cost = np.array([[np.hypot(*(pos[i] - cen[d])) for d in A] for i in range(len(mem))])
    ri, ci = linear_sum_assignment(cost)
    piece = {}      # label index -> (object, mask before the finish)
    for i, j in zip(ri, ci):
        if cost[i, j] > JG_MAXD: continue
        d = A[j]; o, k = mem[i]
        m0 = cyl & (ids_h == d)
        for b, a in bt.items():
            if a == d: m0 = m0 | (teal & (ids_h == b))
        cand = _close(m0 & (cv >= JG_THR), 2)
        L, n = cc(cand); keep = np.unique(L[cores[d]]); keep = keep[keep > 0]
        m = np.isin(L, keep)
        if m.any(): piece[k] = (o, m)
    if not piece: return P
    # stacks: projectiles closer than JG_MERGE x radius (logic centres) are one label, on the lowest object id
    rad = projtab.table()[JG_T]['radius']; ks = sorted(piece, key=lambda k: piece[k][0]['id'])
    par = {k: k for k in ks}
    def find(k):
        while par[k] != k:
            par[k] = par[par[k]]; k = par[k]
        return k
    for a in range(len(ks)):
        for b in range(a + 1, len(ks)):
            la, lb = piece[ks[a]][0]['logic'], piece[ks[b]][0]['logic']
            if np.hypot(float(la[0]) - float(lb[0]), float(la[1]) - float(lb[1])) < JG_MERGE * rad:
                par[find(ks[b])] = find(ks[a])
    stacks = {}
    for k in ks: stacks.setdefault(find(k), []).append(k)
    out = ids.copy(); share = P['share'].copy()
    for k in piece: out[out == k] = -1
    for root, mm in stacks.items():
        U = np.zeros(ids.shape, bool)
        for k in mm: U |= piece[k][1]
        m = _jg_finish(U, JG_STACK_CLOSE if len(mm) > 1 else 0); carrier = mm[0]
        t = m & (out < 0)
        out[t] = carrier; share[t] = np.maximum(cv[t] / 255.0, BL.CUT)
    share[out < 0] = 0
    P2 = dict(P); P2.update(ids=out, share=share)
    return P2
