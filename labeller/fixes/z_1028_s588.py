# Roller ice streak, skin 588 RollerGoodRandoms (Roller005Projectile, type 6000910), revision 3 (round 3, Sonnet).
# Replaces fix.py / fix2.py (install this file only). Installed after z_1024_roller588.py: its POST_BODY hook and its 'hull'
# ONE_LABEL entry are left in place for the objects this file does not rebuild (fallback); the labels of the rebuilt
# flames are replaced here.
#
# THE USER (10-28): "separate, but the labels arent quite accurate": each projectile = its own ice crystal outline, filled
# solid, one part, soft glow / sparkles out, no gap and no overlap at the seam, never on the brawler.
#
# How the two flames are told apart: from the client's own draws, not from geometry. Each flame's clip is drawn as one run of
# consecutive draw calls (draw ids rise in draw order): [translucent under-layers (shape 0 / 1, coverage 10-20 %), then its
# OPAQUE crystal facets (shape 11837, coverage 255), one draw per facet batch]. The next flame starts with its own
# translucent layer, so a run is cut where a translucent draw follows opaque ones (or a second big translucent body follows a
# first one). Measured on every tick of sw2_588 (265-271, 305-315): the two flames' opaque facet draws are two disjoint id runs
# (e.g. tick 305: 83..96 and 100..113), i.e. the draws give a clean per-object split of the crystal facets. A run is matched
# to a flame object by distance of its facets' centroid to the flame's projected anchor (an injective matching, objects that
# are not drawn any more (finished / ended) get no run: ended rule). A tick where only one flame is drawn (271, 315) gives
# one run, matched to the flame that is still alive.
#
# What is a flame's crystal: (a) its opaque facet pixels; (b) the crystal parts the ID pass cannot show because a translucent
# sprite is drawn over them (the 4-point sparkles / twinkles of the clip, the flame's own translucent body layer between the
# facets): pixels of those draws near the flame's facets (<= S588_RMAX px) whose frame colour is crystal (G >= S588_GMIN
# and B >= 180, or a saturated blue B >= S588_BLUE within S588_RB px of a facet: the dark facets; the glow around the crystal
# is darker, B < 190); (c) closed (S588_CLOSE), spikes of the sparkles opened off (S588_OPEN), holes filled. The translucent
# outer fringe / glow of the body layer (measured: coverage 10-20 %, colour (35..65, 70..120, 150..185)) is not taken.
# Character model draws (runtime texture MODEL_TEX: the brawler's body in front of the flame) are never labelled, so the
# crystal is never extended over the brawler's own pixels; crystal drawn over the brawler's clothes is crystal and stays.
# The streak shared between the two flames: facets keep the flame whose draws they are; the pixels no facet shows (under
# sparkles, gaps) go to the flame whose facets are nearest, so there is no overlap; then the union of both labels is closed
# and the remaining gap pixels are shared the same way, so there is no gap at the seam. One part per flame (largest part).
#
# Only type 6000910 (skin 588). No other skin uses it (projectiles_skin row Roller005Projectile, referenced only by skin 588).
import itertools
S588_T = 6000910
S588_FILE = 'sc/effects_brawler.sc'
S588_CLIPS = ('stu_005_atk_projectile', 'stu_005_atk_projectile_red')
S588_OPAQUE_CV = 200      # median coverage from which a draw of the clip is an opaque facet draw
S588_BODY_PX = 150        # translucent draw with at least this many clip pixels = the flame's body layer (starts a new run)
S588_MIN_O = 40           # a run needs at least this many opaque facet pixels (half-res px)
S588_MATCH_R = 70.0       # half-res px: max distance of a run's facet centroid from its flame's anchor
S588_GMIN = 125           # frame colour of a crystal pixel behind a translucent sprite: G >= this and B >= 180 ...
S588_BLUE = 235           # ... or B >= this (the dark blue facets) near a facet
S588_RMAX = 12            # half-res px: max distance of such a pixel from the flame's facets
S588_RB = 20               # half-res px: max distance for the dark-blue criterion
S588_CLOSE = 3            # px: closing of the crystal
S588_OPEN = 2             # px: opening (the sparkles' spikes) -- facets are never opened away
S588_GAP = 3              # px: closing of the union of both flames (the seam), gap pixels shared by nearest label
ONE_LABEL.pop(S588_T, None)
_s588_prev = labels


def _s588_disk(r):
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r + 0.5


def _s588_close(m, r):
    from scipy.ndimage import binary_closing
    p = r + 1
    return binary_closing(np.pad(m, p), _s588_disk(r))[p:-p, p:-p]


def _s588_open(m, r):
    from scipy.ndimage import binary_opening
    p = r + 1
    return binary_opening(np.pad(m, p), _s588_disk(r))[p:-p, p:-p]


def _s588_runs(ids, cv, SP):
    """the clip's draws grouped into per-flame runs: list of (opaque draw ids, mask of their opaque pixels (cv >= 128))"""
    m = SP > 0
    seq = []
    for d in np.unique(ids[m]):
        mm = m & (ids == d)
        seq.append((int(d), bool(np.median(cv[mm]) >= S588_OPAQUE_CV), int(mm.sum())))
    runs = []; cur = None; seen_o = False; seen_body = False
    for d, opq, n in seq:
        if cur is None or (not opq and (seen_o or (n >= S588_BODY_PX and seen_body))):
            cur = []; runs.append(cur); seen_o = False; seen_body = False
        cur.append((d, opq, n))
        if opq: seen_o = True
        elif n >= S588_BODY_PX: seen_body = True
    out = []
    for r in runs:
        od = [d for d, opq, n in r if opq]; td = [d for d, opq, n in r if not opq]
        if sum(n for d, opq, n in r if opq) < S588_MIN_O: continue
        out.append((od, np.isin(ids, od) & (cv >= 128)))
    return out


def labels(objs, cam, R, state=None, ended=(), img=None):
    import masks, bulletlabels as BL
    from scipy.ndimage import binary_fill_holes, label as cc, distance_transform_edt
    P = _s588_prev(objs, cam, R, state=state, ended=ended, img=img)
    allf = [o for o in objs if o['t'] == S588_T]
    if not allf: return P
    alive = sorted([o for o in allf if not o.get('ended') and o.get('drawn', True) is not False], key=lambda o: o['id'])
    ids = P['ids'].copy(); share = P['share'].copy(); lids = list(P['lids'])
    H, W = ids.shape
    ids_h = R['id'][::2, ::2]; cv = R['cv'][::2, ::2]
    SP = _a4_shape_pixels(R, S588_FILE, S588_CLIPS)
    runs = _s588_runs(ids_h, cv, SP)
    # runs -> alive flames (injective, minimum total distance of the facets' centroid to the flame's anchor)
    anc = []
    for o in alive:
        p = masks.project(cam, np.asarray(o['anchor'], float)[None])[0][0] / 2.0; anc.append(p)
    cen = []
    for od, m in runs:
        ys, xs = np.nonzero(m); cen.append(np.array([xs.mean(), ys.mean()]))
    best = None
    k = min(len(runs), len(alive))
    for rs in itertools.permutations(range(len(runs)), k):
        for os_ in itertools.permutations(range(len(alive)), k):
            dd = [float(np.hypot(*(cen[r] - anc[j]))) for r, j in zip(rs, os_)]
            if any(x > S588_MATCH_R for x in dd): continue
            sc = (-len(dd), sum(dd))
            if best is None or sc < best[0]: best = (sc, list(zip(rs, os_)))
    match = dict((j, r) for r, j in best[1]) if best else {}
    # labels of the flames this file rebuilds (matched) or that are not alive any more (ended rule) are removed; an alive flame
    # without a run keeps its previous label
    rebuilt = [o for j, o in enumerate(alive) if j in match] + [o for o in allf if o not in alive]
    ks_all = []
    for o in rebuilt:
        lid = (BL._lid(o), -1, 'body')
        if lid in lids: ks_all.append(lids.index(lid))
    if ks_all:
        clear = np.isin(ids, ks_all); ids[clear] = -1; share[clear] = 0
    if not match:
        P2 = dict(P); P2.update(ids=ids, share=share, lids=lids); return P2
    # draws of the character models: the brawler is never part of the flame
    cdraws = [d for d, dr in R['draws'].items() if dr.get('th') == MODEL_TEX]
    model = np.isin(ids_h, cdraws) if cdraws else np.zeros((H, W), bool)
    od_all = set(d for od, m in runs for d in od)
    fd = [int(d) for d in np.unique(ids_h[SP > 0]) if int(d) not in od_all]
    foot = np.isin(ids_h, fd) & ~model
    free = (ids < 0) & ~model
    js = sorted(match)                       # alive flame indices that have a run
    Ms = [runs[match[j]][1] & free for j in js]
    n = len(Ms)
    dist = np.stack([distance_transform_edt(~m) for m in Ms]); own = np.argmin(dist, 0); dmin = dist.min(0)
    if img is not None:
        bright = (img[..., 1] >= S588_GMIN) & (img[..., 2] >= 180); blue = img[..., 2] >= S588_BLUE
    else:
        bright = blue = np.zeros((H, W), bool)
    Ks = []
    for q in range(n):
        oth = np.any([Ms[i] for i in range(n) if i != q], 0) if n > 1 else np.zeros((H, W), bool)
        c = Ms[q] | (foot & free & (own == q) & ((bright & (dmin <= S588_RMAX)) | (blue & (dmin <= S588_RB))))
        c = (_s588_close(c, S588_CLOSE) & free & ~oth) | Ms[q]
        c = binary_fill_holes(c) & free & ~oth
        c = (_s588_open(c, S588_OPEN) | Ms[q]) & free
        Ks.append(c)
    if n > 1:                                # no overlap: a pixel two flames claim goes to the flame whose facets are nearer
        for q in range(n):
            oth = np.any([Ks[i] for i in range(n) if i != q], 0)
            Ks[q] = Ks[q] & ~(oth & (own != q))
    # no gap: the union closed, the new pixels shared by the nearest label
    U = np.any(Ks, 0)
    G = binary_fill_holes(_s588_close(U, S588_GAP)) & free & ~U
    if G.any() and n > 0:
        dk = np.stack([distance_transform_edt(~k_) for k_ in Ks]); ok = np.argmin(dk, 0)
        Ks = [k_ | (G & (ok == q) & (dk.min(0) <= S588_GAP + 1)) for q, k_ in enumerate(Ks)]
    for q, j in enumerate(js):
        c = Ks[q]
        L, nn = cc(c, structure=np.ones((3, 3)))
        if nn > 1:
            sz = np.bincount(L.ravel())[1:]; c = L == (1 + int(np.argmax(sz)))
        oth = np.any([Ks[i] for i in range(n) if i != q], 0) if n > 1 else np.zeros((H, W), bool)
        c = binary_fill_holes(c) & free & ~oth
        lid = (BL._lid(alive[j]), -1, 'body')
        if lid not in lids: lids.append(lid)
        kk = lids.index(lid)
        c = c & (ids < 0)
        ids[c] = kk; share[c] = np.maximum(cv[c] / 255.0, CUT)
        Ks[q] = c
    P2 = dict(P); P2.update(ids=ids, share=share, lids=lids)
    return P2
