# misc5 fix (round 5, 2026-11-06 evening): owner-flagged pages 216 CrowWhite, 452 Mender199Gem, 590 PowerLevelerBuzz.
# Exec'd in idlabels' namespace (installed as z_1115_misc5, after z_1113_misc4 / z_1111_extra3). Unique prefix _x5_.
import numpy as _x5_np

# ---- 590 PowerLevelerBuzz (Surge008Projectile 6000806 = the flying fist, Surge008Projectile2 6000807 = the bloom; only this skin) ----
# Owner (page 590, attack 90 t313): "should not be labelled". Checked on the capture: the fist (6000806) flies and is labelled up to the tick
# its logic projectile is finished (o['finished'] > 0: t311 -> t313); on exactly that tick the game spawns Surge008Projectile2, the burst:
# a round white bloom with a cyan rim, then sparks (t313-t319), the end effect of the fist. All that was labelled there (z_1113_misc4) was a
# speck / worm of the swirl inside the bloom. Rule (owner: hit / end / landing effects are NEVER labelled): Surge008Projectile2 carries no
# label at all (its body is only ever that end bloom; the flying fist is always Projectile 6000806, which is not touched).
_X5_T590B = 6000807


def _x5_590b(body, ctx):
    return _x5_np.zeros_like(body)


POST_BODY[_X5_T590B] = _x5_590b


# ---- shared helpers ----
from scipy.ndimage import (gaussian_filter as _x5_gauss, binary_fill_holes as _x5_fill, binary_dilation as _x5_dil,
                           binary_erosion as _x5_ero, label as _x5_cc)


def _x5_disk(r):
    y, x = _x5_np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r + 0.5


def _x5_close(m, r):
    c = _x5_np.pad(m, r + 2)
    return _x5_ero(_x5_dil(c, _x5_disk(r)), _x5_disk(r))[r + 2:-r - 2, r + 2:-r - 2]


# ---- 452 Mender199Gem super: the landing WAVE (Mender002UltiProjectile 6001759, the shards; only this skin) ----
# Owner (page 452, super 45 deg, t355 and t357): "the full wave should be labeled". When a shard lands (o['finished'] > 0) the game plays
# the big white-blue wave of the super: a fan of translucent blue water with a white foam edge, drawn from the shards' own atlas (the same
# draws that show the flying shards; most of it is not a clip shape the object tracker knows, the landed shards are not 'drawn' any more,
# so the labels only ever held the landed shard, trimmed by z_1111_extra3). The wave is part of the super: its own outline, solid.
# Rule (labels() wrapper, own type only): while an object of the type has landed, the wave = the pixels the front-most draw shows from
# the draws that carry the type's sprites (the ID pass; so things in front of it, characters / walls, stay out) with coverage >=
# _X5_WCV (the soft glow beyond the sheet is not taken); closed (hidden parts / flame-shaped gaps between the foam streaks), holes
# filled, rounded. The wave keeps its label while it is visibly drawn (round 5b: its fade too, until under _X5_WALIVE px of sheet remain). Each piece of at least _X5_WMIN px becomes ONE label (the lowest-id landed shard's label; a flying shard's label stays
# its own and is only absorbed when it has landed, finished > 0, or lies inside the wave).
_X5_T452U = 6001759
_X5_WCV = 40
_X5_WMIN = 3500
_X5_WCLOSE = 9
_X5_WALIVE = 8000      # px (half res) of the wave at coverage >= _X5_WCV: the wave is labelled while it is drawn, incl. its fade (peak ~33000, t361 ~27500); below ~a quarter of the peak it is gone (round 5b: owner t361 'missing label')
_X5_WOPEN = 6          # thin necks / small bits of other effect sprites joined to the sheet (the little fish / bubble) are opened away
_x5_prev_labels_452u = labels


def _x5_452_wave(P, objs, R, cv):
    import bulletlabels as BL
    os_ = [o for o in objs if o['t'] == _X5_T452U]
    if not any((o.get('finished') or 0) > 0 for o in os_): return P
    ids_h = R['id'][::2, ::2]
    seen = set(); sel = set()
    for o in os_:
        if o['own'] in seen: continue
        seen.add(o['own'])
        C = sprite_pixels(o['t'], o['own'], R)
        sel |= {int(d) for d in _x5_np.unique(ids_h[C > 0]) if d > 0}
    if not sel: return P
    Wm = _x5_np.isin(ids_h, list(sel)) & (cv >= _X5_WCV)
    if not Wm.any(): return P
    if Wm.sum() < _X5_WALIVE: return P
    Wc = _x5_fill(_x5_close(Wm, _X5_WCLOSE)) & (_x5_dil(Wm, iterations=_X5_WCLOSE))
    L, n = _x5_cc(Wc)
    if not n: return P
    sz = _x5_np.bincount(L.ravel())
    ids, lids, share = P['ids'].copy(), list(P['lids']), P['share'].copy()
    lk = {}
    for o in sorted(os_, key=lambda q: q['id']):
        lid = (BL._lid(o), -1, 'body')
        lk[o['id']] = lid
    ks = [lids.index(l) for l in lk.values() if l in lids]
    landed = [o for o in sorted(os_, key=lambda q: q['id']) if (o.get('finished') or 0) > 0]
    for j in range(1, n + 1):
        if sz[j] < _X5_WMIN: continue
        comp = L == j
        over = [k for k in ks if (ids[comp] == k).any()]
        if over: tk_ = min(over)
        else:
            lid = lk[landed[0]['id']]
            if lid not in lids: lids.append(lid)
            tk_ = lids.index(lid)
        free = (ids < 0) | _x5_np.isin(ids, ks)
        s = comp & free
        so = _x5_np.pad(s, _X5_WOPEN + 2); so = _x5_dil(_x5_ero(so, _x5_disk(_X5_WOPEN)), _x5_disk(_X5_WOPEN))[_X5_WOPEN + 2:-_X5_WOPEN - 2, _X5_WOPEN + 2:-_X5_WOPEN - 2]
        if so.sum() >= 0.6 * s.sum():
            L2, n2 = _x5_cc(so)
            if n2 > 1: so = L2 == (_x5_np.bincount(L2.ravel())[1:].argmax() + 1)
            s = s & _x5_dil(so, iterations=2)
        # one silhouette: every own label inside the wave goes into it
        for k in over: ids[ids == k] = -1
        s = _x5_fill(s) & free
        sm = _x5_gauss(s.astype(float), 1.6) > 0.5
        sm &= _x5_dil(s, iterations=1)
        sm |= _x5_ero(s, iterations=1)
        sm &= free
        ids[sm] = tk_; share[sm] = 1.0
    P2 = dict(P); P2['ids'] = ids; P2['lids'] = lids; P2['share'] = share
    return P2


def labels(objs, cam, R, state=None, ended=(), img=None):
    P = _x5_prev_labels_452u(objs, cam, R, state=state, ended=ended, img=img)
    return _x5_452_wave(P, objs, R, R['cv'][::2, ::2])


# ---- 216 CrowWhite (Crow003Projectile 6000317, the three daggers; only this skin) ----
# Owner (page 216, attack 0, t265): "missing label". Checked on the capture: at the shot's first tick the three daggers are stacked flush on
# top of each other (their predicted bodies touch within 2 px, 64 px of contact, against 0-3 px from the next tick on, where the blades
# are separate) under their green skull puffs (the clip's 'over' sprites, code 4). z_1113_misc4 left bites where a puff covers a blade / handle,
# and the middle dagger was only a thin sliver (268 px against 340): it looked unlabelled. Owner rule 223: bullets that touch = ONE label,
# hidden parts kept. Rule (labels() wrapper, own type only): daggers whose predicted bodies touch (dilated by 2 px they share >= _X5_216T px)
# are ONE bullet silhouette: the union of their labels + the clip body / puff sprite pixels (codes 1 and 4, coverage >= _X5_216CV) within 3 px of
# the predicted bodies, closed (disc _X5_216R), holes filled, rounded, never over another type's label. Separate daggers are not touched.
_X5_T216 = 6000317
_X5_216R = 5
_X5_216CV = 64
_X5_216T = 30
_x5_prev_labels_216 = labels


def _x5_216_labels(P, objs, cam, R, cv):
    import bulletlabels as BL
    os_ = [o for o in objs if o['t'] == _X5_T216]
    if len(os_) < 2: return P
    lids = list(P['lids']); ids = P['ids'].copy(); share = P['share'].copy()
    PP = BL.labels(objs, cam, cut=0.01); pids, psh, plids = PP['ids'], PP['share'], list(PP['lids'])
    items = []
    for o in os_:
        lid = (BL._lid(o), -1, 'body')
        if lid not in lids or not (ids == lids.index(lid)).any(): continue
        pk = [k for k, l in enumerate(plids) if l[0] == lid[0] and l[2] == 'body']
        items.append((lids.index(lid), _x5_np.isin(pids, pk) & (psh >= BL.CUT)))
    if len(items) < 2: return P
    par = list(range(len(items)))

    def find(a):
        while par[a] != a: a = par[a]
        return a
    for a in range(len(items)):
        da = _x5_dil(items[a][1], iterations=2)
        for b in range(a + 1, len(items)):
            if (da & items[b][1]).sum() >= _X5_216T: par[find(b)] = find(a)
    groups = {}
    for a in range(len(items)): groups.setdefault(find(a), []).append(a)
    C = None
    for g in groups.values():
        if len(g) < 2: continue
        if C is None:
            C = _x5_np.zeros(ids.shape, _x5_np.int8); seen = set()
            for o in os_:
                if o['own'] in seen: continue
                seen.add(o['own']); C = _x5_np.maximum(C, sprite_pixels(o['t'], o['own'], R))
        gk = [items[a][0] for a in g]; U = _x5_np.isin(ids, gk)
        PB = _x5_np.zeros(ids.shape, bool)
        for a in g: PB |= items[a][1]
        sp = _x5_np.isin(C, (1, 4)) & (cv >= _X5_216CV) & _x5_dil(PB, iterations=3)
        free = (ids < 0) | U
        S = _x5_fill(_x5_close(U | (sp & free), _X5_216R)) & free | U
        S = _x5_gauss(S.astype(float), 1.2) > 0.5
        S = (S & free) | _x5_ero(U, iterations=1)
        for k in gk: ids[ids == k] = -1
        keep = min(gk); ids[S] = keep; share[S] = 1.0
    P2 = dict(P); P2['ids'] = ids; P2['share'] = share
    return P2


def labels(objs, cam, R, state=None, ended=(), img=None):
    P = _x5_prev_labels_216(objs, cam, R, state=state, ended=ended, img=img)
    return _x5_216_labels(P, objs, cam, R, R['cv'][::2, ::2])


# ---- 207 AssaultShotgunDragon attack pellets (Griff006Projectile 6001031 / Griff006ProjectileSp1Buddy 6001032; only this skin) ----
# Owner 11-06 evening: "just label every bullet, but the labels arent good, they have a lot of background included". The chain
# (z_1109_mortis_dragon re-cut + closing 4, z_1110_misc closing 9 + blur 1.6, z_1113_misc4) glues the pellet pieces with wide closings, so
# the labels hold the gaps between the gold legs, the inside of bracelet / ring, the clasp's opening, and (opening away the thin gold parts)
# cut off the thorn / cone. Coordinator: every pellet (crystal WITH its gold cone, bracelet, ring, pendant, thorn ...) gets its own label that
# follows the pellet's own drawn pixels; gaps showing background are NOT part of it; only real body holes are filled; the gold cone is NOT cut.
# Rule (labels() wrapper, own types only, runs after the whole chain): the pellet = the pixels the front-most draw shows from the clip's body
# sprites (sprite code 1, coverage >= _X5_207CV; the shadow ellipse / glow are code 2 and stay out) + their anti-aliased 1-px rim; each
# connected piece (hairline cracks of 1 px joined) that overlaps the chain's labels is taken for the object that owns most of it (an object
# not yet used first, as in z_1109_mortis_dragon, so overlapping pellets keep one label each); a piece with no chain label stays unlabelled
# (ended / hit effects of the same clip). Holes inside a piece are filled only when they are hidden body (<= _X5_207HOLE px, or mostly sprite
# pixels of the clip body / over sprites), never when they show background (bracelet / ring interior stays open); the outline is rounded by a
# 1-px blur (gaps of 1 px close, nothing wider). Only pixels no other label holds are taken.
_X5_207 = (6001031, 6001032)
_X5_207CV = 64
_X5_207GCV = 96     # clip 'glow' sprites (code 2) that are opaque are pellet body (the bracelet's beads, parts of the gem): shadows / aura have coverage ~40
_X5_207HOLE = 12
_X5_207CONTACT = 9   # px: a thin part that lies along its pellet over at least this many px (the bracelet's bead ring along its shell) is part of it
_X5_207THREAD = 14    # px: a leftover thinner than 3 px and at least this big (a gold thread / wisp) is a bullet of its own, a shorter one (a tip) stays with its pellet
_x5_prev_labels_207 = labels


def _x5_207_labels(P, objs, cam, R, cv):
    import bulletlabels as BL
    os_ = [o for o in objs if o['t'] in _X5_207 and o.get('drawn', True)]
    if not os_: return P
    lids = list(P['lids']); ids = P['ids'].copy(); share = P['share'].copy()
    allk = []
    for o in objs:
        if o['t'] in _X5_207:
            lid = (BL._lid(o), -1, 'body')
            if lid in lids and (ids == lids.index(lid)).any(): allk.append(lids.index(lid))
    prior = _x5_np.isin(ids, allk) if allk else _x5_np.zeros(ids.shape, bool)
    PP = BL.labels(objs, cam, cut=0.01); pids, psh, plids = PP['ids'], PP['share'], list(PP['lids'])
    cand = []       # (object, lid, predicted body grown by 3 px)
    for o in os_:
        lid = (BL._lid(o), -1, 'body')
        pk = [k for k, l in enumerate(plids) if l[0] == lid[0] and l[2] == 'body']
        pb = _x5_np.isin(pids, pk) & (psh >= BL.CUT)
        cand.append((o, lid, _x5_dil(pb, iterations=3)))
    C = _x5_np.zeros(ids.shape, _x5_np.int8); seen = set()
    for o in os_:
        if o['own'] in seen: continue
        seen.add(o['own']); C = _x5_np.maximum(C, sprite_pixels(o['t'], o['own'], R))
    free = (ids < 0) | prior
    body = (((C == 1) & (cv >= _X5_207CV)) | ((C == 2) & (cv >= _X5_207GCV))) & free
    # anti-aliased rim: pixels of the same draws touching the body with some coverage
    ids_h = R['id'][::2, ::2]
    dr = _x5_np.unique(ids_h[body]); dr = dr[dr > 0]
    rim = _x5_dil(body, iterations=1) & _x5_np.isin(ids_h, dr) & (cv >= 32) & ((C != 2) | (cv >= _X5_207GCV)) & free
    body = body | rim
    if not body.any(): return P
    # pieces: the body cut where it is thinner than 3 px (a gold thread / chain between two pellets is not part of either); the thin
    # leftovers join the one piece they touch (when short: a tip), or (a thread, a wisp) are a piece of their own
    from scipy.ndimage import binary_opening as _x5_open
    core = _x5_open(_x5_np.pad(body, 2), structure=_x5_np.ones((3, 3), bool))[2:-2, 2:-2]
    if core.sum() < 0.5 * body.sum(): core = body.copy()
    Lc, nc = _x5_cc(core, structure=_x5_np.ones((3, 3), bool))
    thin = body & ~core
    Lt, nt = _x5_cc(thin, structure=_x5_np.ones((3, 3), bool))
    L = Lc.copy(); n = nc
    for j in range(1, nt + 1):
        tj = Lt == j
        nb = _x5_np.unique(Lc[_x5_dil(tj, iterations=1) & (Lc > 0)]); nb = nb[nb > 0]
        contact = int((_x5_dil(Lc == nb[0], iterations=1) & tj).sum()) if len(nb) == 1 else 0
        if len(nb) == 1 and (tj.sum() < _X5_207THREAD or contact >= _X5_207CONTACT): L[tj] = nb[0]
        else: n += 1; L[tj] = n
    pieces = []
    for c in range(1, n + 1):
        pc = L == c
        sz = int(pc.sum())
        if sz < 6: continue
        sc = [int((pc & g).sum()) + 3 * (int((pc & (ids == lids.index(lid))).sum()) if lid in lids else 0) for _, lid, g in cand]
        if max(sc) < 0.15 * sz: continue       # not at any pellet (an end effect of the same clip)
        pieces.append((pc, sc))
    pieces.sort(key=lambda p: -p[0].sum())
    used = set(); assign = []
    for pc, sc in pieces:
        free_i = [i for i in range(len(cand)) if i not in used and sc[i] > 0]
        i = max(free_i, key=lambda q: sc[q]) if free_i else max(range(len(cand)), key=lambda q: sc[q])
        used.add(i); assign.append((pc, cand[i][1]))
    for k in allk:
        mk = ids == k; ids[mk] = -1; share[mk] = 0
    for pc, lid in assign:
        if lid not in lids: lids.append(lid)
        k = lids.index(lid)
        m = pc.copy()
        hole = _x5_fill(m) & ~m
        if hole.any():
            Lh, nh = _x5_cc(hole)
            hid = (C == 1) | (C == 4)
            for j in range(1, nh + 1):
                hj = Lh == j
                if hj.sum() <= _X5_207HOLE or hid[hj].mean() >= 0.5: m |= hj
        s = _x5_gauss(_x5_np.pad(m, 4).astype(float), 1.0)[4:-4, 4:-4] > 0.5
        s = (s & _x5_dil(m, iterations=1)) | _x5_ero(m, iterations=1)
        s &= (ids < 0)
        ids[s] = k; share[s] = 1.0
    P2 = dict(P); P2['ids'] = ids; P2['lids'] = lids; P2['share'] = share
    return P2


def labels(objs, cam, R, state=None, ended=(), img=None):
    P = _x5_prev_labels_207(objs, cam, R, state=state, ended=ended, img=img)
    return _x5_207_labels(P, objs, cam, R, R['cv'][::2, ::2])
