# Skin 254 BlowerTrader (Gale 002), type 6000733 = Gale002Projectile (BlowerProjectile logic). Full replacement of z_1028_s254.py
# (round 4, 2026-10-28, the user's HIDDEN-PARTS rule: "the label should fill in the missing part, that way the model learns the exact
# bullet shape and predicts it even if part of the bullet is covered").
#
# WHAT THE OBJECT IS. One shot = SIX snowballs in a line, logic centres 100 apart (about 10 px on screen), so they stack. Each snowball
# = a head (blue blob with a bright eye, clip shape 57828; the enemy '_red' clip: 57817), a thin curl behind it (57833 / 57831 / 57826)
# and little dashes (37995). A tan dust cloud (clips particle_dust_*) is drawn over them, translucent in places, so many heads are
# only seen through it, some not at all. LABEL = HEADS ONLY (user: no curls, no dust, no dashes), the head's FULL silhouette.
#
# HOW THE FULL HEAD IS PLACED (this round). The clip is stopped on one RANDOM frame per snowball (projtab random_start 2, 6 frames):
# the client's rand() is not in the game data, so the tracker's start_frame (o['start_frame']) is a guess and the placed head (o['parts'])
# is often a different pose than the drawn one (the head sprite is rotated / flipped a bit per frame and moves up to ~4 px; the curl
# and dashes differ per frame too). So per snowball:
#   1. POSES: all 6 frames of the clip (projdraw.Display(start_frame=k): every polygon's clip-local coordinates, shape id, uv) are placed
#      with the object's OWN clip-local -> world transform (an exact affine fit on the object's placed polygons: residual 1e-11), i.e.
#      at the projectile's logic position / angle / scale, and rasterised with the same renderer as the prediction (BL.labels, the head
#      sprite's atlas alpha >= 50 %).
#   2. EVIDENCE from the ID pass: the pixels whose front-most draw shows a shape of the snowball clip (_a4_shape_pixels: shape id per
#      pixel). Candidate score (pose k, shift dy dx): + head pixels seen inside the head raster (x2), + curl / dash pixels seen in the
#      same shape's raster (the shape id also tells the pose), - raster pixels where another clip shape is the front-most draw, - head
#      raster on dark floor (a head would show through the dust), - visible head pixels left outside. The six snowballs of one shot share
#      the frame-timing jitter (the whole line sits a few px off its logic position along the flight direction, G254_COMMON = 5 px,
#      cost 0.6 / px); each snowball has its own pose and a +-1 px own offset (cost 1 / px) around it: the best shot shift is the one
#      with the best sum (_g254_solve). A snowball's visible pixels are never taken by its neighbour (nearest head centre owns them).
#   3. A snowball with (almost) no evidence (hidden under the dust, < G254_MIN_EVID weighted pixels) takes the CONSENSUS pose (the pose
#      most like the others) at the shot's common shift (its logic position plus the line's timing jitter: nothing else is known).
#   LABEL = that head silhouette  U  the visible head pixels of the ID pass (head shape + antialiased bluish edge within 2 px, never
#   the curl / dash pixels), holes filled, ONE solid convex hull per snowball; hulls that overlap by G254_MERGE (50 %) of the smaller
#   are ONE label (stacked snowballs, existing rule), groups that are apart stay separate (where two separate hulls overlap, the pixel goes
#   to the nearer head centre). Cut only by world geometry (walls / crates / characters, opaque depth-writing draws), never by dust.
#   Never labelled: a snowball whose logic ended (o['drawn'] False; the object has no head polygon), dust, curls, dashes.
# TEAMS: own = clip gale_002_atk_projectile (head 57828), enemy (o['own'] False) = gale_002_atk_projectile_red (head 57817): the same code,
# the pose polygons come from the right clip (Display(own)); colour tests only for the 2 px antialiased edge (own blue / enemy crimson, unchanged from round 3) and the 'dark floor' test (max channel < 115).
# History: round 1 cut by clip shape (slivers), fix2 hulled the object's dust-covered body (labelled dust), round 3 (10-28) took the
# visible head pixels only (a fully hidden head got no label, a partly hidden one only its visible part); this round fills.
# Env G254_FILL=0 gives the round-3 behaviour approximately (visible parts only: raster not added), compare only.
# Type touched: 6000733 only (check: other Gale skins use the same type / clip, projtab).

from scipy.ndimage import binary_fill_holes as _g254_fill, label as _g254_cc, binary_dilation as _g254_dil

G254_TYPE = 6000733
G254_FILE = 'sc/effects_brawler.sc'
G254_CLIPS = ('gale_002_atk_projectile', 'gale_002_atk_projectile_red')
G254_HEADS = (57828, 57817)     # head shape ids: own clip / '_red' clip
G254_SHIFT = 6                  # px: largest total shift tabulated
G254_COMMON = 5                 # px: the shot's common timing jitter (flight direction) searched around the logic position
G254_IND = 1                    # px: each snowball's own offset around the shot's common one (pose offsets, rounding)
G254_SHIFT_COST = 0.6           # score per px of common shift
G254_IND_COST = 1.0             # score per px of own offset
G254_MIN_EVID = 4.0             # weighted visible pixels needed to trust a candidate (else: hidden snowball -> consensus pose, shift 0)
G254_FLOOR = 115                # max(r, g, b) below this: dark floor / water (a head, even under dust, is brighter)
G254_EDGE = 2                   # px: antialiased edge pixels around the visible head shape that count
G254_WHITE = 170
G254_BLUE = 50
G254_MERGE = 0.5                # hulls overlapping by this share of the smaller one are one label (stacked snowballs)
G254_SPECK = 20                 # px: loose visible parts smaller than this are specks
G254_FILL = os.environ.get('G254_FILL', '1') == '1'
G254_DEBUG = os.environ.get('G254_DEBUG', '') == '1'
G254_LAST = {}
G254_LOG = []                   # per frame: list of dicts (object id, pose, shift, score, evidence, hidden)

_g254_prev_labels = labels
_G254_POSES = {}


def _g254_poses(own):
    """{pose k: [(shape id, cmd, tex, uv (4, 2), local (4, 2), mul, add)]} for the clip's start frames (the client picks one at random
    per snowball and stops there); local = clip px before the placement"""
    if own in _G254_POSES: return _G254_POSES[own]
    T = projtab.table()[G254_TYPE]; S = PD.sc(T['file']); cid = S.exports.get(T['blue'] if own else T['red'])
    n = PD.start_range(T, S, cid) if cid in S.clips else 1
    out = {}
    for k in range(n if isinstance(n, int) else 1):
        D = PD.Display(G254_TYPE, own, start_frame=k, ground=False); D.update(angle=0, travel=0)
        out[k] = [(q[1], q[2], q[3], np.asarray(q[4]), np.asarray(q[5], float), q[6], q[7]) for q in D.polys()]
    _G254_POSES[own] = out
    return out


def _g254_raster(o, q, cam):
    """half-resolution bool mask of one placed polygon (atlas alpha >= 50 %), by the prediction's own renderer"""
    import bulletlabels as BL
    p = dict(o['parts']); p['clip'] = [q]; p.pop('shadow', None); p.pop('glow', None)
    H = BL.labels([dict(o, parts=p)], cam, cut=0.01)
    return (H['ids'] >= 0) & (H['share'] >= 0.5)


def _g254_pose_rasters(o, cam):
    """{pose k: [(shape id, is_head, mask)]} : every pose of the clip placed at the object's own transform. If the poses cannot be
    built (no clip transform fit, an unexpected clip), the object's own placed polygons are its single pose (key -1)"""
    qs = o['parts'].get('clip', [])
    if not any(q['shape'] in G254_HEADS for q in qs): return None
    try:
        L = np.concatenate([np.asarray(q['local'], float) for q in qs]); Wd = np.concatenate([np.asarray(q['world'], float) for q in qs])
        X = np.c_[L, np.ones(len(L))]; A = np.linalg.lstsq(X, Wd, rcond=None)[0]      # clip-local px -> world render coords (affine)
        if np.abs(X @ A - Wd).max() > 1e-3: raise ValueError('no affine clip transform')
        tm = [q for q in qs if q['shape'] in G254_HEADS][0]
        out = {}
        for k, pl in _g254_poses(o['own']).items():
            row = []
            for shape, cmd, tex, uv, loc, mul, add in pl:
                q = dict(tm, shape=shape, cmd=cmd, tex=tex, uv=uv, local=loc, world=np.c_[loc, np.ones(len(loc))] @ A, mul=mul, add=add)
                m = _g254_raster(o, q, cam)
                if m.any(): row.append((shape, shape in G254_HEADS, m))
            out[k] = row
        if any(any(ih for _, ih, _ in row) for row in out.values()): return out
    except Exception as e_:
        if G254_DEBUG: print('G254 pose fallback', o['id'], repr(e_))
    row = []
    for q in qs:
        m = _g254_raster(o, q, cam)
        if m.any(): row.append((q['shape'], q['shape'] in G254_HEADS, m))
    return {-1: row} if any(ih for _, ih, _ in row) else None


def _g254_shift(a, dy, dx, R):
    ap = np.pad(a, R)
    return ap[R - dy:R - dy + a.shape[0], R - dx:R - dx + a.shape[1]]


def _g254_table(o, pr, SP, S, floor):
    """score tables of one snowball over every pose and every shift of +-G254_SHIFT px (no shift cost yet):
    ({pose: (2R+1, 2R+1) score}, {pose: same, the positive evidence only}, consensus pose)"""
    R = G254_SHIFT
    ys, xs = [], []
    for row in pr.values():
        for shape, ih, m in row:
            y, x = np.nonzero(m); ys += [y.min(), y.max()]; xs += [x.min(), x.max()]
    if not ys: return None
    y0 = max(0, min(ys) - R - 5); y1 = min(S.shape[0], max(ys) + R + 6); x0 = max(0, min(xs) - R - 5); x1 = min(S.shape[1], max(xs) + R + 6)
    sl = (slice(y0, y1), slice(x0, x1)); SPw = SP[sl]; Sw = S[sl]; Fw = floor[sl]; anyshape = SPw > 0
    heads = np.zeros(Sw.shape, bool)
    for row in pr.values():
        for shape, ih, m in row:
            if ih: heads |= m[sl]
    near = Sw & _g254_dil(heads, iterations=R + 2)
    T = {}; TP = {}
    for k, row in pr.items():
        rows = [(shape, ih, m[sl]) for shape, ih, m in row]
        T[k] = np.full((2 * R + 1, 2 * R + 1), -1e9); TP[k] = np.zeros((2 * R + 1, 2 * R + 1))
        for dy in range(-R, R + 1):
            for dx in range(-R, R + 1):
                pos = neg = 0.0; head = None
                for shape, ih, m in rows:
                    r = _g254_shift(m, dy, dx, R)
                    pos += (2.0 if ih else 1.0) * int((r & (SPw == shape + 1)).sum())
                    neg += 0.7 * int((r & anyshape & (SPw != shape + 1)).sum())
                    if ih: head = r; neg += 1.0 * int((r & Fw).sum())
                if head is None: continue
                neg += 0.5 * int((near & ~head).sum())
                T[k][dy + R, dx + R] = pos - neg; TP[k][dy + R, dx + R] = pos
    hk = {k: [m[sl] for shape, ih, m in row if ih] for k, row in pr.items()}
    hk = {k: v[0] for k, v in hk.items() if v}
    iou = {k: np.mean([(a & b).sum() / max(1, (a | b).sum()) for k2, b in hk.items() if k2 != k]) if len(hk) > 1 else 1.0 for k, a in hk.items()}
    ck = max(iou, key=lambda k: iou[k]) if iou else 0
    return T, TP, ck


def _g254_solve(tabs):
    """one shot's snowballs share the frame-timing jitter s (the whole line moves along the flight direction); each snowball has its own
    pose and a +-G254_IND px offset around s. Maximise the sum of the snowballs' best scores minus the shift costs.
    tabs: [(T, TP, consensus pose)]; returns (s, [(pose, dy, dx, pos, hidden)])"""
    R = G254_SHIFT; Rs = G254_COMMON; ind = G254_IND
    best = None
    for sy in range(-Rs, Rs + 1):
        for sx in range(-Rs, Rs + 1):
            tot = -G254_SHIFT_COST * (abs(sy) + abs(sx)); pick = []
            for T, TP, ck in tabs:
                bo = None
                for k in T:
                    for dy in range(-ind, ind + 1):
                        for dx in range(-ind, ind + 1):
                            yy, xx = sy + dy + R, sx + dy * 0 + dx + R
                            if not (0 <= yy < 2 * R + 1 and 0 <= xx < 2 * R + 1): continue
                            v = T[k][yy, xx] - G254_IND_COST * (abs(dy) + abs(dx))
                            if bo is None or v > bo[0]: bo = (v, k, sy + dy, sx + dx, TP[k][yy, xx])
                if bo is None: pick.append(None); continue
                hidden = bo[4] < G254_MIN_EVID
                if hidden: pick.append((ck, sy, sx, 0.0, True))
                else: tot += bo[0]; pick.append((bo[1], bo[2], bo[3], bo[4], False))
            if best is None or tot > best[0] + 1e-9: best = (tot, (sy, sx), pick)
    return best[1], best[2]


def labels(objs, cam, R, state=None, ended=(), img=None):
    import bulletlabels as BL
    P = _g254_prev_labels(objs, cam, R, state=state, ended=ended, img=img)
    if img is None: return P
    ids, lids, share = P['ids'], P['lids'], P['share']
    on = sorted([o for o in objs if o['t'] == G254_TYPE], key=lambda o: o['id'])
    if not on: return P
    live = [o for o in on if o.get('drawn') and not o.get('ended')]     # a snowball whose logic ended is never labelled
    cv = R['cv'][::2, ::2]; H, Wd = ids.shape
    # the old labels of every snowball are replaced (their dust / curl pixels go)
    old = {}
    for o in on:
        lid = (BL._lid(o), -1, 'body')
        if lid in lids and (ids == lids.index(lid)).any(): old[o['id']] = ids == lids.index(lid)
    U = np.zeros(ids.shape, bool)
    for m in old.values(): U |= m
    im = img.astype(np.int16); r_, g_, b_ = im[..., 0], im[..., 1], im[..., 2]
    ids_h = R['id'][::2, ::2]
    SP = _a4_shape_pixels(R, G254_FILE, G254_CLIPS)
    S = np.isin(SP, [s_ + 1 for s_ in G254_HEADS])
    # visible clip pixels that are NOT the head (curls, dashes): never added; head pixels / edge pixels only
    other = (SP > 0) & ~S
    floor = im.max(-1) < G254_FLOOR
    # bluish / bright-white (own) or crimson / white (enemy) antialiased edge pixels around the visible head shape
    own_snow = ((g_ >= r_ + 12) & (b_ >= r_ + 20)) | (b_ >= r_ + G254_BLUE + 10) | ((im.min(-1) >= G254_WHITE) & (b_ >= r_))
    red_snow = ((b_ >= g_ + 8) & (g_ <= 95) & (r_ >= 150)) | ((im.min(-1) >= G254_WHITE) & (r_ >= 150))
    # world geometry (walls, crates, characters: opaque depth-writing draws) cuts a label; blended dust / effects never
    ids_w = ids_h; world = np.zeros(ids.shape, bool)
    for d in np.unique(ids_w[ids_w > 0]):
        dr = R['draws'].get(int(d), {})
        if dr.get('blend') == 0 and dr.get('dmask') == 1: world |= ids_w == d
    world &= ~(SP > 0)
    allowed = ~world & ((ids < 0) | U)
    for l in [(BL._lid(o), -1, 'body') for o in on]:
        if l in lids: m_ = ids == lids.index(l); ids[m_] = -1; share[m_] = 0
    # per snowball: the head silhouette
    heads = []      # (object, head mask, visible head pixels, info)
    log = []
    prs = []
    for o in live:
        pr = _g254_pose_rasters(o, cam)
        if not pr: continue
        tb = _g254_table(o, pr, SP, S, floor)
        if tb is None: continue
        prs.append((o, pr, tb))
    shots = {}
    for o, pr, tb in prs: shots.setdefault((o.get('first'), bool(o.get('own', True))), []).append((o, pr, tb))
    for key, mem in shots.items():
        sh, picks = _g254_solve([tb for o, pr, tb in mem])
        for (o, pr, tb), pk in zip(mem, picks):
            if pk is None: continue
            k, dy, dx, pos, hidden = pk
            hm = [m for shape, ih, m in pr[k] if ih]
            if not hm: continue
            hm = _g254_shift(hm[0], dy, dx, G254_SHIFT) if (dy or dx) else hm[0]
            heads.append((o, hm, k, dy, dx, hidden, pos, sh))
    heads.sort(key=lambda h_: h_[0]['id'])
    if not heads and not S.any(): return P
    # each visible head pixel belongs to ONE snowball: the nearest head centre among those whose silhouette (grown 3 px) holds it
    # (stacked heads must not take each other's visible pixels: their hulls would merge on that alone)
    cents = [np.argwhere(hm).mean(0) for o, hm, *_ in heads]
    nears = [_g254_dil(hm, iterations=3) for o, hm, *_ in heads]
    owner_s = np.full(ids.shape, -1, np.int32); best_d = np.full(ids.shape, np.inf, np.float32)
    for i, (c_, nr) in enumerate(zip(cents, nears)):
        t = S & nr
        d = ((np.nonzero(t)[0] - c_[0]) ** 2 + (np.nonzero(t)[1] - c_[1]) ** 2).astype(np.float32)
        yy_, xx_ = np.nonzero(t); upd = d < best_d[yy_, xx_]
        owner_s[yy_[upd], xx_[upd]] = i; best_d[yy_[upd], xx_[upd]] = d[upd]
    keep_rows = []
    for i, (o, hm, k, dy, dx, hidden, pos, sc) in enumerate(heads):
        vis = S & (owner_s == i)
        edge = _g254_dil(vis, iterations=G254_EDGE) & (red_snow if not o.get('own', True) else own_snow) & ~other & (owner_s < 0)
        reg = ((hm if G254_FILL else np.zeros_like(hm)) | vis | edge) & allowed
        if not reg.any(): continue
        reg = _g254_fill(reg)
        reg2 = hull_mask(reg) & allowed
        keep_rows.append((o, reg2, hm, vis, k, dy, dx, hidden, pos))
    # loose visible head parts that no head silhouette explains (never lose a seen head): their own groups
    covered = np.zeros(ids.shape, bool)
    for row in keep_rows: covered |= _g254_dil(row[1], iterations=2)
    loose = S & allowed & ~covered
    extra = []
    if loose.any():
        Ll, nl = _g254_cc(_g254_dil(loose, iterations=2), structure=np.ones((3, 3), int))
        for i in range(1, nl + 1):
            c = (Ll == i) & loose
            if c.sum() >= G254_SPECK: extra.append(hull_mask(_g254_fill(c)) & allowed)
    # groups: hulls overlapping by G254_MERGE of the smaller one are one label (stacked snowballs)
    grp = []   # [mask, [members (index into keep_rows)]]
    for i, row in enumerate(keep_rows):
        h = row[1]; mem = [i]
        hit = [q for q in grp if (q[0] & h).sum() >= G254_MERGE * min(q[0].sum(), h.sum())]
        for q in hit: h = h | q[0]; mem = mem + q[1]
        grp = [q for q in grp if not any(q is z_ for z_ in hit)]
        grp.append([h, mem])
    grp = [[_g254_fill(hull_mask(q[0]) & allowed), q[1]] for q in grp]
    # a separate group's pixels overlapping another group's: the nearer centre
    cen = [np.argwhere(q[0]).mean(0) for q in grp]
    yy, xx = np.mgrid[0:H, 0:Wd]
    owner = np.full(ids.shape, -1, np.int32); dist = np.full(ids.shape, np.inf, np.float32)
    for gi, q in enumerate(grp):
        d = ((yy - cen[gi][0]) ** 2 + (xx - cen[gi][1]) ** 2).astype(np.float32)
        t = q[0] & (d < dist); owner[t] = gi; dist[t] = d[t]
    lids = list(lids)
    for gi, q in enumerate(grp):
        o = min((keep_rows[i][0] for i in q[1]), key=lambda o_: o_['id'])
        lid = (BL._lid(o), -1, 'body')
        if lid not in lids: lids.append(lid)
        k = lids.index(lid)
        take = (owner == gi) & (ids < 0)
        ids[take] = k; share[take] = np.maximum(cv[take] / 255.0, BL.CUT)
    for c in extra:
        # carrier: the nearest object's lid not yet... (loose part): the object whose head silhouette centre is nearest
        cy, cx = np.argwhere(c).mean(0)
        cand = [(np.hypot(np.argwhere(row[2]).mean(0)[0] - cy, np.argwhere(row[2]).mean(0)[1] - cx), j) for j, row in enumerate(keep_rows)]
        o = keep_rows[min(cand)[1]][0] if cand else (live[0] if live else on[0])
        lid = (BL._lid(o), -1, 'body')
        if lid not in lids: lids.append(lid)
        k = lids.index(lid); take = c & (ids < 0)
        ids[take] = k; share[take] = np.maximum(cv[take] / 255.0, BL.CUT)
    P2 = dict(P); P2['lids'] = lids; P2['ids'] = ids; P2['share'] = share
    if G254_DEBUG:
        G254_LAST['rows'] = [(row[0]['id'], row[2], row[1], row[3]) for row in keep_rows]; G254_LAST['allowed'] = allowed; G254_LAST['world'] = world
        G254_LAST['grp'] = [(q[0], [keep_rows[i][0]['id'] for i in q[1]]) for q in grp]
        G254_LOG.append([dict(id=row[0]['id'], pose=row[4], shift=(row[5], row[6]), hidden=row[7], pos=row[8], hull=int(row[1].sum()),
                              head=int(row[2].sum()), vis=int(row[3].sum())) for row in keep_rows])
    return P2
