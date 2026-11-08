# Cooker Porcelain 612 (Pearl004Projectile_k2 6001368): whole pearl incl. white lobe and the pale round bulb, own outline, no faint streak (user 10-26 Q6 + 'bulb isnt included'; Haiku task 34 v5, reviewed; replaces z_1024_cooker612.py)
# Skin 612 CookerPorcelain: each pearl (Pearl004Projectile_k2, type 6001368) labelled as its bulb plus neck plus the small
# opaque white lobe and the pale round bulb at the neck's end; one solid outline per pearl; attack 0 deg and attack 90 deg
# (Haiku task 34, v5: replaces z_1026_cooker612.py, the v4 copy).
# Cause of the v4 miss: the pale round bulb is translucent (atlas shape 58, coverage ~50) but whitish (min RGB ~150-230),
# while the faint streak it sits on is dim blue (min RGB ~50-60). v4 kept shape 58 only within 2 px of the opaque body.
# Fix (v5 = v4 + pale bulb):
#   opaque body (shapes 61, 62, 72, 47, 53, 56, 71, coverage >= 24) split by geodesic distance from the opaque cores to the
#   nearest pearl game position (anchor, projected); own parts within NEAR_PART = 12 px of the main part kept; specks < 6 px
#   dropped; gaps of up to 8 px between its own parts closed; holes filled; sprite-edge pixels (1 px, same draws) kept.
#   Shape 58 (translucent): kept only within FRINGE_PX = 2 px of the opaque body, and PALE bulbs: shape-58 pixels with min RGB
#   >= PALE_MIN = 150 within PALE_R = 10 px of the main part, in a connected piece of PALE_MIN_SIZE (30) to PALE_MAX (90) px that
#   touches the main part (the dim streak and the brawler's white glove under the haze are not pale bulbs).
#   Last step: only the largest part and parts of >= 60 px are kept (one solid label).
#   No ONE_LABEL: the outline is the pearl's own shape, filled solid (not a convex hull).
#   The labels() wrapper stores the frame's objects and camera (the other pearls' positions); no change of policy.
import numpy as np

PEARL_TYPE = 6001368
PEARL_FILE = 'sc/effects_brawler_pearl.sc'
PEARL_CLIPS = ('pearl_004_atk_projectile_k2', 'pearl_004_atk_projectile_k2_red')
PEARL_BODY_SHAPES = (47, 53, 56, 61, 62, 71, 72)   # opaque body of the pearl: the small white lobe (61), bulb (62, 72), chevrons
PEARL_FRINGE = 58        # the translucent streak / neck (coverage 30-60): NOT the pearl; kept only within FRINGE_PX of the opaque body
FRINGE_PX = 2
MIN_PART = 6            # px: pieces of the pearl's own opaque body smaller than this are specks
PALE_MIN = 150          # min RGB channel of the pale round bulb (translucent shape 58, whitish); the blue streak is ~50-60
PALE_R = 10             # px: the pale bulb lies within this of the pearl's own main body (the neck's end)
PALE_MAX = 90          # px: a pale piece is a bulb only up to this size (the brawler's white glove under the haze is far larger)
PALE_MIN_SIZE = 30     # px: smaller pale pieces are specks between pearls (12-21 px), not a bulb
NEAR_PART = 12         # px: its own opaque parts within this of its main part are kept (the white lobe on a neck)
CLOSE_OWN = 4          # px: gaps of up to 2 x this between its OWN opaque parts are closed (one solid outline)
PEARL_COV = 24            # coverage (0..255) of a body-shape pixel: the translucent lobe is 30-60
_PEARL612 = {}            # the current frame's objects and camera (set by labels() below)
_labels_before_612 = labels


def labels(objs, cam, R, state=None, ended=(), img=None):
    _PEARL612['objs'] = objs; _PEARL612['cam'] = cam
    return _labels_before_612(objs, cam, R, state=state, ended=ended, img=img)


def _pearl612_screen(cam, anchor):
    """screen position (half resolution, 1200 x 540) of a frame object's anchor (render coords)"""
    q = cam @ np.array([anchor[0], anchor[1], anchor[2], 1.0])
    return (q[0] / q[3] * 0.5 + 0.5) * 1200, (1 - (q[1] / q[3] * 0.5 + 0.5)) * 540


def _pearl612_split(cand, opaque, centres, me):
    """the pixels of cand that belong to pearl `me`: every opaque core component of the body goes to the pearl whose
    game position (centres: id -> screen xy) is nearest to it; the owners then spread through cand (8-connected, BFS:
    geodesic distance), so a translucent lobe or neck follows its own bulb and not the neighbour's across the join"""
    from scipy.ndimage import label as cc
    from collections import deque
    ids = list(centres); C = np.array([centres[i] for i in ids], np.float64)
    L, n = cc(opaque, structure=np.ones((3, 3)))
    seeds = np.zeros(cand.shape, np.int64); mine_seed = False
    for k in range(1, n + 1):
        ys, xs = np.nonzero(L == k)
        c = np.array([xs.mean(), ys.mean()]); j = int(np.argmin(((C - c) ** 2).sum(1)))
        seeds[L == k] = ids[j]; mine_seed |= ids[j] == me
    if not mine_seed: return None
    H, W = cand.shape; owner = np.zeros((H, W), np.int64); seen = np.zeros((H, W), bool); q = deque()
    for y, x in zip(*np.nonzero(seeds)):
        owner[y, x] = seeds[y, x]; seen[y, x] = True; q.append((y, x))
    while q:
        y, x = q.popleft()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                yy, xx = y + dy, x + dx
                if 0 <= yy < H and 0 <= xx < W and cand[yy, xx] and not seen[yy, xx]:
                    seen[yy, xx] = True; owner[yy, xx] = owner[y, x]; q.append((yy, xx))
    return owner == me


def _fix_6001368_612(body, ctx):
    from scipy.ndimage import label as cc, binary_fill_holes, binary_closing
    cam = _PEARL612.get('cam'); objs = _PEARL612.get('objs') or []
    me = ctx['o']['id']
    centres = {}
    for o in objs:
        if o['t'] == PEARL_TYPE and o.get('anchor') is not None: centres[o['id']] = _pearl612_screen(cam, o['anchor'])
    if me not in centres or cam is None: return body              # no game position: unchanged
    SP = _a4_shape_pixels(ctx['R'], PEARL_FILE, PEARL_CLIPS)      # shape id + 1 per pixel, 0: none
    cv = ctx['cv']; region = ctx['region']
    op_sh = np.isin(SP, [s + 1 for s in PEARL_BODY_SHAPES]) & (cv >= PEARL_COV) & region
    opaque = op_sh & (cv >= CUT)                                  # the white lobe, bulb and chevrons: the seeds
    fr = np.isin(SP, [PEARL_FRINGE + 1]) & (cv >= PEARL_COV) & region & _dilate(opaque, FRINGE_PX)
    cand = op_sh | fr
    if not cand.any() or not opaque.any(): return body
    own = _pearl612_split(cand, opaque, centres, me)
    if own is None or not own.any(): return body
    L, n = cc(own, structure=np.ones((3, 3)))
    if not n: return body
    sizes = np.bincount(L.ravel())[1:]
    ov = np.bincount(L[ctx['pred_any'] & own].ravel(), minlength=n + 1)[1:]
    k0 = int(np.argmax(ov)) + 1 if ov.max() > 0 else int(np.argmax(sizes)) + 1   # the part holding its predicted label
    near = _dilate(L == k0, NEAR_PART)                                          # its own opaque parts close to it (a lobe on a short neck)
    keep = [k0] + [k + 1 for k in range(n) if k + 1 != k0 and sizes[k] >= MIN_PART and (near & (L == k + 1)).any()]
    main = np.isin(L, keep)
    if not main.any(): return body
    img = ctx.get('img')                                          # the frame: the pale round bulb is whitish
    if img is not None:
        from scipy.ndimage import distance_transform_edt as dt
        dm = dt(~main)
        pale = np.isin(SP, [PEARL_FRINGE + 1]) & (img.astype(np.int16).min(-1) >= PALE_MIN) & (cv >= PEARL_COV) & region & (dm <= PALE_R)
        Lp, npl = cc(pale, structure=np.ones((3, 3)))
        touch = _dilate(main, 1); add = np.zeros(main.shape, bool)
        for kq in range(1, npl + 1):
            comp = Lp == kq
            if PALE_MIN_SIZE <= comp.sum() <= PALE_MAX and (comp & touch).any(): add |= comp
        main = main | add
    pad = CLOSE_OWN
    m = binary_closing(np.pad(main, pad), iterations=pad)[pad:-pad, pad:-pad] | main
    m = binary_fill_holes(m)
    pdraws = np.unique(ctx['ids_h'][cand])                        # the draws that show the pearl's own shapes
    edge = body & (SP == 0) & _dilate(m, 1) & np.isin(ctx['ids_h'], pdraws)   # sprite-edge pixels of those draws, 1 px
    out = binary_fill_holes(m | edge)
    Lf, nf = cc(out, structure=np.ones((3, 3)))            # one solid label: the largest part and the parts of 60 px or more
    if nf > 1:
        szf = np.bincount(Lf.ravel())[1:]
        out = np.isin(Lf, [k + 1 for k in range(nf) if szf[k] >= max(60, 0.5 * szf.max())])
    return out


POST_BODY[PEARL_TYPE] = _fix_6001368_612
ONE_LABEL.pop(PEARL_TYPE, None)   # the task-14 hull is replaced: the outline is the pearl's own, filled solid


# Coordinator 10-26 (the user: 'there are still some white bulbs left out'): the pearls also take the bright white bulbs
# touching or within 4 px of them (grown through near-white pixels, never onto opaque
# world draws: the brawler, crates)
import os as _os
CW_MIN = int(_os.environ.get('CW_MIN', '190')); CW_GROW = int(_os.environ.get('CW_GROW', '10')); CW_GAP = 4; CW_BLOB = 150
def _cw(body, ctx):
    from scipy.ndimage import binary_fill_holes
    img = ctx['img']
    if img is None or not body.any(): return body
    R = ctx['R']; ids = ctx['ids_h']; world = np.zeros(body.shape, bool)
    for d in np.unique(ids[ids > 0]):
        dr = R['draws'].get(int(d), {})
        if (dr.get('blend') == 0 and dr.get('dmask') == 1) or dr.get('th') == MODEL_TEX: world |= ids == d
    white = (img.min(-1) >= CW_MIN) & ~world & _dilate(ctx['region'], 6)
    # characters: never (their white parts under the pearl's glow); a band around every character's screen position
    st = ctx.get('state'); cam = ctx.get('cam')
    if st is not None and cam is not None:
        yy, xx = np.mgrid[0:body.shape[0], 0:body.shape[1]]
        for c in st['chars']:
            cx, cy = _proj(cam, int(c[3]), int(c[4])); hx, hy = _proj(cam, int(c[3]), int(c[4]), 450)
            white &= ~((((xx - cx) / 22.0) ** 2 + ((yy - (cy + hy) / 2) / max(12.0, abs(cy - hy) / 2 + 10)) ** 2) <= 1)
    m = body.copy()
    for _ in range(CW_GROW): m = m | (_dilate(m, 1) & white)
    from scipy.ndimage import label as cc
    L, n = cc(white & ~m)   # compact white blobs (bulbs) a few px away from the pearl join it
    if n:
        near = _dilate(m, CW_GAP); sz = np.bincount(L.ravel())
        for k in range(1, n + 1):
            if sz[k] <= CW_BLOB and (near & (L == k)).any(): m |= (L == k)
    return binary_fill_holes(_close(m, 1))
_cw_prev = POST_BODY.get(6001368)
POST_BODY[6001368] = lambda body, ctx: _cw(_cw_prev(body, ctx) if _cw_prev else body, ctx)
