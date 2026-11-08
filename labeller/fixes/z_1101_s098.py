# hosted review 2026-10-29 fix (Haiku fix, reviewed by the coordinator, installed 10-29 LOCAL ONLY; source out/sweep2/haikutasks/site1029/s098/fix3.py; VoodooPet did 16000244 fade-in templates for 098 and 121; hence z_1101_)
# fix3 for task site1029 / skin 098 (VoodooDefault), hosted review 2026-10-29, super 45 tick 369 (user: "too small / cut off"),
# with the coordinator's decisions (2026-10-29): the whole pet, no atlas separation. Also applied and checked on skin 121
# (VoodooSandsOfTimes, the same pet, the same finding). Both skins are in this file; each has its own template.
#
# Cause (measured, see the report): the VoodooPet spawnable (character did 16000244) fades in and settles over ticks ~363-379.
# Its 3D model is drawn only in part at first (the lower body; the head and leaves are translucent / additive and not labelled),
# and the model label is incomplete, sometimes with interior lines, until the whole pet is drawn solidly. The pet's ground point
# does not move while it fades in (both skins, 359-377), and moves a little at 379-383 (the same movement for the template).
#
# Templates: the spawn label of the skin at a CLEAN tick (098: tick 375, 818 px; 121: tick 379, 891 px, no holes; the 121
# tick 377 label has a 71 px interior hole). Stored as run-length rows relative to an integer anchor, with the ground point
# base (float frame px) of that tick.
#
# Rule, on every tick where the pet is drawn (model label >= S3_MIN px) and some template covers the model label:
#   - scale s of the template about the pet's ground point of this tick: the smallest s in [0.90, 1.15] whose scaled template
#     covers >= S3_COVER of the model label; the template with the best cover is used (a template that covers < S3_COVER at
#     every scale is not used: this keeps the other skin's silhouette out);
#   - the label is INCOMPLETE when it is below S3_FULL (0.75) of the fitted template's area: then label = template (scaled, placed at
#     the ground point) union model label, free pixels only (other labels excluded), one piece (the component holding the model
#     label), holes filled;
#   - always: enclosed holes of the model label (interior lines) are filled, free pixels only;
#   - a complete, solid label (no holes, area >= S3_FULL of the template) is left unchanged.
# Windows per template (state tick): the 098 template 363-375, the 121 template 367-369. The 121 template (from tick 379)
# does not fit the pet at 371-377 (it bulges past the golden horn when scaled to the partial label), so 121 is limited to
# the ticks where its fit was checked; 121 at 371-377 keeps the hole fill only.
# Keyed to did 16000244 only. No type id is touched; the zone fix z_1100_s121 (type 17000802) is independent.
import bulletlabels as _BL

S3_PET = 16000244          # VoodooPet spawnable character did
S3_MIN = 5                 # model-label pixels from which the pet is drawn at all
S3_COVER = 0.97            # template cover of the model label required (smallest scale that reaches it)
S3_FULL = 0.75             # model label below this share of the fitted template's area = incomplete (measured: incomplete
                           # 0.45-0.60, complete 0.92-1.00; a complete 098 label at 379 is 0.92 of the 121 template)
S3_SCALES = [round(0.90 + 0.01 * i, 2) for i in range(26)]   # 0.90 .. 1.15
S3_WIN = 56                # px: template window around the anchor (offsets -S3_WIN .. S3_WIN)
# template runs (dy, dx0, dx1): row dy relative to the anchor, pixels dx0..dx1 inclusive
S3_TEMPLATES = {
    'skin098': dict(base=(694.2205531963164, 336.2331103938205), anchor=(694, 336), window=(363, 375), runs=[(-19, -6, -4),
    (-18, -8, -3),
    (-17, -9, -2),
    (-17, 4, 5),
    (-17, 8, 11),
    (-16, -9, -1),
    (-16, 1, 13),
    (-15, -10, 15),
    (-14, -9, 16),
    (-13, -9, 17),
    (-12, -9, 18),
    (-11, -8, 18),
    (-10, -6, 19),
    (-9, -5, 19),
    (-8, -6, 19),
    (-7, -6, 20),
    (-6, -6, 22),
    (-5, -4, 24),
    (-4, -4, 25),
    (-3, -9, 25),
    (-2, -10, 26),
    (-1, -10, 25),
    (0, -10, 25),
    (1, -8, 24),
    (2, -6, 15),
    (2, 19, 23),
    (3, -5, 15),
    (4, -4, 14),
    (5, -5, 13),
    (6, -5, 4),
    (6, 6, 12),
    (7, -6, 3),
    (7, 8, 9),
    (8, -6, 3),
    (9, -6, 4),
    (10, -6, 4),
    (11, -6, 5),
    (12, -6, 6),
    (13, -6, 7),
    (14, -5, 7),
    (15, -5, 7),
    (16, -5, 5),
    (17, -4, 5),
    (18, -4, 5),
    (19, -3, 5),
    (20, -3, 4),
    (21, -2, 4),
    (22, 0, 2)]),
    'skin121': dict(base=(689.5134888635245, 331.3117044795256), anchor=(690, 331), window=(367, 369), runs=[(-17, -6, -3),
    (-16, -8, -2),
    (-16, 6, 7),
    (-15, -9, -1),
    (-15, 4, 13),
    (-14, -10, 0),
    (-14, 2, 15),
    (-13, -10, 16),
    (-12, -11, 17),
    (-11, -11, 18),
    (-10, -11, 19),
    (-9, -10, 20),
    (-8, -10, 20),
    (-7, -9, 21),
    (-6, -9, 21),
    (-5, -8, 25),
    (-4, -8, 26),
    (-3, -9, 27),
    (-2, -9, 27),
    (-1, -9, 27),
    (0, -10, 27),
    (1, -10, 27),
    (2, -10, 27),
    (3, -10, 16),
    (3, 18, 27),
    (4, -5, 15),
    (4, 24, 27),
    (5, -4, 13),
    (5, 25, 25),
    (6, -5, 11),
    (7, -5, 9),
    (8, -6, 8),
    (9, -6, 7),
    (10, -6, 5),
    (11, -7, 5),
    (12, -7, 5),
    (13, -7, 6),
    (14, -6, 7),
    (15, -6, 7),
    (16, -6, 7),
    (17, -5, 5),
    (18, -5, 5),
    (19, -4, 5),
    (20, -4, 5),
    (21, -3, 5),
    (22, -2, 4),
    (23, 0, 4)]),
}


def _s3_mask(runs):
    A = np.zeros((2 * S3_WIN + 1, 2 * S3_WIN + 1), bool)
    for dy, x0, x1 in runs:
        A[dy + S3_WIN, x0 + S3_WIN: x1 + S3_WIN + 1] = True
    return A


_S3_ARR = {k: _s3_mask(v['runs']) for k, v in S3_TEMPLATES.items()}


def _s3_render(arr, tbase, anchor, b, s, H=540, W=1200):
    """template scaled by s about the ground point b (float frame px): full-frame bool, nearest sampling"""
    out = np.zeros((H, W), bool)
    bx, by = b; r = int(S3_WIN * max(1.0, s)) + 2
    y0, y1 = max(0, int(by) - r), min(H, int(by) + r + 1); x0, x1 = max(0, int(bx) - r), min(W, int(bx) + r + 1)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    px = (xx - bx) / s + tbase[0]; py = (yy - by) / s + tbase[1]
    sx = np.round(px).astype(int) - anchor[0] + S3_WIN; sy = np.round(py).astype(int) - anchor[1] + S3_WIN
    ok = (sx >= 0) & (sx < arr.shape[1]) & (sy >= 0) & (sy < arr.shape[0])
    sub = np.zeros(yy.shape, bool); sub[ok] = arr[sy[ok], sx[ok]]
    out[y0:y1, x0:x1] = sub
    return out


_z3_labels_prev = labels


def labels(objs, cam, R, state=None, ended=(), img=None):
    P = _z3_labels_prev(objs, cam, R, state=state, ended=ended, img=img)
    if state is None or 'lids' not in P: return P
    from scipy.ndimage import label as _cc, binary_closing, binary_fill_holes
    ids = P['ids'].copy(); share = P['share'].copy(); cv = R['cv'][::2, ::2]
    changed = False
    for it in state.get('chars', []):
        cid, did = int(it[0]), int(it[1])
        if did != S3_PET: continue
        ks = [k for k, l in enumerate(P['lids']) if l[2] == 'spawn' and l[0] == cid]
        if not ks: continue
        k = ks[0]; m = ids == k
        if m.sum() < S3_MIN: continue
        free = (ids < 0) | (ids == k)
        b = _proj(cam, int(it[3]), int(it[4]), 0.0)
        best = None
        tick = int(state.get('tick', 0))
        for name, tpl in S3_TEMPLATES.items():
            if not (tpl['window'][0] <= tick <= tpl['window'][1]): continue
            cov = []
            for s in S3_SCALES:
                S = _s3_render(_S3_ARR[name], tpl['base'], tpl['anchor'], b, s)
                cov.append(((m & S).sum() / m.sum(), s, S))
            ok = [c for c in cov if c[0] >= S3_COVER]
            if not ok: continue
            pick = ok[0]
            if best is None or pick[0] > best[0]: best = (pick[0], pick[2], name)
        holes = binary_fill_holes(m) & ~m & free
        new = holes
        if best is not None:
            S = best[1]
            if m.sum() < S3_FULL * S.sum():   # incomplete: the template fills it
                grown = binary_fill_holes(binary_closing(m | (S & free), iterations=1) | m) & free
                L, n = _cc(grown); keep = np.unique(L[m]); keep = keep[keep > 0]
                new = binary_fill_holes(np.isin(L, keep)) & free
        add = new & ~m
        if add.any():
            ids[add] = k; share[add] = np.maximum(cv[add] / 255.0, _BL.CUT); changed = True
    if changed:
        P = dict(P); P['ids'] = ids; P['share'] = share
    return P
