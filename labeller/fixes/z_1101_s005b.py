# hosted review 2026-10-29 fix (Haiku fix, reviewed by the coordinator, installed 10-29 LOCAL ONLY; source out/sweep2/haikutasks/site1029/s005/fix2.py; tested via IDLABELS_OVERRIDE after all fixes incl. z_1100_*, hence z_1101_)
# Skin 005 DuplicatorDarkcircus, the Duplicator's clone (spawnable DuplicatorPet, character type 16000199), spawn ticks.
# Coordinator review of fix.py (2026-10-29): ticks 361-369 are not clean (partial pieces at 363-367, outline bits inside the
# white spawn flash at 361, small holes and loops inside the clone at 369). Hidden-parts rule: an object covered by an effect
# (the green veil, the spawn flash) keeps its FULL shape. Ticks 371-411 stay exactly as fix.py makes them.
#
# (1) TEMPLATE (skin 005, the clone under the veil): the clone's label at tick 371 (fix.py, accepted: the whole clone, cup,
#     horn, fin and body) is the template, as run-length rows relative to the rounded ground anchor (AX0, AY0) of that tick.
#     Placed at the clone's ground point of the tick (anchor drift between 361 and 371: 0.1 px) and used only where
#     - the clone's visible top row is the template's top row (294, +-1 px): the clone's own silhouette, not another pet;
#     - the opaque 3D model covers less than F2_OPAQUE of the template (the clone is veiled: 361 33 %, 363 18 %, 365 3 %,
#       367 3 %, 369 24 %; tick 371 61 % and later 84-95 % keep fix.py's label);
#     Measured per tick (sw2_005): the clone does not scale between 361 and 371: its visible top row is 294-295 from 361 to
#     411 and its bottom row 341-342 from 367 on, so the placement is at scale 1 with no shift (checked, not fitted).
#     The label is the template inside the free pixels (not another label's), solid, one piece: the spawn flash's bits and
#     the old partial pieces are replaced; holes are filled by the template itself.
# (2) PET GLOW (the same pet with another skin look: skins 000 and 002, whose top row is 2-3 px higher, so (1) does
#     not apply): fix.py's hull fill takes the blue glow around the pet's outline into the label (a straight boxy
#     right edge at 363-367). Label pixels that are glow-coloured (G - R > F2_GLOW), not the opaque model and outside
#     the hull of the pet's own pink / violet pixels are removed. Its translucent cyan body and crest (inside that
#     hull) and enclosed holes stay; the one-piece rule keeps the main part.
# Keyed to type 16000199 only. Nothing else in the frame changes.
import numpy as np
from scipy.ndimage import binary_fill_holes as _f2_fill, label as _f2_cc

F2_CHAR = 16000199
F2_ANCHOR_REF = (694.2, 336.22)   # ground anchor of the template clone (skin 005, tick 371)
F2_AX0, F2_AY0 = 694, 336         # rounded anchor: the template's origin
F2_TOP = 294                      # top row of the template (frame px, skin 005)
F2_TOP_TOL = 1                    # the visible top row must be this close to the template's (else: another pet, rule 2)
F2_OPAQUE = 0.5                   # opaque-model share of the template below which the clone is veiled (rule 1)
F2_PINK = 30                      # min(R, B) - G of a clone-pink frame pixel
F2_PINK_ROWS = 3                  # a row counts as the clone's top / bottom when it has this many pink px
F2_PINK_IN = 30                   # pink px inside the template needed to place it
F2_GLOW = 40                      # G - R above which a label pixel is glow / veil / ring (rule 2)
F2_RLE = [(-42, -4, -4), (-41, -7, 4), (-41, 11, 16), (-40, -9, 5), (-40, 8, 17), (-39, -11, 17), (-38, -12, 17), (-37, -13, 17), (-36, -14, 16), (-35, -15, 16), (-34, -16, 15), (-33, -17, 15), (-32, -17, 14), (-31, -18, 15), (-30, -18, 15), (-29, -18, 15), (-28, -18, 14), (-27, -18, 13), (-26, -19, 14), (-25, -18, 15), (-24, -18, 16), (-23, -19, 17), (-22, -20, 18), (-21, -20, 18), (-20, -20, 18), (-19, -21, 18), (-18, -21, 18), (-17, -21, 18), (-16, -21, 17), (-15, -22, 18), (-14, -22, 18), (-13, -22, 18), (-12, -22, 18), (-11, -22, 18), (-10, -21, 18), (-9, -21, 18), (-8, -19, 17), (-7, -19, 17), (-6, -19, 16), (-5, -18, 16), (-4, -17, 15), (-3, -17, 14), (-2, -16, 14), (-1, -16, 13), (0, -15, 13), (1, -14, 12), (2, -13, 9), (3, -12, 7), (4, -10, 5), (5, -8, 3), (6, -5, -2)]   # (y, x0, x1) rows, relative to (AX0, AY0): fix.py's label of tick 371


def _f2_grid():
    ys = [r[0] for r in F2_RLE]; xs0 = [r[1] for r in F2_RLE]; xs1 = [r[2] for r in F2_RLE]
    oy = -min(ys) + 2; ox = -min(xs0) + 2
    G = np.zeros((max(ys) + oy + 3, max(xs1) + ox + 3), bool)
    for y, a, b in F2_RLE: G[y + oy, a + ox:b + ox + 1] = True
    return G, oy, ox


F2_TG, F2_OY, F2_OX = _f2_grid()


def _f2_place(ax, ay, y0, y1, x0, x1):
    """template placed at the clone's anchor (ax, ay), scale 1, no shift: (y1 - y0, x1 - x0) bool"""
    yy, xx = np.mgrid[y0:y1, x0:x1]
    X = np.round(xx - ax + F2_ANCHOR_REF[0]).astype(int) - F2_AX0 + F2_OX
    Y = np.round(yy - ay + F2_ANCHOR_REF[1]).astype(int) - F2_AY0 + F2_OY
    ok = (X >= 0) & (X < F2_TG.shape[1]) & (Y >= 0) & (Y < F2_TG.shape[0])
    out = np.zeros(xx.shape, bool); out[ok] = F2_TG[Y[ok], X[ok]]
    return out


_f2_prev = labels
F2_LOG = []   # (tick-free) per clone decisions of the last frame, for the coordinator's checks


def labels(objs, cam, R, state=None, ended=(), img=None):
    P = _f2_prev(objs, cam, R, state=state, ended=ended, img=img)
    if state is None or img is None: return P
    ids, lids, share = P['ids'], P['lids'], P['share']
    cv = R['cv'][::2, ::2]; ids_h = R['id'][::2, ::2]
    models = [d for d, dr in R['draws'].items() if dr.get('th') == MODEL_TEX]
    im = img.astype(np.int16)
    H, W = ids.shape
    new_ids = None; F2_LOG[:] = []
    for c in state['chars']:
        cid, did, own = int(c[0]), int(c[1]), int(c[2])
        if did != F2_CHAR or own < 0: continue
        lid = (cid, -1, 'spawn')
        if lid not in lids: continue
        k = list(lids).index(lid)
        ax, ay = _proj(cam, int(c[3]), int(c[4]), 0)
        x0 = max(0, int(ax) - 42); x1 = min(W, int(ax) + 43); y0 = max(0, int(ay) - 64); y1 = min(H, int(ay) + 22)
        sl = (slice(y0, y1), slice(x0, x1))
        mine = ids[sl] == k
        if not mine.any(): continue
        if new_ids is None: new_ids = ids.copy(); new_share = share.copy()
        free = (ids[sl] < 0) | mine
        g = im[sl]
        pink_all = (np.minimum(g[..., 0], g[..., 2]) - g[..., 1]) >= F2_PINK
        pink = pink_all
        rows = np.nonzero(pink.sum(1) >= F2_PINK_ROWS)[0]
        top = (y0 + rows.min()) if rows.size else None
        T = _f2_place(ax, ay, y0, y1, x0, x1)
        op = np.isin(ids_h[sl], models) & (cv[sl] >= CUT)
        ta = int(T.sum())
        same_pet = top is not None and abs(top - F2_TOP) <= F2_TOP_TOL
        if same_pet:
            frac = int((op & T).sum()) / max(1, ta); pin = int((pink & T).sum())
            veiled = frac < F2_OPAQUE and pin >= F2_PINK_IN
            F2_LOG.append(dict(lid=lid, rule=1 if veiled else 0, top=top, opaque=round(frac, 2), pink_in=pin))
            if not veiled: continue
            new = T & free
            L, n = _f2_cc(new, structure=np.ones((3, 3)))
            if n > 1:
                sz = np.bincount(L.ravel()); sz[0] = 0; new = L == int(np.argmax(sz))
        else:
            glow = (g[..., 1] - g[..., 0]) > F2_GLOW
            # only glow outside the hull of the pet's own pink / violet pixels: its translucent body (cyan-tinted) stays
            core_p = mine & pink_all
            hp = hull_mask(core_p) if core_p.sum() >= F2_PINK_IN else mine
            rem = mine & glow & ~op & ~hp
            F2_LOG.append(dict(lid=lid, rule=2, top=top, removed=int(rem.sum())))
            if not rem.any(): continue
            core = mine & ~rem
            new = core | (_f2_fill(core) & free)
            L, n = _f2_cc(new, structure=np.ones((3, 3)))
            if n > 1:
                keep = np.unique(L[core]); keep = keep[keep > 0]
                new = np.isin(L, keep)
        sub_ids = new_ids[sl]; sub_sh = new_share[sl]
        drop = mine & ~new; add = new & ~mine
        sub_ids[drop] = -1; sub_sh[drop] = 0
        sub_ids[add] = k; sub_sh[add] = np.maximum(cv[sl][add] / 255.0, CUT)
    if new_ids is None: return P
    P2 = dict(P); P2.update(ids=new_ids, share=new_share)
    return P2
