# human review 2026-10-28 fix for skin 118 (Haiku fix, reviewed by the coordinator, installed 10-29; source out/sweep2/haikutasks/human1028/s118/fix3.py; tested via IDLABELS_OVERRIDE after all fixes incl. z_1099_cleanup, hence z_1100_)
# s118 fix3 (human review 2026-10-28; supersedes fix2.py, load this one alone with IDLABELS_OVERRIDE).
# Skin 118 JesterWoodspirit, type 6001270 (Chester002Projectile, skin 118's own attack; 119 and 120 fire 6001275 / 6001280).
# Finding (task.md): row "attack 90 deg", ticks 305, 307, 309 (frames 1839-1841): "too small / cut off". The ghost's solid
# green rim is outside the label. The rim is kept (the coordinator's review).
#
# Rim: the SOLID green band of the ghost is added to the body. A candidate is a pixel of any shape of the type's clip
# (the glow shape 0 and the rim-ring shapes 63, 64, 407) with frame colour G >= S118H_G and G - B >= S118H_GB (measured on
# sw2_118: the solid rim is R 27, G 255, B 150-160; the dark teal halo beyond it has G 70-130; the blue wall beside the
# left ghost has G 170 / B 225 and fails G - B). Candidates must connect to the body through candidates, within
# S118H_RIM_MAX px of it. fix2 took the clip's glow shape 0 within 3-5 px with no colour test, so the halo of the left
# ghost (next to the wall) and of both ghosts was added: the left outline ran several px into the halo.
#
# Shape: union of the body and the rim, closing (radius-2 disk), fill holes, opening 3x3, blur-threshold (sigma 1,
# threshold 0.5), fill holes; only parts touching the body are kept. One solid piece, one smooth outline.
#
# Overlap (the coordinator's correction, 2026-10-28: one owner per pixel): where two ghosts overlap, the FRONT ghost owns
# the shared pixels and the back ghost gets its visible part only. Front = the ghost whose body draws (the ID pass's draw
# ids of its body) show the shared pixel; a shared pixel that neither or both show goes to the nearer body.
# Only type 6001270; nothing of another label is taken (labels of other objects are never added to).
import numpy as np

S118H_TYPE = 6001270
S118H_RIM = 3          # frame px: the solid green rim of the ghost (flat band; kept for reference)
S118H_RIM_MAX = 4      # frame px: the solid green rim is followed from the body through connected solid pixels up to this far (the spark dots beside the ghost sit 5-7 px out)
S118H_G = 170          # solid green: G at least this
S118H_GB = 50          # solid green: G - B at least this (teal halo and blue wall fail)
S118H_BLUR = 1.0       # sigma of the border smoothing
S118H_BLUR_T = 0.5     # threshold after the blur
_S118H_DISK = np.array([[0, 1, 1, 1, 0], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [0, 1, 1, 1, 0]], bool)
_S118H_SQ = np.ones((3, 3), bool)
_S118H_DISK3 = np.array([[(x * x + y * y) <= 9 for x in range(-3, 4)] for y in range(-3, 4)], bool)  # radius 3


def _s118h_rim_shapes(R):
    """shape id + 1 per pixel of the type's clip (frame resolution); shape 0 is the clip's glow layer (the rim)"""
    T = projtab.table()[S118H_TYPE]; names = tuple(sorted({T[k] for k in ('blue', 'red') if T.get(k)}))
    return _a4_shape_pixels(R, T['file'], names)


def _s118h_keep_touching(V, m):
    """parts of V that touch the body m (dilated by 1 px); detached parts (specks, spark dots) are dropped"""
    from scipy import ndimage as nd
    L, n = nd.label(V, structure=_S118H_SQ)
    if n <= 1: return V
    touch = np.unique(L[nd.binary_dilation(m, structure=_S118H_SQ) & V]); touch = touch[touch > 0]
    return np.isin(L, touch) & V


def _s118h_clean(U, m, dist):
    """one solid piece with one smooth outline: closing (gaps of the rim), fill holes, opening (spurs), blur-threshold, fill"""
    from scipy import ndimage as nd
    pad = 6
    V = np.pad(U, pad)
    V = nd.binary_closing(V, structure=_S118H_DISK)
    V = nd.binary_fill_holes(V)
    V = nd.binary_opening(V, structure=_S118H_SQ)
    g = nd.gaussian_filter(V.astype(np.float32), S118H_BLUR)
    V = nd.binary_fill_holes(g > S118H_BLUR_T)
    V = V[pad:-pad, pad:-pad]
    V = V & (dist <= S118H_RIM_MAX)           # growth limited to the rim's reach from the body
    # beyond the rim band (S118H_RIM px) only thick parts are kept (the green spark dots near the ghost are ~6 px wide and
    # thin necks join them: a radius-3 opening removes them; the rim band itself is always kept)
    V = V & ((dist <= S118H_RIM) | nd.binary_opening(V, structure=_S118H_DISK3))
    V = _s118h_keep_touching(V, m)
    return nd.binary_fill_holes(V) & (dist <= S118H_RIM_MAX)


_s118h_prev = labels


def labels(objs, cam, R, state=None, ended=(), img=None):
    import bulletlabels as BL
    from scipy import ndimage as nd
    P = _s118h_prev(objs, cam, R, state=state, ended=ended, img=img)
    mine = {BL._lid(o) for o in objs if o['t'] == S118H_TYPE}
    if not mine: return P
    ids, lids, share = P['ids'], P['lids'], P['share']
    ks = [k for k, lid in enumerate(lids) if lid[2] == 'body' and lid[0] in mine and (ids == k).any()]
    if not ks: return P
    cv = R['cv'][::2, ::2]; idh = R['id'][::2, ::2]
    rim0 = _s118h_rim_shapes(R) > 0      # any shape of the clip (glow shape 0 and the rim-ring shapes); solid colour decides
    if img is not None:
        rgb = img.astype(np.int16)
        solid = (rgb[..., 1] >= S118H_G) & (rgb[..., 1] - rgb[..., 2] >= S118H_GB)
    else:
        solid = np.ones(ids.shape, bool)
    body = {k: ids == k for k in ks}
    dist = {k: nd.distance_transform_edt(~body[k]) for k in ks}
    other = (ids >= 0) & ~np.isin(ids, ks)     # pixels of labels that are not these ghosts: never taken
    fin = {}
    for k in ks:
        # the solid green rim, connected to the body (the halo stops it: not solid), within S118H_RIM_MAX px
        cand = rim0 & solid & ~other & (dist[k] <= S118H_RIM_MAX)
        L, n = nd.label(cand | body[k], structure=_S118H_SQ)
        add = np.isin(L, np.unique(L[body[k]])) & cand & ~body[k]
        U = body[k] | add
        fin[k] = _s118h_clean(U, body[k], dist[k]) & ~other
    # overlaps: the pixel goes to the ghost whose draw it shows (the front one); else to the nearer body
    for i, a in enumerate(ks):
        for b in ks[i + 1:]:
            O = fin[a] & fin[b]
            if not O.any(): continue
            Da = np.unique(idh[body[a]]); Db = np.unique(idh[body[b]])
            ina = np.isin(idh, Da); inb = np.isin(idh, Db)
            front_b = O & inb & ~ina
            amb = O & ~(ina ^ inb)
            to_b = front_b | (amb & (dist[b] < dist[a]))
            to_a = O & ~to_b
            fin[a] = fin[a] & ~to_b; fin[b] = fin[b] & ~to_a
    for k in ks:
        fin[k] = _s118h_keep_touching(fin[k], body[k]) & ~other
        fin[k] = nd.binary_fill_holes(fin[k]) & ~other
    for k in ks:
        share[body[k]] = 0
    for k in ks:
        ids[body[k]] = -1
    for k in ks:
        f = fin[k]
        ids[f] = k; share[f] = np.maximum(cv[f] / 255.0, BL.CUT)
    return P
