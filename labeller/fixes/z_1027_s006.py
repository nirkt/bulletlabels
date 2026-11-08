# sweep 10-27 fix for skin 006 (Haiku fix, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s006/)
# Skin 006 DuplicatorGiftsToy (sweep 2026-10-27, finding pair 152: super 45 deg, tick 371, 'hole').
# The Duplicator pet (spawnable, character did 16000199, class 'spawn', blue outline) carries a cyan gem on the edge of its
# magenta 3D model. The gem is drawn by translucent additive sprites (coverage 27-73, not the model's draw), so it is not
# labelled, and the outline dents around it: an open notch on the right edge, not a closed hole, so the shared clean-up
# (z_1099, fills closed holes only) does not reach it. The verifier's note: the gem is part of the object.
# Fix: the pet's own label is closed with a 4 px disc, and only unlabelled pixels drawn with the gem's own sprite textures
# are added (the notch is exactly those draws at tick 371). Keyed to this spawnable's type id and to the gem textures
# (atlas texture hashes, stable across captures); other spawnables and all projectiles are untouched.
# Runs after z_1099_cleanup.py (override files are exec'd last), so it sees the final label map.

GEM_TH = {'7694b06742396e51', 'f9aa5e26cf4ee229'}   # texture hashes of the gem's sprites (draws at the notch, ticks 361-411)
SPAWN_CLOSE = {16000199: 4}   # spawnable type (did) -> closing radius, px (full resolution)
SPAWN_HOLE_MAX = 150          # px: enclosed holes of the pet's label drawn only with the gem sprites (sparkles over the body) are filled


def _s006_disk(r):
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r


_s006_prev_labels = labels


def labels(objs, cam, R, state=None, ended=(), img=None):
    from scipy.ndimage import binary_closing, binary_fill_holes, label as cc
    import bulletlabels as BL
    P = _s006_prev_labels(objs, cam, R, state=state, ended=ended, img=img)
    if state is None: return P
    ids, lids, share = P['ids'], P['lids'], P['share']
    cvh = R['cv'][::2, ::2]
    RID = R['id'][::2, ::2]
    gem_d = [d for d, dr in R['draws'].items() if dr.get('th') in GEM_TH]
    isgem = np.isin(RID, gem_d) if gem_d else np.zeros(ids.shape, bool)
    for c in state['chars']:
        cid, did = int(c[0]), int(c[1])
        if did not in SPAWN_CLOSE: continue
        ks = [k for k, l in enumerate(lids) if l[0] == cid and l[2] == 'spawn']
        if not ks: continue
        k = ks[0]; m = ids == k
        if not m.any(): continue
        r = SPAWN_CLOSE[did]; pad = r + 2
        ys, xs = np.nonzero(m)
        y0, y1 = max(0, ys.min() - pad), min(ids.shape[0], ys.max() + pad + 1)
        x0, x1 = max(0, xs.min() - pad), min(ids.shape[1], xs.max() + pad + 1)
        sm = m[y0:y1, x0:x1]
        c_ = binary_closing(np.pad(sm, r + 2), structure=_s006_disk(r))[r + 2:-(r + 2), r + 2:-(r + 2)]
        free = (ids[y0:y1, x0:x1] < 0) & isgem[y0:y1, x0:x1]
        add = c_ & ~sm & free
        # enclosed holes of the pet (sparkles drawn over the body, in the gem's sprites): filled up to SPAWN_HOLE_MAX px
        base = sm | add
        hole = binary_fill_holes(base) & ~base
        if hole.any():
            HL, hn = cc(hole)
            for j in range(1, hn + 1):
                hj = HL == j
                if hj.sum() <= SPAWN_HOLE_MAX and (free & hj).sum() == hj.sum(): add |= hj
        if not add.any(): continue
        ids_c = ids[y0:y1, x0:x1]; sh_c = share[y0:y1, x0:x1]; cv_c = cvh[y0:y1, x0:x1]
        ids_c[add] = k
        sh_c[add] = np.maximum(cv_c[add] / 255.0, BL.CUT)
    P['ids'], P['share'] = ids, share
    return P
