# hosted review 2026-10-29 fix (Haiku fix, reviewed by the coordinator, installed 10-29 LOCAL ONLY; source out/sweep2/haikutasks/site1029/s088/fix.py; tested via IDLABELS_OVERRIDE after all fixes incl. z_1100_*, hence z_1101_)
# Skin 088 (Twins Default) fix, hosted review 2026-10-29 (the user, binding): super 45 row, tick 389, "holes / ragged".
# The flagged frame has two labels, both ragged or holed:
#  (a) the Twins pet's shotgun volley (TwinsShotgunProjectile 6001402, one label per volley from z_1027_s088). Its label took in
#      the light-blue electric cloud beside the blades (shape 0 of the volley clip: sparks and particles, never labelled, rule 3),
#      so its outline was ragged along the cloud and notched. Fix: remove the cloud pixels from each volley label, then smooth
#      it into one solid piece in its own outline: closing 2 px, hole fill, Gaussian sigma 2.0 (threshold 0.5), growth at most
#      1 px beyond the blades, largest piece only. Other shapes of the blade bars are kept as they are. No hull.
#  (b) the Twins pet itself (spawnable 3D model; its label id is the spawn id that shot the volley). Its left arm had a notch
#      under the cloud (the hidden-part rule: the label keeps the full shape). Fix: close the pet label by PET_CLOSE = 4 px; the
#      closing may add unlabelled pixels only (never another label), so the leg gap stays as it is up to the closing reach.
# Keyed to type 6001402 (and to the spawn that shot it): no other type changes. MegaBossTwinsShotgunProjectile 6001707 (same clip),
# the Angel, Twins 002 and Twins 004 pet types are other types and unchanged. Runs as a labels() wrapper after the installed
# fixes (incl. z_1027_s088), so it sees the merged volley labels. Shape pixels from _a4_shape_pixels (the clips of tools/shapediag.py).
_S88_VOLLEY = 6001402
_S88_FILE = 'sc/effects_brawler.sc'
_S88_CLIPS = ('elecDagger_trail_blue1', 'elecDagger_trail_blue2', 'elecDagger_trail_blue3', 'elecDagger_trail_blue4',
              'twin_def_ulti_prj', 'twin_def_ulti_prj_red')
_S88_CLOUD = 1        # SP value of shape 0 (SP = shape id + 1): the light-blue electric cloud beside the blades
_S88_CLOSE = 2        # px, closing of the volley blades (gaps between the blades)
SPECK = 15         # px: pieces of the volley label smaller than this (and under 10 % of the largest) are dropped
SMOOTH = 2.0         # px, Gaussian sigma of the volley outline (tested 1.5 and 2.0)
PET_CLOSE = 4         # px, closing of the pet's label (the notch under the cloud)
PET_ANY = True        # True: the closing may add any unlabelled pixel (the notch's own pixels are not cloud); False: cloud only
_s88_prev_labels = labels


def _s88_disk(r):
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r


def _s88_volley(m, keep_off):
    """one volley label: its blades (the cloud pixels removed), closed by CLOSE_R px, Gaussian-smoothed (sigma SMOOTH, threshold
    0.5: thin spurs go, the blade bars stay), filled, largest piece; growth at most 1 px beyond the blades.
    m: its label pixels; keep_off: the cloud pixels"""
    from scipy.ndimage import gaussian_filter, binary_fill_holes, binary_closing, binary_dilation, label as cc
    keep = m & ~keep_off
    if not keep.any(): return keep
    c = binary_closing(np.pad(keep, _S88_CLOSE + 1), structure=_s88_disk(_S88_CLOSE))[_S88_CLOSE + 1:-_S88_CLOSE - 1,
                                                                                    _S88_CLOSE + 1:-_S88_CLOSE - 1] | keep
    c = binary_fill_holes(c)
    s = binary_fill_holes(gaussian_filter(c.astype(np.float32), SMOOTH) >= 0.5)
    s &= binary_dilation(keep, iterations=1) | c
    L, n = cc(s)                                                    # specks only: a second blade bar of the volley stays
    if n > 1:
        sz = np.bincount(L.ravel())[1:]
        drop = [i + 1 for i in range(n) if sz[i] < SPECK and sz[i] < 0.1 * sz.max()]
        if drop: s &= ~np.isin(L, drop)
    return s


def labels(objs, cam, R, state=None, ended=(), img=None):
    P = _s88_prev_labels(objs, cam, R, state=state, ended=ended, img=img)
    vol = [o for o in objs if o['t'] == _S88_VOLLEY]
    if not vol: return P
    ids, lids, share = P['ids'], P['lids'], P['share']
    ltype, lsrc = {}, {}
    for o in objs:
        ltype.setdefault(BL._lid(o), o['t']); lsrc.setdefault(BL._lid(o), o.get('src'))
    SP = _a4_shape_pixels(R, _S88_FILE, _S88_CLIPS)                    # shape id + 1 per pixel, 0: none
    cloud = SP == _S88_CLOUD
    # (a) the volley labels: blades only, one smooth solid piece per label
    for k, l in enumerate(lids):
        if l[2] != 'body' or ltype.get(l[0]) != _S88_VOLLEY: continue
        m = ids == k
        if not m.any(): continue
        s = _s88_volley(m, cloud)
        ids[m & ~s] = -1; share[m & ~s] = 0
        add = s & ~m & (ids < 0) & ~cloud
        ids[add] = k; share[add] = BL.CUT
    # (b) the pet that shot the volley (its spawn label): close the notch that the cloud makes, adding cloud pixels only
    pets = {o.get('src') for o in vol}
    for k, l in enumerate(lids):
        if l[2] != 'spawn' or l[0] not in pets: continue
        m = ids == k
        if not m.any(): continue
        c = binary_closing(np.pad(m, PET_CLOSE + 1), structure=_s88_disk(PET_CLOSE))[PET_CLOSE + 1:-PET_CLOSE - 1,
                                                                                  PET_CLOSE + 1:-PET_CLOSE - 1]
        add = c & ~m & (ids < 0) & (cloud | PET_ANY)
        ids[add] = k; share[add] = BL.CUT
    return P
