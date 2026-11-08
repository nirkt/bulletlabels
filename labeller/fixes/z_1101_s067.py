# hosted review 2026-10-29 fix (Haiku fix, reviewed by the coordinator, installed 10-29 LOCAL ONLY; source out/sweep2/haikutasks/site1029/s067/fix.py; tested via IDLABELS_OVERRIDE after all fixes incl. z_1100_*, hence z_1101_)
# Skin 067 (Flea, super 45): the FleaBigEgg spawnable (character did 16000204, a 3D model, its label = the model's draw
# pixels, spawn_labels) was cut off at its top. The egg's crown (yellow / lime lobes around a blue hole) is drawn over
# its mesh by translucent layers, so the ID pass leaves those crown pixels unlabelled (only the lower, opaque part is
# the model). The reviewer (the user, 2026-10-29): 'too small / cut off' at super 45 t371, t389, t411.
# Fix: the egg's label gets the crown pixels that are part of the egg: lime / yellow pixels (G - B >= 60, G >= 150,
# frame colour) connected to its label within 10 px; then closed and hole-filled (the blue hole in the crown), limited to
# the hull of the whole, and only unlabelled pixels are taken. The teal haze above the egg (G - B < 60) and the cyan
# zone disc stay out. Keyed to this one character did only (other spawnables and attacks of other skins untouched).

EGG_CHAR_DID_067 = 16000204
_labels_before_067 = labels


def labels(objs, cam, R, state=None, ended=(), img=None):
    P = _labels_before_067(objs, cam, R, state=state, ended=ended, img=img)
    if state is None or img is None or not P.get('lids'): return P
    from scipy.ndimage import label as _cc, binary_fill_holes as _fill
    import bulletlabels as _BL
    did_of = {int(c[0]): int(c[1]) for c in state['chars']}
    ks = [k for k, lid in enumerate(P['lids']) if lid[2] == 'spawn' and did_of.get(lid[0]) == EGG_CHAR_DID_067]
    if not ks: return P
    ids = P['ids']; out_ids = ids.copy(); share = P['share'].copy(); cv = R['cv'][::2, ::2]
    G = img[..., 1].astype(np.int16); B = img[..., 2].astype(np.int16)
    lime = (G - B >= 60) & (G >= 150)
    for k in ks:
        m = ids == k
        if not m.any(): continue
        crown = lime & _dilate(m, 10) & (ids < 0)
        L, n = _cc(m | crown); keep = np.unique(L[m]); keep = keep[keep > 0]
        X = np.isin(L, keep)
        S = _fill(_close(X, 2)) & hull_mask(X)
        add = S & (ids < 0)
        if not add.any(): continue
        out_ids[add] = k; share[add] = np.maximum(cv[add] / 255.0, _BL.CUT)
    P = dict(P); P['ids'] = out_ids; P['share'] = share
    return P
