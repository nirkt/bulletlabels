# hosted review 2026-10-29 fix (Haiku fix, reviewed by the coordinator, installed 10-29 LOCAL ONLY; source out/sweep2/haikutasks/site1029/g_flea/fix.py; tested via IDLABELS_OVERRIDE after all fixes incl. z_1100_*, hence z_1101_)
# Fix task site1029 / g_flea (skins 127 FleaSpaceCactus, 134 FleaBiker), hosted review 2026-10-29. User finding (binding):
# super 45, the Flea super's egg (spawnable, character did 16000204 FleaBigEgg, drawn as a 3D model) is 'too small / cut
# off': skin 127 at t371; skin 134 at t371, t389, t411. (Skin 067's crown is the accepted s067/fix.py: not touched here.)
#
# Cause: the spawn label of the egg is its 3D model's draw (coverage >= CUT). The crown on top of the egg (the lime lobe of
# 127 and the lime cap of 134) is drawn with the shared effect atlas 7694b067 (2560 x 4064, additive) at a translucent
# coverage (about 30-110, below CUT), so the ID pass leaves it unlabelled: the label is cut off at the top.
#
# Fix: the unlabelled pixels of that atlas's own draws, coverage FLEA_EGG_FADE .. CUT, within FLEA_EGG_NEAR px of the egg's
# label, connected to it, join the egg's label; then closed (2 px), hole-filled and limited to the hull of the whole, as in
# s067/fix.py. Not taken: the teal haze above the egg and the cyan zone disc (4096 px atlases, other draws), the zone's blue
# crystals (opaque atlas draws, coverage >= CUT), the black stroke, the glow.
#
# Scope: the same egg character (did 16000204) is spawned by every flea skin, so the rule is keyed to the owner's skin
# (info.skin of the capture the frame comes from, script.json): only skins 127 and 134 get it; with the skin unknown,
# nothing changes. Skin 067 (FleaDefault, 29000460) is therefore unchanged by construction (checked on its super ticks).
import json as _fl_json
import bulletlabels as _FL_BL

FLEA_EGG_DID = 16000204            # FleaBigEgg (characters.csv, Minion_Building spawnable of the Flea super)
FLEA_EGG_SKINS = {29000478, 29000611}   # FleaSpaceCactus (127), FleaBiker (134)
FLEA_EGG_ATLAS = '7694b067'        # hash prefix of the shared effect atlas (2560 x 4064) its crown is drawn with
FLEA_EGG_FADE = 32                 # translucent crown layers: coverage from which they are the egg's (fainter = glow)
FLEA_EGG_NEAR = 10                 # px (frame resolution) around the egg's label in which its crown is searched
_fl_skin_lists = {}


def _fl_player_skin(R, own):
    """skin id of player `own` in the capture the ID-pass frame R comes from (R['dir'] = capture/id/<run>/), None if unknown"""
    import os
    d = R.get('dir')
    if not d: return None
    d = os.path.abspath(d)
    for _ in range(4):
        p = os.path.join(d, 'script.json')
        if os.path.exists(p):
            if p not in _fl_skin_lists:
                try: _fl_skin_lists[p] = [int(s) for s in _fl_json.load(open(p))['info']['skin']]
                except Exception: _fl_skin_lists[p] = []
            sk = _fl_skin_lists[p]
            return sk[own] if 0 <= own < len(sk) else None
        d = os.path.dirname(d)
    return None


_fl_labels_prev = labels


def labels(objs, cam, R, state=None, ended=(), img=None):
    P = _fl_labels_prev(objs, cam, R, state=state, ended=ended, img=img)
    if state is None or not P.get('lids'): return P
    eggs = {int(c[0]) for c in state.get('chars', [])
            if int(c[1]) == FLEA_EGG_DID and _fl_player_skin(R, int(c[2])) in FLEA_EGG_SKINS}
    ks = [k for k, lid in enumerate(P['lids']) if lid[2] == 'spawn' and lid[0] in eggs]
    if not ks: return P
    from scipy.ndimage import label as _fl_cc, binary_fill_holes as _fl_fill
    ids = P['ids']; out_ids = ids.copy(); share = P['share'].copy(); cv = R['cv'][::2, ::2]
    if img is None: return P
    ids_h = R['id'][::2, ::2]
    atlas = [d for d, dr in R['draws'].items() if str(dr.get('th', '')).startswith(FLEA_EGG_ATLAS) and dr.get('blend') == 1]
    # the crown is warm (lime, yellow, orange): the same atlas also draws the teal haze above the egg (not warm), so the
    # colour of the frame decides which of its translucent pixels are the egg's
    im = img.astype(np.int16); warm = ((im[..., 1] - im[..., 2]) >= 60) | ((im[..., 0] - im[..., 2]) >= 100)
    crown = np.isin(ids_h, atlas) & (cv >= FLEA_EGG_FADE) & (cv < CUT)
    for k in ks:
        m = ids == k
        if not m.any(): continue
        # warm pixels of the crown, or any of its pixels inside the hull of the model (the egg's body, not the haze above it)
        cand = _dilate(m, FLEA_EGG_NEAR) & crown & (ids < 0) & (warm | hull_mask(m))
        L, n = _fl_cc(m | cand); keep = np.unique(L[m]); keep = keep[keep > 0]
        X = np.isin(L, keep)
        S = _fl_fill(_close(X, 2)) & hull_mask(X)
        add = S & (ids < 0)
        if not add.any(): continue
        out_ids[add] = k; share[add] = np.maximum(cv[add] / 255.0, _FL_BL.CUT)
    P = dict(P); P['ids'] = out_ids; P['share'] = share
    return P
