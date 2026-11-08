# Owner 2026-11-08 (dataset instances, 131 Ash): "theyre the same attack". An ENDED shot that the labeller keeps while its pieces stay
# visible (z_1107_ended_visible) has its predicted trail starting at the brawler, so the draws of the brawler's NEXT shot of the same
# type near the muzzle were handed to the ended one: two touching labels for one attack (131 attack 90 ticks 309-319: ended attack-0
# shot 2000002 took the start of live shot 2000003).
# Fix: every 8-connected piece of an ended object's label that touches the label of a LIVE object of the same type and the same owner
# goes to that live object. Only the label id changes (same class): label pixels, shares and the review pages stay identical; the
# ended object keeps its own pieces away from the live shot. Runs last (after the smoothing / motion / glow wrappers).
# IDLABELS_SAME_ATTACK=0 switches it off.
import os as _sa_os
import numpy as _sa_np

_sa_prev_labels = labels


def labels(objs, cam, R, state=None, ended=(), img=None):
    P = _sa_prev_labels(objs, cam, R, state=state, ended=ended, img=img)
    if _sa_os.environ.get('IDLABELS_SAME_ATTACK', '1') == '0' or not isinstance(P, dict) or state is None: return P
    try:
        live = {int(p[0]) for p in state['proj']} if len(state['proj']) else set()
    except Exception:
        return P
    info = {int(o['id']): (int(o['t']), int(o.get('owner', -1))) for o in objs}
    ids = P['ids']; lids = P['lids']
    by_obj = {}
    for k, lid in enumerate(lids):
        if lid[2] in ('body', 'ground') and int(lid[0]) in info: by_obj.setdefault(int(lid[0]), []).append(k)
    gone = [o for o in by_obj if o not in live]
    alive = [o for o in by_obj if o in live]
    if not gone or not alive: return P
    from scipy.ndimage import label as _sa_cc, binary_dilation as _sa_dil
    st = _sa_np.ones((3, 3), bool)
    for e in gone:
        te, oe = info[e]
        mates = [l for l in alive if info[l] == (te, oe)]
        if not mates: continue
        me = _sa_np.isin(ids, by_obj[e])
        if not me.any(): continue
        L, n = _sa_cc(me, structure=st)
        for l in mates:
            ml = _sa_np.isin(ids, by_obj[l])
            if not ml.any(): continue
            near = _sa_dil(ml, structure=st)
            kl = by_obj[l][0]   # the live object's first label part (its body)
            for j in _sa_np.unique(L[near & me]):
                if j == 0: continue
                piece = L == j
                ids[piece] = kl; L[piece] = 0; me &= ~piece
    P['ids'] = ids
    return P
