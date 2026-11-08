# sweep 10-27 fix for skin 182 (Haiku fix, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s182/)
# Skin 182 RopeDudeGodzilla (Buzz's super, the boat), sweep 2026-10-27, pair 3183 (super 45 deg, tick 347):
# a small cyan bubble detached at the lower-left of the boat was labelled as a 33 px part of the boat's own label. The
# user's rule: detached bubbles / dots stay out; the boat is the label. Its sprite is shape 63 (clip of the boat), the
# same shape as the bubble, so the cut is by connectivity, not by shape: in the label of this type, every part that is
# detached from the main part (4-connected) and smaller than BUZZ_ULTI_DETACHED_PX is dropped; the largest part is
# always kept. Only type 6000993 (Buzz007UltiProjectile, logic RopeDudeUltiProjectile; the super of this brawler, also in
# its other skins) is touched. At tick 345 the label is one part, so nothing changes there.
BUZZ_ULTI_DETACHED = {6000993}
BUZZ_ULTI_DETACHED_PX = 40

_prev_s182 = labels


def labels(objs, cam, R, state=None, ended=(), img=None):
    from scipy.ndimage import label as _cc
    import bulletlabels as BL
    P = _prev_s182(objs, cam, R, state=state, ended=ended, img=img)
    ids, lids = P['ids'], P['lids']
    keys = {BL._lid(o) for o in objs if o['t'] in BUZZ_ULTI_DETACHED}
    if not keys: return P
    for k in np.unique(ids[ids >= 0]):
        if lids[k][0] not in keys: continue
        m = ids == k
        L, n = _cc(m)
        if n < 2: continue
        sz = np.bincount(L.ravel())[1:]
        big = int(np.argmax(sz)) + 1
        for j in range(1, n + 1):
            if j != big and sz[j - 1] < BUZZ_ULTI_DETACHED_PX:
                d = L == j
                ids[d] = -1
                if 'share' in P: P['share'][d] = 0
    return P
