# sweep 10-27 fix for skin 329 (Haiku fix, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s329/)
# s329 (sweep 2026-10-27, Ice Dude 'Stonetroll' skin 329, attack 0 deg, pairs 6146 (tick 271) and 6147 (tick 273)).
# The snowball (Lou005Projectile, type 6000855) hits the crate and its impact burst is drawn from the snowball's own
# atlas shape (shape 63, the clip body), so no shape or colour rule separates the burst from the body. The projectile is
# still alive in the logic for about two ticks after the impact (finished 4 and ENDED two ticks later), so the burst is
# labelled by the live snowball whose predicted position is the crate (271: the snowball at the crate; 273: the last one).
# Rule 4: impact animations are not labelled; the same burst is unlabelled at tick 275 (pair 6148).
# Fix, this type only: a live projectile of type 6000855 whose label lies at least half inside the hit region of a
# projectile of the same type that ended at most S329_AGE ticks ago (idlabels._ended_region: its end point plus its
# radius, the region the ENDED rule uses) is at its own impact, and its label is dropped. The flying snowballs of the same
# shot lie 30+ px outside that region and keep their labels. Other types are not touched.
S329_TYPES = {6000855}   # Lou005Projectile (skin 329 only among the captures checked)
S329_AGE = 3             # ticks since the other projectile ended
S329_FRAC = 0.5          # share of a live label inside the impact region that makes it an impact
_s329_prev = labels


def labels(objs, cam, R, state=None, ended=(), img=None):
    import bulletlabels as BL
    P = _s329_prev(objs, cam, R, state=state, ended=ended, img=img)
    ends = [e for e in ended if e['t'] in S329_TYPES and e.get('age', 0) <= S329_AGE]
    if not ends: return P
    ended_ids = {e['id'] for e in ended}
    ltype = {}
    for o in objs: ltype.setdefault(BL._lid(o), o['t'])
    ids, lids, share = P['ids'], P['lids'], P['share']
    H, W = ids.shape
    for k in np.unique(ids[ids >= 0]):
        lid = lids[k]
        if lid[2] != 'body' or ltype.get(lid[0]) not in S329_TYPES or lid[0] in ended_ids: continue
        m = ids == k
        reg = _ended_region(ends, cam, (H, W), {lid[0]})
        if (m & reg).sum() >= S329_FRAC * m.sum():
            ids[m] = -1; share[m] = 0
    return P
