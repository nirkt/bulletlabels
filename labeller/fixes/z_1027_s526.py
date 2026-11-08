# sweep 10-27 fix for skin 526 (Haiku fix, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s526/)
# Skin 526 UndertakerMecha (Undertaker Mecha, Mortis 005 super, type 6000301 Mortis005UltiProjectile): sweep 2026-10-27,
# pair 9499 (tick 367, extra_label). The super's blue energy sphere hits the bush / pool edge (logic 'finished' about
# tick 362); at 363 the sphere is still visible and labelled, from 365 it has turned into an electric ring with the bats flying
# off it. That ring is the post-impact burst: an impact animation and the leftovers of an ended attack are never labelled
# (rules 2 and 4). The trail search kept a 6 px speck of the ring on its upper-right edge, and the clean-up does not drop a
# lone part, so the speck stayed (page 526 t367). Its bats after the impact (94 px, t365) are the same leftovers.
# Fix: labels of this projectile are dropped from 2 ticks after its impact (Tracker.ended age: the sphere at its impact
# frame, age 1, keeps its label as before; age 3 and later, the burst and the bats, are unlabelled). Only type 6000301
# uses these clips (sc/effects_brawler.sc mortis_005_ulti_projectile, mortis_005_ulti_trail_elec_01..04); no other skin shares it.
POST_END_DROP = {6000301: 2}   # type -> ticks after the impact (Tracker.ended 'age') from which its labels are dropped
_prev_labels_s526 = labels


def labels(objs, cam, R, state=None, ended=(), img=None):
    import numpy as _np
    import bulletlabels as BL
    P = _prev_labels_s526(objs, cam, R, state=state, ended=ended, img=img)
    age = {e['id']: e.get('age', 0) for e in ended}
    grp = {}
    for o in objs: grp.setdefault(BL._lid(o), []).append(o)
    drop = set()
    for g, mem in grp.items():
        # every member of the label group is a finished POST_END_DROP projectile, past its impact by the given ticks
        if all(o['t'] in POST_END_DROP and o.get('finished') and o['id'] in age and age[o['id']] >= POST_END_DROP[o['t']]
               for o in mem):
            drop.add(g)
    if not drop: return P
    ids, lids = P['ids'], P['lids']
    ks = [k for k, lid in enumerate(lids) if lid[0] in drop and lid[2] == 'body']
    if not ks: return P
    m = _np.isin(ids, ks)
    if not m.any(): return P
    P = dict(P)
    P['ids'] = _np.where(m, -1, ids).astype(ids.dtype)
    P['share'] = _np.where(m, 0, P['share']).astype(P['share'].dtype)
    return P
