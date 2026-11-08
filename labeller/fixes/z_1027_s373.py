# sweep 10-27 fix for skin 373 (Sonnet revision, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s373/)
# Skin 373 DeadMariachiOnceupon (sweep 2026-10-27, Poco), version 2 (replaces fix.py after the review).
# Pairs 7015 (attack 0 deg, tick 283), 7023 (attack 90 deg, tick 323), 7031 (super 45 deg, tick 359): leftover shards / leaves /
# petals of an ended Poco cloud were labelled. Types 6000349 (Poco008Projectile, attack) and 6000350 (Poco super): the cloud's
# wave group (label_id) is labelled as one ZONE. Both type ids are shared by every Poco skin 008 (not checked on other captures).
# (1) End of the cloud. The group stays alive while any member is listed by the tracker without the 'ended' flag. A member whose
#     logic projectile has finished (finished = 1) and which the client no longer draws (drawn False) is still listed for a few
#     ticks and keeps the group's region alive, so the leftovers are labelled through it. A group whose every listed member has
#     finished, is not drawn and has been finished for >= FIN_AGE ticks (tracker 'age' in the ended list) is dropped (rule 4).
#     Checked on every tick of the three rows (sw2_373, compare2.png): attack cloud labelled 265-279 and 305-319, its leaves
#     dropped from 281 / 321; super cloud labelled 345-357, its petals dropped from 359.
# (2) Pair 7018 (tick 305, speck on a detached cyan dot): not present in this capture, current or with the clean-up off: the
#     zone code already drops detached parts (ZONE_MIN) and that frame of the page came from another capture; checked on all
#     ticks that no label of these two types has a part under 100 px other than the fully-ended leftovers that (1) removes.
# (3) Clean edge (rules of 10-24: no spurs, no jagged notches): the zone outline of these two types had 1-3 px thick spurs
#     (a line out to a cyan sparkle, tick 309) and zigzag slits (ticks 305, 309, 313, 317). After the labels are made the label
#     of each group is closed (slits <= 2*CLOSE_S373 px filled, holes filled), then opened with radius OPEN_S373: pieces
#     of the label that the opening removes are dropped when they are spur-like (a connected removed piece of at least
#     SLIVER_S373 px; smaller shaved corners stay), so the outline keeps its shape and only thin spikes go.
import numpy as np
FIN_TYPES_S373 = {6000349, 6000350}
# ticks since the member's logic projectile finished (tracker 'age'): the attack's cloud is gone 4 ticks after its finish
# (checked: 279 kept, 281 leaves only), the super's teal mass is still an attack at 357 (6 ticks: gone at 359)
FIN_AGE_S373 = {6000349: 4, 6000350: 6}
_prev_labels_s373 = labels


def labels(objs, cam, R, state=None, ended=(), img=None):
    age = {e['id']: e.get('age', 0) for e in ended}
    groups = {}
    for o in objs:
        if o['t'] in FIN_TYPES_S373:
            groups.setdefault(o.get('label_id', o['id']), []).append(o)
    drop = set()
    for g, mem in groups.items():
        live = [o for o in mem if not o.get('ended')]
        if live and all(o.get('finished') and not o.get('drawn') and age.get(o['id'], -1) >= FIN_AGE_S373[o['t']] for o in live):
            drop.add(g)
    if drop:
        objs = [o for o in objs if not (o['t'] in FIN_TYPES_S373 and o.get('label_id', o['id']) in drop)]
    return _prev_labels_s373(objs, cam, R, state=state, ended=ended, img=img)


CLOSE_S373 = 3
OPEN_S373 = 2
SLIVER_S373 = 12
_prev2_labels_s373 = labels


def _disk_s373(r):
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r + 1


def labels(objs, cam, R, state=None, ended=(), img=None):
    from scipy.ndimage import binary_closing, binary_opening, binary_fill_holes, label as cc
    import bulletlabels as BL
    P = _prev2_labels_s373(objs, cam, R, state=state, ended=ended, img=img)
    ids, lids, share = P['ids'], P['lids'], P['share']
    ltype = {}
    for o in objs: ltype.setdefault(BL._lid(o), o['t'])
    for k in np.unique(ids[ids >= 0]):
        if ltype.get(lids[k][0]) not in FIN_TYPES_S373: continue
        m = ids == k
        ys, xs = np.nonzero(m)
        y0, y1, x0, x1 = max(0, ys.min() - 8), ys.max() + 9, max(0, xs.min() - 8), xs.max() + 9
        sm = m[y0:y1, x0:x1]; sid = ids[y0:y1, x0:x1]; ssh = share[y0:y1, x0:x1]
        free = sid < 0
        # close slits (never over another label), fill holes
        cl = sm | (binary_closing(np.pad(sm, CLOSE_S373 + 1), _disk_s373(CLOSE_S373))[CLOSE_S373 + 1:-CLOSE_S373 - 1, CLOSE_S373 + 1:-CLOSE_S373 - 1] & free)
        cl |= binary_fill_holes(cl) & free
        # spur-like pieces the opening cuts off
        op = binary_opening(cl, _disk_s373(OPEN_S373))
        L, n = cc(cl & ~op)
        if n:
            sz = np.bincount(L.ravel())
            drop = np.isin(L, np.nonzero((sz >= SLIVER_S373) & (np.arange(len(sz)) > 0))[0])
            # a cut that would split the label (a thin neck to a lobe) is not made
            new = cl & ~drop
            if cc(new)[1] <= 1: cl = new
        add = cl & ~sm; rem = sm & ~cl
        sid[add] = k; ssh[add] = BL.CUT
        sid[rem] = -1; ssh[rem] = 0
    return P
