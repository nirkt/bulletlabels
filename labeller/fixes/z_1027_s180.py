# sweep 10-27 fix for skin 180 (Haiku fix, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s180/)
# sweep 10-27 fix for skin 180 (RocketGirlVirus, sweep 2026-10-27, task out/sweep2/haikutasks/sweep1027/s180/)
# Findings: pairs 3129 (tick 289), 3130 (301), 3132 (305), 3135 (329), 3140 (super 363), 3141 (super 377): tiny yellow
# outlines on small green particles, detached from the zone.
# Cause: the lingering burn zone BurnMainAttackHacker (zone data id 17000181, brawler RocketGirl; item 3000007 at
# ticks 275-331, item 3000009 at 325-377) labels every sprite pixel of its burn effect inside the logic circle
# (zone_labels keeps all zone sprite pixels). The effect's green particle bars and dots lie on the blue map blocks and
# beside the cube, so they become detached pieces of the zone label: 50-100 px bars and 1-px dots (measured on all ticks
# 263-395 of this capture, component sizes 1-106 px). The zone's main body is 500-2700 px and its real parts (the arc,
# the cube, the bar at 281 of 219 px) are at least 200 px.
# Approaches considered:
#   1. drop the zone's effect shapes by atlas shape id: rejected, the same shapes make up the real bars of the zone.
#   2. clip the zone to the logic circle: rejected, the specks are inside the circle (blocks and cube are inside it).
#   3. CHOSEN: drop every detached piece of this zone's label under S180_MIN px, keeping pieces of S180_MIN px and more.
#      Counts of the pieces of S180_MIN px and more are unchanged (checked on every tick).
#      Not done: the green bar hanging under the cube at ticks 289-301 (about 70-85 px) is joined to the zone body by a
#      neck about 7 px wide; an opening that would cut it also cuts the arc's tail (about 50 px of real body).
# Keyed to the zone's data id only (17000181, BurnMainAttackHacker); no projectile type is touched.
# Shared: the same zone data id is used by the other RocketGirl skins that carry this burn zone (not captured here).
S180_ZONE_DID = {17000181}   # BurnMainAttackHacker (RocketGirl)
S180_MIN = 120               # px: a detached piece of the zone label smaller than this is a speck


def _s180_zone_specks(P, state):
    from scipy.ndimage import label as cc
    if not state or 'items' not in state: return P
    ids = P['ids'].copy(); share = P['share'].copy(); lids = P['lids']; changed = False
    for it in state.get('items', []):
        if int(it[1]) not in S180_ZONE_DID: continue
        ks = [k for k, l in enumerate(lids) if l[0] == int(it[0]) and l[2] == 'area']
        for k in ks:
            m = ids == k
            if not m.any(): continue
            L, n = cc(m)
            sz = np.bincount(L.ravel())
            small = np.nonzero(sz < S180_MIN)[0]
            small = small[small > 0]
            if small.size:
                drop = np.isin(L, small)
                ids[drop] = -1; share[drop] = 0.0; changed = True
    if changed:
        P = dict(P); P['ids'] = ids; P['share'] = share
    return P


_s180_labels_prev = labels


def labels(objs, cam, R, state=None, ended=(), img=None):
    P = _s180_labels_prev(objs, cam, R, state=state, ended=ended, img=img)
    return _s180_zone_specks(P, state)
