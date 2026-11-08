# sweep 10-27 fix for skin 167 (Haiku fix, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s167/)
# sweep 10-27 fix for skin 167 (CactusLog, Haiku task s167, finding pair 2850: sw2_167 super 45 deg, tick 371, spur_or_speck).
# The lingering zone of Spike's super (area effect did 17000042, Spike_005_ulti_explosion, effect spike_005_ulti_explosion) is
# labelled with the zone's sprite pixels (zone_labels: every zone clip, its effect clips included). The zone's effect emits
# ground-fragment debris (clips groundFrag_1..6, the brown pebbles); a pebble lying on the cloud's lower-left edge is labelled
# with the cloud, so the outline bulges round it as a small loop (zone outline drawn round a brown pebble, tick 371).
# Rule 3: debris is never labelled, and rule 7: effects never cut the zone (zone = solid, one piece). So:
#  - the groundFrag debris pixels of this zone's label that touch the OUTSIDE of the label are dropped (the pebble bump);
#  - debris pixels enclosed by the cloud stay (the pebbles lying inside the zone are part of the solid zone label).
# Checked (probe_debris.py): debris on the outer edge 7 / 22 / 10 px at ticks 367 / 371 / 375; none from 379 on.
# Only did 17000042 (its effect spike_005_ulti_explosion is used by no other zone row). Other Spike ulti zones (Spike_002..010)
# have their own effect clips and were not checked here. Only zone labels are touched; no projectile type.
S167_ZONE_DIDS = {17000042}   # Spike_005_ulti_explosion (CactusLog super zone, pair 2850)


def _s167_gf_clips(Z):
    """(swf, clip name) of the zone's ground-fragment debris clips (every zone clip and effect clip whose name has 'groundfrag')"""
    clips = [(Z.get('FileName'), Z.get(c)) for c in ZONE_EXPORTS if Z.get(c) and Z.get('FileName')]
    for eff in (Z.get('Effect'), Z.get('LoopingEffect')): clips += effect_clips(eff)
    return sorted({(s, n) for s, n in clips if n and 'groundfrag' in n.lower()})




def _s167_zone_debris(P, state, R):
    from scipy.ndimage import binary_fill_holes
    import owners
    if not state or 'items' not in state: return P
    ids = P['ids'].copy(); share = P['share'].copy(); lids = P['lids']; changed = False
    for it in state.get('items', []):
        did = int(it[1])
        if did not in S167_ZONE_DIDS: continue
        Z = owners.zone(did)
        if not Z: continue
        ks = [k for k, l in enumerate(lids) if l[0] == int(it[0]) and l[2] == 'area']
        if not ks: continue
        k = ks[0]; m = ids == k
        if not m.any(): continue
        deb = np.zeros(m.shape, bool)
        for swf, nm in _s167_gf_clips(Z):
            deb |= _a4_shape_pixels(R, swf, (nm,)) > 0
        D = m & deb
        if not D.any(): continue
        # debris enclosed by the rest of the zone stays (solid zone); debris touching the outside is cut off
        keep = binary_fill_holes(m & ~D) & m
        drop = m & ~keep
        if drop.any():
            ids[drop] = -1; share[drop] = 0.0; changed = True

    if changed:
        P = dict(P); P['ids'] = ids; P['share'] = share
    return P


_s167_labels_prev = labels


def labels(objs, cam, R, state=None, ended=(), img=None):
    P = _s167_labels_prev(objs, cam, R, state=state, ended=ended, img=img)
    return _s167_zone_debris(P, state, R)
