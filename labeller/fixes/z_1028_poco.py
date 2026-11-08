# Task poco (q1028, 2026-10-28): Poco's clouds end with their logic, for EVERY Poco skin.
# Types: POCO_P6 = idlabels POCO_ATK | POCO_SUPER (23 ids: 6000335..6000355 odd = attack, 6000336..6000356 even + 6001521 = super).
# Cause (measured on capture sw2_028, tools/variants.py + scratch/probe_state.py): these are trail-body types
# (TRAIL_BODY_EXTRA). The members of one shot form one wave group (label_id); the group's attack hull (wave.attack_tris over the
# path points of ALL members, attached to the group's first member) is the label's region. A member left that label only once it
# had LEFT the battle state (tracker gone -> ended). After its logic finished (state row field 'finished' != 0, track.J), the
# member stayed in the state for 6 to 10 ticks (drawn False), so the group stayed alive and its hull kept labelling the leftover
# cloud through its trail sprites. ENDED_DROP does not apply to trail-body or ZONE types.
# User's decision (2026-10-28): "drop when it's logic/hitbox ends". Rule implemented here: a member whose state row has
# finished != 0 (or that has left the state) has ended its logic. From that tick it contributes nothing: it stays in the list
# only as a hidden placeholder (drawn False, no parts, no attack; labelling keeps its id for the group's instance rows), and the
# group's attack hull is rebuilt from the members still running (finished == 0 and not ended). A group with no running member
# has no label at once. Where a finished member and a running member share one label (attack shots fired together), the
# finished member's own swept path region (its trail, grown by its radius) is removed from the label wherever no running
# member's own region covers it (_p6_exclude). Other types and skins are untouched; the earlier per-skin fixes (z_1027_s311,
# s346, s373 ...) run on the result.
import wave as _wave_p6, projdraw as _PD_p6

POCO_P6 = {6000335, 6000337, 6000339, 6000341, 6000343, 6000345, 6000347, 6000349, 6000351, 6000353, 6000355,   # attacks
           6000336, 6000338, 6000340, 6000342, 6000344, 6000346, 6000348, 6000350, 6000352, 6000354, 6000356, 6001521}   # supers
_prev_labels_p6 = labels


def _p6_path_pts(o):
    """the path points the attack hull covers for one running object: wave.path_points on its own tracker history"""
    e = dict(hist=[dict(pos=p) for p in (o.get('path') or [])], t=o['t'], gone=0)
    return _wave_p6.path_points(e, _PD_p6.DT)


def _p6_placeholder(o):
    """an object that has ended its logic: no geometry (not drawn, no clip parts, no attack hull), same id and type"""
    h = dict(o); h.update(drawn=False, attack=None, attack_members=None, parts={}, pellets=[], mesh_tris=None, mesh=None,
                          file=None, ended=True)
    return h


def _p6_exclude(P, cam, pairs):
    """pixels of a finished member's own swept region (its path, grown by its radius) that no running member's own region
    covers are dropped from the group's label: the leftover puffs / hit flash of an ended member are not the attack.
    pairs: [(gid, finished objects, running objects)] of the original (pre-placeholder) objects."""
    import bulletlabels as BL
    synth = []; keymap = {}
    for gid, fin, run in pairs:
        for role, L in (('f', fin), ('r', run)):
            for o in L:
                pts = _p6_path_pts(o)
                if not pts: continue
                key = -(len(synth) + 1000)
                s = dict(o); s.update(id=key, label_id=key, attack=_wave_p6.attack_tris(pts, o['radius']), attack_members=[o['id']],
                                      drawn=False, parts={}, pellets=[], mesh_tris=None, mesh=None, file=None, ended=True)
                synth.append(s); keymap[key] = (gid, role)
    if not any(r == 'f' for _, r in keymap.values()): return P
    Q = BL.labels(synth, cam, cut=0.01)
    ids_q, lids_q = Q['ids'], Q['lids']
    fin_k = [k for k, l in enumerate(lids_q) if l[0] in keymap and keymap[l[0]][1] == 'f']
    run_k = [k for k, l in enumerate(lids_q) if l[0] in keymap and keymap[l[0]][1] == 'r']
    F = np.isin(ids_q, fin_k) if fin_k else np.zeros(ids_q.shape, bool)
    Rr = np.isin(ids_q, run_k) if run_k else np.zeros(ids_q.shape, bool)
    E = F & ~Rr
    if not E.any(): return P
    gids = {g for g, _, _ in pairs}
    ids, lids, share = P['ids'], P['lids'], P['share']
    grp_k = [k for k, l in enumerate(lids) if l[0] in gids]
    m = np.isin(ids, grp_k) & E if grp_k else np.zeros(ids.shape, bool)
    ids[m] = -1; share[m] = 0
    return P


def labels(objs, cam, R, state=None, ended=(), img=None):
    grp = {}
    for o in objs:
        if o['t'] in POCO_P6: grp.setdefault(o.get('label_id', o['id']), []).append(o)
    if not grp: return _prev_labels_p6(objs, cam, R, state=state, ended=ended, img=img)
    ended_ids = set(); carry = {}   # id() of objects whose logic ended; group id -> (carrier, new attack region, running ids)
    for gid, mem in grp.items():
        run = [o for o in mem if not o.get('ended') and not o.get('finished')]
        for o in mem:
            if not any(o is r for r in run): ended_ids.add(id(o))
        if not run: continue
        pts = []
        for o in run: pts += _p6_path_pts(o)
        r = max(o['radius'] for o in run)
        reg = _wave_p6.attack_tris(pts, r) if pts else None
        carrier = next((o for o in run if o['id'] == gid), run[0])
        carry[gid] = (carrier, reg, sorted(o['id'] for o in run))
    out = []
    for o in objs:
        if o['t'] not in POCO_P6: out.append(o); continue
        g = o.get('label_id', o['id'])
        if id(o) in ended_ids: o = _p6_placeholder(o)
        elif g in carry and o is carry[g][0]: o = dict(o, attack=carry[g][1], attack_members=carry[g][2])
        out.append(o)
    P = _prev_labels_p6(out, cam, R, state=state, ended=ended, img=img)
    pairs = []
    for gid, mem in grp.items():
        fin = [o for o in mem if id(o) in ended_ids and not o.get('ended')]
        run = [o for o in mem if id(o) not in ended_ids]
        if fin and run: pairs.append((gid, fin, run))
    return _p6_exclude(P, cam, pairs) if pairs else P
