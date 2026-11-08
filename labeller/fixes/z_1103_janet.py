# AI-review batch 2026-11-01 fix (Haiku fix, reviewed by the coordinator, installed 11-01 LOCAL ONLY; source out/sweep2/haikutasks/siteai/janet/fix2.py)
# Janet per-shot hull, 2026-11-01 (coordinator decision, after the owner's 068 finding). Standalone: replaces fix.py.
# Janet's attack types of the six skins: 6001160 (304, JetpackGirl002Projectile), 6001162 (324, 003), 6001166 (337, 005),
# 6001168 (391, 006), 6001170 (399, 007), 6001172 (423, 008). Each shot = 4 projectile objects of one attack with the same owner
# and the same first tick (2000012-15 at attack 0 deg, 2000016-19 at attack 90 deg); each had its own outline.
# Owner's rule (2026-10-29, skin 068, z_1101_s068.py): ONE convex hull per shot, per tick.
# Same mechanism as z_1101_s068.py, keyed to these type ids: the members of one shot are the objects that are not ended and
# have a labelled body piece this tick (a member that finished / stopped at a hit stays out: ended rule; a member under a flash
# with a visible piece is included: hidden-parts rule). The union of the members' body labels is replaced by its convex hull, on
# the lowest object id of the shot. Pixels of other labels are never taken (only unlabelled pixels and the members' own pixels
# are re-assigned); every member pixel lies inside the hull, so no label loses a pixel.
# Runs after the installed fixes (z_1027_s399 / z_1027_s423 / z_1101_s068 do their per-object work first; the hull replaces their
# per-object outlines on these types).
JG2_TYPES = {6001160, 6001162, 6001166, 6001168, 6001170, 6001172}

_jg2_prev_labels = labels


def labels(objs, cam, R, state=None, ended=(), img=None):
    import bulletlabels as BL
    P = _jg2_prev_labels(objs, cam, R, state=state, ended=ended, img=img)
    on = [o for o in objs if o['t'] in JG2_TYPES and not o.get('ended')]
    if not on: return P
    ids, lids = P['ids'], P['lids']
    shots = {}
    for o in on:
        lid = (BL._lid(o), -1, 'body')
        if lid in lids: shots.setdefault((o.get('owner'), o.get('first'), o['t']), []).append((o, lids.index(lid)))
    if not shots: return P
    cv = R['cv'][::2, ::2]
    out = ids.copy(); share = P['share'].copy()
    for key, mem in shots.items():
        ks = [k for _, k in mem]
        U = np.isin(ids, ks)
        if not U.any(): continue
        H = hull_mask(U)
        carrier = min(mem, key=lambda m: m[0]['id'])[1]
        for k in ks: out[out == k] = -1
        t = H & (out < 0)
        out[t] = carrier; share[t] = np.maximum(cv[t] / 255.0, BL.CUT)
    P2 = dict(P); P2.update(ids=out, share=share)
    return P2
