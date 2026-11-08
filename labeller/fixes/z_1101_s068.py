# hosted review 2026-10-29 fix (Haiku fix, reviewed by the coordinator, installed 10-29 LOCAL ONLY; source out/sweep2/haikutasks/site1029/s068/fix.py; tested via IDLABELS_OVERRIDE after all fixes incl. z_1100_*, hence z_1101_)
# Skin 068 JetpackGirlDefault, review 2026-10-29 (the user's finding, reviewer note "should get a hull").
# Janet's basic attack: type 6001158 (JetpackGirlProjectile, clip jetgirl_def_normal_projectile_blue). One shot = 4
# projectiles of one attack: same owner and same 'first' tick (attack 0 deg: first 1872, objects 2000012-2000015; attack
# 90 deg: first 1912, objects 2000016-2000019). Each projectile had its own outline on its bright cyan segment, so one shot
# showed as 4 small separate outlines. The user: ONE convex hull per shot, per tick, over the bright segments.
# Fix (a labels() wrapper, after the installed fixes): per tick, the body labels of the projectiles of one shot are united,
# the convex hull of that union is taken (hull_mask), and the hull becomes one label on the shot's lowest object id.
# The pale purple / yellow copies of the same projectiles are not in the body labels, so they stay out unless they fall
# inside the hull (the coordinator's rule). Nothing else in the frame is touched; other types are returned unchanged.
JG_SHOT_T = 6001158

_labels_s068fix_prev = labels


def labels(objs, cam, R, state=None, ended=(), img=None):
    import bulletlabels as BL
    P = _labels_s068fix_prev(objs, cam, R, state=state, ended=ended, img=img)
    on = [o for o in objs if o['t'] == JG_SHOT_T and not o.get('ended')]
    if not on: return P
    ids, lids = P['ids'], P['lids']
    shots = {}
    for o in on:
        lid = (BL._lid(o), -1, 'body')
        if lid in lids: shots.setdefault((o.get('owner'), o.get('first')), []).append((o, lids.index(lid)))
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
