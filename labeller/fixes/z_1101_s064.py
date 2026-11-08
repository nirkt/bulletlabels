# hosted review 2026-10-29 fix (Haiku fix, reviewed by the coordinator, installed 10-29 LOCAL ONLY; source out/sweep2/haikutasks/site1029/s064/fix.py; tested via IDLABELS_OVERRIDE after all fixes incl. z_1100_*, hence z_1101_)
# Skin 064 KnightDefault (hosted review 2026-10-29, task s064). THE USER: 'attack should get a hull'.
# The Knight's attack, KnightProjectile1 (type 6001037, a wave: one object per shot, object 2000002 at attack 0 deg ticks
# 269-275, 2000003 at attack 90 deg ticks 309-315), is a spray of blue crystal shards drawn from its trail sprites
# (trail ash_def_atk_trail). Before: each cluster of shards got its own ragged outline (many red fragments per shot).
# Fix (rule 5/6 and the coordinator's clarification): ONE convex hull over the shards of one shot, per tick.
# The hull is built from the object's own trail body only (the shard sprites); the pale blue puffs beside it (ID pass 'over'
# sprites) are not body and are not used as hull vertices. The hull is cut by depth like the other wave hulls (fix s132):
# solid map geometry nearer the camera than the ground (crates, walls) that the hull reaches into is taken out again,
# the shards themselves are always kept. Ended rule: once the logic projectile has finished (the game's own state
# 'finished' = 1, from the tick the shards hit and shatter on) the leftover shards and dust are an end animation: no label.
# Only type 6001037 (the Knight's KnightProjectile1). The super (KnightPet / KnightUltiProjectile) is not touched.
_K064_ATK = 6001037
_K064_CUT = 0.35     # depth margin (as _K132_CUT): hull pixels nearer than the ground by more than this are solid geometry


def _fix_k064_atk(body, ctx):
    if ctx['o'].get('finished'):   # the logic projectile reached its end: its leftover shards are not labelled
        return np.zeros_like(body)
    if not body.any(): return body
    hm = hull_mask(body)   # one solid convex hull over the shards of this shot (the user: 'attack should get a hull')
    R = ctx['R']; o = ctx['o']; wd = R.get('wd'); wref = R['meta'].get('wref', 0.0)
    if wd is not None and o.get('logic') is not None:
        gd = ground_depth(ctx['cam'], float(o['logic'][0]), float(o['logic'][1]), 300.0, wref)
        hm = hm & ((wd >= gd - _K064_CUT) | body)
    return hm


POST_BODY[_K064_ATK] = _fix_k064_atk
