# hosted review 2026-10-29 fix (Haiku fix, reviewed by the coordinator, installed 10-29 LOCAL ONLY; source out/sweep2/haikutasks/site1029/g_knight/fix.py; tested via IDLABELS_OVERRIDE after all fixes incl. z_1100_*, hence z_1101_)
# Ash (Knight) skins 129 KnightPizza, 131 KnightPinata, 150 KnightAdidas: hosted review 2026-10-29 (task g_knight).
# THE USER on all three: 'the attack should have a hull' / 'all ash attacks should have a hull' / 'like before, use a hull'.
# Same request and same solution as the accepted skin 064 (s064/fix.py, type 6001037 KnightProjectile1):
# one convex hull over the attack's own body per tick, depth-cut against solid geometry (crates, walls) the hull reaches
# into and that is nearer the camera than the ground (the shards / pieces themselves are always kept), and no label once
# the logic projectile has finished (the game's own state 'finished' = 1: from the hit on, the leftover pieces are the end
# animation, rule 2).
# Types (each Ash attack is a wave, one object per shot, trail clips ash_0xx_atk_*):
#   129 -> 6001043 Ash003Projectile1 (trail ash_003_atk_trail, sc/effects_brawler.sc); already has z_1027_s129.py (flame +
#          topping clips, closed, specks dropped). That cleaned body is kept: this fix chains to it (POST_BODY from before)
#          and takes the hull of it.
#   131 -> 6001046 Ash005Projectile1 (trail ash_005_atk_trail, sc/effects_brawler.sc): no earlier per-type fix.
#   150 -> 6001055 Ash008Projectile1 (trail ash_008_atk_trail, sc/effects_brawler_ash.sc): no earlier per-type fix.
# Keyed to these three type ids only. Skin 064 (6001037) and every other type are untouched. Super rows are untouched
# (KnightPet / Ash00xUltiProjectile types 6001058..6001069 are not in this list).
_GK_TYPES = (6001043, 6001046, 6001055)
_GK_CUT = 0.35     # depth margin (as s064 / s132): hull pixels nearer than the ground by more than this are solid geometry


def _gk_hull(t, prev):
    def fn(body, ctx):
        if ctx['o'].get('finished'):   # the logic projectile reached its end: its leftover pieces are not labelled
            return np.zeros_like(body)
        if prev is not None: body = prev(body, ctx)   # an earlier per-type cleaning of this type (129: z_1027_s129)
        if not body.any(): return body
        hm = hull_mask(body)   # one solid convex hull over the attack's body of this shot (the user: 'use a hull')
        R = ctx['R']; o = ctx['o']; wd = R.get('wd'); wref = R['meta'].get('wref', 0.0)
        if wd is not None and o.get('logic') is not None:
            gd = ground_depth(ctx['cam'], float(o['logic'][0]), float(o['logic'][1]), 300.0, wref)
            hm = hm & ((wd >= gd - _GK_CUT) | body)   # nearer solid geometry (crate, wall) is cut out of the hull; body kept
        return hm
    return fn


for _t in _GK_TYPES:
    if ('_gk_prev_%d' % _t) not in globals(): globals()['_gk_prev_%d' % _t] = POST_BODY.get(_t)   # the hook before this file
    POST_BODY[_t] = _gk_hull(_t, globals()['_gk_prev_%d' % _t])
