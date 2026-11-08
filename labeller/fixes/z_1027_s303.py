# sweep 10-27 fix for skin 303 (Sonnet revision, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s303/)
# Skin 303 BlackHoleDevious (Tara 004's super), sweep 2026-10-27, task s303 (pair 5674, super 45 deg, tick 365: the yellow
# zone outline split into petal loops with an empty centre, rule 1 / rule 7 violated). Second attempt (the first was rejected).
# Zone = area effect row 'Tara004UltiExplosion' (area 17000159, logic row 'BlackHoleUltiExplosion': DelayedDamage, radius 450).
# Its drawn sprites are additive rings and petals: the ID pass sees only opaque bits between them, so the label was the petal
# regions. Rule 10 / 7: labelled as the logic ground disc, cut by what is drawn in front of the ground (ZONE_DISC).
# Only this ONE zone item is affected. The in-flight projectile 6000418 (Tara004UltiProjectile, class super, magenta) is a
# different object and is NOT touched: its label is pixel-identical before / after on every flight tick (345-359; checked,
# diff 0). (The first attempt's compare image looked as if the projectile had turned red -> yellow only because variants.py
# colours by variant index, not by class.)
# The area item lives until tick 387, but the explosion is on screen only until about tick 377: at 379 only a fading blue
# glow is left and from 381 nothing is drawn on the ground. The disc is kept for the first 16 ticks of the item (ticks
# 361-377, covering the white flash at 375 where the zone is active), later the fade is an end animation (rule 2) and is
# not labelled (before the fix it was not labelled either).
ZONE_DISC.add('Tara004UltiExplosion')
_FIX303_MAXAGE = 16
_fix303_zone_labels = zone_labels
def zone_labels(state, cam, R, taken, own_team=None):
    import owners
    out = _fix303_zone_labels(state, cam, R, taken, own_team)
    tick = int(state.get('tick', 0))
    for it in state.get('items', []):
        k = int(it[0])
        if k in out and owners.zone(int(it[1])) and owners.zone(int(it[1])).get('Name') == 'Tara004UltiExplosion':
            if tick - _ZONE_SEEN.get(k, tick) > _FIX303_MAXAGE: del out[k]
    return out
