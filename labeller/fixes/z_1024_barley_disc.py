# Barley puddle zones of every remaining skin = the FULL LOGIC CIRCLE from birth, cut only by walls / crates (the user,
# 2026-10-24, option C on out/sweep2/coord_1024/barley_question.png; same as Barley Wizard 002 and the other circle zones,
# z_1023_zone_solid.py). Scale 1.0: the logic radius matched the drawn puddle on sw2_021 (ticks 293-381).
for _n in ('Barley_def_atk', 'Barley_def_ulti', 'Barley_004_atk', 'Barley_004_ulti', 'Barley_005_atk', 'Barley_005_ulti',
           'Barley_006_atk', 'Barley_006_ulti', 'Barley_007_atk', 'Barley_007_ulti', 'Barley_008_atk', 'Barley_008_ulti',
           'Barley_009_atk', 'Barley_009_ulti', 'BarleyOverchargedUlti', 'BarleyMutationAreaEffect'):
    ZONE_DISC.add(_n); A3_DISC_SCALE.setdefault(_n, 1.0)
