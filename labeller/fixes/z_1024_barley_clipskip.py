# Barley zones of the remaining skins (coordinator 2026-10-24; agent 5 did 004/005/007/009, its question Q2 "apply to the
# others?" settled by the user's rules: impacts, shards, debris, smoke and sparkles are hit / end effects, not the zone).
# The puddle, its cloud / bubble, the rings and stars on the ground and the octopus legs (008) stay.
_Z24_SKIP = ('impact', 'shard', 'debris', 'particle_gas', 'twinkle', 'fire_spark', 'particle_stars', 'gen_star',
             'gene_particle_star', 'glowingdot', 'bottle_particle', 'storn_particle', 'lightning')
for _n in ('Barley_def_atk', 'Barley_def_ulti', 'Barley_006_atk', 'Barley_006_ulti', 'Barley_008_atk', 'Barley_008_ulti',
           'BarleyOverchargedUlti', 'BarleyMutationAreaEffect'):
    A5_ZONE_CLIP_SKIP[_n] = _Z24_SKIP
