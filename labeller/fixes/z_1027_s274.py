# sweep 10-27 fix for skin 274 (Haiku fix, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s274/)
# Fix task s274 (sweep 2026-10-27, skin 274 DeadMariachiPunk = Poco skin 004): two per-type trail skips.
# Both types are this skin's own rows (Poco004Projectile 6000341 = basic attack, Poco004UltiProjectile 6000342 = super;
# ParentProjectileForSkin DeadMariachiProjectile / DeadMariachiUltiProjectile). Keyed to these type ids only.
#
# 6000341 (pair 5063, attack 0 deg, tick 279, extra_label): the cyan / purple electric zigzags of its trail
#   (clips shocked_long_1..4 and shocked_long_purple_1..4, shapes 53508 / 693) were labelled as body, so a zigzag
#   spark lying among the confetti after the star cloud had ended got an outline. Sparks and electric bolts are not the
#   attack (rule 3: sparks are never labelled). The clip prefix 'shocked_long' covers all eight clips. The cloud puffs
#   (cloud_particle_flat_blue_01/02) and the stars are kept.
# 6000342 (pair 5079, super 45 deg, tick 359, extra_label): its trail clips particle_patch_1..3 are single atlas quads
#   (shapes 55473, 55471, 55469) drawn as static blue rings and teal cards on the field. They are ground decals
#   (rule 3), not the super's cloud. They were labelled only on the ticks where the super's zone region reached them
#   (357, 359), and not on 349-353 or 361 (the same spots stayed unlabelled). They are never labelled now. The green
#   and yellow cloud pieces (shocked_green / shocked_yellow, cloud_green_punk, poco_004_ulti) are kept.
TRAIL_SKIP[6000341] = ('shocked_long',)
TRAIL_SKIP[6000342] = ('particle_patch',)
