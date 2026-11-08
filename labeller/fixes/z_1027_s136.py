# sweep 10-27 fix for skin 136 (Haiku fix, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s136/)
# sweep 2026-10-27, skin 136 WeaponThrower Darkcircus (Bronson004, type 6001228), pair 2216 (attack 0 deg, tick 269).
# Finding: the swoosh label had inward notches on its left edge where the striped rear blocks sit, a spur at the bottom,
# so it was not one solid hull (rules 1, 5, 6). The projectile is one object (the thrown weapon, its 3D model is
# bronson_renamed_004_atk_projectile.glb) whose bright trail is the swoosh; its striped blocks travel with it.
# Fix: ONE label = the convex hull of the attack's own pixels (the a4 POST_BODY: bright pixels of the trail clip
# bronson_004_atk_trail_afterimage only, never the impact clip, the orange particles or the dim motion band).
# Only type 6001228 changes (the attack, both attacks 0 deg and 90 deg): the super is type 6001230 and is unchanged.
# Checked with maskdiff: ticks 265-315 differ only at 269 (+283 px) and 309 (+463 px, the same hull on the 90 deg attack).
ONE_LABEL[6001228] = 'hull'
