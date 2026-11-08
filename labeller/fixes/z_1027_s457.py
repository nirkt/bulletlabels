# sweep 10-27 fix for skin 457 (Haiku fix, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s457/)
# Skin 457 IceDudeStrikers (sweep 2026-10-27, pairs 8572, 8573, 8574: super 45 deg, ticks 371, 389, 411 and the
# rest of the zone ticks 369-411): the super's lingering zone IceDude007UltiArea (yellow, logic radius 1100, 10 s) was cut
# by the green bush band on the right side of the pitch. The zone's sprites are hidden under the bushes (drawn in front),
# so the outline had a gap there. Rule 7: bushes never cut a zone; a zone is the full logic circle, cut only by solid map
# blocks (crates, walls, barrels). Fix: this zone name is treated as a ZONE_DISC zone (the logic disc, minus whatever is
# drawn in front of the ground that is not a bush cut: the solid cut of z_1023_zone_solid then leaves only the map blocks).
# Keyed to this one zone name only. The other IceDude ulti areas (IceDudeUltiArea, IceDudeKingUltiArea, IceDude006UltiArea,
# ...) are not changed: no capture of them here, so their own bush behaviour is not verified.
ZONE_DISC.add('IceDude007UltiArea')
