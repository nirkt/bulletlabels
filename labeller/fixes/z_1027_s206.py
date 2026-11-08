# sweep 10-27 fix for skin 206 (Haiku fix, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s206/)
# Skin 206 AmbusherSandsOfTime (sweep 2026-10-27, Haiku fix task out/sweep2/haikutasks/sweep1027/s206/, findings
# pairs 3647, 3648, 3650 (attack 0 deg, ticks 265, 267, 271) and 3653, 3654 (attack 90 deg, ticks 305, 307): extra_label).
# Ambusher004Projectile (type 6001461, the lily_004 wave, a wave-kind type: its trail particles are its body) has two
# trail clips of the Buzz 009 effects file, buzz_009_atk_twinkle_01 and buzz_009_atk_twinkle_02: the gold starburst
# sparkles that flash beside and ahead of the blue wave (and at the brawler's hand) are these twinkles. They were labelled
# as body (the sparkle draws show the twinkle sprites in the ID pass). Rule 3: twinkles are never labelled. The wave's own
# clip (lily_004_atk_projectile) is not touched.
# Fix: the two twinkle trail clips are skipped for this type only (TRAIL_SKIP, matched by name prefix). The same two clips
# are trail clips of other types (Bo009 6000373/6000374, Arcade007 6000571, Buzz009 6000999/6001000, Gunslinger buddies
# 6001771/6001772, Hammer buddy 6001780): they are NOT changed here (per-type decision, not generalised).
TRAIL_SKIP[6001461] = ('buzz_009_atk_twinkle',)
