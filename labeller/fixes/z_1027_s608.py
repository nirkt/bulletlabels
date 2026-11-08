# sweep 10-27 fix for skin 608 (Haiku fix, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s608/)
# sweep 10-27 fix for skin 608 (SplitterBrawloween, Splitter 005 attack), installed as an override for the task
# out/sweep2/haikutasks/sweep1027/s608/ (row "sw2_608 attack 90 deg", pair 10397, tick 311, kind hole).
# The flame of the attack (type 6001316, Splitter005Projectile, logic SplitterProjectile, brawler rt_005) is one solid
# object. Its body label had two closed inner loops in the lower half (about 42-64 px at tick 309 and 96-101 px at tick
# 311, around the lighter lobe and the dark teeth), so the outline was a ring instead of one filled piece. The shared
# clean-up (z_1099_cleanup.py) fills holes of at most 40 px only, so these stayed.
# Fix: a POST_BODY hook keyed to type 6001316 fills the body's enclosed holes inside its own region. Holes that are open
# to the outside are not touched (binary_fill_holes only fills enclosed ones). Other types are unchanged.
# Shared type: 6001316 is the Splitter brawler's own attack projectile. Other Splitter skins (077, 535, 567) are not
# checked here; see the report.

S608_TYPES = {6001316}


def _s608_fill_holes(body, ctx):
    from scipy.ndimage import binary_fill_holes
    if not body.any():
        return body
    return binary_fill_holes(body) & ctx['region']


for _t608 in S608_TYPES:
    POST_BODY[_t608] = _s608_fill_holes
