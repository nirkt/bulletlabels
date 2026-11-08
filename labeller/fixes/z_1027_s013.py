# sweep 10-27 fix for skin 013 (Sonnet revision fix2.py after a rejected Haiku fix, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s013/)
# Skin 013 RosaToon, sweep 2026-10-27 findings 256 (tick 277), 261 (tick 307), 264 (tick 317); type 6000590 (Rosa007Projectile:
# the donut attack, trail clips rosa_007_atk_* of sc/effects_brawler.sc).  Second version (the first, fix.py, was rejected:
# its 10 px growth left the upper-right frosting lobe of the right donut at tick 307 outside the label).
#
# 256 and 264 (extra_label): the donuts have landed and their dough is gone, but the frosting splats they leave on the ground
# still carried red outlines (rule 4: leftovers of an ended attack are not labelled). A donut in view always shows its dough,
# the atlas shape 32106 of sc/effects_brawler.sc. The splats have only frosting shapes (63, 85, 407) and no dough, so label
# parts without dough pixels are dropped (this also removes the 4 px speck at tick 281). Donut stacks are kept.
#
# 261 (too_small): the label is limited to the region around the attack's predicted body (NEAR px); a drawn donut's frosting
# reaches up to ~20 px (half resolution) beyond it, where the region's edge cuts it along a straight diagonal. After the dough
# test, the donut's own sprite pixels (clip / trail code, coverage >= CUT, frosting shapes 63 / 85 / 407, never the flying
# sprinkles 7231) connected to the label are added up to DONUT_GROW px beyond the region (a bound so that nothing far away is
# reached). Then the interior is filled solid, enclosed gaps (sprinkles) included.
#
# Scope: keyed to type 6000590 only (every skin that uses this projectile type). Other types are untouched.

DONUT_T = 6000590
DONUT_FILE = 'sc/effects_brawler.sc'
DONUT_DOUGH = 32106      # shape id of the donut's dough (the ID pass gives shape id + 1 in _a4_shape_pixels)
DONUT_GROW = 24          # px (half resolution): bound of the donut's own frosting beyond the label's region
DONUT_CLOSE = 4          # px (half resolution): notches / gaps closed in a donut label
DONUT_FROST = (63, 85, 407)   # frosting shapes (sprinkles 7231 fly off the donut and are not part of it)


def _donut_parts(body, ctx):
    from scipy.ndimage import label as cc, binary_fill_holes
    if not body.any(): return body
    C, cv, R = ctx['C'], ctx['cv'], ctx['R']
    names = tuple(sorted({nm for f, nm in _trail_clips(DONUT_T) if f == DONUT_FILE}))
    SP = _a4_shape_pixels(R, DONUT_FILE, names)
    # 1. every label part must show dough (a donut in view); frosting splats left on the ground, specks: dropped
    dough = body & (SP == DONUT_DOUGH + 1)
    L, n = cc(body)
    keep = np.unique(L[dough]); keep = keep[keep > 0]
    body = body & np.isin(L, keep)
    if not body.any(): return body
    # 2. the donut's own frosting cut by the region's edge: connected sprite pixels beyond it
    own = ((C == 1) | (C == 3)) & (cv >= CUT) & np.isin(SP, [s + 1 for s in DONUT_FROST])
    cand = own & _dilate(body, DONUT_GROW)
    L, n = cc(cand | body)
    keep = np.unique(L[body]); keep = keep[keep > 0]
    body = body | (np.isin(L, keep) & cand)
    # 3. one solid piece: notches of a few px that effect sprites (ground glow, sparkles) or the character cut into the stack
    #    are closed, holes filled. Only pixels under a blended effect sprite are added: the narrow waists between touching
    #    donuts of one shot (background floor) must stay open, or the donuts would be bridged.
    from scipy.ndimage import binary_closing
    r = DONUT_CLOSE; yy, xx = np.mgrid[-r:r + 1, -r:r + 1]; disk = (yy * yy + xx * xx) <= r * r
    cl = binary_closing(np.pad(body, r), structure=disk)[r:-r, r:-r]
    return binary_fill_holes(body | (cl & blended_sprite(R, ctx['ids_h'])))


POST_BODY[DONUT_T] = _donut_parts
