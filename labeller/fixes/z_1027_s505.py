# sweep 10-27 fix for skin 505 (Sonnet revision, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s505/)
# sweep 10-27 fix for skin 505 PercenterGladiator, task s505 (Haiku fix, sweep 2026-10-27; task folder
# out/sweep2/haikutasks/sweep1027/s505/)
# Confirmed defect (sweep_1027 pair 9248, row 'sw2_505 attack 90 deg', tick 305, kind 'extra_label'): the red attack outline
# covers a small flat cyan-yellow ellipse on the ground just below the Gladiator's green ring. It is the launch ring of the
# attack (a muzzle / ground ring), which rule 4 says is never labelled. It is present only on the launch frame: at tick 305 the
# label holds it, at 307 it is gone, and at 303 and 315 the object has no label.
#
# Cause: the ellipse is drawn by the same clip as the heart projectile (Colette004Projectile, type 6000822, clip
# colette_004_atk_projectile in sc/effects_brawler.sc), so its pixels pass the body rule. Its sprite shape is 55092 (flat
# ellipse, 47 px at tick 305, bbox 582-598 x 275-283); the heart's own shapes (55054 yellow arrow, 55059, 55088, 55095, 55096,
# 146) are different shapes and sit below it, with a gap of several px.
#
# Fix: the pixels of shape 55092 are dropped from the body, together with the sprite-edge pixels (no shape) within 2 px of it,
# so that its anti-aliased rim does not stay as a speck. Shape ids are stable across captures, the ids come from the clip's
# own atlas uv, and no other shape of this type is touched. Only type 6000822 is keyed here; the other Colette types and
# other skins keep their labels.
# Round 2 (reviewer: detached label chunks, 30 px, stayed over the ring's left and centre at tick 305): they are pixels of
# the heart's own shapes (55088, 55054) that overlap the ring's rows, so cutting shape 55092 does not reach them. After the
# ring cut, every small connected part (< 150 px) of the body that lies within 5 px of the ring is dropped as well; the heart
# itself is far larger and is not touched, and nothing is dropped on frames that have no ring pixels.
# Checked (see report.md): tick 305 loses the 55092 ellipse and keeps the heart (pixel counts of the other shapes unchanged);
# ticks 303 (no label), 307-313 (no 55092 pixels in the label, so no change), 315 (no label) are unchanged.

GLADIATOR_LAUNCH_RING = {6000822: dict(file='sc/effects_brawler.sc',
                                       clips=('colette_004_atk_projectile', 'colette_004_atk_projectile_red'),
                                       ring=(55092,), edge=2, near=5, small=150)}


from scipy.ndimage import label as _cc


def _gladiator_launch_ring(body, ctx):
    if not body.any(): return body
    p = GLADIATOR_LAUNCH_RING[ctx['t']]
    SP = _a4_shape_pixels(ctx['R'], p['file'], p['clips'])   # shape id + 1 per pixel, 0: none of the clip's shapes
    ring = np.isin(SP, [s + 1 for s in p['ring']])
    if not ring.any(): return body
    edge = _dilate(ring, p['edge']) & (SP == 0)   # the ring's sprite edge (no shape), 2 px around it
    out = body & ~ring & ~edge
    near = _dilate(ring, p['near'])
    L, n = _cc(out, np.ones((3, 3), int))
    for i in range(1, n + 1):
        c = L == i
        if c.sum() < p['small'] and (c & near).any(): out = out & ~c
    return out


for _t in GLADIATOR_LAUNCH_RING: POST_BODY[_t] = _gladiator_launch_ring
