# sweep 10-27 fix for skin 485 (Sonnet revision, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s485/)
# Surge004 projectile (PowerLevelerKnight skin 485, sweep 2026-10-27, pair 8956: attack 0 deg ticks 265-271, and the
# attack 90 deg row, ticks 305-313): the label is the cone with its puff trail (type Surge004Projectile, 6000798, clip
# surge_004_atk1_projectile_s1 / _red, sc/effects_brawler.sc). Second attempt (fix.py was rejected: the detached puff at
# tick 269 kept its check-mark notch, the thin tail streak was cut, spurs / notch / hook stayed at ticks 307-311).
#
# Defect (ragged): the label's outline zigzags between the puffs (notches, spurs, thin hooks: the sprite edges between the
# puffs), and small detached puff pieces are labelled as hollow check marks / loops.
#
# Fix, for this type only (POST_BODY, before the clean-up z_1099): the body is split into parts (8-connected pixels).
# Main part (the largest, with the cone): thin spurs opened away (r 2), gaps between puffs closed (r 3), holes
# filled, kept within 3 px of the predicted region; the cone's own pixels (shape 52732 of the clip, per pixel by the atlas
# uv: _a4_shape_pixels) are never removed. Every other part (a detached puff) of at least 40 px (half resolution) is replaced
# by its own convex hull (a puff is a round blob: a solid, notch-free outline, the thin streak of it included); smaller
# parts (flecks, loops of specks) are dropped, as the clean-up would do.
S485_SURGE004 = 6000798
S485_SMOOTH = 2.5
S485_CONE_SHAPE = 52732   # the cone (the projectile's own sprite shape, clip surge_004_atk1_projectile_s1)


def _s485_surge004(body, ctx):
    from scipy.ndimage import binary_fill_holes, gaussian_filter as _gauss, label as cc
    if not body.any(): return body
    L, n = cc(body, structure=np.ones((3, 3)))
    sz = np.array([int((body & (L == k)).sum()) for k in range(1, n + 1)])
    T = projtab.table()[S485_SURGE004]
    SP = _a4_shape_pixels(ctx['R'], T['file'], (T['blue'], T['red']))   # shape id + 1 per pixel, 0: none
    cone = body & (SP == S485_CONE_SHAPE + 1)
    main = 1 + int(np.argmax(sz))
    out = np.zeros_like(body)
    reg3 = _dilate(ctx['region'], 3)
    for k in range(1, n + 1):
        part = body & (L == k)
        if k == main:
            b = _close(part, 3) & _dilate(part, 3)
            b = binary_fill_holes(b) & reg3
            b = binary_fill_holes(_gauss(b.astype(np.float32), S485_SMOOTH) > 0.5) & reg3   # spurs / hooks / notches rounded
            b = binary_fill_holes(b | cone) & reg3 | cone
            out |= b
        elif sz[k - 1] >= 40:
            out |= hull_mask(part) & _dilate(ctx['region'], 4)
    return out


POST_BODY[S485_SURGE004] = _s485_surge004
