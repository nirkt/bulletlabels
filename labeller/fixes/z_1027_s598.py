# sweep 10-27 fix for skin 598 (Haiku fix, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s598/)
# sweep 2026-10-27, skin 598 UndertakerGreekGod (task out/sweep2/haikutasks/sweep1027/s598/, pairs 10272 and 10273,
# super 45 deg, ticks 343-369; capture W/runs/gc/sw2_598).
# extra_label: the magenta super label (type 6000307 Mortis009UltiProjectile, the Undertaker's super, file
# sc/effects_brawler_mortis.sc) also took in small leaf and petal particles of its trail, away from the skull orb.
# 6000307 is a trail-body type (TRAIL_BODY_EXTRA), so its trail particles were body. Rule 3 (trails and particles are
# never labelled) and rule 4 (particles left behind are not the attack) apply; the orb itself stays as it was.
#
# Leaf / flower trail clips of 6000307 (idlabels._trail_clips; shapes from shapediag):
#   mortis_009_leaves_trail (shapes 24, 148), janet_004_atk_trail_leaf_1 and _3 (shapes 13379-13397),
#   mortis_009_flower and mortis_009_flower_leaves (shapes 1, 2, 3: the orb's own clip uses shapes 1, 2, 3 too).
# Approaches considered:
#   1. skip the leaf clips by name (TRAIL_SKIP): the leaves with their own shapes are removed. Used (part 1).
#      It cannot remove the flower and the leaves that share shapes 1 and 3 with the orb's clip, so part 2 is needed.
#   2. shape ids alone: rejected, shapes 1 and 3 are the orb's own sprites too.
#   3. distance to the prediction (pred_any / pred_body): rejected, the prediction of this trail-body type covers most
#      of the frame (about 24000 to 39000 px), so every particle lies inside it.
#   4. connected parts of the label (chosen, part 2): the orb is several parts 2 to 5 px apart, while the flower and the
#      loose leaves are separate blobs 40 to 110 px away from it. A part is kept when it links to the largest part through
#      parts at most MORTIS009_NEAR px apart (single linkage); the largest part is always kept (a projectile never loses
#      its label). Measuring to the largest part alone dropped the right lobe of the orb at tick 367.
#   5. colour / brightness thresholds: rejected, the least robust.
# Part 3: a leaf that lies over the orb's top edge (tick 361) leaves an enclosed hole of about 80 px in the orb's label
# (no sprite code: an effect drawn over the attack). Holes of at most MORTIS009_HOLE px are filled (the addendum's rule
# for effects drawn over an attack). No other hole of the row is this small (checked on all 14 ticks), and the dark
# centre of the orb is not a hole of the label (it opens to the outside), so it is unchanged.
# Part 4: the outline is closed (MORTIS009_CLOSE) only next to the skipped leaves, so the leaves over the edge do not leave notches.
# Other skins of this type (none found in the kept captures): the same rules would apply there, not verified.

TRAIL_SKIP[6000307] = ('mortis_009_leaves_trail', 'mortis_009_flower', 'janet_004_atk_trail_leaf')

MORTIS009_TYPE = 6000307
MORTIS009_NEAR = 12   # px (label array units = frame px): a part this close to the largest part belongs to the orb
MORTIS009_HOLE = 120  # px: holes of the orb's label this small are filled (an effect drawn over the orb)
MORTIS009_CLOSE = 10  # px: closing that bridges the notches a leaf over the orb's edge cuts into the outline


def _fix_mortis009_loose_parts(body, ctx):
    from scipy.ndimage import label as _m9_cc, distance_transform_edt as _m9_edt, binary_fill_holes as _m9_fill
    if not body.any():
        return body
    L, n = _m9_cc(body, structure=np.ones((3, 3), bool))
    if n >= 2:
        sz = np.bincount(L.ravel())
        main = 1 + int(np.argmax(sz[1:]))
        # the orb's pieces chain together (a lobe, a middle piece and the bowl lie 4 to 5 px apart): a part belongs to the
        # orb when it links to the largest part through parts each at most MORTIS009_NEAR px from the one before
        keep = np.zeros(n + 1, bool)
        keep[main] = True
        grown = True
        while grown:
            grown = False
            d = _m9_edt(~keep[L])   # distance of every pixel to the orb's parts kept so far
            for k in range(1, n + 1):
                if not keep[k] and (d[L == k] <= MORTIS009_NEAR).any():
                    keep[k] = True
                    grown = True
        body = body & keep[L]
    if body.any():
        body = body | small_holes(body, MORTIS009_HOLE)
    # leaves lying over the orb's edge (their sprites are 'over', code 4) cut notches in its outline (tick 361): the orb
    # is behind them, so the outline is closed there, only next to those leaves (MORTIS009_CLOSE px, the leaves' margin 3 px)
    C = ctx['C']
    if body.any() and (C == 4).any():
        from scipy.ndimage import binary_closing as _m9_close, binary_dilation as _m9_dil
        P = MORTIS009_CLOSE
        cl = _m9_fill(_m9_close(np.pad(body, P), iterations=P)[P:-P, P:-P])
        lv = _m9_dil(C == 4, iterations=3) & ctx['region']
        body = body | (cl & lv)
    return body


POST_BODY[MORTIS009_TYPE] = _fix_mortis009_loose_parts
