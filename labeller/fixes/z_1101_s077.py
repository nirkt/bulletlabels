# hosted review 2026-10-29 fix (Haiku fix, reviewed by the coordinator, installed 10-29 LOCAL ONLY; source out/sweep2/haikutasks/site1029/s077/fix2.py; tested via IDLABELS_OVERRIDE after all fixes incl. z_1100_*, hence z_1101_)
# Skin 077 (SplitterDefault), fix2 (coordinator review 2026-10-29): the streak is ONE convex hull per object.
# The coordinator: the outline still had notches / bumps where the straight column meets the round head (t305-t311, and
# the left end at t265). Thin lines / streaks = one convex hull over the whole attack (the user's global rule).
# This file is fix.py (kept unchanged next to it) plus one wrapper: the final label of each object (after fix.py's
# body) is replaced by its convex hull (idlabels.hull_mask, one piece), clipped only by props in front (the ID pass's
# depth: a draw nearer the camera than the projectile, as idlabels' own visibility test). Everything else, incl.
# FADE_CUT, EXTEND_BODY, the opaque-beam gate and the column extent, is exactly fix.py.
# Keyed to type 6001312 only.

# Skin 077 (SplitterDefault, hosted review 2026-10-29), type 6001312 (SplitterProjectile, clip rt_def_projectile of
# effects_brawler.sc): a long blue beam with a bright round head. Attack 0 flies along x, attack 90 flies down the frame.
#
# What the reviewer / the user saw: the label covered only part of the beam. The beam body is translucent away from its
# bright core (coverage 32..127), so the CUT of 128 kept only the part near the head: attack 0 t265 (left part missing,
# flagged), attack 90 t305-t311 (a flat top above the head; t311 'too small / cut off' = the head's rim outside the
# outline and the beam's top cut). The prediction window NEAR also ends before the beam's tail, so the beam behind the head
# was never reached. The green reticle drawn over the beam, and the cyan / yellow sparks on its edge, left notches, bumps
# and holes in the outline (the user's 'holes / ragged').
#
# Fix (keyed to 6001312 only; other types unchanged):
#  1. FADE_CUT: the beam's translucent body (coverage 32 and up) is body, as for other translucent bodies.
#  2. EXTEND_BODY (the existing mechanism, used for Beamer's super and Colette): the connected body pixels of the beam's
#     own draws are followed wherever the beam reaches, not only inside the prediction window.
#  3. POST_BODY gate: the label is kept only while the beam's OPAQUE body (its clip shape 666 at coverage >= 128, in the
#     object's region) is drawn. At attack 90 t313 the head has hit the water and is gone; the translucent tail left on the
#     water is an end leftover and must not be labelled (rules 2 and 4). At t315 nothing is opaque any more.
#  4. POST_BODY shape, for a beam that runs down the frame: its straight part is one column. The rows from the first to the
#     last full-width row of the beam's clip shape 666 are set to the column's edges (median of its left and right extents,
#     coverage >= 32). Inside that range sparks on the edge are out and the notches they leave are filled. The round head
#     below is the body as computed. Attack 0's horizontal beam keeps its outline (its round head is not squared).
#  5. POST_BODY rim: the head's core rim under the head (shape 0 of the clip at coverage >= 32) that the ownership test
#     left out is grown into from the label, at most 5 px, inside the beam's own columns (never into the water beside it).
#  6. POST_BODY smooth: closing of radius 2 and hole fill (the reticle and the chevron rings drawn over the beam are filled:
#     the label is one solid piece in the beam's own outline).
#
# Shape 666 of effects_brawler.sc is the beam itself (attack 0 t265 and attack 90 t305-t311: its pixels are the beam from the
# shooter's ring down to the head). Shape ids are stable for this sc file; nothing depends on draw ids or prog.

FADE_CUT[6001312] = 32
EXTEND_BODY.add(6001312)

S077_TYPE = 6001312
S077_BEAM_SHAPE = 666
S077_MIN_OPAQUE = 30     # px of opaque beam body in the region; below this the projectile has ended (its tail fades)


def _fix_6001312(body, ctx):
    from scipy.ndimage import binary_fill_holes, binary_closing, binary_dilation
    if not body.any():
        return body
    T = projtab.table()[S077_TYPE]
    names = tuple(sorted({n for n in (T.get('blue'), T.get('red')) if n}))
    SP = _a4_shape_pixels(ctx['R'], T['file'], names)
    beam = ((SP - 1) == S077_BEAM_SHAPE) & (ctx['cv'] >= 32) & binary_dilation(body, iterations=3)
    if (beam & (ctx['cv'] >= CUT) & ctx['region']).sum() < S077_MIN_OPAQUE:
        return np.zeros_like(body)
    body = binary_fill_holes(body)
    out = body.copy()
    brows = np.nonzero(beam.any(1))[0]
    if brows.size:
        bl = {int(y): int(np.nonzero(beam[y])[0].min()) for y in brows}
        br = {int(y): int(np.nonzero(beam[y])[0].max()) for y in brows}
        cl_ = int(np.median(list(bl.values()))); cr_ = int(np.median(list(br.values())))
        full = [int(y) for y in brows if bl[int(y)] <= cl_ + 2 and br[int(y)] >= cr_ - 2]
        if full and (max(full) - min(full)) > (cr_ - cl_):   # a beam running down the frame, not across it
            for y in range(min(full), max(full) + 1):
                if out[y].any() or beam[y].any():
                    out[y] = False; out[y, cl_:cr_ + 1] = True
    if out.any():
        core = ((SP - 1) == 0) & (ctx['cv'] >= 32)
        ys = np.nonzero(out.any(1))[0]; xs = np.nonzero(out.any(0))[0]
        box = np.zeros_like(out); box[:ys.max() + 6, max(0, xs.min() - 2):xs.max() + 3] = True
        grow = core & box & ~out
        for _ in range(5):
            add = binary_dilation(out) & grow
            if not add.any(): break
            out |= add; grow &= ~add
    yy, xx = np.mgrid[-2:3, -2:3]
    return binary_fill_holes(binary_closing(out, structure=(yy * yy + xx * xx) <= 4, border_value=0))


POST_BODY[S077_TYPE] = _fix_6001312


# ---- fix2: one convex hull per object (replaces fix.py's final label of 6001312 by its hull, clipped by depth) ----
_fix1_6001312 = POST_BODY[S077_TYPE]


def _fix2_6001312(body, ctx):
    body = _fix1_6001312(body, ctx)          # fix.py's final body for this object
    if not body.any():
        return body
    hull = hull_mask(body)                   # the convex hull of the whole body, as one piece
    R = ctx['R']; o = ctx['o']; wd = R.get('wd'); wref = R['meta'].get('wref', 0.0)
    if wd is not None and o.get('anchor') is not None:   # props in front of the beam cut the hull, nothing else does
        dz = _eye_depth(ctx['cam'], o['anchor'], wref)[0]
        hull = hull & (wd >= dz - DEPTH_TOL)
    return hull


POST_BODY[S077_TYPE] = _fix2_6001312
