# sweep 10-27 fix for skin 085 (Haiku fix, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s085/)
# Skin 085 CocoonerDefault, attack 0 deg (CocoonerProjectile, type 6001376), sweep 2026-10-27, pairs 1479 (tick 267)
# and 1480 (tick 269) plus the same notch at tick 265: the label of the swirl disc is C-shaped, open to the left where
# the cyan beam crosses the disc.
# Cause: the beam is a 16x16 solid texture drawn additively in front of the disc. Its pixels show no sprite of the
# attack (sprite code 0), so the ID pass gives them no label and the disc's outline has a notch of beam width.
# Fix (POST_BODY, after the per-type rules): the beam's pixels that lie in the notch become body, i.e. beam pixels with the
# disc's body in the same column above AND below (within COCOONER_BEAM_GAP rows; the beam band is 6 px tall), clipped to
# the convex hull of the body. A beam running out of the disc has body on one side only, so it stays unlabelled. The
# disc's own low-coverage pixels one row beside the notch (drawn at 64-72 coverage, no sprite code: the beam's antialiased
# edge) are added too. (A closing across the notch did not work: the notch opens into the disc's dark centre, not body.)
# Only pixels of 16x16 blended draws with no sprite code, or the disc's own draws beside them, are added; a frame without
# the beam over the disc is unchanged. Other types are not touched.
COCOONER_BEAM_GAP = 8   # px (half resolution): beam pixels with body within this many rows above AND below (the beam band is 6 px) are the notch


def _fix_6001376(body, ctx):
    if not body.any(): return body
    R = ctx['R']; ids_h = ctx['ids_h']; cv = ctx['cv']; C = ctx['C']
    beam_d = [d for d, dr in R['draws'].items() if dr.get('tw') == 16 and dr.get('tht') == 16 and dr.get('blend')]
    if not beam_d: return body
    beam = np.isin(ids_h, beam_d) & (cv >= CUT) & (C == 0) & ctx['region']
    if not beam.any(): return body
    # the notch: beam pixels with the disc's body in the same column above and below (a beam running into the disc from the
    # outside has body on one side only; the hull fill would take it in, so it is not used alone)
    up = np.zeros_like(body); dn = np.zeros_like(body)
    for k in range(1, COCOONER_BEAM_GAP + 1):
        up[k:] |= body[:-k]; dn[:-k] |= body[k:]
    fill = beam & up & dn
    if not fill.any(): return body
    # the disc's own pixels one row beside the notch (drawn at 64-72 coverage, no sprite code: the beam's antialiased edge)
    disc_d = np.unique(ids_h[body & (ids_h > 0)])
    edge = _dilate(fill, 1) & np.isin(ids_h, disc_d) & (cv >= FADE_CUT.get(6001376, CUT)) & (C == 0) & ctx['region']
    add = (fill | edge) & hull_mask(body) & ~body
    if not add.any(): return body
    return body | add


POST_BODY[6001376] = _fix_6001376
