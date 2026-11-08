# sweep 10-27 fix for skin 021 (Sonnet revision fix2.py after a rejected Haiku fix, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s021/)
# Barkeep mug (BarkeepProjectile, type 6000248; skin 021 BarkeepDefault, sweep 2026-10-27 task s021, second attempt).
# Pairs 409-412, 418: detached puff specks / frayed bits beside the mug (rule 1: no specks, spurs; rule 3: trail puffs never labelled).
# Per pixel (shapediag, sc/effects_brawler.sc, clips barley_def_atk_projectile / _red, the type's own clip):
#   shape 58819 = the mug itself (cup, rim, white sleeve, cyan spout)            -> the label
#   shape 63    = blue puff sprites (the trail puffs, also dots drawn over the cup) -> not the mug
#   shape 407   = the blue splash / drip that hangs off the spout end (a blue puff-coloured blob, NOT the spout; the first
#                 attempt kept it as 'spout', leaving a frayed tail below the spout)
#   'none'      = sprite edge pixels (1-2 px around the shapes)
# Fix: the label is the mug shape (largest connected part of shape 58819 inside the label) plus
#   - the label's edge pixels within 2 px of it,
#   - the label's puff pixels (shape 63) that lie INSIDE the convex hull of the mug: a puff drawn over the cup's rim or face
#     hides the cup in the ID pass, the cup is there (no notch, no hole). Puffs outside the hull (trail) go,
#   then holes are filled.  Shape 407 and puffs apart from the mug never stay.
# If the label has no mug-shape pixel (another skin, hidden mug) the body is left as it is.
# Other skins sharing type 6000248 (not captured here): 218, 226, 234, 235, 278, 293, 366, 429 (same type id, same clip).
BARKEEP_MUG = 6000248
BARKEEP_CLIP_FILE = 'sc/effects_brawler.sc'
BARKEEP_CLIPS = ('barley_def_atk_projectile', 'barley_def_atk_projectile_red')
BARKEEP_PUFF = 63
BARKEEP_DRIP = 407
BARKEEP_BODY = 58819


def _barkeep_mug(body, ctx):
    if not body.any(): return body
    from scipy.ndimage import binary_fill_holes, label as cc
    SP = _a4_shape_pixels(ctx['R'], BARKEEP_CLIP_FILE, BARKEEP_CLIPS)   # shape id + 1 per pixel, 0: none of the clip's shapes
    mug = body & (SP == BARKEEP_BODY + 1)
    if not mug.any(): return body
    L, n = cc(mug)
    if n > 1:
        sz = np.bincount(L.ravel())[1:]; mug = L == (1 + int(np.argmax(sz)))
    hull = hull_mask(mug)
    puff = body & (SP == BARKEEP_PUFF + 1) & hull
    edge = body & (SP == 0) & _dilate(mug, 2)
    out = mug | puff | edge
    out = binary_fill_holes(out)
    L, n = cc(out)
    if n > 1:
        sz = np.bincount(L.ravel())[1:]; out = L == (1 + int(np.argmax(sz)))
    return out


POST_BODY[BARKEEP_MUG] = _barkeep_mug
