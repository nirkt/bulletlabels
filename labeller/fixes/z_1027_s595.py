# sweep 10-27 fix for skin 595 (Haiku fix, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s595/)
# Skin 595 SniperJapan (Piper, sweep 2026-10-27, pairs 10242 / 10243): Piper009Projectile, type 6000382 (the orb
# attack, 90 deg). The green/cyan swirl inside the orb is drawn translucently over the body sprite: its pixels are the
# body's own sprite (sprite code 1) at coverage 27..126, under the CUT of 128, so they fell out of the body and left
# closed holes in the label (tick 309: one 69 px hole, tick 311: one 69 px hole, lower-left of the orb). The label must
# be the object's own outline filled solid (no holes). Fix: fill the holes of this type's body, limited to its region.
# Ticks 305 / 307 have no holes and are unchanged; no other type is touched.
def _fix_6000382(body, ctx):
    from scipy.ndimage import binary_fill_holes
    if not body.any():
        return body
    return body | (binary_fill_holes(body) & ctx['region'])

POST_BODY[6000382] = _fix_6000382
