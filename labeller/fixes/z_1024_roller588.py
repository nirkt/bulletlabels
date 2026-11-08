# Roller Good Randoms 588 (Roller005Projectile 6000910): whole crystal flame (translucent bulk incl.), own part, one hull per flame (Haiku task 29, reviewed 10-24)
# Task 29: skin 588 RollerGoodRandoms (Roller005 flame, type 6000910, attack 0 and 90 deg).
# The user's rules: the flame is the projectile's whole visible body; a bright flame part left unlabelled is a defect.
# Diagnosis (probe.py / probe2.py, capture sw2_588): the flame is two sprite shapes of its clip. Shape 11837 (opaque
# head) was labelled. Shape 0 (the translucent bulk and tail, texture 2560x4064 shared with its trail, so classed
# 'shared' and cut at coverage < 128) was almost entirely unlabelled. Fix: shape 0 pixels of the clip with coverage >= 32,
# connected to the label, inside the object's region; holes filled (the trail's twinkles drawn over the body).
# Halo / haze is not a sprite pixel of the clip and is not taken. Only this type id.
ROLLER005_FLAME = {6000910: dict(file='sc/effects_brawler.sc', clips=('stu_005_atk_projectile', 'stu_005_atk_projectile_red'),
                                 body=(0,), cut=32)}


def _roller005_flame(body, ctx):
    # keep only this flame's own part: the connected piece of (body + tail) that overlaps its own predicted body most.
    # The other flame's underside / tail pieces lie in this flame's region too (their predictions overlap) and were
    # labelled as this one; with a hull they would also pull the other flame into this label (capture sw2_588, tick 265).
    if not body.any(): return body
    from scipy.ndimage import binary_fill_holes, label as cc
    p = ROLLER005_FLAME[ctx['t']]
    SP = _a4_shape_pixels(ctx['R'], p['file'], p['clips'])   # shape id + 1 per pixel, 0: none of the clip's shapes
    tail = np.isin(SP, [s + 1 for s in p['body']]) & (ctx['cv'] >= p['cut']) & ctx['region']
    m = body | tail
    L, n = cc(m)
    if n == 0: return body
    ov = np.bincount(L[ctx['pred_body'] & m].ravel(), minlength=n + 1); ov[0] = 0
    if ov.max() == 0: return body
    return binary_fill_holes(L == int(ov.argmax()))


for _t in ROLLER005_FLAME: POST_BODY[_t] = _roller005_flame
# one clean label per flame (the user's rule: one convex hull when the label is several parts or frays; without it the
# flame's label had 2-10 parts at ticks 269-313, the gaps between its translucent parts). Applied after POST_BODY, per object.
ONE_LABEL[6000910] = 'hull'
