# sweep 10-27 fix for skin 286 (Haiku fix, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s286/)
# sweep 10-27 fix for skin 286 StickyBombPotato (task out/sweep2/haikutasks/sweep1027/s286/, finding pair 5301).
# Type 6000938 Squeak003Projectile (the sticky bomb; only this skin uses it). Attack 90 row, pair 5301, tick 311: a red loop
# around one detached cyan bubble just above the bomb. The clean-up step (z_1099) already drops that 23 px speck. The same
# bubbles are still labelled where they make a group above the speck size: attack 90 tick 305 (36 + 3 + 14 px above the bomb,
# a gap of 4 px) and attack 0 tick 265 (29 + 23 px left of the bomb). Rules 1, 3 and 4: the bubbles and their trail are effects,
# not the bomb body, so these detached groups are dropped too.
# Shapes (sc/effects_brawler.sc, per pixel via _a4_shape_pixels, checked on sw2_286 ticks 265-311):
#   55054 = the cyan bubble sprite (one of the bomb's own clip shapes; its pieces away from the body are the bubbles, a few
#   pixels stay attached to the body piece and are kept with it);
#   55386 = the bubble trail (trail clip particle_stickyBubble). The other trail clip, particle_potato_chip, is body-like here
#   and is kept (conservative).
# The bomb body is every other shape (93, 427, 635, 888, 55290, 55364, 55371, 55372 ...) and the bomb's stalk at tick 309
# (427 / 55290 pieces) is kept.
# Rule: the label's pieces (8-connected, not bridged) that hold no body shape pixel (only bubble / trail shapes and
# shape-less pixels) are dropped. The largest piece is always kept; when the object has no body shape pixel at all, nothing
# is dropped (a projectile never loses its label to this fix).
from scipy.ndimage import label as _s286_cc

S286_FILE = 'sc/effects_brawler.sc'
S286_BUBBLE = (55054, 55386)   # cyan bubble sprite, bubble trail (shape ids, not +1)


def _s286_bubbles(body, ctx):
    t = ctx['t']; T = projtab.table()[t]
    names = tuple(sorted({n for f, n in _trail_clips(t) if f == S286_FILE} | {n for n in (T['blue'], T['red']) if n}))
    SP = _a4_shape_pixels(ctx['R'], S286_FILE, names)   # shape id + 1 per pixel, 0: none
    if not body.any() or not SP.any(): return body
    dot = np.isin(SP, [s + 1 for s in S286_BUBBLE])
    bodysh = (SP > 0) & ~dot                              # any other shape of the sprites: the bomb's own body
    if not (body & bodysh).any(): return body
    L, n = _s286_cc(body, structure=np.ones((3, 3), bool))
    if n < 2: return body
    sz = np.bincount(L.ravel())
    has = np.bincount(L[body & bodysh], minlength=n + 1) > 0
    keep = has.copy(); keep[0] = False
    keep[1 + int(np.argmax(sz[1:]))] = True               # the largest piece is always the bomb
    return body & keep[L]


POST_BODY[6000938] = _s286_bubbles
