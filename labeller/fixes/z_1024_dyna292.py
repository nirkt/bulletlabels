# Dynamike Sticky Bomb Ghost 292: solid attack blobs; super cube + window + handle one piece (Haiku task 18, reviewed)
# Task 18: skin 292 StickyBombGhost (Dynamike). Attack 6000941 (Squeak004Projectile, the ghost blob) and super 6000942
# (Squeak004UltiProjectile, the ice cube with its handle). User's decision (2026-10-23, BRIEF + addendum): one solid label
# per live projectile, its visible body, fully labelled inside, no holes or slits; end animations stay unlabelled.
# Self-check found enclosed holes drawn over the body (eye / hat features of the ghost at ticks 267, 305-311; loops inside
# the cube at 349 and 355) and a narrow slit between the cube and its handle (359). Fix: close gaps of up to 2 (attack) /
# 3 (super) half-resolution px, then fill enclosed holes; pixels added only inside the projectile's region.
# The projectile ends at its max range / on landing (tracker ended, drawn False: 269, 313 and 367 stay unlabelled).


def _t18_solid(r):
    def f(body, ctx):
        if not body.any(): return body
        from scipy.ndimage import binary_fill_holes
        m = _close(body, r) | body
        return binary_fill_holes(m) & (ctx['region'] | body)
    return f


POST_BODY[6000941] = _t18_solid(2)   # Squeak004Projectile (attack, skin 292)
# super (cube + handle): the cube's own clip body shapes (the window, the faces and the handle are all body-class shapes of
# squeak_004_ulti_projectile; glow shape 0 and the trail particle shapes are not) are added where they lie within 6 px of
# the label (the window's notch), then closed (3 px) and holes filled. Trail sparks stay out (not clip shapes).
_T18_SUPER_CLIPS = ('squeak_004_ulti_projectile', 'squeak_004_ulti_projectile_red')
_T18_SUPER_BODY = {sid for (sid, cmd), c in _layer_classes(6000942, True).items() if c == 'body'}


def _t18_super(body, ctx):
    if not body.any(): return body
    from scipy.ndimage import binary_fill_holes
    SP = _a4_shape_pixels(ctx['R'], 'sc/effects_brawler.sc', _T18_SUPER_CLIPS) - 1   # shape id, -1 none
    cube = np.isin(SP, sorted(_T18_SUPER_BODY)) & ctx['region']   # the cube + handle's own shapes (window included)
    add = cube & _dilate(body, 6)
    m = body | add
    m = _close(m, 3) | m
    # the cube's silhouette closed on its own shapes (a notch in the window's top edge is not closed by the label alone);
    # added only within 4 px of the label, so nothing far from the bomb is taken
    cs = _close(cube, 4) & _dilate(m, 4) & ctx['region']
    m = m | cs
    return binary_fill_holes(m) & (ctx['region'] | body)


POST_BODY[6000942] = _t18_super   # Squeak004UltiProjectile (super, skin 292)
