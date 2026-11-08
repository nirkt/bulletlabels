# sweep 10-27 fix for skin 271 (Haiku fix, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s271/)
# Skin 271 ElectroSniperBoss (sweep 2026-10-27, task s271): Belle002Projectile (type 6000917), the boss bolt.
# Defects (sw2_271, pairs 4996 / 4997 attack 0 deg ticks 265-267; 5000-5004 attack 90 deg ticks 305-313): the body is the
# orange shell head (atlas shape 63889, body layer) with a yellow flame column behind it (shape 65, also a body layer). At its
# back the column fades into the translucent teal trail, and the body took in thin spikes and strips there; above the flame
# there were detached specks and a small loop on the column top. The head itself was right.
# Fix: the body is cut to its solid part: a 2-iteration opening removes strips and spikes narrower than about 5 px, the largest
# connected part is kept (detached specks and sparkles out), its holes are filled. Every pixel of the shell (shape 63889) is kept
# as it was. The label stays one solid shape (the shell and the solid core of the flame).
# Approaches considered:
#   1. shape ids: rejected, the spikes and the flame column are both shape 65 (the column is body, the spikes are the same shape).
#   2. distance to the predicted body (ctx['pred_body']): rejected, every shape-65 pixel lies inside the predicted body.
#   3. frame colour (yellow core vs teal glow): rejected, a threshold on the frame is fragile across frames and skins.
#   4. morphology on the body, largest solid part, holes filled, shell kept: CHOSEN. It relies only on the geometry of the
#      solid part; it fails only for a genuinely thin part attached to the solid core (none on these ticks).
# Only this type (6000917). The super (6000919, Belle002UltiProjectile) is another type and is untouched.
BELLE002_TYPE = 6000917
BELLE002_SHELL = 63889     # atlas shape of the bolt head (_a4_shape_pixels returns shape id + 1)
BELLE002_OPEN = 2          # iterations of the cross-shaped opening (strips narrower than about 5 px go)
def _belle002_solid(body, ctx):
    from scipy.ndimage import binary_opening, binary_fill_holes, label as cc
    if not body.any(): return body
    T = projtab.table()[ctx['t']]
    SP = _a4_shape_pixels(ctx['R'], T['file'], (T['blue'], T['red']))   # shape id + 1 per pixel, 0: none
    shell = body & (SP == BELLE002_SHELL + 1)
    op = binary_opening(body, iterations=BELLE002_OPEN)
    if not op.any(): return body                                       # never lose the projectile
    L, n = cc(op); sz = np.bincount(L.ravel()); sz[0] = 0
    core = L == int(np.argmax(sz))
    out = (body & _dilate(core, 1)) | core | shell
    return binary_fill_holes(out)
POST_BODY[BELLE002_TYPE] = _belle002_solid
