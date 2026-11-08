# sweep 10-27 fix for skin 379 (Haiku fix, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s379/)
# Skin 379 MinigunDudeZombie, attack 0 deg (sweep 2026-10-27, finding pair 7129, tick 287; same defect on ticks 265-285).
# Pam007Projectile (6000394): the bone-shaped attack (clip pam_007_atk_projectile / _red). Its label took (a) detached blobs of
# the teal smear beside the bone and (b) small spurs under the bone. The smear's own sprite shapes (trail clips
# particle_gas_seq_1 / _2: shapes 622-785 and 64062-64078) share draws with the bone, so the body rules (any drawn pixel of
# the body's draws next to the prediction) took them: (a). Spurs under the bone are clip shape 0, a faint ground shadow
# drawn under the bone (not part of the attack body; ground decals are not labelled): (b). Shape 0 is also drawn inside the
# bone outline, so only shape-0 pixels touching the bone's own shapes are kept.
# Fix (cut by sprite shape, per pixel, see _a4_shape_pixels): smear shapes dropped; shape 0 kept only where it touches the
# bone's other shapes; sprite-edge pixels with no shape stay only within 2 px of the bone; parts without bone pixels are
# dropped; enclosed holes (smear drawn over the bone) are filled. The bone's own shapes are kept unchanged.
# Applies to every skin whose projectile is type 6000394 (same brawler, same clip): the cut is by shape, not by skin.

PAM007_TYPE = 6000394
PAM007_FILE = 'sc/effects_brawler.sc'
PAM007_BONE = ('pam_007_atk_projectile', 'pam_007_atk_projectile_red')
PAM007_SMEAR = ('particle_gas_seq_1', 'particle_gas_seq_2')   # the trail clips of this type (shapes 622-785, 64062-64078)
PAM007_SHADOW = 0      # clip shape 0 (texture 1): the ground shadow under the bone, also drawn inside its outline
PAM007_EDGE = 2        # half-resolution px: sprite edges (no shape) kept only this close to the bone's own shapes


def _fix_6000394(body, ctx):
    if not body.any(): return body
    from scipy.ndimage import binary_fill_holes, label as cc
    SP = _a4_shape_pixels(ctx['R'], PAM007_FILE, PAM007_BONE)    # shape id + 1 of the bone's own clip, 0: none of them
    TP = _a4_shape_pixels(ctx['R'], PAM007_FILE, PAM007_SMEAR)   # shape id + 1 of the smear's trail clips, 0: none
    S0 = SP == PAM007_SHADOW + 1                                  # the ground-shadow shape
    bone = body & (SP > 0)
    core = bone & ~S0                                             # the bone's own shapes
    if not core.any(): return body                                # no bone shapes here: nothing to judge, unchanged
    m = body & ~((TP > 0) & (SP == 0))                            # the smear's shapes out (the bone's own shapes win)
    m = m & ((SP > 0) | _dilate(core, PAM007_EDGE))               # no-shape pixels: only at the bone's edge
    m = m & (~S0 | _dilate(core, 1))                              # shadow pixels: only where they touch the bone's shapes
    core = m & (SP > 0) & ~S0
    L, n = cc(m); keep = np.unique(L[core]); keep = keep[keep > 0]
    m = np.isin(L, keep)                                          # parts with bone shapes only (detached blobs and spurs gone)
    m = binary_fill_holes(m) & ctx['region']                      # holes the smear drew over the bone: filled
    return m


POST_BODY[PAM007_TYPE] = _fix_6000394
