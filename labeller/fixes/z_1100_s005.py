# human review 2026-10-28 fix for skin 005 (Haiku fix, reviewed by the coordinator, installed 10-29; source out/sweep2/haikutasks/human1028/s005/fix.py; tested via IDLABELS_OVERRIDE after all fixes incl. z_1099_cleanup, hence z_1100_)
# Duplicator Darkcircus (skin 005), Lolla006Projectile (type 6001099): the TAIL is part of the projectile.
# User finding (human review 2026-10-28, page 005_DuplicatorDarkcircus_01): "the tail of the attack should also be labeled".
# The decision (task.md): the tail is part of the projectile and is labelled together with the head as ONE label per
# projectile; the tail's own solid shape, not the soft glow around it.
#
# Cause (sw2_005, ticks 265-279 and the attack 90 row, 303-325): the tail's segments are clip-body sprites (the same clip
# shapes 13414 ribbon / 13412, 13419 head) drawn translucent: their coverage is 96-255 along the tail, but below CUT (128)
# at the far end, so only the part near the head passed. The lola005 hook (z_1024_lola005.py) then kept body pixels within 2 px
# of the head only. The soft glow around the tail is trail-sprite haze (particle_gas_seq_1, shape 624-628, coverage 16-48)
# and clip shape 0 (glow, coverage 16-64): both stay out.
# Fix: (1) the clip body is taken from coverage 64 (FADE_CUT) so the translucent tail is body; the haze (16-48) and glow stay out;
# (2) POST_BODY replaces the lola005 hook for this type: the body minus the trail haze, the gaps of 2 px closed (only pixels of
# its own clip draws with coverage >= 64), holes filled (also those the haze lies over), and only the part connected to the head
# kept, so one label per projectile. Ended projectiles' leftover beads are still dropped by drop_ended_parts (runs after this).
LOLA006_ATK = 6001099
LOLA006_FILE = 'sc/effects_brawler.sc'
LOLA006_CLIP = ('lolla_006_atk_projectile', 'lolla_006_atk_projectile_red')
LOLA006_TRAIL = ('particle_gas_seq_1',)   # the trail's particles: the green haze around the tail (never the projectile)
LOLA006_BODY = (13412, 13414, 13419)      # clip shapes of the head (13412, 13419) and the tail's segments / ribbon (13414)
LOLA006_CUT = 64                          # coverage from which the clip body (translucent tail) is the projectile
FADE_CUT[LOLA006_ATK] = LOLA006_CUT
LOLA006_GAP = 2                           # px: gaps between the tail's own pieces closed


def _lola006_atk(body, ctx):
    from scipy.ndimage import binary_fill_holes, label as cc
    if not body.any(): return body
    R = ctx['R']; C = ctx['C']; cv = ctx['cv']; region = ctx['region']; ids_h = ctx['ids_h']
    SP = _a4_shape_pixels(R, LOLA006_FILE, LOLA006_CLIP)              # clip shape id + 1 per pixel, 0: none of the clip's shapes
    TR = _a4_shape_pixels(R, LOLA006_FILE, LOLA006_TRAIL) > 0         # the trail's haze / particles: never the projectile
    core = body & np.isin(SP, [s + 1 for s in LOLA006_BODY])
    if not core.any(): return body & ~TR
    m = body & ~TR & region
    own_d = np.unique(ids_h[core]); own_d = own_d[own_d > 0]          # the draws that show its own head / tail
    own_pix = np.isin(ids_h, own_d) & (cv >= LOLA006_CUT) & ~TR & region & ((C == 0) | (C == 1))
    m = _close(m, LOLA006_GAP) & (m | own_pix)                        # gaps between the tail's own pieces, own draws only
    m = binary_fill_holes(m) & region                                 # holes (also those the haze lies over) are solid
    L, n = cc(m, structure=np.ones((3, 3)))                           # one piece: the part connected to the head
    if n > 1:
        keep = np.unique(L[core & m]); keep = keep[keep > 0]
        m = np.isin(L, keep)
    return m


POST_BODY[LOLA006_ATK] = _lola006_atk
