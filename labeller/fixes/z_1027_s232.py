# sweep 10-27 fix for skin 232 (Haiku fix, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s232/)
# ShamanShiba (skin 232, Nita's shiba dog; sweep 2026-10-27, pairs 4183-4187, ticks 307-315, attack 90 deg).
# Type 6000226 Shaman003Projectile: the projectile is a 3D shiba head (a mesh, no sprite). Its footprint trail (clip
# nita_013_atk_footprint_trail of sc/effects_brawler_nita.sc: the blue paw-print sparks that follow the dog) is drawn
# 'over' the head, and the 'over' rule (aligned to the predicted body) took the paw print lying at the head's upper-left
# into the label: one small separate red speck beside the stem, 2-20 px per frame. Paw prints are the trail's effects,
# never the attack (rule 3): the label keeps the head's own draw and loses every pixel that the footprint trail's
# sprites show. Holes left in the head by a paw print lying on it are filled again, and the speck that remains (a few
# sprite-edge pixels of the paw print, not bridged to the head by the clean-up) is dropped as a separate part: parts
# smaller than 25 px (half resolution) and under a tenth of the largest part.

def _s232_fix_6000226(body, ctx):
    from scipy.ndimage import binary_fill_holes, label as cc
    SP = _a4_shape_pixels(ctx['R'], 'sc/effects_brawler_nita.sc', ('nita_013_atk_footprint_trail',))
    m = body & (SP == 0)
    if not m.any(): return m
    m = binary_fill_holes(m)
    L, n = cc(m)
    if n > 1:
        sz = np.bincount(L.ravel())[1:]
        keep = 1 + np.nonzero((sz >= 25) | (sz >= 0.1 * sz.max()))[0]
        m = np.isin(L, keep)
    return m
POST_BODY[6000226] = _s232_fix_6000226
