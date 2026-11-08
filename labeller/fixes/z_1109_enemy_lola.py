# enemy-label fix 2026-11-04 (task away_1104/enemy, group lola: skin 002 Lolla003Projectile 6001093, skin 005 Lolla006Projectile 6001099)
# Cause: the installed fixes (z_1027_s002, z_1100_s002, z_1024_lola005, z_1100_s005, z_1107_dup_crow_poco_gale) key the projectile on
# the BLUE clip's shape ids (cylinder 35441 / head+tail 13412, 13414, 13419). The red clips (lolla_003_atk_projectile_red,
# lolla_006_atk_projectile_red) are the same layouts with other shape ids (35440 / 13411, 13413, 13415..13418), so for an enemy shot
# no pixel was recognised as the cylinder / head / tail: no label at all (002) and a head with a ragged tail piece (005).
# Fix: the shape-pixel map of these clips (_a4_shape_pixels) reports the red shape ids as their blue counterparts, so every rule
# written for the blue art applies unchanged to the red art. Blue shape ids are never touched (the map only renames red ids,
# which are never drawn in an own shot), so own labels are pixel-identical.
import numpy as _np
_EN4L_MAP = {35440: 35441,
             13411: 13412, 13413: 13414, 13415: 13419, 13416: 13420, 13417: 13421, 13418: 13422}    # red clip shape -> blue counterpart
_EN4L_CLIPS = ('lolla_003_atk_projectile_red', 'lolla_006_atk_projectile_red')
_en4l_prev_sp = _a4_shape_pixels
_EN4L_LUT = _np.arange(65536, dtype=_np.int32)
for _r, _b in _EN4L_MAP.items(): _EN4L_LUT[_r + 1] = _b + 1      # maps are shape id + 1


def _a4_shape_pixels(R, swf, names):
    out = _en4l_prev_sp(R, swf, names)
    if any(n in _EN4L_CLIPS for n in names if isinstance(n, str)) and out.size:
        out = _EN4L_LUT[out]
    return out
