# 2026-11-08 enemy-label fix v2, group v9 (skins 130 140 156 295). Prefix _f9_. Installed as py/fixes/z_1118_enemy_v9.py.
# Own (blue) labels are never touched: see each part.
import numpy as _f9_np

# ---------------------------------------------------------------------------------------------------------------------------------
# (156 CactusMecha, Spike003Projectile 6000117) CAUSE: z_1023_spike_mecha keys the ball-only rule on the BLUE clip's shape ids (ball 58990, chain
# links 58408, chain rod 58409). The red clip spike_003_atk1_projectile_red draws the same layout with other shape ids (ball 58989, links 58405,
# rod 58406), so for an enemy shot the ball was not recognised and the label fell back to 'body minus the blue chain ids': the whole chain was
# in the label. FIX: the shape-pixel map of the clip reports the red shape ids as their blue counterparts (as z_1109_enemy_lola does), so the
# rule written for the blue art applies unchanged. Red shape ids are never drawn in an own shot, so own labels are pixel-identical.
_f9_MAP = {58989: 58990, 58405: 58408, 58406: 58409}                     # red clip shape -> blue counterpart
_f9_CLIPS = ('spike_003_atk1_projectile_red',)
_f9_LUT = _f9_np.arange(65536, dtype=_f9_np.int32)
for _r, _b in _f9_MAP.items(): _f9_LUT[_r + 1] = _b + 1                  # maps hold shape id + 1
_f9_prev_sp = _a4_shape_pixels


def _a4_shape_pixels(R, swf, names):
    out = _f9_prev_sp(R, swf, names)
    if any(n in _f9_CLIPS for n in names if isinstance(n, str)) and out.size:
        out = _f9_LUT[out]
    return out

# No change for 130 (Grom003UltiProjectile2: scattered fins, count / position differ per run), 140 (Digger shrapnel / bounce: random scatter and
# gem rotation phase, verdict flips between two captures), 295 (Bo005Projectile: launch-star phase at rel tick 3; fresh capture SAME): false positives.
