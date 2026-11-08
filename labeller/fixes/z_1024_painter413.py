# Painter Greek God 413 (Painter003Projectile 6001475): no hole where translucent pixels close the horseshoe notch (Haiku task 25, reviewed 10-24)
# Task 25 (agent t25): skin 413 PainterGreekGod. Painter003Projectile (6001475, attack 0 and attack 90 rows).
# The user's decisions: label the projectile's whole visible body (its solid horseshoe, no holes); soft translucent glow
# and ground marks not labelled.
# Tick 277 (attack 0 row): the translucent glow in the opening of the horseshoe's notch (coverage 69-123, below CUT 128)
# was labelled and closed the notch: one enclosed 102 px hole in the label. The notch is open at 265-275, 279 and at
# 305-319. Fix: only where the label encloses a hole, its labelled pixels under the body cut within 3 px of that hole
# are dropped, so the notch opens to the outside. Ticks without an enclosed hole are not changed.
import numpy as np

PAINTER_PROJ = 6001475
PAINTER_CUT = 128   # the projectile's body cut (idlabels CUT)
PAINTER_NOTCH_R = 3   # px (half-resolution grid) around an enclosed hole in which translucent label pixels are dropped


def _painter_open_notch(body, ctx):
    """drop the labelled translucent pixels (coverage < PAINTER_CUT) that lie within PAINTER_NOTCH_R px of an enclosed hole of
    the label: they close the horseshoe's notch from the outside (tick 277)"""
    if not body.any(): return body
    from scipy.ndimage import binary_fill_holes, binary_dilation
    holes = binary_fill_holes(body) & ~body
    if not holes.any(): return body
    ring = binary_dilation(holes, iterations=PAINTER_NOTCH_R) & body & (ctx['cv'] < PAINTER_CUT)
    return body & ~ring


POST_BODY[PAINTER_PROJ] = _painter_open_notch
