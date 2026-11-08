# sweep 10-27 fix for skin 092 (Haiku fix, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s092/)
# Skin 092 AmbusherDefault (sweep 2026-10-27, Haiku fix task out/sweep2/haikutasks/sweep1027/s092/, finding pair 1537:
# attack 90 deg, tick 305, 'too_big').
# The default Ambusher attack (type 6001452, AmbusherProjectile: a purple streak drawn translucent, labelled by the STREAK
# rule) has a label that is bow-tied: its upper part reaches above the beam's own sprite into the brawler's body (a violet
# flare drawn over the magenta body, the brawler's pink leg), and the green disc's tint on the beam's sides leaves a waist
# (a notch at the disc's lower edge) between that upper part and the solid beam below. The STREAK rule takes any pixel of
# the purple range near the beam, so the brawler's pixels and the flare come in; the notch comes from the tint.
# Fix (POST_BODY of this type only, after the STREAK rule and the per-type rules, so it sees their result):
#  - pixels above the top row of the beam's own sprite (its clip body, sprite code 1, inside the label's region) are dropped
#    unless they are clip body themselves: the flare above the beam start is not the attack (not labelled: rule 4);
#  - clip-body pixels of the beam (sprite code 1, coverage >= 32, inside the region) are added wherever a vertical gap of at
#    most 8 px lies between label pixels: the waist is closed, so the label is one solid column from the beam's start to its tip.
# Checked on the capture: tick 305 loses 27 px of the flare (draw 65, no sprite code) and gains 77 px of the beam's own
# clip body; tick 265 (attack 0 deg, same type) loses 17 glow pixels above the beam (code 2, the halo above its top row)
# and gains 9 clip-body pixels. Other types are untouched. Not fixed (needs_user): how far down the beam's upper part runs
# behind the brawler's legs. Tick 305 keeps ~60 non-blue pixels of the pink leg at the left edge of the column; removing
# them (a colour cut in the first 12 rows) leaves a ragged top-left corner, so it is left to the user's decision.

import numpy as np

_s092_TYPE = 6001452    # AmbusherProjectile (AmbusherDefault, skin 29000854)
_s092_CLOSE = 8         # px: vertical gaps between the label's pixels closed inside the beam's clip body
_s092_CV = 32           # coverage from which a clip-body pixel of the beam is part of its gap closing


def _s092_fix(body, ctx):
    from scipy.ndimage import binary_closing
    C = ctx['C']; mine = ctx['mine']; cv = ctx['cv']; region = ctx['region']
    if not body.any(): return body
    clip_rows = np.nonzero(((C == 1) & region).any(1))[0]
    if not clip_rows.size: return body
    H = body.shape[0]; rr = np.arange(H)[:, None]
    top = clip_rows.min()
    body = body & ~((rr < top) & ~mine)
    cl = binary_closing(body, structure=np.ones((_s092_CLOSE + 1, 1), bool))   # gaps of up to _s092_CLOSE px
    return body | (cl & (C == 1) & (cv >= _s092_CV) & region)


POST_BODY[_s092_TYPE] = _s092_fix
