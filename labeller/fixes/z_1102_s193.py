# hosted review 2026-11-01 fix (Haiku fix, reviewed by the coordinator, installed 11-01 LOCAL ONLY; source out/sweep2/haikutasks/site1101/s193/fix.py; runs after z_1101_*)
# Skin 193 (Twins Angel) fix, hosted review 2026-11-01 (task s193, THE USER, binding): "holes / ragged" on attack 0 ticks
# 269, 277, 283 and attack 90 ticks 307, 317, 323. All attack ticks of both rows were checked (263-285, 303-325).
# Keyed to type 6001408 Twins003ThrowerProjectile (attack, clip twin_003_atk_projectile, skin 193 only). No other type or skin.
#
# What the projectile is (the arrow): a blue triangle with a round blue core, and two yellow wing crescents. The triangle is
# partly translucent: its pixels with ID-pass coverage 32..127 are below the body cut CUT (128), so they were holes in the
# label, and the outline around the holes was ragged. At tick 283 part of the triangle was missing from the label and its tip
# was a separate piece (3 parts in all). Measured on sw2_193: the unlabelled arrow-coloured pixels next to the label are the
# arrow's clip body at ID coverage 32..127 (tick 283: 27 of 33; tick 323: 35 of 42; tick 267: 23 of 27). Tick 307's
# unlabelled cyan is the big cyan machine behind the arrow (not the arrow), and it stays unlabelled.
#
# Fix (the coverage cut is the game's own measure, so this is the most stable of the options; the others are noted in the
# report: hull of the whole arrow (not asked for, it fills the notch between the wings), closing + hole fill of the opaque label
# alone (did not reach the tip hole at tick 283), colour rules on the frame (least robust)):
#  1. FADE_CUT 32 for this type: the translucent clip-body pixels of the arrow are body (as FADE_CUT of z_1024_ash138 and
#     z_1027_s129). Only its own clip body (C == 1) enters; its glow (C == 2) does not.
#  2. Translucent trail-clip pixels (C == 3 below CUT) are removed again: they are not the arrow.
#  3. Only the parts of the body that touch the opaque body (within 2 px) are kept. The UI pip beside the ammo bar (the
#     blue dot that z_1027_s193 removes) and other detached translucent bits stay out. z_1027_s193 still runs after this.
#  4. One smooth solid outline: closing 1 px, hole fill, Gaussian sigma 1.0 (threshold 0.5), growth at most 1 px beyond the
#     body, specks under 15 px dropped.
# Measured (sw2_193, 12 flagged/checked ticks): the opaque parts of the arrow keep all their pixels (lost 0-3 px per tick, the
# 1-px tips the smoothing trims); gained 17-132 px per tick, all of them the triangle's translucent body or the 1-px edge.
# The label is one part per tick (before: up to 3 parts at 283).

FADE_CUT[6001408] = 32
_S193_SIG = 1.0          # px, Gaussian sigma of the arrow's outline (threshold 0.5)


def _s193_arrow_body(body, ctx):
    from scipy.ndimage import binary_fill_holes, gaussian_filter, binary_dilation, label as cc
    if not body.any(): return body
    cv, C = ctx['cv'], ctx['C']
    body = body & ~((C == 3) & (cv < CUT))                      # translucent trail-clip pixels are not the arrow
    strict = body & (cv >= CUT)
    L, n = cc(body)
    if n == 0: return body
    near = binary_dilation(strict, iterations=2) if strict.any() else np.zeros_like(body)
    keep = [k for k in range(1, n + 1) if (near & (L == k)).any()]  # parts that touch the opaque body (the UI pip does not)
    m = np.isin(L, keep) if keep else body
    m = binary_fill_holes(_close(m, 1) | m)                     # 1 px gaps closed, enclosed holes filled
    sm = gaussian_filter(m.astype(np.float32), _S193_SIG) >= 0.5
    sm &= binary_dilation(m, iterations=1)                      # never more than 1 px beyond the body
    sm = binary_fill_holes(sm)
    L2, n2 = cc(sm)
    if n2 > 1:                                                  # specks only: a part under 15 px is dropped
        sz = np.bincount(L2.ravel()); sz[0] = 0
        sm = np.isin(L2, np.nonzero(sz >= 15)[0])
    return sm


POST_BODY[6001408] = _s193_arrow_body
