# Belle SCmake 382 (Belle005Projectile 6000926): hull = flame only, solid-fill sparks out (user 10-26; Haiku task 37, reviewed; replaces z_1023_belle382.py)
# Belle SCmake (skin 382, ElectroSniperSCmake, Belle005Projectile type 6000926): the attack is the flame only, no sparks.
# Task 37 (2026-10-26), replaces z_1023_belle382.py. The user's decision (binding): the cyan crescents, slivers and diamonds
# around the flame head are SPARKS and stay OUT: the hull is the flame only (no spark sets its edge), on every tick of both
# attack rows (attack 0 deg, attack 90 deg). Only this type; the super (another type) is untouched.
# Cause: the sparks are drawn as a SOLID colour fill, atlas shape 47354 of sc/effects_brawler.sc (one texel: every pixel
# showing it is flat colour), while the flame is textured (its strands are shapes 47376-47380, 47358, 47359, 47377-47378).
# The rule of z_1023_belle382 kept every part of 60 px or more, so the 111 px cyan sliver at tick 313 (56 px solid, 43 px of
# the flame's shape 47380 at its tip) stayed and set the top-right edge of the hull.
# Approaches considered:
#   1. shape ids of the flame alone: rejected, shape 47380 and 47379 are both in the flame and at the tips of the sparks.
#   2. colour (cyan vs blue): rejected, the flame's core is cyan too, and the spark tips are as blue as the flame.
#   3. distance to the predicted body (ctx['pred_body']): rejected, the sparks lie inside the prediction.
#   4. size alone: rejected, sparks are as big as flame strands (t5's failure), and thin flame tongues are small.
#   5. solid fill (shape 47354) + connectivity to the flame + distance to the sparks: CHOSEN. The solid pixels are sparks
#      (removed); the flame is the largest part left; other parts stay when textured, attached to the flame (within
#      BELLE382_FAR px), not fringe of a spark, and not a small sliver beside a spark.
# The rule only REMOVES pixels of the current body (never adds), so the flame's own pixels are unchanged.
BELLE382_TYPE = 6000926
BELLE382_SOLID = 47354     # atlas shape of the clip's solid-colour sparks (flat fill)
BELLE382_MIN = 3           # px: smaller parts are dust
BELLE382_FAR = 30          # px: a part further than this from the flame is not attached to it (the violet tongue tips: 19-25 px)
BELLE382_PB = 0.5          # share of a part's pixels that must lie on the predicted body (beside-the-flame specks are not the flame)
BELLE382_NEAR = 2          # px: a part this close to a solid spark is that spark's fringe
BELLE382_SLIVER = 12       # px: a smaller part within this of a solid spark is that spark's sliver
BELLE382_SLIVER_BIG = 30   # px: a part this big beside a spark is flame (a tongue next to a crescent)


def _belle382_flame(body, ctx):
    from scipy.ndimage import label as cc, distance_transform_edt as edt
    if not body.any(): return body
    T = projtab.table()[ctx['t']]
    SP = _a4_shape_pixels(ctx['R'], T['file'], (T['blue'], T['red']))   # shape id + 1 per pixel, 0: none
    sw = body & (SP == BELLE382_SOLID + 1)                              # the sparks' solid fill
    if not sw.any(): return body
    flame = body & ~sw
    L, n = cc(flame)
    if not n: return flame
    sizes = np.bincount(L.ravel())[1:]; main = int(np.argmax(sizes)) + 1
    pb = ctx['pred_body']
    dsw = edt(~sw); dmain = edt(L != main)                              # distances (px) to the sparks / to the flame
    keep = [main]
    for k in range(1, n + 1):
        if k == main: continue
        m = L == k; size = int(sizes[k - 1]); gs = float(dsw[m].min()); gm = float(dmain[m].min())
        if size < BELLE382_MIN or gm > BELLE382_FAR: continue
        if (m & pb).sum() <= BELLE382_PB * size: continue     # beside the flame, off its predicted body: not the flame
        if gs <= BELLE382_NEAR: continue
        if gs <= BELLE382_SLIVER and size < BELLE382_SLIVER_BIG: continue
        keep.append(k)
    return np.isin(L, keep)


POST_BODY[BELLE382_TYPE] = _belle382_flame
