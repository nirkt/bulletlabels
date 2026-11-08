# sweep 10-27 fix for skin 152 (Sonnet revision, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s152/)
# sw2_152 RocketGirlBeach, zone BurnMainAttackWater (did 17000241, the blue water blob, yellow in the page).
# Defect (pairs 2516 tick 303, 2519 tick 319; reviewer of fix.py: loops / holes still left, ragged notches): the zone label is
# the blob's own sprite pixels, but wherever the translucent blob is drawn over a map block (the purple block top) or a
# ground-level opaque draw, the ID pass shows that draw as front-most, so the label opens into enclosed holes, loops and
# notches (the label was a thin outline in places). The blob is visibly drawn over these spots, so the label is the blob's
# own outline filled solid (the user's 10-24 rule): the label is closed (gaps up to ~2 * S152_CLOSE px) and EVERY enclosed hole
# is filled (any size; only pixels of other labels stay out). The closing never reaches outside the zone's circle region
# (the previous label's own bounding region grown by S152_CLOSE) and never takes other labels' pixels.
# Only did 17000241 is touched; no other zone or projectile label changes.
S152_DID = 17000241
S152_CLOSE = 4      # px (half resolution)
_s152_prev_zone_labels = zone_labels


def zone_labels(state, cam, R, taken, own_team=None):
    out = _s152_prev_zone_labels(state, cam, R, taken, own_team)
    if not out: return out
    from scipy.ndimage import binary_fill_holes as _f, label as _cc
    for iid, (m, did, own) in list(out.items()):
        if int(did) != S152_DID or not m.any(): continue
        # keep the parts that were there: close small gaps between and inside parts, then fill every enclosed hole
        new = _close(m, S152_CLOSE) & ~taken | m
        new = (_f(new) & ~taken) | new
        out[iid] = (new, did, own)
    return out
