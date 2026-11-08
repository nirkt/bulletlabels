# sweep 10-27 fix for skin 172 (Sonnet revision, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s172/)
# sweep 10-27 fix (v2) for skin 172 (RocketGirlDelivery, task out/sweep2/haikutasks/sweep1027/s172/, pair 2963, sw2_172 super
# 45 deg, tick 377, extra_label). Replaces fix.py (rejected: it cut holes in the star heads).
# Type 6000064 (Brock006UltiProjectile, clip brock_def_ulti_tail_blue, trail brock_def_ulti_trail / rocketSmoke_short_blue_*):
# the super's rockets. Each rocket = blue star head (clip body, codes 1 / 2) on an orange box (mesh, no sprite code), trailed by
# dark blue smoke puffs.
# Cause: the puff sprite is in the body clip AND in the trail clips (code 5 'shared') and the trail's puffs drawn over the body
# are code 4; labels() takes them as body, so the label has smoke lobes around the star head. But the puffs are also drawn IN
# FRONT of the star head, so the ID pass shows only fragments of the star (codes 1/2); dropping all code 4/5 pixels (fix.py)
# leaves a frayed star with holes.
# Rule (keyed to type 6000064 only): the star head = the convex hull of its own visible star pixels (codes 1 and 2) of each
# rocket (clusters within STAR_LINK px); the puff pixels (codes 4 / 5) are dropped everywhere EXCEPT inside that hull. The
# result is closed (STAR_CLOSE px) and holes filled, so the star head stays one solid piece joined to the box; smoke outside
# the star is never labelled (user rule 3).
# Approaches considered:
#   1. drop code 4/5 pixels (fix.py): rejected by review, hole-ridden star.
#   2. drop code 4/5 + closing/fill only: the star's pixels hidden behind a puff are a large part; closing at a small radius
#      does not restore the star outline. Rejected.
#   3. hull of the star's visible pixels + drop puffs outside it (chosen): uses the star's own sprite pixels; the star head is
#      compact (about 20 px), so its hull is a faithful solid outline.
#   4. distance to the prediction / brightness: rejected, least robust.
STAR_TYPE = 6000064
STAR_LINK = 4
STAR_CLOSE = 2
STAR_PRED = 3      # px around the predicted body in which star pixels count as the rocket's own
STAR_EDGE = 5      # px around the predicted body kept (the mesh box is drawn inside it)
STAR_PART = 150    # px: a part this big is kept even without star pixels (the box)
STAR_HIDE = 3      # px around the star's hull in which puff pixels inside the predicted body belong to the hidden star


def _fix_6000064_star_solid(body, ctx):
    from scipy.ndimage import binary_fill_holes, label as cc
    if not body.any(): return body
    raw = sprite_pixels(ctx['t'], ctx['own'], ctx['R'])
    puff = np.isin(raw, (4, 5))
    seeds = body & np.isin(raw, (1, 2))
    pred = ctx['pred_body']
    keep = body & ~puff
    own = seeds & _dilate(pred, STAR_PRED)      # the rocket's own star pixels: neighbours' stars / debris drawn with the
    if own.any():                                # same sprite lie outside its predicted body
        keep = keep & ~(seeds & ~own) & _dilate(pred, STAR_EDGE)   # also debris edge pixels (code 0) strung along it
        seeds = own
    if not seeds.any(): return body & ~puff
    L, n = cc(_dilate(seeds, STAR_LINK))
    hull = np.zeros_like(body)
    for i in range(1, n + 1):
        s = seeds & (L == i)
        if not s.any(): continue
        h = hull_mask(s)
        # the star's part hidden behind a puff drawn in front of it: puff pixels inside the predicted body next to the star
        s2 = s | (body & puff & pred & _dilate(h, STAR_HIDE))
        hull |= hull_mask(s2)
    out = keep | hull
    out = _close(out, STAR_CLOSE)
    out = binary_fill_holes(out) & (out | ctx['region'])
    # specks / strips left over (debris edge pixels, e.g. a rocket buried in an explosion): only parts that hold the star
    # head or are box-sized stay
    L, n = cc(out)
    ok = [i for i in range(1, n + 1) if (L == i).sum() >= STAR_PART or (hull & (L == i)).any()]
    return np.isin(L, ok) if ok else out


POST_BODY[STAR_TYPE] = _fix_6000064_star_solid
