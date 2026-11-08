# AI-review batch 2026-11-01 fix (Haiku fix, reviewed by the coordinator, installed 11-01 LOCAL ONLY; source out/sweep2/haikutasks/siteai/undertaker/fix.py)
# AI-review findings 2026-11-01, task undertaker: skins 482 UndertakerBazaar, 526 UndertakerMecha, 546 UndertakerUndertakerHockey,
# 578 UndertakerSB, 598 UndertakerGreekGod. Their 'super 45 deg' rows (ticks 343-369) carry Mortis's super, one projectile type per skin:
#   482 -> 6000300 Mortis003UltiProjectile (tornado)     526 -> 6000301 Mortis005UltiProjectile (blue sphere + bats)
#   546 -> 6000302 Mortis006UltiProjectile (blue discs)  578 -> 6000303 Mortis007UltiProjectile (cyan vortex)
#   598 -> 6000307 Mortis009UltiProjectile (skull orb)
# Owner rules applied (coordinator 2026-11-01, as z_1100_s026.py / z_1102_s195.py for skins 026 and 195):
#   1. Mortis super: ONE convex hull over the super per tick (the hull of its own body, after the per-type clean-up).
#   2. Its label is dropped once the super's logic has ended: object finished = 1 (the logic reached its hit at tick 363 on all
#      five captures, sw2_482 / 526 / 546 / 578 / 598), while the client keeps drawing the object until tick 369.
#
# Mechanism per type (all five): POST_BODY = the installed hook of that type first, then _mort_clean:
#   1. the main part of the body and the parts chained to it (MORT_NEAR px, the body's own connectivity);
#   2. of those, only the super's own sprite pixels (codes 1, 3, 5), their edge (MORT_EDGE px) and the glow they enclose;
#      glow outside the solid body (aura, light over the brawler, flashes beside a disc) is dropped;
#   3. ONE_LABEL 'hull': one convex hull of the cleaned body per tick;
#   4. labels() wrapper: no label once finished = 1 (the logic has ended; the client draws the object until tick 369).
# Findings it addresses (verified on the compare images compare_NNN.png in this folder):
#   482 t351-t361 (ragged, spurs, loop round the leaf): one hull; t363-t367 (leftovers after the logic ended): no label.
#   526 t355 (follows the splash, interior loop): one hull of the sphere and its bats; t363 on: no label (see below).
#   546 t349 (too big, reaches into the glow / flash beside the disc) and t355 (ragged): one hull of the discs; the glow
#       beside the discs is out. The cyan starburst beside the small disc at t355 is the super's own sprite (code 1) and stays
#       in the hull: needs_user.
#   578 t347 (interior lines): one hull of the vortex. t367 (should not be labelled): no label from 363 (finished).
#       At t345 the fade cut 64 gives no label (the yellow swoosh beside it is not the super).
#   598 t361 (ragged): one hull. t367 (reviewer: should get a hull): no label, the super's logic has ended (finished from 363),
#       so the ended rule removes it; a hull at 367 would contradict that rule: needs_user. The dark-blue ring above the orb at
#       t357-t361 is the super's own trail sprites (code 3) and stays inside the hull: needs_user.
#
# Decision flagged for the coordinator: on 526 the impact frame (tick 363, finished = 1, the blue sphere still drawn) keeps its
# label in z_1027_s526 (age-1 rule). The logic-ended rule drops it from tick 363, as on skins 026, 195, 482, 546, 578 and 598.
# This file drops it; to keep the old behaviour, remove 6000301 from MORT_FIN.
#
# Installed hooks for these types are kept: the body hook of each type (a3_mortis003, a3_mortis005, MORTIS_TWINKLE for 546,
# the loose-part clean-up for 598, the age-based drop of z_1027_s526) runs first, then this file's clean-up, then the hull.
# Other skins with these types: owners.projectile names one skin per type (checked: 6000300..6000303 and 6000307 each list
# only the skin above). Type 6000307 is also named in z_1027_s598.py (same skin).

MORT_SUPER = {6000300, 6000301, 6000302, 6000303, 6000307}
MORT_FIN = set(MORT_SUPER)     # types whose label ends with the logic (finished), not with the client's drop
MORT_NEAR = 3                  # px (full frame resolution): a body part within this distance of the kept parts belongs to the super
MORT_EDGE = 2                  # px: sprite edge pixels next to the super's own sprites stay
# fading-in unit (sw2_578 t347: the vortex's translucent rings have coverage 32-127, the body was 1433 px of ~7000 predicted):
# the lower cut keeps its silhouette (rule: fading-in units keep their full silhouette)
FADE_CUT[6000303] = 64

_mort_prev_post = {t: POST_BODY[t] for t in MORT_SUPER if t in POST_BODY}
_mort_prev_labels = labels


def _mort_clean(body, ctx):
    """1. the main part of the body and the parts chained to it (within MORT_NEAR px, the body's own connectivity: a disc
    joined to the other through glow stays);
    2. of those pixels, only the super's own sprite pixels (clip body, trail body, shared sprites: codes 1, 3, 5), their edge
    pixels (MORT_EDGE px) and the glow they enclose. Glow lying outside the solid body (the aura around the orb, the light over
    the brawler, a flash beside a disc) and unsprited pixels away from the sprites are not the super."""
    from scipy.ndimage import label as _mc_cc, distance_transform_edt as _mc_edt, binary_fill_holes as _mc_fill, binary_dilation as _mc_dil
    if not body.any():
        return body
    L, n = _mc_cc(body, structure=np.ones((3, 3), bool))
    if n > 1:
        sz = np.bincount(L.ravel())
        keep = np.zeros(n + 1, bool)
        keep[1 + int(np.argmax(sz[1:]))] = True
        grown = True
        while grown:
            grown = False
            d = _mc_edt(~keep[L])
            for k in range(1, n + 1):
                if not keep[k] and (d[L == k] <= MORT_NEAR).any():
                    keep[k] = True
                    grown = True
        body = body & keep[L]
    own = body & np.isin(ctx['C'], (1, 3, 5))
    if not own.any():
        return body & False
    return body & (_mc_dil(own, iterations=MORT_EDGE) | _mc_fill(own))


def _mort_hook(body, ctx):
    t = ctx['t']
    if t in _mort_prev_post:
        body = _mort_prev_post[t](body, ctx)
    return _mort_clean(body, ctx)


for _t in MORT_SUPER:
    POST_BODY[_t] = _mort_hook
    ONE_LABEL[_t] = 'hull'


def _mort_fin(o):
    return o['t'] in MORT_FIN and o.get('finished') and not o.get('ended')


def labels(objs, cam, R, state=None, ended=(), img=None):
    # the super's logic has ended (finished = 1): no label for it; a group with a member still running keeps its label
    alive = {o.get('label_id', o['id']) for o in objs if not _mort_fin(o)}
    objs = [o for o in objs if not (_mort_fin(o) and o.get('label_id', o['id']) not in alive)]
    return _mort_prev_labels(objs, cam, R, state=state, ended=ended, img=img)
