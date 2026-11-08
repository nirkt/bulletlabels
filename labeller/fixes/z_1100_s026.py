# human review 2026-10-28 fix for skin 026 (Haiku fix, reviewed by the coordinator, installed 10-29; source out/sweep2/haikutasks/human1028/s026/fix2.py; tested via IDLABELS_OVERRIDE after all fixes incl. z_1099_cleanup, hence z_1100_)
# human review 2026-10-28 (skin 026 UndertakerDefault, page 026_UndertakerDefault_01.png, marked good), revised 2026-10-29.
# The user's note on the page: "mortis should get a hull for it's super". Coordinator decision 2026-10-29: the bats left
# after the super's logic has ended are NOT labelled (ended rule), from the first tick the logic is finished.
#
# Type 6000299 = UndertakerUltiProjectile (Mortis's base super: clip bat_clump_projectile in sc/effects.sc, trail 'bats').
# Its label is the bat swarm: the bats are the attack (TRAIL_BODY_EXTRA), labelled as many separate parts per frame.
#
# 1. Hull: ONE label per frame, the convex hull of the whole swarm (ONE_LABEL 'hull').
#    The body is cleaned first (POST_BODY): only pixels of the attack's own sprites (codes 1, 3, 5) and their sprite edges
#    (<= 3 px) stay; crumbs under 3 px go. On sw2_026 this removes nothing (every label pixel is a bat sprite or an edge).
#
# 2. Ended rule: why the bats stayed labelled after the logic finished (sw2_026, tick 363: object 2000000 has finished=1,
#    the tracker's ended entry has age 1, but the client still draws the object, so it is not in tr.gone and its 'ended'
#    flag is not set until the client drops it at tick 369). The label drop in idlabels.labels() only fires on o['ended'],
#    and drop_ended_parts() skips trail-body types (not trail_body(t)), so nothing removed the object's label before 369.
#    Fix: this type's objects with finished=1 are removed from the objects passed to labels() (same grouping rule as the
#    existing ended drop: a group with a member still running keeps its label). Ticks 345-361 (finished=0) are unchanged.
#
# Other fixes in py/fixes/ do not touch 6000299 (checked). Other skins with this type: see the handback report.

FINISHED_DROP = {6000299}   # projectile types whose label ends with the logic (finished), not with the client's drop
_prev_labels_s26 = labels


def _is_finished_drop(o):
    return o['t'] in FINISHED_DROP and o.get('finished') and not o.get('ended')


def labels(objs, cam, R, state=None, ended=(), img=None):
    alive = {o.get('label_id', o['id']) for o in objs if not _is_finished_drop(o)}
    objs = [o for o in objs if not (_is_finished_drop(o) and o.get('label_id', o['id']) not in alive)]
    return _prev_labels_s26(objs, cam, R, state=state, ended=ended, img=img)


def _fix_6000299_clean(body, ctx):
    from scipy.ndimage import label as _s26_cc, distance_transform_edt as _s26_edt
    if not body.any():
        return body
    own = np.isin(ctx['C'], (1, 3, 5))   # the attack's own sprite pixels (clip body, trail body, shared sprites)
    if own.any():
        body = body & (own | (_s26_edt(~own) <= 3))   # sprite edges of those sprites stay, anything further away goes
    if body.any():
        L, n = _s26_cc(body, structure=np.ones((3, 3), bool))
        if n > 1:
            sz = np.bincount(L.ravel())
            crumb = (sz < 3) & (np.arange(len(sz)) > 0)   # lone pixels and tiny specks
            body = body & ~crumb[L]
    return body


POST_BODY[6000299] = _fix_6000299_clean
ONE_LABEL[6000299] = 'hull'
