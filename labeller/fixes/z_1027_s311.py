# sweep 10-27 fix for skin 311 (Haiku fix, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s311/)
# sweep 10-27 fix for skin 311 DeadMariachiOutlaw (Poco 006: attack 6000345 'Poco006Projectile', super 6000346
# 'Poco006UltiProjectile'; both types are used by this skin only). Pairs 5810 (attack 0 deg, tick 283), 5818 (attack 90 deg,
# tick 323), 5826 (super 45 deg, tick 359); the same leftover puffs also show on the other ticks of those rows (attack 0 deg
# 279-281, attack 90 deg 319-321, super 355-357).
# The cloud is ONE trail-body group per shot (its members share label_id). A member's logic lands ('finished'), the client
# hides its body (drawn False), and the cloud's puffs drift on as after-effects. The group label stayed alive as long as ANY
# member was still in the battle state, so the hidden, finished members kept the leftover puffs labelled (tracker-gone
# rows of the group, pixels in the group's trail region): 5 to 10 ticks of specks and blobs after the cloud had gone.
# Rule (the user's 10-23 decision: once an attack's logic has ended its leftover effects are unlabelled; Q1 Poco super
# leftovers: unlabel; 10-20: end and fade animations are unlabelled): a Poco 006 group keeps its label only while at
# least one of its members is still drawn (live, not ended, body visible). When every member is hidden, the whole group
# (its live members and its ended trail remainder) is dropped. Groups with a visible body are unchanged.
# Only the two types above are touched; no other type, no other skin. The shared clean-up (z_1099) runs on the result.
S311_P6 = {6000345, 6000346}
_s311_prev_labels = labels


def labels(objs, cam, R, state=None, ended=(), img=None):
    shown = set()
    for o in objs:
        if o.get('t') in S311_P6 and not o.get('ended') and o.get('drawn', True):
            shown.add(o.get('label_id', o['id']))
    keep = [o for o in objs if not (o.get('t') in S311_P6 and o.get('label_id', o['id']) not in shown)]
    return _s311_prev_labels(keep, cam, R, state=state, ended=ended, img=img)
