# hosted review 2026-11-01 fix (Haiku fix, reviewed by the coordinator, installed 11-01 LOCAL ONLY; source out/sweep2/haikutasks/site1101/s195/fix.py; runs after z_1101_*)
# Skin 195 UndertakerDemon (Mortis 008 Demon), page 195_UndertakerDemon_01.png, row "super 45" (ticks 343-369).
# The user's finding (hosted review 2026-11-01): "label's wrong. also, it should get a hull". Coordinator (task.md): the same
# rule as skin 026's Mortis super (z_1100_s026.py): ONE convex hull over the whole super per tick, and no label once the
# super's logic has finished.
#
# Type 6000304 = Mortis008UltiProjectile (logic UndertakerUltiProjectile, file sc/effects_brawler.sc). Its parts, as drawn:
#  - the ghost: a 3D mesh (draws of program 129, no atlas uv, opaque) = the attack's body;
#  - its demon bats: the sprites of its trail clips mortis_008_ulti_trail_demon / _flip (bats with yellow eyes) = the attack;
#  - the blue follow trail (mortis_008_ulti_follow_trail_01/02: fire wisps behind the ghost), the fang fire sprites
#    (fang_005_atk_trail_fire) and the ground fire (mortis_008_ulti_ground_fire_01/02) = effects, not labelled (rule 3).
# Before, the label was ragged: the ghost with notches, parts of the blue trail (orange in the diagnosis, e.g. shape 28407)
# and only fragments of the bats (red rings on the bats' eyes on the page, tick 349).
#
# Fix: the body is the ghost's mesh pixels plus the pixels of the demon bat sprites (sprite shape ids of the two demon
# clips, shape 0 excluded: it is the generic glow circle that other clips use too). Everything else is dropped from the
# body. ONE_LABEL 'hull' then gives one convex hull of the ghost and its bats. The bats after the logic ended are dropped
# (in capture sw2_195: finished = 1 from tick 362, client-ended at 369; no label from 362 on): the same rule as skin 026.
#
# Other skins with this type: owners.projectile(6000304) names only skin UndertakerDemon. The client's availability list also
# has UndertakerDemonChroma1 / Chroma2, not captured or paged in this sweep; whether they use this type was not checked.
# Installed fixes that touch 6000304: a3_agent3 (_a3_mesh_only_proj, POST_BODY). This file replaces that hook for this type
# (it keeps only the ghost's mesh pixels; the bats it dropped are now kept, see above).

S195_SUPER = 6000304
S195_BAT_CLIPS = ('mortis_008_ulti_trail_demon', 'mortis_008_ulti_trail_demon_flip')
S195_MESH_PROG = 129        # the client's program of the Mortis 008 ghost mesh (also used by a3 for the same mesh)
S195_BAT_NEAR = 150         # px: bats are taken within this distance of the ghost / the predicted region (the swarm drifts, e.g. tick 359-361)


def _s195_super_body(body, ctx):
    R, ids, cv, region = ctx['R'], ctx['ids_h'], ctx['cv'], ctx['region']
    uvneg = R['uv'][..., 0] < 0
    # the ghost: opaque pixels of its mesh draws (program 129, no atlas uv), drawn in the super's region
    mesh = np.zeros(body.shape, bool)
    for d in np.unique(ids[region & (ids > 0)]):
        r = R['draws'].get(int(d), {})
        if r.get('prog') != S195_MESH_PROG: continue
        m = ids == d
        if uvneg[m].mean() < 0.5: continue
        mesh |= m
    mesh &= region & (cv >= CUT)
    # the demon bats: the front-most draw's pixels that show a sprite of the two bat clips (shape id + 1 per pixel, so
    # 1 = shape 0, the halo disc, excluded), opaque. The swarm drifts: bats up to S195_BAT_NEAR px from the ghost / region
    # are taken (the prediction's region misses the far bats at ticks 351-361). Only the bat clips' shapes are taken.
    SP = _a4_shape_pixels(R, 'sc/effects_brawler.sc', S195_BAT_CLIPS)
    from scipy.ndimage import distance_transform_edt as _s195_edt
    ref = region | mesh
    near = _s195_edt(~ref) <= S195_BAT_NEAR if ref.any() else np.zeros(body.shape, bool)
    bats = (SP > 1) & near & (cv >= CUT)
    out = mesh | bats
    return out


POST_BODY[S195_SUPER] = _s195_super_body
ONE_LABEL[S195_SUPER] = 'hull'


# ended rule (as skin 026's z_1100_s026.py): once the super's logic has finished (object finished = 1, the client still draws
# it), its label is removed from the objects passed to labels(); a group with a member still running keeps its label.
_S195_FINISHED = {S195_SUPER}
_prev_labels_s195 = labels


def _s195_finished(o):
    return o['t'] in _S195_FINISHED and o.get('finished') and not o.get('ended')


def labels(objs, cam, R, state=None, ended=(), img=None):
    alive = {o.get('label_id', o['id']) for o in objs if not _s195_finished(o)}
    objs = [o for o in objs if not (_s195_finished(o) and o.get('label_id', o['id']) not in alive)]
    return _prev_labels_s195(objs, cam, R, state=state, ended=ended, img=img)
