# installed 11-02 from out/sweep2/haikutasks/siteai5/janet_shield_twins_mender/fix.py (073 shield plates one hull per shot, 111 Mender super shards one hull before landing; 088 spawn-label drop DISABLED)
# Owner complaints (sweep 4, 2026-11-02), skins 068 Janet, 073 ShieldTank, 088 Twins, 111 Mender. One file, exec'd after all
# installed fixes (IDLABELS_OVERRIDE). Each part is keyed to explicit type ids and changes nothing else.
#
# (A) 068 Janet: NO change in this file. The installed one-hull-per-shot label stays. Tested: a hull that also takes the
#     translucent copies of the card sprite (trail-like afterimages) reached the green orb and the brawler's head at t265-t309,
#     so it was rejected. The copies are reported as needs_user.
#
# (B) 073 ShieldTank (type 6001254). Owner: "hull around the blue rectangles", one hull per shot. The shot's plates split into
#     two outlines at t307 / t309 because a crate (opaque world draw) stands between the top plate and the bottom plate. The
#     fix takes the convex hull of all labelled plate pixels of one shot (same owner, same first tick), and fills the
#     unlabelled pixels inside it (the crate between the plates is taken in, the hidden-part rule). Single label, one outline.
#
# (C) 088 Twins (types 6001403 super, 6001402 volley; spawn 1000026 = the Twins model that shoots the volley). Owner: "NO label
#     on the brawler's body" (super t371 and t389 were labelled on the body). From the first tick the Twins super exists
#     (the star, 6001403), the spawn label of the Twins model (1000026) is dropped; the volley and the star stay as they are.
#     The state is kept for the rest of the run, so the drop continues after the star has gone.
#
# (D) 111 Mender super (type 6001756, MenderUltiProjectile). Owner / AI: "the super shards are split into several outlines at
#     t349-t351, should get a hull". Before any shard has landed (the area label of z_1100_s111 is off), the labelled shards
#     of one shot are one convex hull on the lowest shard id. Once the area label is on, nothing changes.

import numpy as _kn
import bulletlabels as _KBL
from scipy.ndimage import label as _kcc, binary_dilation as _kdil

_K_JG = 6001158          # 068 Janet basic attack
_K_SH = 6001254          # 073 ShieldTank plates
_K_TW_SUPER = 6001403    # 088 Twins super star
_K_TW_PET = 1000026      # 088 Twins model (spawn) that shoots the volley
_K_SHARD = 6001756       # 111 Mender super shards

_k_twins_on = [False]    # set once the Twins super has been seen in this run (088)
_k_prev_labels = labels


def _k_merge_hull(P, R, objs, typ, grow=None):
    """one convex hull per shot (same type, same owner, same first tick): the labelled body pixels of its members, optionally
    grown by grow(U, ks); the hull takes the unlabelled pixels inside it, the pixels of other labels stay theirs"""
    ids, lids, share = P['ids'], P['lids'], P['share']
    shots = {}
    for o in objs:
        if o['t'] != typ or o.get('ended'): continue
        lid = (_KBL._lid(o), -1, 'body')
        if lid in lids: shots.setdefault((o.get('owner'), o.get('first')), []).append((o, lids.index(lid)))
    if not shots: return P
    cv = R['cv'][::2, ::2]
    out = ids.copy(); sh = share.copy()
    for key, mem in shots.items():
        ks = [k for _, k in mem]
        U = _kn.isin(ids, ks)
        if not U.any(): continue
        if grow is not None: U = grow(U, ks, ids)
        H = hull_mask(U)
        carrier = min(mem, key=lambda m: m[0]['id'])[1]
        for k in ks: out[out == k] = -1
        t = H & (out < 0)
        out[t] = carrier; sh[t] = _kn.maximum(cv[t] / 255.0, _KBL.CUT)
    P2 = dict(P); P2.update(ids=out, share=sh)
    return P2


def labels(objs, cam, R, state=None, ended=(), img=None):
    P = _k_prev_labels(objs, cam, R, state=state, ended=ended, img=img)
    # (B) ShieldTank: one hull per shot over its plates (the crate between the plates is filled in)
    if any(o['t'] == _K_SH for o in objs):
        P = _k_merge_hull(P, R, objs, _K_SH)
    # (C) Twins: no label on the Twins model's body once the super has started
    # COORDINATOR 11-02: (C) DISABLED -- the blue spawn label (1000026) is the Twins spawnable, which stays labelled; the owner's 088 complaint was only 'super t389 ragged'
    # if any(o['t'] == _K_TW_SUPER for o in objs): _k_twins_on[0] = True
    if _k_twins_on[0]:
        lids = P['lids']
        ks = [k for k, l in enumerate(lids) if l[0] == _K_TW_PET and l[2] == 'spawn']
        if ks and _kn.isin(P['ids'], ks).any():
            ids = P['ids'].copy(); sh = P['share'].copy()
            for k in ks:
                m = ids == k; ids[m] = -1; sh[m] = 0
            P = dict(P); P.update(ids=ids, share=sh)
    # (D) Mender super before any shard has landed: one hull per shot over the shards
    if any(o['t'] == _K_SHARD for o in objs) and globals().get('_MENDER_AREA') is None:
        P = _k_merge_hull(P, R, objs, _K_SHARD)
    return P
