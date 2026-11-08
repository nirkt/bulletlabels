# sweep 10-27 fix for skin 296 (Haiku fix, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s296/)
# sweep 10-27 fix for skin 296 SandstormKoya (Sandy), task s296 (Haiku fix, sweep 2026-10-27; pair 5518, attack 90 deg,
# tick 313). Defect (extra_label): the Sandy cloud's label (type 6000672, Sandy004Projectile, a zone-type attack: one solid
# label over the whole cloud, rule 7) also took in a small detached clump of green leaves about 137 px (frame) above the
# cloud at tick 313 (351 half-res px, drawn by the attack's trail clip, shape 99, with no contact to the cloud's clip body).
# Rule 3: leaves that have drifted off the cloud are debris, not labelled.
#
# Fix (POST_BODY of this type only, chained after the existing hook _sandy_end of a2_agent2, which removes the bursting
# end animation and is kept as it is): every connected part of the body that touches none of the cloud's own clip-body
# pixels (C == 1) and lies more than S6672_DETACH half-res px (32 frame px) from them is dropped. The cloud's own parts
# (clip body touching, the stars and bubbles of the cloud included) are kept. Checked on the capture: the trail-only part of
# the attack 0 row (tick 273, object 2000006, 222 px) lies 7.1 px from the clip body and stays; the detached leaves at tick
# 313 lie 68.4 px away and are dropped. If the body has no clip-body pixels at all, nothing is dropped (never lose a
# projectile). The other Sandy types (6000666 ... 6000678) are not changed by this file.
#
# Other approaches considered: position against the predicted body (pred_any): relies on the prediction, which lags a
# drifting cloud; a size threshold: the leaves clump (351 px) is as large as the cloud's own bubbles (333 px), so it fails;
# cutting by shape id 99: the same leaf shape is attached to the cloud at ticks 305-311, so it would cut real parts.

S6672_TYPE = 6000672
S6672_DETACH = 16     # half-res px (32 frame px): trail-only parts farther than this from the clip body are debris
_p6672_prev = POST_BODY.get(S6672_TYPE)


def _fix_6000672_detached(body, ctx):
    if _p6672_prev is not None:
        body = _p6672_prev(body, ctx)
    if not body.any(): return body
    from scipy.ndimage import label as _cc, distance_transform_edt as _edt
    C = ctx['C']
    clip = body & (C == 1)
    if not clip.any(): return body
    L, n = _cc(body)
    if n <= 1: return body
    dclip = _edt(~clip)
    out = body.copy()
    for k in range(1, n + 1):
        m = L == k
        if (C[m] == 1).any(): continue   # touches the cloud's own clip body: part of the cloud
        if float(dclip[m].min()) > S6672_DETACH: out[m] = False
    return out


POST_BODY[S6672_TYPE] = _fix_6000672_detached
