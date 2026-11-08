# sweep 10-27 fix for skin 221 (Haiku fix, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s221/)
# sweep 10-27 fix for skin 221 (MechanicSummer; attack type 6000141 Jessie002Projectile1, logic MechanicProjectile1,
# clip jessie_002_atk_projectile = the spiky blue ball, trail jessie_002_atk_trail = the blue flame behind it).
# Confirmed findings (task out/sweep2/haikutasks/sweep1027/s221/task.md): pair 3948 (attack 90 deg, tick 309) and pair 3950
# (attack 90 deg, tick 317): small detached red loops on the blue flame above the ball, i.e. specks (rule 1).
# Cause: the flame is the trail (not labelled, rule 3), but its trail sprites share atlas shapes with the ball's body, and the
# 'own trail drawn over its body' rule takes the flame pixels that fall inside the aligned predicted disc. Those pixels form
# detached pieces above the ball: flame-only pieces (no clip-body pixel, ticks 269, 309, 311, 317) and one 34 px clip-body
# fragment floating on the flame (tick 315). The same loops show at ticks 269 and 271 of the attack 0 deg row.
# Fix: the label of this projectile is the ball's own piece only. The largest body part of the frame is the ball (kept as it
# is, so its outline and its flame-touching bits are unchanged); every other body part is dropped when it has no clip-body
# pixel (flame) or is smaller than MECH_DETACHED_MAX px (a fragment). Nothing is changed when the largest part is small (a
# ball almost out of view). Only this type is touched; the rest of the label pipeline (cleanup, ONE_LABEL) is unchanged.
import numpy as _np

MECH_BALL = 6000141
MECH_MAIN_MIN = 200       # px (frame resolution): the ball's part must be at least this big, else nothing changes
MECH_DETACHED_MAX = 60    # px: a detached body fragment this small is not the ball


def _mech_ball(body, ctx):
    from scipy.ndimage import label as cc
    if not body.any(): return body
    L, n = cc(body)
    if n <= 1: return body
    sz = _np.bincount(L.ravel()); sz[0] = 0
    main = int(sz.argmax())
    if sz[main] < MECH_MAIN_MIN: return body
    C = ctx['C']
    out = body.copy()
    for j in range(1, n + 1):
        if j == main: continue
        m = L == j
        flame_only = not (C[m] == 1).any()          # no clip-body pixel: flame / trail pixels only
        if flame_only or sz[j] < MECH_DETACHED_MAX: out[m] = False
    return out


POST_BODY[MECH_BALL] = _mech_ball
