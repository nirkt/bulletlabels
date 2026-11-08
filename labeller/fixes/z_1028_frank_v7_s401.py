# Frank v7, skin 401 HammerDudeSandsOfTimes (attack type 6000487): per-tau keyframe label (task t42 round 7, 2026-10-28).
#
# What the skin draws (studied frame by frame at 17 aims, see study/): light-blue flames. Per update one row of four flames (the four
# lanes: aim +-8.1 deg inner, +-23.2 deg outer, rays from the attack origin O) appears at the projectiles' positions: at the first
# update only an ember, one update later a light-cyan flame, from age 2 a dark-blue swoosh that is an after-effect (not the attack,
# the old pixel labels already leave it out: A = 1). The flames are much SMALLER on the first drawn frames (tau 2: 4 small flames,
# visual about 2000 px) than on tau 4 (4 large flames, about 6500 px), so the drawn attack changes size and shape quickly with tau, and
# the old pixel label is, in the ground plane, a consistent shape between the age-1 row and the front of the wave at every aim (the
# aim dependence that v6 modelled with harmonics was mostly the size growth: no harmonic helps held-out here).
# v6 used ONE parameter set with a start-up ramp (r0 0.1 -> 1 over tau 1..3) and aim harmonics: its tau 2 label was 1.9x the drawn
# rows (IoU 0.55, 3740 px for 1950 px of visual), its tau 4 label missed 11 % of the drawn rows (coverage 0.89).
# v7: the same ground-plane footprint family (rows tau-1 and tau on the four predicted lanes, each lane point grown by an egg,
# convex hull, raised to a visual height that goes from hn at the newest row to ho at the oldest labelled row, projected with the
# frame's camera; mirror-symmetric about the aim axis by construction; walls / bushes / solid blocks cut by the template's own
# _ft42_mask) but with TWO KEYFRAMES of its parameters, fitted separately on the tau 2 frames and on the tau 4 frames of the 17 aims
# (8 parameters each, leave-one-aim-out checked), linear in tau between them (tau 3), constant for tau >= 4, and an ember-size
# scale r1 on tau 1 (not imaged: extrapolated). Per keyframe:
#   fo  : the age-1 row (the row drawn one update ago) is drawn fo further out along its lane than the template's lagged point s(k - 0.34)
#   fo0 : the same for the age-0 row (the newest);  g0: egg scale of the age-0 row
#   mf / mb / mv : forward / backward / sideways half axes of the egg around an age-1 row point (units)
#   hn / ho : visual height (logic z) at the newest / the oldest labelled row, linear in between
# Predictable: only O, the aim, tau, the skin and the static map (no pixels). Nothing before tau 1 (the first drawn row) and nothing once
# the wave's logic has ended (alive: a projectile still travels), as the template. The egg margins mf / mb / mv of both keyframes carry
# +6 units over the fitted optimum (coverage of the drawn rows 0.942 -> 0.953 on the 17 aims, IoU 0.700 -> 0.692).

FT7_401 = dict(
    first=1,                 # first drawn row (no label at tau 0); the flames are embers at tau 1
    P2={'fo': 58.5, 'fo0': -11.3, 'mf': 75.4, 'mb': 61.9, 'mv': 141.0, 'g0': 1.0, 'hn': 85.8, 'ho': 0.0},   # keyframe tau 2 (fitted on the tau 2 frames of 17 aims)
    P4={'fo': -37.1, 'fo0': 86.9, 'mf': 177.2, 'mb': 85.4, 'mv': 166.5, 'g0': 0.8, 'hn': 523.6, 'ho': 185.5},   # keyframe tau >= 4 (fitted on the tau 4 frames)
    r1=0.3)                  # tau 1 (not imaged, extrapolated): egg and heights scale of the embers on the first row
FT7_HARM = ()                # (param names with first-order aim harmonics, see the keyframes' '_c' / '_s' entries)


def _ft7_resolve(p, th):
    import numpy as np
    q = {k: v for k, v in p.items() if not (k.endswith('_c') or k.endswith('_s'))}; a = np.radians(th)
    for k in list(q):
        if k + '_c' in p or k + '_s' in p: q[k] = q[k] + p.get(k + '_c', 0.0) * np.cos(a) + p.get(k + '_s', 0.0) * np.sin(a)
    lim = dict(fo=(-300, 400), fo0=(-300, 400), mf=(0, 500), mb=(0, 500), mv=(0, 500), g0=(0.0, 2.0), hn=(0, 1200), ho=(0, 1200))
    for k, (lo, hi) in lim.items():
        if k in q: q[k] = min(hi, max(lo, q[k]))
    return q


def _ft7_params(tau, th):
    """keyframe parameters at update tau (linear between tau 2 and tau 4, constant outside; tau 1: the tau 2 keyframe, scaled)"""
    P = FT7_401; a = _ft7_resolve(P['P2'], th); b = _ft7_resolve(P['P4'], th)
    w = min(1.0, max(0.0, (tau - 2) / 2.0)); p = {k: a[k] + (b[k] - a[k]) * w for k in a}
    return p


def _ft7_footprint(kin, inc, tau, p):
    """convex footprint (n, 2) attack-local (u, v) and u of the oldest / newest labelled row"""
    import numpy as np
    A = 1; k0 = min(tau, max(FT7_401['first'], tau - A)); pts = []; urow = {}
    ang = np.linspace(0, 2 * np.pi, 24, endpoint=False); ca, sa = np.cos(ang), np.sin(ang)
    rs = FT7_401['r1'] if tau < 2 else 1.0
    for k in range(k0, tau + 1):
        age = tau - k; sc = (1.0 if age >= 1 else p['g0']) * rs; fo = p['fo'] if age >= 1 else p['fo0']; us = []
        for c in (0, 1):
            ph = np.radians(kin['phi'][c]); s = max(0.0, _ft42_sdraw(kin, inc[c], k) + fo)
            egg = np.c_[ca * np.where(ca > 0, p['mf'], p['mb']), sa * p['mv']] * sc
            for sg in (1, -1): pts.append(np.array([s * np.cos(ph), sg * s * np.sin(ph)]) + egg)
            us.append(s * np.cos(ph))
        urow[k] = float(np.mean(us))
    return _ft42_hull(np.concatenate(pts)), urow[k0], urow[tau], rs


def _ft7_401(mem=None, cam=None, state=None, kin=None, O=None, th=None, tau=None, inc=None, alive=None, grid=None, solid=None, shape=(540, 1200)):
    if not alive or tau < FT7_401['first']: return None
    p = _ft7_params(tau, th); F, u0, u1, rs = _ft7_footprint(kin, inc, tau, p)
    return _ft42_mask(cam, O, th, F, (u0, u1, p['ho'] * rs, p['hn'] * rs), grid=grid, solid=solid, shape=shape)


FT42_CUSTOM[6000487] = _ft7_401
