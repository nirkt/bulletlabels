# hosted review 2026-11-01 fix (Haiku fix, reviewed by the coordinator, installed 11-01 LOCAL ONLY; source out/sweep2/haikutasks/site1101/g_brock2/fix.py; runs after z_1101_*)
# g_brock2 fix (hosted review 2026-11-01, coordinator clarification in task.md): RocketGirl (Brock) skins 175, 179, 180,
# 184, 185, 186, 187, 188, 205. Skins 152 / 154 / 159 / 172 are not touched.
#
# 1. ZONE (every skin's burn area, user notes: 175 'the zone effects have to be ignored. only the circle matters', 179
#    'just label the visible circle', 185 'the man should be ignored', 186 / 187 / 188 'like before, only the circle matters').
#    The zone label = the plain projected logic circle (radius 1.0 of the zone's logic radius, as the accepted zone fix of
#    skin 152, g_brock fix2 'G2' section of z_1101_s152brock.py), from the zone's first tick: ONE solid piece, no holes, not
#    following splashes / flames / cubes / figures drawn on it. Only pixels of other labels (taken) are left out.
#    Keyed to the zone data ids of these skins (one id per skin's burn area; the 1.15-scaled disc of a3_agent3 and the
#    sprite / speck fixes of z_1027_s175 / z_1027_s180 are replaced for these ids only).
G2_ZONE_DID = {17000179: 'BurnMainAttackElec',       # 175 RocketGirlZeus
               17000180: 'BurnMainAttackAngry',      # 179 RocketGirlBrown
               17000181: 'BurnMainAttackHacker',     # 180 RocketGirlVirus
               17000182: 'BurnMainAttackRangerRed',  # 184 RocketGirlRanger red
               17000183: 'BurnMainAttackRangerBlue', # 185 RocketGirlRanger blue
               17000184: 'BurnMainAttackRangerYellow',  # 186
               17000185: 'BurnMainAttackRangerPink', # 187
               17000186: 'BurnMainAttackRangerBlack',   # 188
               17000187: 'BurnMainAttackSteampunk'}  # 205 RocketGirlSteamPunk
G2_ZONE_RADIUS = 1.0

_g2_zone_labels_prev = zone_labels


def zone_labels(state, cam, R, taken, own_team=None):
    import owners
    out = _g2_zone_labels_prev(state, cam, R, taken, own_team)
    items = [it for it in state.get('items', []) if int(it[1]) in G2_ZONE_DID]
    if not items: return out
    H, W = R['id'][::2, ::2].shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    for it in items:
        iid, did = int(it[0]), int(it[1])
        Z = owners.zone(did)
        if not Z or owners.area(did)['kind'] != 'zone': continue
        rz = float(Z['logic_row'].get('Radius') or 300) * G2_ZONE_RADIUS
        x, y = int(it[3]), int(it[4])
        cx, cy = _proj(cam, x, y); dx = abs(_proj(cam, x + rz, y)[0] - cx); dy = abs(_proj(cam, x, y + rz)[1] - cy)
        if dx < 1 or dy < 1: continue
        part = (((xx - cx) / dx) ** 2 + ((yy - cy) / dy) ** 2 <= 1) & ~taken
        if part.any(): out[iid] = (part, did, int(it[2]))
        else: out.pop(iid, None)
    return out


# 2. SUPER ROCKETS (user findings: 184 t377 holes / ragged and t393 cut off; 185-188 t377 'too big'; 175 / 180 / 205 no reason).
#    'too big' at 377 = the blue ground ring of the super area and smoke next to the rocket are inside its label (rule 2 / 4).
#    Fix: a label pixel that is NOT one of the rocket's own clip sprites (sprite code 0) and lies beyond G2_RK_NEAR half-res px of
#    the rocket's predicted body is dropped. Clip-sprite pixels (codes 1-5) and everything near the prediction stay, so a rocket
#    whose prediction lags keeps its body (tick 393). Nothing is added. A label that would be empty is left as it was.
G2_RK_TYPES = (6000066,        # 175 Brock007UltiProjectile (super rockets, sprite bolt)
               6000070,        # 180 Brock009UltiProjectile
               6000076, 6000077, 6000078, 6000079, 6000080,   # 184-188 Brock010UltiProjectile{Red,Blue,Yellow,Pink,Black}
               6000082)        # 205 Brock011UltiProjectile
G2_RK_NEAR = 6          # half-res px around the predicted body that a rocket's non-clip pixels may reach


def _g2_rocket(body, ctx):
    if not body.any(): return body
    pb = ctx['pred_body']
    if not pb.any(): return body
    C = ctx['C']
    far = ~_dilate(pb, G2_RK_NEAR)
    out = body & ~(far & (C == 0))
    return out if out.any() else body


for _t in G2_RK_TYPES:
    _g2_prev = POST_BODY.get(_t)
    if _g2_prev is None:
        POST_BODY[_t] = _g2_rocket
    else:
        POST_BODY[_t] = (lambda p: (lambda b, c: _g2_rocket(p(b, c), c)))(_g2_prev)
