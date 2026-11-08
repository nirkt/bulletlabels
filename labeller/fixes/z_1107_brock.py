# installed 11-02 from out/sweep2/haikutasks/siteai5/brock/fix.py with coordinator edits (Ranger zone scale kept 1.15, landing-ring section removed): zones cut by walls/crates/barrels, 175 super bolt body, 018 rocket notch, Ranger attack flames smooth
# owner complaints task brock (2026-11-02): RocketGirl (Brock) skins 018 175 184 185 186 187 188 205.
#
# 1. ZONE, every Brock burn area (175 17000179, 179 / 180 17000180-81, 184-188 17000182-86, 205 17000187):
#    owner ruling (binding): the zone is the PLAIN LOGIC CIRCLE (radius 1.0 of the logic radius) from its first tick, cut ONLY
#    by solid map blocks (walls, crates, barrels). Characters, bushes and effects never cut it.
#    Owner complaints: 'Brock zone circle 1.15' (184-188: the Ranger zones were drawn at 1.15 x the radius by
#    z_1102_s175brock_b; replaced here by 1.0), and 205 'the zone still runs over the wall blocks instead of being cut by
#    them' (z_1102_s175brock_a's circle has no solid cut on any skin).
#    Mechanism: the solid tiles of the capture's own map (map.npy, the z_1023 tile codes) are projected to the screen as their
#    footprint at ground level and at BZ_WALL_H (the block's top), convex hull per tile = the block's silhouette. The zone
#    circle minus that silhouette, minus the labels of projectiles (taken), small holes filled. Map-based, so it is not
#    affected by a character or a bush standing in front of the blocks (a depth-based cut took bush tops and figures).

BROCK_ZONE_DID = {17000179,   # 175 RocketGirlZeus: BurnMainAttackElec
                  17000180,   # 179 RocketGirlBrown: BurnMainAttackAngry
                  17000181,   # 180 RocketGirlVirus: BurnMainAttackHacker
                  17000182,   # 184 RocketGirlRanger red: BurnMainAttackRangerRed
                  17000183,   # 185 RocketGirlRanger blue: BurnMainAttackRangerBlue
                  17000184,   # 186 RocketGirlRanger yellow: BurnMainAttackRangerYellow
                  17000185,   # 187 RocketGirlRanger pink: BurnMainAttackRangerPink
                  17000186,   # 188 RocketGirlRanger black: BurnMainAttackRangerBlack
                  17000187}   # 205 RocketGirlSteamPunk: BurnMainAttackSteampunk
BROCK_ZONE_RADIUS = 1.0
# COORDINATOR 11-02: the owner asked for the Ranger zones at 1.15 x the logic radius (the visible ring, z_1102_s175brock_b) -- keep it.
BZ_ZONE_SCALE = {17000182: 1.15, 17000183: 1.15, 17000184: 1.15, 17000185: 1.15, 17000186: 1.15}
BZ_WALL_H = 600.0       # logic height of a solid block's top (walls, crates, barrels: tops at about 400 to 800)

_bz_zone_labels_prev = zone_labels


def _bz_block_mask(cam, R, x, y, rz):
    """(H, W) bool: the screen silhouette of the solid map tiles that the circle (x, y, rz) reaches (per tile: convex hull of
    its footprint at z = 0 and at z = BZ_WALL_H)"""
    from scipy.spatial import ConvexHull
    from PIL import Image as _bz_im, ImageDraw as _bz_dr
    g = _zs_grid(R)
    H, W = R['id'][::2, ::2].shape
    m = _bz_im.new('L', (W, H), 0); d = _bz_dr.Draw(m)
    i0 = max(0, int(np.floor((x - rz) / ZS_TILE))); i1 = min(59, int(np.floor((x + rz) / ZS_TILE)))
    j0 = max(0, int(np.floor((y - rz) / ZS_TILE))); j1 = min(59, int(np.floor((y + rz) / ZS_TILE)))
    for j in range(j0, j1 + 1):
        for i in range(i0, i1 + 1):
            if int(g[j, i]) not in ZS_SOLID: continue
            x0, x1, y0, y1 = i * ZS_TILE, (i + 1) * ZS_TILE, j * ZS_TILE, (j + 1) * ZS_TILE
            P = np.array([_proj(cam, a, b, c) for a, b in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)) for c in (0.0, BZ_WALL_H)])
            try: hull = ConvexHull(P)
            except Exception: continue
            d.polygon([(float(P[k][0]), float(P[k][1])) for k in hull.vertices], fill=1)
    return np.array(m) > 0


def zone_labels(state, cam, R, taken, own_team=None):
    import owners
    out = _bz_zone_labels_prev(state, cam, R, taken, own_team)
    items = [it for it in state.get('items', []) if int(it[1]) in BROCK_ZONE_DID]
    if not items or R.get('wd') is None: return out
    H, W = R['id'][::2, ::2].shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    for it in items:
        iid, did = int(it[0]), int(it[1])
        Z = owners.zone(did)
        if not Z or owners.area(did)['kind'] != 'zone': continue
        rz = float(Z['logic_row'].get('Radius') or 300) * BZ_ZONE_SCALE.get(did, BROCK_ZONE_RADIUS)
        x, y = int(it[3]), int(it[4])
        cx, cy = _proj(cam, x, y); dx = abs(_proj(cam, x + rz, y)[0] - cx); dy = abs(_proj(cam, x, y + rz)[1] - cy)
        if dx < 1 or dy < 1: continue
        disc = ((xx - cx) / dx) ** 2 + ((yy - cy) / dy) ** 2 <= 1
        wall = _bz_block_mask(cam, R, x, y, rz)
        part = disc & ~wall & ~taken
        part = _zs_fill_small(part, disc, wall, taken)
        if part.any(): out[iid] = (part, did, int(it[2]))
        else: out.pop(iid, None)
    return out


# 2. SUPER BOLT, skin 175 RocketGirlZeus (type 6000066, Brock007UltiProjectile; owner complaint: 'super bolt body still NOT
#    labelled'). At t363 the big white-cyan bolt drawn from the top right down to the impact is labelled only at its top tip:
#    its body is the bolt's own trail-over sprite (code 4, one draw), and z_1102_brock_b only added the pieces within 10 px of
#    the short prediction, so the rest of the body stayed out. Fix: the bolt's own trail-over pixels (code 4, coverage >=
#    CUT, in the same draws as the labelled body) that are 4-connected to the body and lie within BZ_BOLT_REACH px of the
#    object's predicted region are added, then holes filled. Nothing else (the blue glow, the impact flash of the ground)
#    is added: those are other draws or not code 4. Other types untouched.
BZ_BOLT_TYPE = 6000066
BZ_BOLT_REACH = 60      # frame px around the predicted region that the bolt's body may reach
_bz_bolt_prev = POST_BODY.get(BZ_BOLT_TYPE)


def _bz_bolt(body, ctx):
    from scipy.ndimage import label as _bz_cc, binary_fill_holes as _bz_fill
    if _bz_bolt_prev is not None: body = _bz_bolt_prev(body, ctx)
    if not body.any(): return body
    C = ctx['C']; ids = ctx['ids_h']; cv = ctx['cv']; pa = ctx['pred_any']
    if not pa.any(): return body
    big = _dilate(pa, BZ_BOLT_REACH)
    own_d = np.unique(ids[body & (ids > 0)]); own_d = own_d[own_d > 0]
    cand = (C == 4) & (cv >= CUT) & big & np.isin(ids, own_d)
    if not cand.any(): return body
    L, n = _bz_cc(cand | body)
    keep = np.unique(L[body & (L > 0)]); keep = keep[keep > 0]
    out = body | (np.isin(L, keep) & cand)
    return _bz_fill(out) & (out | big)


POST_BODY[BZ_BOLT_TYPE] = _bz_bolt


# 3. RocketGirlDefault super rocket, skin 018 (type 6000054, RocketGirlUltiProjectile). Owner complaint: 'some rocket parts arent
#    labeled'. At t383 the top-right rocket's label has a straight notch: its cream bulb (the missile's bright end between the
#    white body and the dark ring) is covered by a smoke puff drawn over it, so the ID pass does not show it as the rocket and
#    the label is cut across the bulb (hidden-parts rule: the label keeps the full shape). Fix: opaque pixels (coverage >= CUT)
#    of the predicted body that are not yet in the label are added, then the gaps of 2 px closed and holes filled. Only
#    inside the rocket's own predicted body (pred_body): smoke beyond the prediction stays out. Keyed to 6000054 only.
BZ_ROCKET_SUPER18 = 6000054
_bz_r18_prev = POST_BODY.get(BZ_ROCKET_SUPER18)


def _bz_rocket18(body, ctx):
    from scipy.ndimage import binary_fill_holes as _bz_fill
    if _bz_r18_prev is not None: body = _bz_r18_prev(body, ctx)
    if not body.any(): return body
    pb = ctx['pred_body']; cv = ctx['cv']
    if not pb.any(): return body
    add = pb & (cv >= CUT) & ~body
    if not add.any(): return body
    out = body | add
    out = _bz_fill(_close(out, 2)) & (out | _dilate(pb, 2))
    return out | body


POST_BODY[BZ_ROCKET_SUPER18] = _bz_rocket18


# 4. RANGER ATTACK (Brock 010 attack, types 6000071-6000075 = Red / Blue / Yellow / Pink / Black clip). Owner complaint on 184,
#    185 and 186 (attack 0 deg, tick 267): the flame part of the rocket outline is ragged, with inner loops (rule: one clean
#    piece, filled solid, smooth edges). Fix: the body is closed by 3 px and its holes filled, inside the object's predicted
#    region (pred_any dilated by 6 px) so nothing outside it is added. Only the loops and notches of the own label change: the
#    nose and the outer outline stay as they are. Keyed to the five attack types.
BZ_RANGER_ATTACK = (6000071, 6000072, 6000073, 6000074, 6000075)
BZ_RANGER_CLOSE = 3
_bz_ra_prev = {t: POST_BODY.get(t) for t in BZ_RANGER_ATTACK}


def _bz_ranger_attack(body, ctx):
    from scipy.ndimage import binary_fill_holes as _bz_fill
    prev = _bz_ra_prev.get(ctx['t'])
    if prev is not None: body = prev(body, ctx)
    if not body.any(): return body
    pa = ctx['pred_any']
    if not pa.any(): return body
    reg = _dilate(pa, 6)
    out = _bz_fill(_close(body, BZ_RANGER_CLOSE)) & reg
    return out | body


for _t in BZ_RANGER_ATTACK: POST_BODY[_t] = _bz_ranger_attack


# 5. (landing-ring labels removed by the coordinator 11-02: the owner's 'rocket ring out' means the blue ground ring stays OUT of all labels)
