# hosted review 2026-10-29 fix (Haiku fix, reviewed by the coordinator, installed 10-29 LOCAL ONLY; source out/sweep2/haikutasks/site1029/g_brock/fix2.py; tested via IDLABELS_OVERRIDE after all fixes incl. z_1100_*, hence z_1101_)
# g_brock fix2 (coordinator review of fix.py: rocket part unchanged, zone part replaced; hosted review 2026-10-29, task out/sweep2/haikutasks/site1029/g_brock/, skins 152 / 154 / 159 / 172, RocketGirl).
#
# 1. SUPER ROCKETS (user finding, super 45 rows flagged 'missing label' at ticks 347 / 363 / 377 / 393 on all four skins; the
#    user's words in task.md: the rocket body, the grey / blue / white missile behind the bright head flare, is unlabelled).
#    Cause: the missile body is a 3D mesh (sc3d glb of the attack model) drawn as ONE instanced draw for all rockets in flight
#    (uv-less, opaque, one texture hash per skin). idlabels assigns a draw to one owning object only, so the other rockets kept
#    just their star-head flare (additive sprite, codes 1 / 2 of the clip). Measured on the captures: the mesh draw holds the
#    bodies of 2 to 5 rockets at once (152 t363: 2553 px, the owner got 600 px, the others 0).
#    Fix (keyed to the four super types only): the rocket's own mesh pixels are the pixels of the skin's mesh texture hash
#    (blend 0, dmask 1; the rocket's own draws, not the ground / smoke draws) that are in the object's predicted region and whose
#    connected part touches the object's predicted body. Added to the existing body (the flare and whatever the earlier fixes
#    keep), then closed by 2 px and holes filled, so the label is one solid outline of the whole missile. Smoke, puffs and
#    other draws are not mesh draws and stay out (rule 3); parts that touch no predicted body of this rocket stay out.
#    Types: 6000060 Brock004UltiProjectile (152 RocketGirlBeach), 6000056 Brock002UltiProjectile (154 RocketGirlCny),
#    6000062 Brock005UltiProjectile (159 RocketGirlRanger), 6000064 Brock006UltiProjectile (172 RocketGirlDelivery; its earlier
#    star-hull fix z_1027_s172 runs first and is kept).
#    Other skins use other types, so they are untouched. Keyed by texture hash: the hash of the mesh draw of each type on its
#    capture (152 8371787794963ea9, 154 25da29a95031ec59, 159 9122958aa6b4a41c, 172 c5a635924fc6b3f3).
#
# 2. BURN WATER ZONE, skin 152 (user finding on page 152_01: 'attacks are good, for the zone, just label the circle from the
#    start, ignoring the other effects (at the start) since everything just gets messy'). The zone BurnMainAttackWater (zone data
#    id 17000241, the only zone of that name) was labelled from its sprites (splash, drops, explosion at the start). Now it is the
#    FULL LOGIC CIRCLE from the zone's first tick, cut only by solid map blocks (the circle rule, ZONE_DISC of z_1023_zone_solid).
#    z_1027_s152's closing / hole fill is superseded for this zone: it would fill the holes that solid crates make inside the
#    circle, so its zone id is cleared (S152_DID = -1). Skins 154 / 159 / 172 use BurnMainAttack (17000178), already a disc (a3).

G29_MESH = {6000060: ('8371787794963ea9',),        # 152 RocketGirlBeach: Brock004UltiProjectile
            6000056: ('25da29a95031ec59',),        # 154 RocketGirlCny: Brock002UltiProjectile
            6000062: ('9122958aa6b4a41c',),        # 159 RocketGirlRanger: Brock005UltiProjectile
            6000064: ('c5a635924fc6b3f3',)}        # 172 RocketGirlDelivery: Brock006UltiProjectile
G29_CLOSE = 2          # px: gaps between the parts of one missile closed


def _g29_rocket(body, ctx):
    from scipy.ndimage import label as _g29_cc, binary_fill_holes as _g29_fill
    R = ctx['R']; ids = ctx['ids_h']; cv = ctx['cv']; pb = ctx['pred_body']; region = ctx['region']
    ths = G29_MESH[ctx['t']]
    md = [d for d, dr in R['draws'].items() if dr.get('th') in ths and dr.get('blend') == 0 and dr.get('dmask') == 1]
    if not md: return body
    m = np.isin(ids, md) & (cv >= CUT) & region
    if not m.any(): return body
    L, n = _g29_cc(m)
    touch = np.unique(L[_dilate(pb, 2) & m]); touch = touch[touch > 0]
    if not touch.size: return body
    out = body | np.isin(L, touch)
    out = _g29_fill(_close(out, G29_CLOSE)) & (out | region)
    return out | body


for _t in G29_MESH:
    _g29_prev = POST_BODY.get(_t)
    if _g29_prev is None:
        POST_BODY[_t] = _g29_rocket
    else:
        POST_BODY[_t] = (lambda p: (lambda b, c: _g29_rocket(p(b, c), c)))(_g29_prev)


# zone (skin 152, did 17000241 BurnMainAttackWater): the plain logic circle (projected ellipse, radius 1.0 of the zone's logic
# radius, as the other ZONE_DISC circles) from the zone's first tick, ONE solid piece with no interior holes or lines.
# Coordinator review (2026-10-29): the solid-block cut and the z_1028_zone184 ray cut made the label loops and lines at 281 / 297
# ('everything just gets messy'). Here no solid block cuts the circle (the puddle is drawn over the wall tops there). The only
# holes left are pixels of other labels (a projectile flying over the puddle keeps its own label; the zone takes the rest).
# z_1027_s152 (closing / fill of the sprite blob) is superseded for this zone: S152_DID = -1.
G29_WATER_DID = 17000241
ZONE_DISC.add('BurnMainAttackWater')   # base chain on the circle branch; this wrapper (outermost) sets the label
S152_DID = -1
_g29_zone_prev = zone_labels


def zone_labels(state, cam, R, taken, own_team=None):
    out = _g29_zone_prev(state, cam, R, taken, own_team)
    items = [it for it in state.get('items', []) if int(it[1]) == G29_WATER_DID]
    if not items or R.get('wd') is None: return out
    import owners
    H, W = R['id'][::2, ::2].shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    for it in items:
        iid, did = int(it[0]), int(it[1])
        Z = owners.zone(did)
        if not Z or owners.area(did)['kind'] != 'zone': continue
        rz = float(Z['logic_row'].get('Radius') or 300)
        x, y = int(it[3]), int(it[4])
        cx, cy = _proj(cam, x, y); dx = abs(_proj(cam, x + rz, y)[0] - cx); dy = abs(_proj(cam, x, y + rz)[1] - cy)
        if dx < 1 or dy < 1: continue
        part = (((xx - cx) / dx) ** 2 + ((yy - cy) / dy) ** 2 <= 1) & ~taken
        if part.any(): out[iid] = (part, did, int(it[2]))
        else: out.pop(iid, None)
    return out
