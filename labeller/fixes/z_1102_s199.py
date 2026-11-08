# hosted review 2026-11-01 fix (Haiku fix, reviewed by the coordinator, installed 11-01 LOCAL ONLY; source out/sweep2/haikutasks/site1101/s199/fix2.py; runs after z_1101_*)
# Skin 199 (ChronomancerKaiju), hosted review 2026-11-01. Reviewer findings (the user, binding):
#  - attack 0, ticks 265 / 267 and attack 90, ticks 305 / 307: "missing label" on the side pieces (Chronomancer002
#    ProjectileSecondary, 6001611): each side shard had only a 1-3 px sliver on its left edge, the triangle was unlabelled.
#    Cause: the shard is a translucent triangle (coverage mostly 16-128, below the CUT of 128) drawn with atlas shape 235,
#    which its sub trail particles (finx_002_sub_trail_par) also show, so the 'shared' rule kept only the pixels within
#    SHARED_NEAR of the prediction (the sliver). Fix: its body = the connected parts of its own shard shape (235 of its
#    own clip finx_002_atk_sub_projectile, coverage >= 16) that touch its predicted body (trail copies lie apart from it),
#    holes filled. The centre projectile (6001610) is not touched.
#  - super 45, ticks 343-349: the time-bend orb (Chronomancer002UltiProjectile, 6001612) was labelled with a yellow spark
#    wedge beside it (atlas shape 231 of its clip, outside the orb's disc; "extra stuff" in the label). Fix: shape 231 is
#    dropped from the orb's body. The orb disc (shapes 5, 146, 200-202) is unchanged.
# The super's zone (Chronomancer002UltiTimebend) is the z_1101_s102 disc, not changed here.
# Other skins of these brawlers: 211 (Chronomancer003, StarrGirls) uses 6001614/6001615, a different type, not touched.
# Type ids touched: 6001611 (ProjectileSecondary, all skins), 6001612 (UltiProjectile, all skins).

_S199_SEC_CLIP = ('finx_002_atk_sub_projectile', 'finx_002_atk_sub_projectile_red')
_S199_SEC_SHAPE = 236     # shape 235 + 1 (_a4_shape_pixels returns shape id + 1)
_S199_ORB_CLIP = ('finx_002_ulti_projectile', 'finx_002_ulti_projectile_red')
_S199_ORB_WEDGE = 232     # shape 231 + 1: the spark wedge beside the orb


def _s199_secondary(body, ctx):
    from scipy.ndimage import label as _cc, binary_fill_holes as _fill
    SP = _a4_shape_pixels(ctx['R'], 'sc/effects_brawler_finx.sc', _S199_SEC_CLIP)
    tri = (SP == _S199_SEC_SHAPE) & (ctx['cv'] >= 16) & ctx['region']
    if not tri.any(): return body
    L, n = _cc(tri, structure=np.ones((3, 3), bool))
    touch = _dilate(ctx['pred_any'], 3) | body
    keep = np.unique(L[touch & tri]); keep = keep[keep > 0]
    if not keep.size: return body
    out = np.isin(L, keep)
    return _fill(out) & ctx['region'] | out


POST_BODY[6001611] = _s199_secondary


def _s199_orb(body, ctx):
    SP = _a4_shape_pixels(ctx['R'], 'sc/effects_brawler_finx.sc', _S199_ORB_CLIP)
    return body & (SP != _S199_ORB_WEDGE)


POST_BODY[6001612] = _s199_orb

# an installed wrapper of labels() re-adds a few wedge pixels after the body is built (tick 355: 17 px at the wedge's
# root touching the disc), so the wedge is also cleared from the final label ids of the orb (outermost wrapper)
_s199_prev_labels = labels


def labels(objs, cam, R, state=None, ended=(), img=None):
    import bulletlabels as BL
    P = _s199_prev_labels(objs, cam, R, state=state, ended=ended, img=img)
    orb = {BL._lid(o) for o in objs if o['t'] == 6001612}
    if not orb: return P
    idx = [k for k, l in enumerate(P['lids']) if l[0] in orb and l[2] == 'body']
    if not idx: return P
    A = _a4_shape_pixels(R, 'sc/effects_brawler_finx.sc', _S199_ORB_CLIP)
    m = np.isin(P['ids'], idx) & (A == _S199_ORB_WEDGE)
    if m.any():
        P['ids'] = P['ids'].copy(); P['ids'][m] = -1
        P['share'] = P['share'].copy(); P['share'][m] = 0
    return P


# ---- time-bend zones (fix2, 2026-11-01, the coordinator's decision): bushes never cut a zone. Only the solid map tiles'
# own drawn pixels cut the four Chronomancer time-bend rings. z_1023_zone_solid's solid test cuts a pixel whose surface
# point lies in a solid tile (or in its side-face margin); a bush pixel (raised top) can fall into such a tile, so bush
# draws (draws with at least half of their pixels on Forest tiles; measured on sw2_199 t389: bush draw 0.97 Forest, wall /
# crate draws 0.00) are never cut here. Only those pixels are restored; ring and other draws keep the solid cut. The 7x7
# opening of z_1101_s102 (spurs) is kept. Other zones are unchanged.
from scipy.ndimage import binary_opening as _s199_open
_S199_TB = {'ChronomancerUltiTimebend', 'ChronomancerOverchargedUltiTimebend', 'Chronomancer002UltiTimebend',
            'Chronomancer003UltiTimebend'}
_S199_S7 = np.ones((7, 7), bool)
_S199_BLOCK_FRAC = 0.5
_S199_BLOCK_MIN = 40
_s199_prev_zone_labels2 = zone_labels


def _s199_bush_draws(cam, R):
    """draws (ids) whose drawn pixels mostly stand on Forest tiles (bushes): the only draws the time-bend solid cut must
    not take. Measured on sw2_199 t389: bush draw 43 has 0.97 of its pixels on Forest, walls / crates 0.00 on Forest"""
    X, Y, Z = _zs_world(cam, R); g = _zs_grid(R)
    ids = R['id'][::2, ::2]
    tx = np.floor(X / ZS_TILE).astype(int); ty = np.floor(Y / ZS_TILE).astype(int)
    ok = (tx >= 0) & (tx < 60) & (ty >= 0) & (ty < 60)
    ex = np.full(ids.shape, -1); ex[ok] = g[ty[ok], tx[ok]]
    sel = (ids > 0) & (Z > ZS_Z[0]) & (Z < ZS_Z[1])
    fo = ex == 7
    out = set()
    for d in np.unique(ids[sel]):
        m = sel & (ids == d)
        if m.sum() >= _S199_BLOCK_MIN and fo[m].mean() >= _S199_BLOCK_FRAC: out.add(int(d))
    return out


def zone_labels(state, cam, R, taken, own_team=None):
    import owners
    out = _s199_prev_zone_labels2(state, cam, R, taken, own_team)
    if R.get('wd') is None: return out
    H, W = R['wd'].shape
    items = []
    for it in state.get('items', []):
        iid, did = int(it[0]), int(it[1]); Z = owners.zone(did)
        if not Z or owners.area(did)['kind'] != 'zone' or Z.get('Name') not in _S199_TB or iid not in out: continue
        rz = float(Z['logic_row'].get('Radius') or 300) * float(globals().get('A3_DISC_SCALE', {}).get(Z.get('Name'), 1.0))
        x, y = int(it[3]), int(it[4])
        cx, cy = _proj(cam, x, y); dx = abs(_proj(cam, x + rz, y)[0] - cx); dy = abs(_proj(cam, x, y + rz)[1] - cy)
        if dx < 1 or dy < 1: continue
        items.append((iid, did, int(it[2]), cx, cy, dx, dy))
    if not items: return out
    solid = _zs_solid(cam, R)
    bushes = _s199_bush_draws(cam, R)
    ids = R['id'][::2, ::2]
    bush_solid = solid & np.isin(ids, list(bushes))   # solid by the tile test, but drawn by a bush draw
    if not bush_solid.any(): return out
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    D = np.stack([np.where(((xx - cx) / dx) ** 2 + ((yy - cy) / dy) ** 2 <= 1,
                           ((xx - cx) / dx) ** 2 + ((yy - cy) / dy) ** 2, np.inf)
                  for (_, _, _, cx, cy, dx, dy) in items])
    best = np.argmin(D, axis=0); anyin = np.isfinite(D.min(axis=0))
    for k, (iid, did, own, cx, cy, dx, dy) in enumerate(items):
        owned = anyin & (best == k)
        add = owned & bush_solid & ~taken
        if not add.any(): continue
        m = out[iid][0] | add
        p = 4
        c = _s199_open(np.pad(m, p), structure=_S199_S7)[p:-p, p:-p] & m
        if c.any(): out[iid] = (c, did, own)
    return out
