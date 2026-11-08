# AI-review batch 2026-11-01 fix (Haiku fix, reviewed by the coordinator, installed 11-01 LOCAL ONLY; source out/sweep2/haikutasks/siteai/bo_ruffs_shaman/fix.py)
# AI-review fixes, task bo_ruffs_shaman (skins 257, 295, 264, 404, 276; 393 needs_user), 2026-11-01.
# Override file, exec'd in idlabels' namespace after all installed fixes (IDLABELS_OVERRIDE). Each block says what it fixes.
# Owner rules used: one convex hull per attack object for thin / splintered streaks (one object = one outline, separate
# projectiles stay separate); holes / ragged / cut-off labels: one solid piece in the object's own outline; hidden parts filled in;
# characters (brawlers) never labelled.
from scipy.ndimage import binary_fill_holes as _bo_fill, binary_opening as _bo_open, label as _bo_cc


def _bo_brawler_draws(R, state, cam):
    """pixels of the brawlers (heroes) on screen: their 3D model draws, each model draw assigned to the nearest character as in
    spawn_labels; a draw nearest to a hero is that brawler's"""
    import owners
    ids_h = R['id'][::2, ::2]; out = np.zeros(ids_h.shape, bool)
    chars = []
    for c in state['chars']:
        try: kind = owners.character(int(c[1]))['kind']
        except Exception: continue
        a = np.array(_proj(cam, int(c[3]), int(c[4]), int(c[5])), float)
        b = np.array(_proj(cam, int(c[3]), int(c[4]), int(c[5]) + CHAR_TOP), float)
        chars.append((kind, a, b))
    if not any(k == 'hero' for k, _, _ in chars): return out
    for d, dr in R['draws'].items():
        if dr.get('th') != MODEL_TEX: continue
        m = ids_h == d
        if not m.any(): continue
        ys, xs = np.nonzero(m); p = np.array([xs.mean(), ys.mean()]); best = (CHAR_NEAR, None)
        for kind, a, b in chars:
            ab = b - a; t = np.clip(np.dot(p - a, ab) / max(1e-6, np.dot(ab, ab)), 0, 1)
            dist = np.linalg.norm(a + t * ab - p)
            if dist < best[0]: best = (dist, kind)
        if best[1] == 'hero': out |= m
    return out


def _bo_hull_clear(body, ctx):
    """the convex hull of the object's body, minus the brawlers' own model pixels: a hull over a splintered attack can reach a
    brawler standing beside it (Shaman's spray under the brawler's ring, tick 305). The object's own body pixels always stay"""
    if not body.any(): return body
    h = hull_mask(body)
    st, cam = ctx.get('state'), ctx.get('cam')
    if st is None or cam is None: return h
    br = _bo_brawler_draws(ctx['R'], st, cam)
    return (h & ~br) | body


def _bo_hull_post(t):
    """a hull over the object's body (as ONE_LABEL 'hull'), chained after the type's earlier POST_BODY hook, brawler bands out"""
    prev = POST_BODY.get(t)
    def fn(body, ctx):
        if prev is not None: body = prev(body, ctx)
        return _bo_hull_clear(body, ctx)
    POST_BODY[t] = fn


# (1) 257 BowDudeDemonic, attack 0 deg and attack 90 deg, type 6000363 (Bo004Projectile: one arrow of ice shards per object,
#     three arrows in a row at attack 0: 2000009 at tick 265, 2000010 at 267, 2000011 at 271). Reviewer tick 271: "outlined as
#     several dashed pieces instead of one hull". Each object's label is one body of ice-shard pieces joined by thin lines.
#     Owner rule: thin / splintered attack of one object = one convex hull per shot; the arrows stay separate outlines.
_bo_hull_post(6000363)


# (2) 264 RuffsRonin, attack 0 and attack 90 deg, type 6000890 (Ruffs002Projectile: a blue flame with a white-hot bulb, two flames
#     per shot). Reviewer ticks 305 and 311: "label covers only the white-hot lower bulb, the blue flame body above is outside".
#     The flame is clip shape 85 (ruffs_002_atk_projectile body layer, sprite shape+1 = 86). Its blue upper part is drawn with the
#     atlas sprites its trail shares (code 5), so the body rule left 231 px of shape 85 at coverage >= 128 outside at tick 311.
#     Fix: shape-85 pixels touching the flame's body are body (one flame = one solid piece, filled); glow shapes stay out.
#     Attack 0 (265, 267) is the same type and gets the same blue part (small change, checked on the compare image).
RUFFS002_CLIPS = ('ruffs_002_atk_projectile', 'ruffs_002_atk_projectile_red')
RUFFS002_BODY_SHAPE = 85


def _bo_ruffs002_flame(body, ctx):
    if not body.any(): return body
    SP = _a4_shape_pixels(ctx['R'], 'sc/effects_brawler.sc', RUFFS002_CLIPS)   # shape id + 1, 0: none of its shapes
    add = (SP == RUFFS002_BODY_SHAPE + 1) & (ctx['cv'] >= CUT) & ctx['region']
    m = body | add
    L, n = _bo_cc(m, structure=np.ones((3, 3), bool))
    keep = np.unique(L[body]); keep = keep[keep > 0]
    m = np.isin(L, keep) & m                       # only the parts joined to this flame's own body
    return _bo_fill(m) & ctx['region'] | body


POST_BODY[6000890] = _bo_ruffs002_flame


# (3) 276 ShamanSummer, type 6000228 (Shaman004Projectile: one ice spray per object; attack 0 at tick 267 (2000002), attack 90 at
#     tick 311 (2000003)). Reviewer: "outlined as many thin fragments with interior lines and holes instead of one hull"; "the attack
#     90 column at t311 is ragged with holes". Owner rule: splintered attack of one object = one convex hull per shot (brawler bands
#     out, see _bo_hull_clear). The fill of enclosed holes from a4_agent4 still runs first (chained).
_bo_hull_post(6000228)


# (4) 295 BowDudeSanta, super 45 deg, type 6000366 (Bo005UltiProjectile: one blue mace per shot: head, shaft and tip, drawn from
#     several sprites). Reviewer tick 345: "ragged, notched outline with an interior bite and a spur". A white-cyan flash lies over
#     part of the head (hidden part filled in). Fix: notches closed (2 px), holes filled inside the object's region, 1-2 px spurs
#     opened away (3x3 opening). The shaft (3 px and wider at half resolution) survives the opening.
def _bo_super_solid(body, ctx):
    if not body.any(): return body
    m = _close(body, 2) | body                     # notches and bites closed
    m = _bo_fill(m) & ctx['region'] | body         # holes filled (inside the object's region only)
    m = _bo_open(m, np.ones((3, 3), bool))         # 1-2 px spurs out; the shaft stays
    return _bo_fill(m) & ctx['region']


POST_BODY[6000366] = _bo_super_solid


# (5) 404 RuffsSandsOfTimes, attack 0 deg (and attack 90 deg, same type), type 6000898 (Ruffs006Projectile: two blade projectiles per
#     shot, side by side: 2000006 upper and 2000007 lower). Reviewer ticks 265 and 267: "the blade fan is drawn as several outlines
#     with interior lines ... one hull per shot is needed". Each projectile's body is a fan of blade pieces. Owner rule: thin /
#     splintered attack of one object = one convex hull; the two projectiles stay separate outlines. The glow removal of z_1027_s404
#     runs first (chained).
_bo_hull_post(6000898)
