# installed 11-01 from out/sweep2/haikutasks/siteai2/arcade_samurai/fix.py (2nd AI sweep: 268 note stream one hull, 448 glyph chain hull + zigzag joint, 412 zone birth disc plain circle)
# AI-review sweep 2 (2026-11-01), task folder arcade_samurai (skins 245, 268, 412, 448, 583, 630). Exec'd in idlabels' namespace
# (IDLABELS_OVERRIDE) after all installed fixes (py/fixes/z_1103_*, z_1104_*). Changes keyed to explicit type ids / zone did.
#
# 1. 268 ArcadePiano, Arcade003Projectile (6000562): the note streams (one object per stream: beams + notes) were outlined as
#    several ragged pieces with comb teeth (AI flags: attack 90 t315, t317, 'want one hull per stream'; RULES 5: beams split into
#    strips = ONE convex hull). Fix: small stray parts of the object's body are dropped first (parts < 10% of the largest closed
#    part: the cyan swoosh's bits at t267-t275 would otherwise stretch the hull), then ONE_LABEL hull per stream object.
#    Verified: only type 6000562 changes (ticks 265-275, 305-323); the skin's other rows are untouched.
#
# 2. 448 ArcadeSandsOfTime, Arcade007Projectile (6000570), both attack rows: the glyph / zigzag objects were labelled as many
#    small outlines (AI flag: attack 90 t307, t309) and the zigzag's lower joint was missing (attack 0 t267, open from the last
#    review). Fix: the same stray-part drop as 1, then the convex hull of the object's body per object (one label per chain or
#    zigzag, separate objects stay separate). The hull fill excludes the purple ground squares: they share shape id 1 with the
#    glyph, so shape-1 pixels that are not body are cut out of the hull (the squares are never labelled, RULES 4).
#    Computed inside POST_BODY (the hull is produced here, so no ONE_LABEL entry for this type).
#
# 3. 412 SamuraiJapanOvercharge, Samurai003 super zone (17000777, Samurai003CrossSlashArea): at its birth (age <= 4 ticks,
#    SAM_BIRTH_DISC) the label is the plain logic disc, but it was cut by the dissolving creature's remains (effect draw, blend
#    1: not a wall or crate) and by ground speckles inside the splash (AI flag: 412 super45 t363, 'holes / ragged', 'follows the
#    splash instead of the plain circle'). Fix: in that birth window the disc is cut only by opaque draws (walls, crates,
#    barrels: blend 0), and enclosed holes of the disc are filled. Zones of 412 after their birth window (the blade hull) are
#    untouched; the birth-window ticks of this zone (t361, t363, t365) become the plain logic circle.
#
# Not changed (see report.md): 245 a0 t267 (two separate projectiles, one hull each, as decided), 583 / 630 (zone colour is the
# game's zone class; under-name-plate parts = needs_user), 412 creature at t345-349 (needs_user).

# ---- 1. 268: Arcade003 note streams, one hull per stream object ---------------------------------------------------------
from scipy.ndimage import label as _cc268, binary_closing as _bc268


def _stream268_body(body, ctx):
    """the object's body without its small stray parts (parts < 10% of the largest closed part)"""
    if not body.any(): return body
    cl = _bc268(np.pad(body, 3), iterations=2)[3:-3, 3:-3] | body
    L, n = _cc268(cl)
    if n <= 1: return body
    sz = np.bincount(L[body], minlength=n + 1)
    keep = sz >= 0.1 * sz.max(); keep[0] = False
    return body & keep[L]


POST_BODY[6000562] = _stream268_body
ONE_LABEL.update({6000562: 'hull'})   # one convex hull per stream object (RULES 5: beams split into strips)


# ---- 2. 448: Arcade007 glyph / zigzag projectile, one hull per object, no ground squares ---------------------------------
from scipy.ndimage import label as _cc448, binary_closing as _bc448


def _zig448_label(body, ctx):
    if not body.any(): return body
    cl = _bc448(np.pad(body, 3), iterations=2)[3:-3, 3:-3] | body
    L, n = _cc448(cl)
    if n > 1:
        sz = np.bincount(L[body], minlength=n + 1)
        keep = sz >= 0.1 * sz.max(); keep[0] = False
        body = body & keep[L]
    if not body.any(): return body
    T = projtab.table()[ctx['t']]
    clips = tuple(c for c in (T.get('blue'), T.get('red')) if c)
    SP = _a4_shape_pixels(ctx['R'], T['file'], clips)
    decal = (SP == 2)                         # shape id 1: the purple ground squares share it; kept only where body
    return hull_mask(body) & ~(decal & ~body)


POST_BODY[6000570] = _zig448_label


# ---- 3. 412: Samurai 003 super zone, birth disc cut only by opaque draws, holes filled ----------------------------------
_z412_prev_zone_labels = zone_labels


def zone_labels(state, cam, R, taken, own_team=None):
    out = _z412_prev_zone_labels(state, cam, R, taken, own_team)
    if not state or R.get('wd') is None: return out
    import owners
    from scipy.ndimage import binary_fill_holes as _bfh
    tick = int(state.get('tick', 0))
    ids_h = R['id'][::2, ::2]; H, W = ids_h.shape
    yy, xx = np.mgrid[0:H, 0:W]
    opaque = np.isin(ids_h, [d for d, dr in R['draws'].items() if not dr.get('blend')])
    for it in state.get('items', []):
        did = int(it[1])
        if did != 17000777: continue
        iid = int(it[0]); Z = owners.zone(did)
        if not Z: continue
        age = tick - _ZONE_SEEN.get(iid, tick)
        if age > SAM_BIRTH_DISC.get(did, -1): continue           # after the birth window: the blade hull (z_1103) stays
        x, y = int(it[3]), int(it[4]); rz = float(Z['logic_row'].get('Radius') or 300); own = int(it[2])
        cx, cy = _proj(cam, x, y)
        dx = abs(_proj(cam, x + rz, y)[0] - cx); dy = abs(_proj(cam, x, y + rz)[1] - cy)
        if dx < 1 or dy < 1: continue
        disc = ((xx - cx) / dx) ** 2 + ((yy - cy) / dy) ** 2 <= 1
        g = ground_depth(cam, x, y, rz, R['meta'].get('wref', 0.0), H, W)
        dm = disc & ~((R['wd'] < g - DEPTH_TOL) & opaque) & ~taken
        dm = _bfh(dm) & disc & ~taken                             # ground speckles showing through the splash: holes
        if dm.any(): out[iid] = (dm, did, own)
        else: out.pop(iid, None)
    return out
