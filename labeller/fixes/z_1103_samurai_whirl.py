# AI-review batch 2026-11-01 fix (Haiku fix, reviewed by the coordinator, installed 11-01 LOCAL ONLY; source out/sweep2/haikutasks/siteai/samurai_whirl/fix.py)
# AI-reviewer findings on Samurai / Whirlwind skins (review 2026-11-01, task samurai_whirl: skins 412, 583, 630, 471, 478, 592).
# Written by the Haiku fix agent for samurai_whirl; test with IDLABELS_OVERRIDE=<this file>.
#
# 1. Samurai super zones (did 17000776 Samurai002CrossSlashArea, 17000777 Samurai003CrossSlashArea, 17000778
#    Samurai004CrossSlashArea; used by the Samurai 002 / 003 / 004 skins, e.g. SamuraiStarrForce 583, SamuraiJapanOvercharge 412,
#    SamuraiWindstock 630). The lingering slash ('super 45' from the zone birth, 1500 ms) had a label cut by the zone's
#    sprite pixels inside the logic circle: ragged, notched, cut off at the blade tips (review: 583 t345 is the projectile,
#    630/412 zones ragged). Same fix as the default Samurai area (z_1101_s096.py, did 17000775): ONE convex hull over the
#    bright blade pixels of the zone's sprite cluster; the hull is cut only by props drawn in front of the blade (walls,
#    crates: DEPTH_TOL), pieces smaller than 4 px dropped, other labels stay out.
#    412 zone birth (review 412 t363, "zone outline follows the lumpy creature remains with interior lines instead of the
#    plain logic circle"): the first ticks of did 17000777 show the dissolving creature (the ground impact sprite), not the
#    slash. Zones are the plain logic circle from birth (RULES 7): for ages <= SAM_BIRTH_DISC ticks since the zone was first
#    seen, the label is the logic disc cut only by what is drawn in front of the ground (as ZONE_DISC).
#    (the sprite cluster's bright body, threshold SAM_HALF = 0.75 of the bright core, not 0.5 as z_1101_s096: 0.5 added the teal halo)
# 2. (no change) Samurai 002 / 004 super projectiles 583 t345-351 and 630 t345-359: the lower-left part of the blob outside the label
#    is not reached by FADE_CUT (tested 32: identical pixel counts) or by a glow-class body rule (tested: identical); not fixed.
# 3. Whirlwind hammer (6000607 Carl003Projectile, skin 471 WhirlwindHogrider): the hammer is one object of several pieces (the
#    checkered block, the blue handle end, the orange connector); the label split into two pieces at t269/t271 (review:
#    "hammer is split into two outlines"). Multi-piece attack of one object = ONE convex hull per shot (RULES 5 / owner rules).
#    Closing (radius 6) did not join the pieces at t271, so the hull is used.
# 4. Whirlwind anchor (6000611 Carl005Projectile, skin 478 WhirlwindPirate): the grey chain links next to the anchor were inside
#    the label at t265-t271 (review: "should not be labelled"). The chain is the same chain sprite as Spike Mecha's (shapes 58408
#    links and 58409 rod, the user's decision Q9: "ball only, no chain"), so the chain shapes are dropped the same way; the anchor
#    body (its clip shapes) is kept. Only this type; no other skin's type is touched.
# 5. Whirlwind disc (6000619 Carl009Projectile, skin 592 WhirlwindSupercellMake): the label had bites and an interior cut-out on
#    every labelled frame (review: "holes / ragged"). The disc is several arcs around a centre (multi-piece of one object):
#    ONE convex hull per shot. Closing (radius 6) left the bottom notch, so the hull is used.

SAM_ZONE_DIDS = {17000776, 17000777, 17000778}   # Samurai 002 / 003 / 004 ulti slash zones (the default area is z_1101_s096's did 17000775)
SAM_BIRTH_DISC = {17000777: 4}                   # 412: while the zone is at most 4 ticks old (its creature impact sprite), the logic disc
SAM_HALF = 0.75                                  # fraction of the zone's bright core (95th percentile of luminance) at which the blade edge sits: 0.75 (0.5 took the teal halo, the tail and the bots' tops, tested 412 t367-389)
SAM_ANCHOR = 6000611                             # Whirlwind Pirate anchor (Carl005Projectile)
SAM_CHAIN_SHAPES = (58408, 58409)                # chain links, chain rod (shape ids, as Spike Mecha's chain: z_1023_spike_mecha.py)

ONE_LABEL.update({6000607: 'hull',               # Whirlwind hammer, skin 471: one hull per shot
                  6000619: 'hull'})              # Whirlwind disc, skin 592: one hull per shot

_sm_img = [None]
_sm_prev_labels = labels
_sm_prev_zone_labels = zone_labels


def labels(objs, cam, R, state=None, ended=(), img=None):
    """idlabels.labels with the frame kept for the luminance of the zone's blade (zone_labels)"""
    _sm_img[0] = img
    return _sm_prev_labels(objs, cam, R, state=state, ended=ended, img=img)


def zone_labels(state, cam, R, taken, own_team=None):
    out = _sm_prev_zone_labels(state, cam, R, taken, own_team)
    if not state: return out
    import owners
    from scipy.ndimage import label as _cc
    from scipy.spatial import ConvexHull
    from PIL import Image as _Im, ImageDraw as _Dr
    img = _sm_img[0]
    tick = int(state.get('tick', 0))
    ids_h = R['id'][::2, ::2]; H, W = ids_h.shape
    yy, xx = np.mgrid[0:H, 0:W]
    items = {int(it[0]): it for it in state.get('items', [])}
    for iid, it in items.items():
        did = int(it[1])
        if did not in SAM_ZONE_DIDS: continue
        Z = owners.zone(did)
        if not Z: continue
        own = int(it[2]); ow = True if own_team is None else own_team(own)
        x, y = int(it[3]), int(it[4]); rz = float(Z['logic_row'].get('Radius') or 300)
        age = tick - _ZONE_SEEN.get(iid, tick)
        if did in SAM_BIRTH_DISC and age <= SAM_BIRTH_DISC[did]:
            # the plain logic disc on the ground, cut by what is drawn in front of it (ZONE_DISC's rule)
            cx, cy = _proj(cam, x, y)
            dx = abs(_proj(cam, x + rz, y)[0] - cx); dy = abs(_proj(cam, x, y + rz)[1] - cy)
            if dx < 1 or dy < 1: continue
            disc = ((xx - cx) / dx) ** 2 + ((yy - cy) / dy) ** 2 <= 1
            g = ground_depth(cam, x, y, rz, R['meta'].get('wref', 0.0), H, W)
            dm = disc & ~(R['wd'] < g - DEPTH_TOL) & ~taken
            if dm.any(): out[iid] = (dm, did, own)
            else: out.pop(iid, None)
            continue
        if iid not in out or img is None: continue
        m = out[iid][0]
        if not m.any(): continue
        # the blade: the bright pixels of the zone's sprite cluster that touches the label (as z_1101_s096)
        lum = 0.3 * img[..., 0].astype(np.float32) + 0.59 * img[..., 1].astype(np.float32) + 0.11 * img[..., 2].astype(np.float32)
        C = sprite_pixels(('zone', did), ow, R) > 0
        thr = SAM_HALF * np.percentile(lum[m], 95)
        cand = C & ~taken
        L, n = _cc(cand, structure=np.ones((3, 3), bool))
        keep = np.unique(L[m & cand]); keep = keep[keep > 0]
        cluster = np.isin(L, keep) | m
        bright = (_close((lum >= thr) & cluster, 2) & cluster) | (m & (lum >= thr))
        bright &= ~taken
        L2, n2 = _cc(bright, structure=np.ones((3, 3), bool))
        if n2:
            sz = np.bincount(L2.ravel()); small = np.nonzero((sz < 4) & (np.arange(len(sz)) > 0))[0]
            bright &= ~np.isin(L2, small)
        pts = np.argwhere(bright)[:, ::-1].astype(float)   # (x, y)
        if len(pts) < 3: out.pop(iid, None); continue
        try:
            hull = ConvexHull(pts)
        except Exception:   # collinear: nothing to hull
            out.pop(iid, None); continue
        im = _Im.new('L', (W, H), 0)
        _Dr.Draw(im).polygon([tuple(pts[v]) for v in hull.vertices], fill=1, outline=1)
        hm = np.asarray(im, bool)
        if R.get('wd') is not None:                       # cut only by props drawn in front of the blade
            g = ground_depth(cam, x, y, rz, R['meta'].get('wref', 0.0), H, W)
            hm &= ~(R['wd'] < g - DEPTH_TOL)
        hm &= ~taken
        L3, n3 = _cc(hm, structure=np.ones((3, 3), bool))
        if n3:
            sz = np.bincount(L3.ravel()); small = np.nonzero((sz < 4) & (np.arange(len(sz)) > 0))[0]
            hm &= ~np.isin(L3, small)
        if hm.any(): out[iid] = (hm, did, own)
        else: out.pop(iid, None)
    return out


def _sm_anchor(body, ctx):
    """Whirlwind Pirate anchor (478): the chain links and rod (shapes SAM_CHAIN_SHAPES) are not the anchor; the anchor's
    own shapes are kept, its sprite-edge pixels only within 2 px of them; specks dropped; holes filled"""
    if not body.any(): return body
    from scipy.ndimage import binary_fill_holes, label as cc
    T = projtab.table()[ctx['t']]
    clips = tuple(c for c in (T.get('blue'), T.get('red')) if c)
    SP = _a4_shape_pixels(ctx['R'], T['file'], clips)  # shape id + 1, 0: none of the clip's shapes
    chain = np.isin(SP, [s + 1 for s in SAM_CHAIN_SHAPES])
    shape = body & (SP > 0) & ~chain
    if not shape.any(): return body & ~chain
    m = body & ~chain & ((SP > 0) | _dilate(shape, 2))
    L, n = cc(m); sz = np.bincount(L.ravel())
    small = np.nonzero((sz < 12) & (np.arange(len(sz)) > 0))[0]
    m &= ~np.isin(L, small)
    # the chain lay over the anchor: the anchor pixels under it are hidden, so the notches it leaves are filled in (closing)
    return binary_fill_holes(_close(m, 5)) & ctx['region']


POST_BODY[SAM_ANCHOR] = _sm_anchor
