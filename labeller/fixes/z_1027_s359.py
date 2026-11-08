# sweep 10-27 fix for skin 359 (Haiku fix, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s359/)
# sweep 10-27 fix for skin 359 (SandstormMake, Sandy's skin 005): Haiku task s359 (sweep 2026-10-27, out/sweep2/haikutasks/sweep1027/s359/).
# Two confirmed defects (sweep_1027 pairs 6752, 6760, 6761), both on the Sandy 005 family:
#
# (1) attack 90 deg, tick 313 (pair 6752): the attack's sand cloud (Sandy005Projectile, type 6000674, ZONE: one solid label over
#     the whole cloud, rule 7) has thin protrusions on its right side: a tab at the top right, a notch-lobe at the lower right,
#     small pieces joined by 1-2 px necks ("spur_or_speck"). They are the cloud's ragged edge, not the cloud. Fix: the cloud's
#     own outline is opened with a disc of radius S359_OPEN px (frame resolution): protrusions narrower than about 2 * S359_OPEN + 1
#     px go; the solid body and the puffs of the cloud stay. The opening is closed again by a small disc (puffs that touch
#     through a 1-px channel stay one piece) and kept inside the region. A piece the opening would remove whole and that is
#     still a puff (at least S359_KEEP px) is put back, so no puff of the attack is lost. Only type 6000674 is changed.
#
# (2) super 45 deg, ticks 389 and 411 (pairs 6760, 6761): the yellow zone Sandy_005_ulti (area effects data id 17000221, logic
#     name Sandy_005_ulti, Radius 2000) is labelled from its own sprites, so the green bushes drawn in front of the zone cut the
#     label: a jagged outline along the bush edges, the bushes taken out of the zone. The user's rule 7 (RULES.md): the zone is the
#     full circle from the game data, cut ONLY by solid map objects (walls, crates, barrels); bushes, characters, UI and effects
#     never cut it. Fix: the same rule as z_1023_sandy270.py (skin 270) and z_1027_s042.py (skin 042), keyed to this zone's did and
#     name: the label is the exact projected logic circle (its ground points through idlabels._proj, filled polygon, scale 1.0)
#     minus the solid map blocks (_zs_solid, z_1023_zone_solid) and minus the other labels (taken: projectiles, spawnables).
#     Pinholes under ZS_HOLE px that are not solid and not another label are filled (_zs_fill_small); specks under 1 % of the
#     largest part are dropped. Only the zone with this did and name is changed; every other zone keeps the earlier rules.

S359_ATK = 6000674            # Sandy005Projectile: Sandy's sand-cloud attack of skin 359 (ZONE)
S359_OPEN = 3                 # px (frame): disc radius of the opening; protrusions narrower than about 7 px go
S359_CLOSE = 4                 # px (frame): disc radius of the closing that rejoins the cloud's parts through the necks the opening cut
S359_KEEP = 40                # px: a puff the opening would remove whole is put back when it is at least this big
S359_TAB = 0.03               # share of the largest piece: a loose fragment under this (and under S359_SPECK_PX) is a tab, dropped
S359_SPECK_PX = 25            # px: the speck size of the shared clean-up (z_1099_cleanup), used for the tab test

S359_ZONE_DID = 17000221      # area_effects_skin row of skin 359's super zone
S359_ZONE_NAME = 'Sandy_005_ulti'
S359_ZONE_SCALE = 1.0         # drawn edge = logic radius (checked on the frames, see the report)
S359_RING = 1440              # ground points of the logic circle projected to the screen
S359_SPECK = 0.01             # parts of the zone label under this share of its largest part are specks: dropped


def _s359_disc(r):
    """(2r+1, 2r+1) bool disc of radius r"""
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    return (xx * xx + yy * yy) <= r * r


_s359_atk_prev = POST_BODY.get(S359_ATK)   # a2_agent2's _sandy_end (every Sandy attack type): the end burst has no label (rule 2)


def _s359_atk(body, ctx):
    """the cloud's own outline without its thin protrusions (opening), puffs kept, rejoined through 1-px channels; the hook
    that was there before (a2_agent2: no label once the cloud is only its end burst) runs first"""
    from scipy.ndimage import binary_opening, binary_closing, binary_fill_holes, label as cc
    if _s359_atk_prev is not None: body = _s359_atk_prev(body, ctx)
    if not body.any(): return body
    op = binary_opening(body, structure=_s359_disc(S359_OPEN))
    # a whole puff that the opening would remove and that is big enough is put back (never lose a part of the attack)
    L, n = cc(body)
    if n:
        touched = np.bincount(L[op].ravel(), minlength=n + 1)
        sz = np.bincount(L.ravel(), minlength=n + 1)
        lost = np.nonzero((touched == 0) & (sz >= S359_KEEP))[0]
        lost = lost[lost > 0]
        if lost.size: op = op | (np.isin(L, lost) & body)
    if not op.any(): return body
    P = S359_CLOSE + 1                       # padding so the closing does not erode at the frame edge
    op = binary_closing(np.pad(op, P), structure=_s359_disc(S359_CLOSE))[P:-P, P:-P] & body
    op = binary_fill_holes(op) & ctx['region']
    # a tab the opening cut off a puff (a loose loop beside the cloud) is a speck, not a part of the attack: dropped. A whole
    # puff of the input body is never dropped here (it is a part of the attack, however small)
    Lf, nf = cc(op)
    if nf > 1:
        szf = np.bincount(Lf.ravel())[1:]; big = szf.max()
        for j in range(1, nf + 1):
            if szf[j - 1] >= max(S359_SPECK_PX, S359_TAB * big): continue
            ys, xs = np.nonzero(Lf == j)
            k = L[ys[0], xs[0]]
            if k > 0 and (L == k).sum() == szf[j - 1]: continue   # a whole puff of the input
            op[Lf == j] = False
    return op


POST_BODY[S359_ATK] = _s359_atk


def _s359_circle(cam, x, y, rz, H, W):
    """(H, W) bool: the exact screen polygon of the logic circle (centre x, y, radius rz, on the ground plane)"""
    th = np.linspace(0.0, 2.0 * np.pi, S359_RING, endpoint=False)
    pts = [tuple(_proj(cam, x + rz * np.cos(t), y + rz * np.sin(t))) for t in th]
    im = Image.new('L', (W, H), 0); ImageDraw.Draw(im).polygon(pts, fill=1)
    return np.asarray(im, bool)


def _s359_specks(part):
    from scipy.ndimage import label as cc
    L, n = cc(part)
    if n <= 1: return part
    sz = np.bincount(L.ravel())[1:]
    return np.isin(L, 1 + np.nonzero(sz >= S359_SPECK * sz.max())[0])


_s359_zone_labels_prev = zone_labels


def zone_labels(state, cam, R, taken, own_team=None):
    import owners
    out = _s359_zone_labels_prev(state, cam, R, taken, own_team)
    if R.get('wd') is None: return out
    H, W = R['id'][::2, ::2].shape
    solid = None
    for it in state.get('items', []):
        iid, did = int(it[0]), int(it[1])
        if did != S359_ZONE_DID: continue
        Z = owners.zone(did)
        if not Z or Z.get('Name') != S359_ZONE_NAME or owners.area(did)['kind'] != 'zone': continue
        if solid is None: solid = _zs_solid(cam, R)
        x, y = int(it[3]), int(it[4]); rz = float(Z['logic_row'].get('Radius') or 300) * S359_ZONE_SCALE
        disc = _s359_circle(cam, x, y, rz, H, W)
        part = disc & ~solid & ~taken
        if part.any():
            part = _zs_fill_small(part, True, solid, taken)   # pinholes (not solid, not another label)
            part = _s359_specks(part)
        if part.any(): out[iid] = (part, did, int(it[2]))
        else: out.pop(iid, None)
    return out
