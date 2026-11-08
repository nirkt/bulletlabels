# hosted review 2026-10-29 fix (Haiku fix, reviewed by the coordinator, installed 10-29 LOCAL ONLY; source out/sweep2/haikutasks/site1029/s096/fix2.py; tested via IDLABELS_OVERRIDE after all fixes incl. z_1100_*, hence z_1101_)
# sweep 10-29 coordinator follow-up for skin 096 (SamuraiDefault, sw2_096, task out/sweep2/haikutasks/site1029/s096/).
# Replaces fix.py (kept, superseded): run this file alone (IDLABELS_OVERRIDE=<this file>), it is self-contained.
# The super "super 45" (Samurai's ulti: a cyan slash streak with the white/orange flame head) is labelled as the lingering
# area SamuraiCrossSlashArea (did 17000775, kind zone, 1500 ms), from tick 367 on (its logic projectile has ended).
# Base label (zone_labels): the zone's sprite pixels inside the zone's logic circle, which cut the flame tip and the
# cyan tip (user finding, hosted review 2026-10-29: super 45 t375 and t389 "too small / cut off").
# fix.py took the zone-sprite cluster touching the label, kept the BRIGHT body (luminance >= half of the label's own bright
# core, so the blue halo and the faint translucent tail stay out), closed it and filled its holes: its outline had a jagged
# spur and a notch at the tips (coordinator, 2026-10-29).
# Fix2 (the user's rule for slash streaks: ONE convex hull over the whole attack, not the thin lines):
#  1. the bright blade pixels: the bright body of the zone-sprite cluster that touches the label (as fix.py: luminance >=
#     half the label's bright core, closed 2 px), plus the label's own bright pixels; one-pixel and tiny specks dropped;
#  2. the label is the convex hull of those pixels (one piece). Whatever falls inside the hull stays (the pale flame haze
#     inside it stays; nothing extra is added for it); the faint translucent tail beyond the bright tip is outside the hull;
#  3. the hull is cut only by props drawn in front of the blade: pixels whose depth (R['wd'], the front-most draw) is nearer
#     than the ground plane behind the zone by more than DEPTH_TOL (barrels, crates, walls); no other object or effect cuts it;
#  4. other labels' pixels (taken) stay out; pieces smaller than 4 px left by the cut are dropped.
# Only did 17000775 is touched (the default Samurai area; Samurai 002/003/004/Overcharged use dids 17000776-779).
# Class (zone, yellow) is kept as it is.
S096_DID = 17000775
S096_HALF = 0.5          # fraction of the label's bright core (95th percentile of luminance) at which the body edge sits
_s096_img = [None]
_s096_prev_labels = labels
_s096_prev_zone_labels = zone_labels


def labels(objs, cam, R, state=None, ended=(), img=None):
    """idlabels.labels with the frame kept for the luminance of zone_labels (as z_1023_zone_solid)"""
    _s096_img[0] = img
    return _s096_prev_labels(objs, cam, R, state=state, ended=ended, img=img)


def zone_labels(state, cam, R, taken, own_team=None):
    out = _s096_prev_zone_labels(state, cam, R, taken, own_team)
    if not out: return out
    import owners
    from scipy.ndimage import label as _cc
    from scipy.spatial import ConvexHull
    from PIL import Image as _Im, ImageDraw as _Dr
    img = _s096_img[0]
    items = {int(it[0]): it for it in state.get('items', [])}
    for iid, (m, did, own) in list(out.items()):
        if int(did) != S096_DID or not m.any() or img is None: continue
        ow = True if own_team is None else own_team(int(own))
        C = sprite_pixels(('zone', int(did)), ow, R) > 0
        lum = 0.3 * img[..., 0].astype(np.float32) + 0.59 * img[..., 1].astype(np.float32) + 0.11 * img[..., 2].astype(np.float32)
        thr = S096_HALF * np.percentile(lum[m], 95)
        cand = C & ~taken
        L, n = _cc(cand, structure=np.ones((3, 3), bool))
        keep = np.unique(L[m & cand]); keep = keep[keep > 0]
        cluster = np.isin(L, keep) | m
        bright = (_close((lum >= thr) & cluster, 2) & cluster) | (m & (lum >= thr))
        bright &= ~taken
        L2, n2 = _cc(bright, structure=np.ones((3, 3), bool))   # 1. specks out before the hull
        if n2:
            sz = np.bincount(L2.ravel()); small = np.nonzero((sz < 4) & (np.arange(len(sz)) > 0))[0]
            bright &= ~np.isin(L2, small)
        pts = np.argwhere(bright)[:, ::-1].astype(float)   # (x, y)
        if len(pts) < 3: continue
        try:
            hull = ConvexHull(pts)
        except Exception:   # collinear bright pixels: nothing to hull
            continue
        H, W = bright.shape
        im = _Im.new('L', (W, H), 0)
        _Dr.Draw(im).polygon([tuple(pts[v]) for v in hull.vertices], fill=1, outline=1)
        hm = np.asarray(im, bool)                           # 2. the convex hull of the bright blade pixels
        if R.get('wd') is not None and iid in items:        # 3. cut only by props drawn in front of the blade
            it = items[iid]; Z = owners.zone(int(did))
            rz = float(Z['logic_row'].get('Radius') or 300) if Z else 300.0
            g = ground_depth(cam, int(it[3]), int(it[4]), rz, R['meta'].get('wref', 0.0), H, W)
            hm &= ~(R['wd'] < g - DEPTH_TOL)
        hm &= ~taken                                        # 4. other labels stay out
        L3, n3 = _cc(hm, structure=np.ones((3, 3), bool))
        if n3:
            sz = np.bincount(L3.ravel()); small = np.nonzero((sz < 4) & (np.arange(len(sz)) > 0))[0]
            hm &= ~np.isin(L3, small)
        out[iid] = (hm, did, own)
    return out
