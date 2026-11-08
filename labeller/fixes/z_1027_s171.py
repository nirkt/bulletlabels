# sweep 10-27 fix for skin 171 (Haiku fix, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s171/)
# Skin 171 CactusFireproof (sweep 2026-10-27, Haiku task s171; findings 2927, 2935, 2943-2945). Two defects, two type ids.
#
# (1) Spike006Projectile (Cactus's attack, type 6000126, every Cactus skin that uses it): a 13 px speck, a loop on a
#     teal spark beside the head, detached from the attack body (tick 271: (692-696, 243-246)). The shared clean-up
#     (z_1099_cleanup) bridges parts closer than 6 px, and the speck is joined to the body through a 9 px spur on the
#     head's left edge, so the clean-up keeps it. The rule here: parts of the body smaller than S171_SPECK px and smaller
#     than S171_FRAC of the largest part are dropped, when they lie more than 2 px away from the rest of the body (the
#     bridging is 1 px here, not 3), and only while the largest part is at least S171_MIN_MAIN px (the head is never
#     dropped). The spur (2 px from the head) stays. The other speck findings of this skin are gone: at tick 273 there is
#     no speck in the new capture, and at ticks 313 and 321 the shared clean-up already removed them; this rule changes
#     nothing there.
#
# (2) Spike_006_ulti_explosion (Cactus's super, lingering zone, zone did 17000045, item 3000010): the zone fill has two
#     small holes from tick 363 to 411 (27-107 px at ticks 371, 389, 411): spike clusters drawn over the ground inside the
#     zone, so the ID pass shows the ground there and not the zone's sprite. Effects never cut a zone (the user's rule).
#     Here: holes of up to S171_ZHOLE px that lie on the ground (not nearer the camera than the ground plane of the zone,
#     so no crate, wall or barrel) and are no other label's pixels are filled. Walls and crates stay cut out (the larger
#     openings, e.g. the wall at the top of the zone at ticks 363-369, are over the size limit and stay). The zone's
#     other shapes are unchanged.
#
# Shared: Spike006Projectile (6000126, Cactus's attack clip spike_006_atk1) and the zone 17000045 (Cactus's super) are the
# brawler's types, so the fix is keyed to the type / zone. Other Cactus skins were not re-captured here: the same defects
# are likely there, not verified.

S171_TYPE = 6000126        # Spike006Projectile
S171_SPECK = 25            # px: a detached part smaller than this is a speck
S171_FRAC = 0.1            # ... and smaller than this share of the largest part
S171_MIN_MAIN = 100        # px: the largest part must be at least this (the head is never dropped)
S171_BRIDGE = 1            # px (dilation): parts with a gap of 2 px or less are one part
S171_ZONE_DID = 17000045   # Spike_006_ulti_explosion (Cactus super zone)
S171_ZHOLE = 150           # px: holes of the zone label up to this size are filled (effects on the ground)


def _s171_spike_specks(body, ctx):
    from scipy.ndimage import label as _s171_cc, binary_dilation as _s171_dil
    if not body.any(): return body
    st = np.ones((3, 3), bool)
    L, n = _s171_cc(_s171_dil(body, structure=st, iterations=S171_BRIDGE), structure=st)
    if n < 2: return body
    sz = np.bincount(L[body], minlength=n + 1)[1:]
    if sz.max() < S171_MIN_MAIN: return body
    drop = [k + 1 for k in range(n) if 0 < sz[k] < S171_SPECK and sz[k] < S171_FRAC * sz.max()]
    if not drop: return body
    return body & ~np.isin(L, drop)


POST_BODY[S171_TYPE] = _s171_spike_specks


_s171_zone_labels_prev = zone_labels


def zone_labels(state, cam, R, taken, own_team=None):
    """zone labels, with the small ground-level holes of the Cactus super zone filled (see the header, defect 2)"""
    from scipy.ndimage import binary_fill_holes as _s171_fill, label as _s171_cc
    import owners
    out = _s171_zone_labels_prev(state, cam, R, taken, own_team)
    if R.get('wd') is None or not out: return out
    items = {int(it[0]): it for it in state.get('items', [])}
    for iid, (m, did, own) in list(out.items()):
        if did != S171_ZONE_DID or iid not in items: continue
        f = _s171_fill(m) & ~m
        if not f.any(): continue
        it = items[iid]; Z = owners.zone(did)
        rz = float(Z['logic_row'].get('Radius') or 300)
        gd = ground_depth(cam, int(it[3]), int(it[4]), rz, R['meta'].get('wref', 0.0))
        on_ground = R['wd'] >= gd - DEPTH_TOL
        L, n = _s171_cc(f)
        add = np.zeros(m.shape, bool)
        for j in range(1, n + 1):
            hj = L == j
            if hj.sum() > S171_ZHOLE: continue
            if (on_ground[hj]).mean() < 0.5: continue    # nearer than the ground: a block, a wall or a crate
            if (taken[hj]).any(): continue                # another label's pixels
            add |= hj
        if add.any(): out[iid] = (m | add, did, own)
    return out
