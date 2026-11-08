# sweep 10-27 fix for skin 175 (Haiku fix, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s175/)
# sweep 10-27 fix for skin 175 RocketGirlZeus (Rocket Girl, Brock 007 electric skin), task s175 (sweep 2026-10-27).
# Confirmed defects (sw2_175): pair 3018, attack 0 deg, tick 281 (extra_label): thin yellow loops around detached blue blobs and two
# tiny rings above the zone; pair 3025, attack 90 deg, tick 329 (extra_label): small yellow rings around droplet specks above the pool edge.
#
# Cause: the lingering electric zone BurnMainAttackElec (area effect 17000179, logic BurnMainAttack, radius 300) is labelled from its
# sprites (zone_sprites), and the zone_labels region reaches up to ZONE_TOP. The particles the zone throws out (blue blobs, droplets,
# sparks) lie outside the zone's drawn ring, detached from it, and were labelled as part of the zone. Particles are never labelled
# (rule 3), and a zone is its circle (rule 7).
#
# Fix, for this zone only (did 17000179, the only area row of this name): a connected part of the zone label that lies ENTIRELY beyond
# the zone's drawn ring is dropped. The ring is the zone's ground ellipse (logic radius projected at ground level, as in
# idlabels.zone_labels); the zone's own fire ring (brock_def_sp_fire_ground) lies at 1.13 to 1.17 radii on every tick of both rows
# (measured on this capture), so S175_OUT = 1.22 is beyond it. Parts that touch the ring or its inside stay exactly as they were
# (the lightning, the fire ring, the dust, and the attack bolt that reaches the pool at tick 323-325), so only detached particles go.
# Measured on the capture (ticks 263-341): the detached blobs at tick 281 sit at 1.8 to 2.5 radii, the droplets at tick 329 at 1.7 to
# 2.3 radii, each as its own part; the zone's rim pieces (tick 329, the arc at the top right) are at 0.98 to 1.15 and stay.
# Other zones, and other skins, are untouched.

S175_ZONE_DID = 17000179
S175_ZONE_NAME = 'BurnMainAttackElec'
S175_OUT = 1.22          # ground-ellipse radii beyond which a label pixel is outside the zone's ring (the ring's outer edge is 1.13-1.17)

_s175_zone_labels_prev = zone_labels


def zone_labels(state, cam, R, taken, own_team=None):
    import owners
    from scipy.ndimage import label as _s175_cc
    out = _s175_zone_labels_prev(state, cam, R, taken, own_team)
    H, W = R['id'][::2, ::2].shape
    yy = xx = None
    for it in state.get('items', []):
        iid, did = int(it[0]), int(it[1])
        if did != S175_ZONE_DID or iid not in out: continue
        Z = owners.zone(did)
        if not Z or Z.get('Name') != S175_ZONE_NAME or owners.area(did)['kind'] != 'zone': continue
        if yy is None: yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
        x, y = int(it[3]), int(it[4]); rz = float(Z['logic_row'].get('Radius') or 300)
        cx, cy = _proj(cam, x, y); dx = abs(_proj(cam, x + rz, y)[0] - cx); dy = abs(_proj(cam, x, y + rz)[1] - cy)
        if dx < 1 or dy < 1: continue
        r = np.sqrt(((xx - cx) / dx) ** 2 + ((yy - cy) / dy) ** 2)
        old, odid, own = out[iid]
        if not old.any(): continue
        L, n = _s175_cc(old)                      # 4-connected parts of this zone's label
        if n == 0: continue
        size = np.bincount(L.ravel(), minlength=n + 1)
        beyond = np.bincount(L[old & (r > S175_OUT)].ravel(), minlength=n + 1)
        detached = np.nonzero((size > 0) & (beyond == size))[0]
        detached = detached[detached > 0]         # label 0 is the background
        if not detached.size: continue
        new = old & ~np.isin(L, detached)
        if new.any(): out[iid] = (new, odid, own)
        else: out.pop(iid, None)
    return out
