# sweep 10-27 fix for skin 273 (Haiku fix, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s273/)
# sweep 10-27 fix for skin 273 IceDudeBurger, task s273 (Haiku fix, sweep 2026-10-27; task folder out/sweep2/haikutasks/sweep1027/s273/)
# Confirmed defect (sweep_1027 pair 5054, row 'sw2_273 super 45 deg', tick 371, kind 'hole'): the yellow lingering zone of the
# skin's super, IceDudeBurgerUltiArea (area effect data id 17000395; the other IceDude skins have their own ids, 17000392-398),
# has closed loops inside the vortex around a pale-blue droplet. Effects drawn over the zone (the droplet, ice swirls) are not
# the zone's sprites in the ID pass, so their pixels are cut out as holes of the zone label.
#
# The user's rule 7 (RULES.md): a zone is the full circle from the game data, cut ONLY by solid map objects standing on it
# (walls, crates, barrels); characters, bushes, UI and effects never cut it. Fix: holes of this zone's label that are enclosed
# by it (binary_fill_holes), contain no solid map block (_zs_solid of z_1023_zone_solid.py, the map's solid tiles above the
# ground) and are not another label's pixels are filled. A hole with any solid pixel stays cut (walls and crates). Outer
# notches (the zone edge where bushes or walls cut in) are not enclosed, so they are not touched.
# Checked at tick 371 (the defect): holes of 226 px and 96 px around the droplet and one of 13 px, none solid. Other ticks of the
# row carry the same kind of holes over ice swirls and droplets in the vortex (ticks 361-369, 385-395, 409), all solid-free.
# Other zones and other labels are not changed (keyed to the one area data id).
S273_ZONE_DIDS = {17000395}   # IceDudeBurgerUltiArea (skin 273's super zone)
_s273_zone_labels_prev = zone_labels


def zone_labels(state, cam, R, taken, own_team=None):
    from scipy.ndimage import binary_fill_holes as _bfh, label as _cc
    out = _s273_zone_labels_prev(state, cam, R, taken, own_team)
    if R.get('wd') is None or not out: return out
    dids = {int(it[0]): int(it[1]) for it in state.get('items', [])}
    targets = [iid for iid in out if dids.get(iid) in S273_ZONE_DIDS]
    if not targets: return out
    solid = _zs_solid(cam, R)
    for iid in targets:
        m, did, own = out[iid]
        h = _bfh(m) & ~m
        if not h.any(): continue
        L, n = _cc(h)
        fill = np.zeros_like(m)
        for j in range(1, n + 1):
            c = L == j
            if not (c & solid).any(): fill |= c   # an enclosed hole with no solid block in it: effects over the zone
        fill &= ~taken & ~solid
        if fill.any(): out[iid] = (m | fill, did, own)
    return out
