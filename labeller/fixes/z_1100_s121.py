# human review 2026-10-28 fix for skin 121 (Haiku fix, reviewed by the coordinator, installed 10-29; source out/sweep2/haikutasks/human1028/s121/fix3.py; runs after z_1099_cleanup like the tests)
# human review 2026-10-28, skin 121 (VoodooSandsOfTimes), zone labels of zone type 17000802 (Voodoo002AttackEarth, area row
# askin 'Voodoo002AttackEarth'). fix3 = fix2 + the solid-fill of the puddle where the swirl stands in it (user decision 2026-10-29).
#
# User findings (task out/sweep2/haikutasks/human1028/s121/task.md):
#  - "unlabel the thing in the middle" (289, 301): the blue swirl (a cyan flame with a yellow rim) in the middle of the sand
#    zone is taken in by the zone's SPRITE label. Cut by atlas polygon: polygons whose atlas texels are cyan (the swirl), and
#    the opaque polygons within Z121_RIM px of it (its yellow rim).
#  - "some problems with the label" (301): translucent glow of the zone (the warm haze above and beside the sand, a detached
#    blob above the crate, a dark patch under the swirl) is taken in. Cut by atlas polygon: polygons whose alpha is under
#    Z121_GLOW_ALPHA (translucent glow or shadow). What remains is the sand puddle; flecks under Z121_SPECK px are dropped.
#
# Fill rule (user decision 2026-10-29, binding): where the swirl stands INSIDE the sand puddle, the sand continues under
# it, so the puddle label is filled solid there: every enclosed hole of the puddle label that holds swirl or rim pixels
# (the swirl and its translucent blue halo) is filled, and only where it is not another object's label (taken). Where the
# swirl sticks out beyond the sand edge, it is not enclosed by the label, so it stays out. Fade rule (user, 2026-10-29): the
# translucent fading pieces of a puddle (atlas alpha under Z121_GLOW_ALPHA) stay unlabelled, as in fix2.
# Phase rule (the coordinator's review, 2026-10-29): the cut applies to every item of zone type 17000802 (attack 0's
# 3000002, attack 90's 3000003 and the items of the super row), and only in the PUDDLE phase: age > ZONE_DOME (8 ticks, the
# same age zone_labels uses for the dome disc; the state gives no dome field). Dome-phase labels (the disc) are untouched.
# In this capture: 3000002 lives 283-321 (dome 283-291, puddle 292-321); 3000003 lives 323-361 (dome 323-331, puddle 332-361).
# The change therefore also reaches the attack 90 row's puddle frames (303-321 attack 0 puddle, 332-341 attack 90 puddle)
# and the super row's puddle frames (343-361, zone 3000003). Projectile types are not touched.
Z121_DID = 17000802          # zone type id (Voodoo002AttackEarth)
Z121_GLOW_ALPHA = 80         # atlas alpha (0..255) of a zone sprite polygon below which it is translucent glow or shadow
Z121_CYAN = 0.5              # share of cyan texels that makes a polygon the blue swirl
Z121_RIM = 2                 # px: opaque polygons this close to the swirl are its yellow rim
Z121_SPECK = 20              # px: a puddle-phase piece of the label smaller than this is a speck
_Z121_INFO = {}              # (zone did, own) -> [(file, tex, region map, {polygon id: kind})] ; kind 1 glow, 2 swirl, 3 opaque
_z121_zone_prev = zone_labels


def _z121_info(did, own):
    key = (did, bool(own))
    if key in _Z121_INFO: return _Z121_INFO[key]
    out = []
    for (file, tex), polys in sprites(('zone', did), own).items():
        rm = region_map(('zone', did), own, file, tex)
        A = sctex.texture(PD.sc(file), tex)
        alpha = A[..., 3].astype(np.float32); rgb = A[..., :3].astype(np.float32)
        kind = {}
        for i in range(1, len(polys) + 1):
            m = rm == i
            if not m.any(): continue
            a = float(alpha[m].mean()); c = rgb[m]
            cy = float(((c[:, 2] > 150) & (c[:, 1] > 150) & (c[:, 0] < 120)).mean())
            kind[i] = 2 if cy >= Z121_CYAN else (1 if a < Z121_GLOW_ALPHA else 3)
        out.append((file, tex, rm, kind))
    _Z121_INFO[key] = out
    return out


def _z121_bad(did, own, R):
    """(glow, swirl): (H, W) bool at frame resolution. glow = zone sprite pixels of translucent polygons; swirl = the blue
    swirl and its yellow rim (opaque polygons within Z121_RIM px of it)"""
    from scipy.ndimage import binary_dilation
    ids_h = R['id'][::2, ::2]; uv = R['uv']; draws = R['draws']
    kind = np.zeros(ids_h.shape, np.int8)
    for file, tex, rm, pk in _z121_info(did, own):
        tw, th = tex_size(file, tex); Hh, Ww = rm.shape
        dd = [d for d, dr in draws.items() if (dr.get('tw'), dr.get('tht')) == (tw, th)]
        if not dd: continue
        m = np.isin(ids_h, dd) & (uv[..., 0] >= 0)
        if not m.any(): continue
        yy, xx = np.nonzero(m)
        uu = np.clip((uv[yy, xx, 0] * Ww).astype(int), 0, Ww - 1); vv = np.clip((uv[yy, xx, 1] * Hh).astype(int), 0, Hh - 1)
        k = rm[vv, uu]
        lut = np.zeros(int(rm.max()) + 1, np.int8)
        for i, c in pk.items(): lut[i] = c
        code = lut[k]
        hit = code > 0
        kind[yy[hit], xx[hit]] = code[hit]
    swirl = kind == 2
    rim = (kind == 3) & binary_dilation(swirl, iterations=Z121_RIM)
    return (kind == 1), (swirl | rim)


def zone_labels(state, cam, R, taken, own_team=None):
    import owners
    from scipy.ndimage import label as _cc, binary_fill_holes
    out = _z121_zone_prev(state, cam, R, taken, own_team=own_team)
    if not state or R.get('wd') is None or not out: return out
    tick = int(state.get('tick', 0))
    for it in state.get('items', []):
        did = int(it[1])
        if did != Z121_DID: continue
        iid = int(it[0])
        if iid not in out: continue
        Z = owners.zone(did) or {}
        age = tick - _ZONE_SEEN.get(iid, tick)
        if age <= ZONE_DOME.get(Z.get('Name'), -1): continue          # dome phase: the disc, unchanged
        own = True if own_team is None else own_team(int(it[2]))
        m, d0, o0 = out[iid]
        glow, swirl = _z121_bad(did, own, R)
        if not (m & (glow | swirl)).any(): continue
        keep = m & ~glow & ~swirl
        L, n = _cc(keep, structure=np.ones((3, 3), bool))
        if n > 1:
            sz = np.bincount(L.ravel())
            keep = keep & (sz[L] >= Z121_SPECK)
        # fill: the enclosed holes that the swirl (and its halo) left in the puddle; holes without swirl pixels are untouched
        holes = binary_fill_holes(keep) & ~keep
        if holes.any():
            HL, hn = _cc(holes)
            hit = np.unique(HL[swirl & holes]); hit = hit[hit > 0]
            if hit.size: keep = keep | (np.isin(HL, hit) & ~taken)
        if keep.any(): out[iid] = (keep, d0, o0)
        else: del out[iid]
    return out
