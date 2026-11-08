# Penny's super (the user, 2026-10-24, on out/sweep2/regress_1024/sw2_034_t401.png): FOUR different classes
#   1. the cannon (turret character): 'spawnable' (unchanged)
#   2. the bomb the cannon fires: 'spawnable_attack' (new kind: every projectile / explosion of a spawnable, not only Penny)
#   3. the drawn landing zone (the turret projectile's pre-explosion target marker, gadget_primo_meteor_target, drawn on
#      the ground at the shot's end point while the bomb flies): 'target' (new kind), its label the marker's own circle
#      (the drawn ring's size, NOT the burn zone's bigger logic circle) in full, cut only by solid map blocks, like zones
#   4. the burn zone (BurnUlti) once the bomb has landed: 'zone', not labelled while a turret bomb aimed at it is drawn
#      (at sw2_034 t401 the item already existed while only the landing ring and the falling bomb were on screen)
# Only Penny's turret projectiles have a pre-explosion marker (8 types, projtab pre_file).
PENNY_TURRET = {t for t, d in projtab.table().items() if d['pre_file'] and d['name'].endswith('TurretProjectile')}
PENNY_TARGET_ID = 500000000   # label id of a marker: the bomb's object id + this (lid[0] must differ from the bomb's)
PENNY_MARK_CV = 64            # coverage from which a marker sprite pixel counts for the size fit
PENNY_MARK_MIN = 30           # px: fewer marker pixels than this, no marker label
PENNY_MARK_Q = 98             # percentile of the marker pixels' distance from the end point = the ring's radius
PENNY_BURN_NEAR = 200         # logic units: a zone item this close to a drawn turret bomb's end point waits for it

_penny_prev_sprites = sprites


def sprites(t, own):
    """('pre', type): the sprite set of a projectile type's pre-explosion marker clip"""
    if isinstance(t, tuple) and t[0] == 'pre':
        T = projtab.table()[t[1]]
        return _clips_sprites([(T['pre_file'], T['pre_blue'] if own else T['pre_red'])])
    return _penny_prev_sprites(t, own)


_penny_prev_meta = meta


def meta(objs, state, P):
    out = _penny_prev_meta(objs, state, P)
    for k, v in list(out.items()):
        if isinstance(v, dict) and v.get('kind') == 'spawnable':
            # a spawnable's projectile or explosion (not the spawnable character itself: its lid part is 'spawn')
            is_char = any(l[0] == k and l[2] == 'spawn' for l in P['lids'])
            if not is_char:
                out[k] = dict(v, kind='spawnable_attack')
    for oid, info in P.get('targets', {}).items():
        out[oid] = info
    return out


def _penny_ellipse(cam, x, y, r, shape):
    H, W = shape; cx, cy = _proj(cam, x, y)
    rx = abs(_proj(cam, x + r, y)[0] - cx); ry = abs(_proj(cam, x, y + r)[1] - cy)
    if rx < 1 or ry < 1: return None
    yy, xx = np.mgrid[0:H, 0:W]
    return ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1


_penny_prev_labels = labels


def labels(objs, cam, R, state=None, ended=(), img=None):
    import bulletlabels as BL
    bombs = [o for o in objs if o['t'] in PENNY_TURRET and o.get('drawn') and o.get('target')]
    if not bombs or state is None:
        return _penny_prev_labels(objs, cam, R, state=state, ended=ended, img=img)
    P = _penny_prev_labels(objs, cam, R, state=state, ended=ended, img=img)
    ids, lids, share = P['ids'], list(P['lids']), P['share']; H, W = ids.shape
    # 4. zone items under a falling bomb: not yet (the burn starts once the bomb has landed)
    import owners
    for it in state.get('items', []):
        iid = int(it[0]); x, y = int(it[3]), int(it[4])
        if owners.area(int(it[1]))['kind'] != 'zone': continue
        if any(np.hypot(o['target'][0] - x, o['target'][1] - y) <= PENNY_BURN_NEAR for o in bombs):
            lid = (iid, -1, 'area')
            if lid in lids: ids[ids == lids.index(lid)] = -1
    # 3. the landing marker of each falling bomb
    try: solid = _zs_solid(cam, R)
    except Exception: solid = np.zeros((H, W), bool)
    cv = R['cv'][::2, ::2]; targets = {}
    for o in bombs:
        M = (sprite_pixels(('pre', o['t']), o['own'], R) > 0) & (cv >= PENNY_MARK_CV)
        if M.sum() < PENNY_MARK_MIN: continue
        x, y = o['target']; cx, cy = _proj(cam, x, y)
        rx = abs(_proj(cam, x + 100, y)[0] - cx) / 100.0; ry = abs(_proj(cam, x, y + 100)[1] - cy) / 100.0
        if rx <= 0 or ry <= 0: continue
        ys, xs = np.nonzero(M)
        r = float(np.percentile(np.hypot((xs - cx) / rx, (ys - cy) / ry), PENNY_MARK_Q))   # logic units
        disc = _penny_ellipse(cam, x, y, r, (H, W))
        if disc is None: continue
        m = disc & ~solid & (ids < 0)
        if not m.any(): continue
        oid = o['id'] + PENNY_TARGET_ID; lid = (oid, -1, 'target')
        lids.append(lid); k = len(lids) - 1
        ids[m] = k; share[m] = BL.CUT
        r0 = owners.runtime(o['t'], None, None)
        targets[oid] = dict(kind='target', brawler=r0.get('brawler'), skin=r0.get('skin'), name=projtab.table()[o['t']]['pre_blue'])
    P['lids'] = lids
    if targets and 'meta' in P:
        P['meta'].update(targets)
        for k2, v in list(P['meta'].items()):   # meta() ran inside the wrapped labels(): apply the kind rename here too
            if isinstance(v, dict) and v.get('kind') == 'spawnable' and not any(l[0] == k2 and l[2] == 'spawn' for l in lids):
                P['meta'][k2] = dict(v, kind='spawnable_attack')
    P['targets'] = targets
    return P


# Shadowdemon 112 (the user, 2026-10-26): the super's minion landing circle (area ShadowdemonMinionTarget) is a landing
# marker too: class 'target', labelled as the full circle (ZONE_DISC: z_1023_zone_solid cuts it only by solid blocks).
TARGET_ZONES = {'ShadowdemonMinionTarget'}
ZONE_DISC.update(TARGET_ZONES)
# the drawn circle is 1/3 of the logic radius (500 of 1500 units: measured on sw2_112 t349-389, p95 0.332 in every direction)
A3_DISC_SCALE['ShadowdemonMinionTarget'] = 0.34
_penny_meta2 = meta


def meta(objs, state, P):
    out = _penny_meta2(objs, state, P)
    import owners
    for k, (did, own) in P.get('zones', {}).items():
        if (owners.zone(did) or {}).get('Name') in TARGET_ZONES and isinstance(out.get(k), dict):
            out[k] = dict(out[k], kind='target')
    return out
