# Frank (HammerDude), every skin, attack and super waves (the user, 2026-10-26: "youre also keeping the after attack effects
# on all of frank's attacks. just label the attack"): a wave is labelled only while it is the attack, i.e. while at least one
# of its logic projectiles is still travelling. Once every member is finished (reached its range / hit: state 'finished')
# or ended, the spikes, rocks, flames, smoke and debris left on screen are after-attack effects: the wave's label goes.
# While it travels the wave is ONE convex hull (the user's Frank rule; idempotent where an earlier Frank fix already hulled).
FRANK_WAVE_TYPES = {t for t, d in projtab.table().items()
                    if d['logic'] in ('HammerDudeProjectile', 'HammerDudeUltiProjectile')
                    or (d['name'].startswith('Frank') and d['name'].endswith('Projectile'))}
import os as _os
# While the wave travels, only its rows that are still the attack (the user, 2026-10-27: "it should cover every row that is
# still an attack, while gradually excluding the first rows as they turn into just post attack effects"). The wave draws
# one row of spikes / tiles / rocks per update where its projectiles are (o['path']: their ground positions, one per update);
# a row is the attack for its first FRANK_AO_AGE updates, then it turns into an after-effect (measured on sw2_035 / 220 /
# 242 / 321, t275-319, rows coloured by age: ages 0-2 = white ice spikes (035, 220), bright tiles (242), standing spikes
# (321); from age 3 = cyan blobs and bones (220), blue tiles (242), sinking spikes (321)). Each body pixel takes the age of
# the nearest path point of its wave (spikes reach up on screen from their base: distance above a point counts
# 1 / FRANK_AO_UP); pixels older than FRANK_AO_AGE, or further than FRANK_AO_NEAR px from any path point (a speck at the
# hammer), are out, so the hull covers the attack rows only. A member that is finished / ended / hidden has no body of its
# own. Attack waves only: the super's logic points run ahead of its drawn spikes (sw2_344 t367 / t369).
FRANK_AO_AGE = int(_os.environ.get('FRANK_AO_AGE', '2'))      # updates
# per skin (age sheets of sw2_242 / 344 / 401 / 450, 10-27): 242's tiles are still the attack at age 3, the first blue row
# (the user, 10-27, on q_age_242.png: "the first blue row gets no label"); 344's spikes turn into flames in rings, 401's
# light flames into dark blue swooshes and 450's sprouts grow from age 2
FRANK_AO_AGE_T = {6000477: 3, 6000483: 1, 6000487: 1, 6000489: 1}
FRANK_AO_NEAR = float(_os.environ.get('FRANK_AO_NEAR', '25'))  # px (half resolution)
FRANK_AO_UP = float(_os.environ.get('FRANK_AO_UP', '3'))
FRANK_AO_FAR = float(_os.environ.get('FRANK_AO_FAR', '35'))   # px
FRANK_AO_WHOLE = float(_os.environ.get('FRANK_AO_WHOLE', '0.75'))   # share of a piece's pixels that decides it whole
FRANK_AO_FLAT = {6000477, 6000489}   # flat pieces (242's floor tiles, 450's star outlines): distance counts the same in every direction (rows stack up the screen)
FRANK_AO_ATTACK = {t for t in FRANK_WAVE_TYPES if projtab.table()[t]['logic'] == 'HammerDudeProjectile'}
# skins whose spikes are not in the ID pass (additive sprites: Frank 344's cyan spikes 6000483, 450's blue star outlines
# 6000489): bright, blue-saturated pixels joined to the kept body within FRANK_AO_EXT px are the spikes. Not for the other
# attacks: their body has the spikes already (401: the extension took in its left-behind blue swooshes, t279), or an
# earlier Frank fix cleans them to their own shapes (035 / 220 / 339 z_1026_frank_a, 242, 321)
FRANK_AO_EXT = int(_os.environ.get('FRANK_AO_EXT', '12'))
FRANK_AO_EXT_V = 170; FRANK_AO_EXT_SAT = 120   # max channel, blue - red (344 t277: spikes in, the soft halo out)
FRANK_AO_EXT_TYPES = {6000483, 6000489}
_FRANK_AO_PATH = {}   # label id of a wave -> (n, 3) screen x, y, age (updates) of its members' path points (set by labels() below)


def _frank_ao_body(body, ctx):
    from scipy.ndimage import label as cc
    o = ctx['o']
    if o.get('finished') or o.get('ended') or not o.get('drawn'): return body & False
    P = _FRANK_AO_PATH.get(o.get('label_id', o['id']))
    if not body.any() or P is None or not len(P): return body
    ys, xs = np.nonzero(body)
    dx = xs[:, None] - P[None, :, 0]; dy = ys[:, None] - P[None, :, 1]
    up = 1.0 if ctx['t'] in FRANK_AO_FLAT else FRANK_AO_UP
    d = dx ** 2 + np.where(dy < 0, dy / up, dy) ** 2; k = d.argmin(1)
    ok = (P[k, 2] <= FRANK_AO_AGE_T.get(ctx['t'], FRANK_AO_AGE)) & (d[np.arange(len(k)), k] <= FRANK_AO_NEAR ** 2)
    # whole pieces: a connected piece of the body that is mostly attack (or mostly older) is kept (dropped) whole, so a row's
    # edge nearer the next row's path point is not split off (344 t279: the flame column's right edge); mixed pieces (rows
    # drawn into each other) are cut per pixel
    L, n = cc(body); lab = L[ys, xs]
    if n:
        frac = np.bincount(lab, weights=ok, minlength=n + 1) / np.maximum(np.bincount(lab, minlength=n + 1), 1)
        f = frac[lab]; ok = np.where(f >= FRANK_AO_WHOLE, True, np.where(f <= 1 - FRANK_AO_WHOLE, False, ok))
        # a piece that nowhere comes within FRANK_AO_FAR px of the wave's path is not a row (the hammer's swing streak at
        # 220 / 035 t277: 44 px; every row piece measured <= 29 px, 344's column top)
        iso = np.sqrt(dx ** 2 + dy ** 2).min(1)
        mn = np.full(n + 1, np.inf); np.minimum.at(mn, lab, iso)
        ok &= mn[lab] <= FRANK_AO_FAR
    body = np.zeros_like(body); body[ys[ok], xs[ok]] = True
    img = ctx['img']
    if body.any() and img is not None and ctx['t'] in FRANK_AO_EXT_TYPES and FRANK_AO_EXT:
        im = img.astype(np.int16)
        ext = (im.max(-1) >= FRANK_AO_EXT_V) & (im[..., 2] - im[..., 0] >= FRANK_AO_EXT_SAT) & _dilate(body, FRANK_AO_EXT) & ctx['region']
        L, n = cc(ext | body); body = body | (ext & np.isin(L, np.unique(L[body])))
    return body


if _os.environ.get('FRANK_AO_BODY', '1') == '1':
    for _t in FRANK_AO_ATTACK:
        _pp = POST_BODY.get(_t)
        POST_BODY[_t] = (lambda prev: (lambda body, ctx: _frank_ao_body(prev(body, ctx) if prev else body, ctx)))(_pp)
_frank_ao_prev = labels


def labels(objs, cam, R, state=None, ended=(), img=None):
    import bulletlabels as BL
    # a wave's members are fired on one tick, so path index k is the same update for all of them; a member that stopped
    # (hit a wall) stops recording, its rows age with the wave (age from the longest path, not its own)
    _FRANK_AO_PATH.clear(); wl = {}
    for o in objs:
        if o['t'] in FRANK_AO_ATTACK and o.get('path'):
            g = o.get('label_id', o['id']); wl[g] = max(wl.get(g, 0), len(o['path']))
    for o in objs:
        if o['t'] in FRANK_AO_ATTACK and o.get('path'):
            g = o.get('label_id', o['id'])
            _FRANK_AO_PATH.setdefault(g, []).extend(_proj(cam, x, y) + (wl[g] - 1 - k,) for k, (x, y) in enumerate(o['path']))
    for g in list(_FRANK_AO_PATH): _FRANK_AO_PATH[g] = np.array(_FRANK_AO_PATH[g], float)
    P = _frank_ao_prev(objs, cam, R, state=state, ended=ended, img=img)
    groups = {}
    for o in objs:
        if o['t'] in FRANK_WAVE_TYPES: groups.setdefault(o.get('label_id', o['id']), []).append(o)
    if not groups: return P
    ids, lids = P['ids'], P['lids']
    for gid, mem in groups.items():
        if any(not o.get('ended') and not o.get('finished') for o in mem):   # the wave still travels: ONE hull over it
            ks = [lids.index(l) for l in {(BL._lid(o), -1, 'body') for o in mem} | {(gid, -1, 'body')} if l in lids]
            if ks:
                U = np.isin(ids, ks)
                if U.any():
                    H = hull_mask(U) & ((ids < 0) | U)
                    ids[H] = ks[0]; P['share'][H & ~U] = BL.CUT
            continue
        for o in mem:
            for lid in ((BL._lid(o), -1, 'body'), (o.get('label_id', o['id']), -1, 'body')):
                if lid in lids: ids[ids == lids.index(lid)] = -1
    return P
