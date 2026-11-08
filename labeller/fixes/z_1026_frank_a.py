# Frank skins 035 (default; type shared with Gold / Silver / Xmas / Vacation), 220 Caveman (+ CavemanRed), 339 Hindu: Frank rule, one hull per wave while it lives, leftovers out (Haiku task 39a, reviewed 10-26; coordinator added the density rule for scattered remnants)
# Task 39a (2026-10-26): Frank skins 035 FrankDefault, 220 FrankCaveman, 339 HammerDudeHindu. User's Frank decision (the same as
# Frank 321 / Frank 242, z_1023_frank321.py and z_1023_frank242.py): ONE hull per wave while it lives, for the attack AND the
# super; no per-spike / per-rock labels, no holes, not carved by objects in front; leftovers after the wave ended are
# unlabelled (group label dropped once every member has ended; z_1023_frank242.py drops on any member, changed here, see report).
#
# Types (projtab, from the page headers of sw2_035 / sw2_220 / sw2_339 and skin_confs.csv Projectiles columns):
#   035 FrankDefault:    attack 6000475 HammerDudeProjectile, super 6000476 HammerDudeUltiProjectile
#   220 FrankCaveman:    attack 6000485 Frank007Projectile,   super 6000486 Frank007UltiProjectile
#   339 HammerDudeHindu: attack 6000481 Frank005Projectile,   super 6000482 Frank005UltiProjectile
# SHARED with other skins (keyed to type ids, as asked): 6000475 / 6000476 are the default HammerDude projectile (also
# HammerDudeGold, HammerDudeSilver, HammerDudeXmas, HammerDudeVacation); 6000485 / 6000486 are also FrankCavemanRed's.
#
# Shapes (sc/effects_brawler.sc, per pixel with _a4_shape_pixels: shape id + 1; checked with col.py / ana.py / renders):
#   6000475 (035 attack):   61 = the spike / streak body (bright). Dropped: 1 (dark ground marks); dark copies of 61 go by DARK_DROP.
#   6000476 (035 super):    61 = the spike body. Dropped: 567 and 689-694 (yellow shock sparks: rule 3, sparks are not labelled;
#                            kept, the hull spread over the smoke at tick 373), 0, 1 (dark ground marks).
#   6000485 (220 attack):   61 = blue-white ice spikes, 884 / 886 = the cyan electric body, 6671 = cyan-white highlights.
#                            Dropped: 62, 63, 0, 2777, 140, 6655 (cream bones and skulls, leftovers), 407 (brown pebbles),
#                            1 (dark ground marks), 64 and 7022 (small, not body).
#   6000486 (220 super):    61 = cream spike cones, 28721 / 28722 = the yellow flame body. Dropped: 884 / 886 (cyan flecks,
#                            as on Frank 321's super), 64064-64076 (dark copies), 1, 3688, 59292.
#   6000481 (339 attack):   42772, 42787, 42790 = brown rock bodies, 42794, 42798 = their green moss tops, 785 = shaded rock
#                            sides. Dropped: 622, 625, 729 (dark chips and green specks), 1.
#   6000482 (339 super):    42772, 42786, 42787, 42789, 42790, 42792 = rock bodies, 42794 / 42798 green tops, 64065 = the
#                            yellow-orange rock body, 1 = dark rock sides. Dropped: 64069, 64070, 729 (dark copies, specks).
# No-shape edge pixels are kept only within 2 px of a kept shape.
FRANK39A_KEEP = {6000475: (61,),
                 6000476: (61,),
                 6000485: (61, 884, 886, 6671),
                 6000486: (61, 28721, 28722),
                 6000481: (785, 42772, 42787, 42790, 42794, 42798),
                 6000482: (1, 42772, 42786, 42787, 42789, 42790, 42792, 42794, 42798, 64065)}
# ENEMY (red-tinted) shots (task red1028, 2026-10-28; installed as py/fixes/z_1026_frank_a.py, same name): the other team's super of 339 is drawn
# with the SAME clips (the team only tints them) but its rocks are the orange body 64066 ("yellow-orange rock" family) with no green tops
# (shapes 42772 / 42786 / 42789 / 42792 do not occur), so the own keep list kept only the dark sides (1), 42787 / 42790: at tick 369 (en_339)
# 2981 kept px in a 20552 px hull = 0.145 < FRANK39A_DENSE and the whole wave lost its label (own 369: 4458 / 21963 = 0.20, labelled).
# Enemy keep = own keep + the orange rock body 64066 (measured 2612 px of 5601 at en_339 t369, rocks are fully orange; 64066 is also drawn in the own
# clip but is NOT kept there, so the own result stays pixel-identical). Only types listed here are changed, other skins' lists are untouched.
FRANK39A_KEEP_RED = {6000482: FRANK39A_KEEP[6000482] + (64066,)}
# Enemy only: scattered specks of the kept shapes far from the rocks (en_339 t369: 3 px at the water, a 23 px spark on the turret, 11 px from the rocks) would stretch the hull;
# kept pieces under RED_SPECK px that lie farther than RED_FAR px from every piece of at least RED_BIG px are dropped (own shots untouched).
FRANK39A_RED_SPECK = 40; FRANK39A_RED_BIG = 100; FRANK39A_RED_FAR = 8    # px, half resolution
FRANK39A_FILE = 'sc/effects_brawler.sc'
FRANK39A_DENSE = 0.15      # body / hull below this with a large hull: scattered remnants (measured: live waves >= 0.17, remnants <= 0.104)
FRANK39A_DENSE_HULL = 2000  # px: smaller hulls (a wave's first frame) are never dropped by the density rule
FRANK39A_STRAGGLER = 300   # px (half resolution): a wave with an ended member and less kept body than this is leftovers (bones, skulls), not labelled


def _f39a_clean(body, ctx):
    """body pixels of the kept shapes only (and no-shape edge pixels touching them, 2 px)"""
    if not body.any(): return body
    t = ctx['t']; T = projtab.table()[t]
    names = tuple(sorted({n for f, n in _trail_clips(t) if f == FRANK39A_FILE} | {n for n in (T['blue'], T['red']) if n}))
    SP = _a4_shape_pixels(ctx['R'], FRANK39A_FILE, names)   # shape id + 1 per pixel, 0: none
    ids = FRANK39A_KEEP[t] if ctx.get('own', True) else FRANK39A_KEEP_RED.get(t, FRANK39A_KEEP[t])   # enemy shots: see FRANK39A_KEEP_RED
    keep = np.isin(SP, [s + 1 for s in ids]) & body
    out = keep | (body & (SP == 0) & _dilate(keep, 2))
    if not ctx.get('own', True) and t in FRANK39A_KEEP_RED and out.any():   # enemy: drop far-off specks (see FRANK39A_RED_SPECK)
        from scipy.ndimage import label as _cc39, distance_transform_edt as _edt39
        L, n = _cc39(out, structure=np.ones((3, 3), bool)); sz = np.bincount(L.ravel()); sz[0] = 0
        big = np.isin(L, np.nonzero(sz >= FRANK39A_RED_BIG)[0])
        if big.any():
            dist = _edt39(~big)
            for i in np.nonzero((sz > 0) & (sz < FRANK39A_RED_SPECK))[0]:
                m = L == i
                if dist[m].min() > FRANK39A_RED_FAR: out[m] = False
    return out


for _t in FRANK39A_KEEP: POST_BODY[_t] = _f39a_clean


# ONE label per wave: the labels of a wave group (its members share one label id) are one convex hull over all their
# cleaned pixels, not one hull per member. A group whose members have all ended (the wave's logic is over) gets no label;
# while one member lives the wave lives (base labels() rule), so its hull stays.
_f39a_labels_orig = labels


def labels(*args, **kwargs):
    import bulletlabels as BL
    P = _f39a_labels_orig(*args, **kwargs)
    objs = args[0] if args else kwargs['objs']
    R = args[2] if len(args) > 2 else kwargs['R']
    lids, ids, share = P['lids'], P['ids'], P['share']
    cv = R['cv'][::2, ::2]
    groups = {}
    for o in objs:
        if o['t'] in FRANK39A_KEEP: groups.setdefault(o.get('label_id', o['id']), []).append(o)
    for gid, mem in groups.items():
        key = (gid, -1, 'body')
        if key not in lids: continue
        k = lids.index(key); cur = ids == k
        if all(o.get('ended') for o in mem):   # the wave has ended: every member's logic projectile is over
            ids[cur] = -1; share[cur] = 0; continue
        if any(o.get('ended') for o in mem) and cur.sum() < FRANK39A_STRAGGLER:   # a member ended and little of the wave is left
            ids[cur] = -1; share[cur] = 0; continue                              # (bones / skulls around a finished wave)
        if not cur.any(): continue
        hm = hull_mask(cur); dens = cur.sum() / max(1, hm.sum())
        if dens < FRANK39A_DENSE and hm.sum() > FRANK39A_DENSE_HULL:   # only scattered remnants left (coordinator 10-26): the hull would be mostly empty ground
            ids[cur] = -1; share[cur] = 0; continue
        h = hm & (ids < 0)
        ids[h] = k; share[h] = np.maximum(cv[h] / 255.0, BL.CUT)
    return P
