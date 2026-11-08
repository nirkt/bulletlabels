# Emz (Mummy) attack, EVERY skin: ONE label per ATTACK in the clouds' own shape, filled (the user, 2026-10-24: "one hull per attack", then: the cloud shape is what they wanted): an Emz attack spawns 5
# cloud projectiles on the same tick (sw2_044: 2000010-14 at tick 264); their claimed pixels are merged (holes filled) and given to
# the attack's lowest object id. Types 6000683-6000689 (one per skin) and the overcharged attacks 6001488 / 6001765.
# Skin 536 (6000686) also keeps Haiku task 26's body rule (blue rim in, twinkles out). Earlier version (per stack, skin 536
# only) kept at out/sweep2/coord_1024/z_1024_mummy536_stack.py.
# Emz Mummy Harajuku 536 (Emz004Projectile 6000686, skin 536 only): ONE hull per stack of overlapping clouds incl. blue rim (user 10-24; Haiku task 26 redo, reviewed)
# Task 26 (redo, 2026-10-24), skin 536 MummyHarajuku: Emz004Projectile (type 6000686), the hairspray cloud of attack 0 deg
# (ticks 263-287) and 90 deg (303-327). User decision (binding, 2026-10-24): ONE label per STACK of overlapping clouds. Clouds
# whose bodies touch or overlap form one stack, labelled as ONE convex hull over the stack's clouds (each cloud's whole visible
# body incl. its blue outer rim; no holes). A cloud that touches no other keeps its own hull. The yellow twinkle stars and the
# soft see-through glow outside the cloud stay out (inside a hull they are covered: fine). Blobs after the projectiles ended
# (ticks 287, 327) stay unlabelled (their clips are hidden: no body, no label). The stack's pixels go to one of its label ids
# (the lowest object id); the other ids of the stack are empty by the user's decision.
#
# Shapes of the clip (sc/effects_brawler.sc, emz_004_atk_projectile), by the atlas shape of the front-most draw:
#   cloud body: the puffs, face and bright rim 140, 143, 3688, 7442, 8152-8163, 19780, 24771, 38316-38318 and 63 (the cloud's
#               own layer; the trail's gas puffs share 63 but lie outside the clouds: only connected body counts)
#   0: the clip's glow layer. Its OPAQUE part is the cloud's blue outer rim (coverage >= EMZ_RIM_CV, 1-4 px outside the body,
#      connected to it: measured on tick 275, scratch t26/s2/rimstat.py); the translucent haze beyond it (coverage 24-63,
#      5-12 px out) is not the cloud
#   137: the yellow twinkle stars (on the face and free outside the cloud): not body
EMZ_TYPE = 6000686
EMZ_CLOUD = {140, 143, 3688, 7442, 19780, 24771, 38316, 38317, 38318, 63} | set(range(8152, 8164))
EMZ_GLOW = 0            # shape id of the clip's glow layer (its opaque outer rim is body)
EMZ_TWINKLE = 137
EMZ_RIM_CV = 64         # coverage (0..255) from which a glow-layer pixel is the opaque blue rim
EMZ_RIM_R = 4           # px: the rim grows from the body at most this far (connected)
EMZ_SPECK = 0.1         # parts smaller than this share of the main part of a cloud are dropped
EMZ_BODY = {}           # object id -> the cloud's body after the per-type rules (set by POST_BODY in the current frame)
EMZ_STACKS = []         # per frame: (carrier id, member ids, hull px, px of the stack's label, px claimed by other labels inside)


def _emz_body(body, ctx):
    o = ctx['o']
    if not body.any():
        EMZ_BODY[o['id']] = body; return body
    T = projtab.table()[EMZ_TYPE]
    names = (T['blue'], T['red']) + tuple(n for f, n in _trail_clips(EMZ_TYPE) if f == T['file'])
    SP = _a4_shape_pixels(ctx['R'], T['file'], names)     # shape id + 1 per pixel, 0: none (frame resolution)
    sid = SP - 1
    keep = (SP == 0) | np.isin(sid, list(EMZ_CLOUD | {EMZ_GLOW}))
    m = body & keep & ~((SP > 0) & (sid == EMZ_TWINKLE))  # the twinkle stars are not the cloud
    rim = (SP > 0) & (sid == EMZ_GLOW) & (ctx['cv'] >= EMZ_RIM_CV)
    for _ in range(EMZ_RIM_R): m = m | (_dilate(m, 1) & rim)   # the opaque blue outer rim, connected to the body
    if m.any():
        from scipy.ndimage import label as cc
        L, n = cc(m, structure=np.ones((3, 3), bool))
        if n > 1:
            sz = np.bincount(L.ravel())[1:]; m = np.isin(L, 1 + np.nonzero(sz >= EMZ_SPECK * sz.max())[0])
    EMZ_BODY[o['id']] = m
    return m


POST_BODY[EMZ_TYPE] = _emz_body      # no ONE_LABEL for this type: the hull is taken per stack in labels() below

_emz_prev_labels = labels


EMZ_ATTACK = {6000683, 6000684, 6000685, 6000686, 6000687, 6000688, 6000689, 6001488, 6001765}
from scipy.ndimage import binary_fill_holes
EMZ_CLOSE = 2           # px: gaps between the attack's clouds closed before filling
EMZ_SPECK_ATK = 0.05    # parts of an attack's union smaller than this share of its largest part are dropped before the hull


def labels(objs, cam, R, state=None, ended=(), img=None):
    import bulletlabels as BL
    from scipy.ndimage import label as cc
    EMZ_BODY.clear(); EMZ_STACKS.clear()
    P = _emz_prev_labels(objs, cam, R, state=state, ended=ended, img=img)
    if not any(o['t'] in EMZ_ATTACK for o in objs): return P
    ids, lids, share = P['ids'], P['lids'], P['share']
    groups = {}
    for o in objs:   # the clouds of one attack: same type, same owner, first shown on the same frame
        if o['t'] not in EMZ_ATTACK: continue
        lid = (BL._lid(o), -1, 'body')
        if lid not in lids: continue
        k = lids.index(lid); m = ids == k
        if m.any(): groups.setdefault((o['t'], o.get('owner'), o.get('first')), []).append((o['id'], k, m))
    if not groups: return P
    # solid map blocks in front (walls, crates: z_1023_zone_solid's map test) are not the cloud: the hull does not take them.
    # The floor is not excluded (excluding every opaque world pixel, the floor included, undid the hull: sw2_044)
    try: world = _zs_solid(cam, R)
    except Exception: world = np.zeros(ids.shape, bool)
    for key, mem in sorted(groups.items(), key=lambda kv: min(i for i, _, _ in kv[1])):
        mem = sorted(mem); ks = [k for _, k, _ in mem]; carrier = ks[0]
        U = np.zeros(ids.shape, bool)
        for _, _, m in mem: U |= m
        L, n = cc(U, structure=np.ones((3, 3), bool))
        if n > 1:
            sz = np.bincount(L.ravel())[1:]; U = np.isin(L, 1 + np.nonzero(sz >= EMZ_SPECK_ATK * sz.max())[0])
        # the attack's clouds merged, in their own cloud shape (the user, 10-24: 'i dont mind if its the cloud shape, thats
        # exactly what i wanted'), holes filled; not a convex hull
        H = binary_fill_holes(_close(U, EMZ_CLOSE)) & (_dilate(U, EMZ_CLOSE) | U)
        mine = np.isin(ids, ks)
        take = H & (((ids < 0) & ~world) | mine)
        ids[mine & ~take] = -1                             # specks of the attack outside its hull
        new = take & ~mine
        ids[take] = carrier; share[new] = BL.CUT
        EMZ_STACKS.append((mem[0][0], tuple(i for i, _, _ in mem), int(H.sum()), int(take.sum()), 0))
    return P
