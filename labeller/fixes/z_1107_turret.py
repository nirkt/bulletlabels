# installed 11-02 from out/sweep2/haikutasks/siteai5/turret/fix.py (owner ruling turret: whole model for SpawnerDudeTurret 16000173 on 045 291 369 439; footprint clip of z_1100_s045 switched off)
# OWNER RULING 2026-11-02, second batch (RULINGS_1102.md, "turret"): the Spawner turret (SpawnerDudeTurret, did 16000173, the
# super's turret of SpawnerDude skins 045 / 291 / 369 / 439) is labelled as its WHOLE MODEL, not as the 120-unit ground circle.
# Binding and implemented exactly; the constants below are data-derived (type / character ids, the game's model texture, depth).
#
# What it replaces: z_1100_s045.py clipped the turret's spawn label to the ellipse of its collision circle (the footprint). That
# cut off the barbell (439), the upper part of the hex (291) and the flower / top parts; this file switches that clip off
# (S045_TURRET is read at call time and set to -1, no character has that id) and labels the turret's own model instead.
#
# The turret's model is one 3D model draw (texture MODEL_TEX, the same kind of draw as all characters) per frame, e.g.
#   045: draw of the dumbbell + dark slab; 439: the barbell (purple plates, silver bar) + slab (+ the spinning top at 411);
#   291: the dark-red hexagon (+ the red spike on top at 411); 369: the flower.
# Measured on the four kept captures (sw2_045, 291, 369, 439; ticks 363-411): the turret's model draw is owned by the turret from
# tick 363 (its growth) to 411, and its screen draws do not change identity.
#
# Ownership: spawn_labels gives a model draw to the NEAREST character (competition). A draw belongs to the turret when its centroid
# lies within TURRET_OWN px of the turret's own ground-to-top segment (projected); the other characters' draws (heroes, walls) lie
# 100+ px away (measured: hero draws 107-119 px, lavender walls 185+ px), so no competition is needed. This fixes the 439 barbell
# (the purple plates were the turret's model and could fall to another character).
#
# Decal (291): the orange ground hexagon at the lower right is in the SAME model draw as the hexagon (draw 70, 873 px in one
# blob). Rule: an orange blob of the model draw, at least TURRET_DECAL_MIN px, whose median depth lies TURRET_DECAL (w units)
# behind the turret's ground plane (R['wd'] against the fitted ground depth, as the zone labels) is cut, with a TURRET_DECAL_RIM
# px rim. Measured: the blob is at +0.36 median at ticks 371 and 411; at tick 399 the same colour is the box model's gold side
# at ground depth and is NOT cut (the blob test needs both colour and depth). The hex and its gold dots stay (213-900 px kept).
# Not done: the 045 ground shadow (dark, +0.12..+0.18 behind the plane, bottom right of the ring) stays in the label: a dark-blob
# cut took a chord through the ring, so it is disabled (TURRET_DARK_V = -1). The black bump on top of the 045 dome at 411 stays too.
# Crumbs left by the cut are dropped (parts under TURRET_KEEP of the largest).
#
# Hidden parts: the model's holes (enclosed interior, the barbell's gap under the slab) are filled inside the free pixels.
# Free = not labelled by a projectile (taken pixels stay theirs). The HUD health bar with its number (sprite draws at the top of
# the turret, textures 6649 / 0a31 / 7a98, elevated) is NOT a turret part: only MODEL_TEX draws are used.
#
# Types / characters touched: character 16000173 (SpawnerDudeTurret) only. Other skins: none (its characters are only the turrets).
# Other labels are not changed: only the turret's own spawn label pixels are rewritten (checked on ticks 363-411 of all four skins).

import numpy as _t5_np
from scipy.ndimage import binary_fill_holes as _t5_fill, label as _t5_cc, binary_dilation as _t5_dil

TURRET_DECAL_MIN = 100             # px: a colour blob (orange or dark) of the model draw at least this big can be a decal / shadow
TURRET_DARK_V = -1                 # dark-blob cut disabled: it cut a chord through 045 ring (see report); -1 = orange only
TURRET_DECAL_RIM = 1               # px: the decal's black rim next to the blob is cut too
TURRET_DID = 16000173              # SpawnerDudeTurret (characters.csv)
S045_TURRET = -1                   # z_1100_s045's footprint clip switched off (its check reads this constant at call time)
TURRET_OWN = 60.0                  # px: a model draw this close to the turret's ground-to-top segment is the turret's
TURRET_DECAL = 0.12                # w units (wd - ground): pixels farther than this behind the ground plane are a decal, not the model
TURRET_KEEP = 0.15                 # parts smaller than this share of the largest are crumbs
TURRET_GROUND_R = 120.0            # the turret's collision radius (logic units) for the ground-plane fit
TURRET_MIN_DRAW = 40               # px: model draws smaller than this are not the turret


def _t5_disc(r):
    yy, xx = _t5_np.mgrid[-r:r + 1, -r:r + 1]
    return (xx * xx + yy * yy) <= r * r


def _t5_turret_body(cid, x, y, z, cam, R, ids_h, cv, free, img):
    """{pixels} of the turret's own model: MODEL_TEX draws whose centroid is within TURRET_OWN of its segment, opaque, free,
    not behind its ground plane (decal), the largest parts, holes filled inside the free pixels"""
    a = _t5_np.array(_proj(cam, x, y, z)); b = _t5_np.array(_proj(cam, x, y, z + CHAR_TOP)); ab = b - a
    owned = []
    for d, dr in R['draws'].items():
        if dr.get('th') != MODEL_TEX: continue
        m = ids_h == d
        if m.sum() < TURRET_MIN_DRAW: continue
        ys, xs = _t5_np.nonzero(m); p = _t5_np.array([xs.mean(), ys.mean()])
        t = _t5_np.clip(_t5_np.dot(p - a, ab) / max(1e-6, _t5_np.dot(ab, ab)), 0, 1)
        if _t5_np.linalg.norm(a + t * ab - p) <= TURRET_OWN: owned.append(d)
    if not owned: return None
    m = _t5_np.isin(ids_h, owned) & (cv >= CUT) & free
    if not m.any(): return m
    wd = R.get('wd')
    if wd is not None:
        gd = ground_depth(cam, x, y, TURRET_GROUND_R, R['meta'].get('wref', 0.0))
        dif = wd - gd
        if img is not None:
            # the ground decal (orange hexagon of 291) is a separate orange blob in the same draw, far behind the ground plane
            # (median wd - ground >= TURRET_DECAL). Its blob (plus a 2 px rim) is cut; the dark hex and its gold dots stay
            r_, g_, b_ = img[..., 0].astype(int), img[..., 1].astype(int), img[..., 2].astype(int)
            # colour classes of a decal / shadow: orange (291's hexagon) or very dark (045's ground shadow, 0.12-0.18 behind the plane)
            cls = m & (((r_ > g_ + 40) & (g_ > b_ + 30)) | (img.max(-1).astype(int) < TURRET_DARK_V))
            L0, n0 = _t5_cc(cls); sz0 = _t5_np.bincount(L0.ravel())
            dec = _t5_np.zeros_like(m)
            for k in range(1, n0 + 1):
                if sz0[k] >= TURRET_DECAL_MIN and _t5_np.median(dif[L0 == k]) >= TURRET_DECAL: dec |= (L0 == k)
            if dec.any(): m &= ~_t5_dil(dec, structure=_t5_disc(TURRET_DECAL_RIM))
    if not m.any(): return m
    L, n = _t5_cc(m); sz = _t5_np.bincount(L.ravel())[1:]
    keep = [k + 1 for k in range(n) if sz[k] >= TURRET_KEEP * sz.max()]
    m = _t5_np.isin(L, keep)
    return _t5_fill(m) & free


_t5_prev_labels = labels   # the labels chain before this file (z_1100_s045's clip is off, see S045_TURRET)


def labels(objs, cam, R, state=None, ended=(), img=None):
    P = _t5_prev_labels(objs, cam, R, state=state, ended=ended, img=img)
    if state is None: return P
    import bulletlabels as BL, owners
    ids, lids, share = P['ids'], list(P['lids']), P['share']
    ids_h = R['id'][::2, ::2]; cv = R['cv'][::2, ::2]
    for c in state['chars']:
        cid, did, own = int(c[0]), int(c[1]), int(c[2])
        if did != TURRET_DID or own < 0: continue
        x, y, z = int(c[3]), int(c[4]), int(c[5])
        ks = [k for k, l in enumerate(lids) if l[0] == cid and l[2] == 'spawn']
        k = ks[0] if ks else None
        # free: unlabelled, or the turret's own label (projectile labels keep their pixels)
        free = (ids < 0) | (ids == k) if k is not None else (ids < 0)
        m = _t5_turret_body(cid, x, y, z, cam, R, ids_h, cv, free, img)
        if m is None: continue
        if k is None:
            lids.append((cid, -1, 'spawn')); k = len(lids) - 1
            nm = owners.character(did)['name']
            P.setdefault('meta', {})[cid] = dict(kind='spawnable', brawler=owners.spawnable_owner(nm), skin=None, name=nm)
        old = ids == k
        ids[old & ~m] = -1; share[old & ~m] = 0
        ids[m] = k; share[m] = _t5_np.maximum(cv[m] / 255.0, BL.CUT)
    P['ids'], P['share'], P['lids'] = ids, share, lids
    return P
