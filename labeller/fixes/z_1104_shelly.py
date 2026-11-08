# installed 11-01 from out/sweep2/haikutasks/siteai2/shelly/fix.py (2nd AI sweep: Shelly dense clusters one outline per pellet with seams 364/240/341, 280 per-pellet disc)
# Shelly dense-cluster fix (AI-review sweep 2026-11-01, task shelly: skins 240, 280, 341, 364; 362 checked, no defect).
# Findings it fixes (per-skin status in shelly/report.md):
#  - 364 attack (6000004, BanditGirlStar_remake) t265-t269 and attack90 t305-t309: one merged, notched outline over several stars.
#  - 240 super (6000007, ShotgunGirlWitch) t349-t353: leaves outlined as broken pieces, some leaves missing.
#  - 341 super (6000013, ShotgunGirlBrawloween) t345-t353: many small magenta fragments instead of one solid piece per fish.
#  - 280 attack (6000010, ShotgunGirlPrincess) t265-t269 and attack90 t305: merged, notched outline over the round pellets.
# Owner rule for Shelly: no hull; each pellet keeps its own clean, solid outline. No hull is used here.
# Method A (SHF_TYPES = 6000004, 6000007, 6000013): the body of each object is split by the front-most draw it shows (idbuf
# draw id = one sprite instance). Each draw's own pixels are closed (1 px, frame resolution), holes filled, limited to the
# object's region. Then seams: along each shared edge between two different objects, the pixel of the farther one (larger wd
# depth) is left out, so each pellet shows its own outline. The seams never cut inside one object (rings seen on 280 when they did).
# Method B (6000010, 280 attack): fix2's per-pellet disc, switched on here (FX2_TYPES.add). See the end of this file.
# The hooks chain the installed POST_BODY of the same type (z_1103_shelly.py per-pellet closing) and run after it.
# Types: 6000004 = 364 attack (Shelly009), 6000007 = 240 super (Shelly003Ulti), 6000013 = 341 super (Shelly006Ulti),
# 6000010 = 280 attack (Shelly005). Other skins with these ids: none (owners.projectile: one skin each).

import numpy as _shf_np
from scipy.ndimage import binary_fill_holes as _shf_fill

SHF_TYPES = {6000004, 6000007, 6000013}      # method A (per-draw split + object seams); 6000010 uses method B (end of file)
SHF_CLOSE = 1          # px (frame resolution): closing that bridges the gaps inside one pellet's own sprite
_SHF_PREV = {t: POST_BODY.get(t) for t in SHF_TYPES}


def _shf_per_draw(body, ctx):
    t = ctx['t']; prev = _SHF_PREV.get(t)
    if prev is not None: body = prev(body, ctx)
    if not body.any(): return body
    ids_h = ctx['ids_h']; region = ctx['region']
    out = body & (ids_h <= 0)                                  # pixels with no draw id (kept as they are)
    for d in _shf_np.unique(ids_h[body]):
        if d <= 0: continue
        g = body & (ids_h == d)
        c = _close(g, SHF_CLOSE) | g
        c = _shf_fill(c) & region
        out |= c
    return out


for _t in SHF_TYPES:
    POST_BODY[_t] = _shf_per_draw


# Seams between touching pellets. The review page draws the outline of the union of an attack's labels, so two neighbouring
# pellets of one dense cluster read as one outline with no visible line between them. Along each shared edge between two
# DIFFERENT objects (pellets) of the SHF types, the pixel of the farther pellet (larger depth in the ID pass's depth buffer wd)
# is left out, so the nearer pellet keeps its full outline and each pellet shows its own outline. Seams are only drawn between
# objects, never inside one pellet (a seam inside a pellet made concentric rings, 280 t265-t309, so draws are not split here).
_shf_prev_labels = labels


def labels(objs, cam, R, state=None, ended=(), img=None):
    P = _shf_prev_labels(objs, cam, R, state=state, ended=ended, img=img)
    lab_ok = {BL._lid(o) for o in objs if o['t'] in SHF_TYPES and not o.get('finished', 0)}
    ids, share, lids = P['ids'], P['share'], P['lids']
    H, W = ids.shape
    lab_ok_k = np.array([(l[2] == 'body' and l[0] in lab_ok) for l in lids] + [False])
    k_ok = np.nonzero(lab_ok_k[:-1])[0]
    if k_ok.size < 2: return P
    own = np.isin(ids, k_ok)
    wd = R.get('wd')
    if wd is None: return P
    rm = np.zeros((H, W), bool)
    for dy, dx in ((0, 1), (1, 0)):
        Aa = (slice(0, H - dy), slice(0, W - dx)); Bb = (slice(dy, H), slice(dx, W))
        pair = own[Aa] & own[Bb] & (ids[Aa] != ids[Bb])
        if not pair.any(): continue
        da = wd[Aa]; db = wd[Bb]
        rm[Aa] |= pair & (da > db)                          # A is the farther pellet: its pixel goes
        rm[Bb] |= pair & (da <= db)                         # B is the farther pellet (or equal depth): its pixel goes
    if rm.any():
        ids[rm] = -1; share[rm] = 0
    return P


# 280 attack (ShotgunGirlPrincess, type 6000010): five round pellets in a column (t265-t269) and a row (t305-t309) were outlined
# as one merged, notched shape. The per-pellet disc of fix2 (already accepted for 341 and 349: one solid disc per pellet, the
# pellets' own drawn pixels set the centre and size) is switched on for this type too. No seams: the disc outlines already show
# one scalloped outline per pellet.
FX2_TYPES.add(6000010)
