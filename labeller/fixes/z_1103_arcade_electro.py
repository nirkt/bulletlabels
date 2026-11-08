# AI-review batch 2026-11-01 fix (Haiku fix, reviewed by the coordinator, installed 11-01 LOCAL ONLY; source out/sweep2/haikutasks/siteai/arcade_electro/fix.py)
# AI-review fixes, skins 245 / 268 / 300 / 382 (2026-11-01, task folder arcade_electro; reviewer findings from
# out/sweep2/ai/haiku_review_1101/remaining/<skin>/). Exec'd in idlabels' namespace after all installed fixes.
# Each change is keyed to explicit type ids (or the spawnable's character id). Skin 448 and skin 271 are not changed here
# (see report.md, needs_user).
#
# 1. 245 ArcadeVirus, attack 6000560 (Arcade002Projectile, clip arcade_002_projectile): the chain of shards per projectile was
#    labelled as many small outlines (AI flags: attack 0 t267, attack 90 t307). Same structure as the Arcade default attack
#    (6000551, z_1100_s041.py: 5 tiles per projectile, shape 1 = tile, shape 709 = halo, shape 0 = streak). Owner decision
#    'Arcade attack: one hull per projectile' (041, 10-28): each projectile object gets ONE convex hull of its own tiles
#    (rasterised from the game's tile geometry; halos and the streak are left out), never one hull over the whole attack.
#    Neighbouring projectiles' tiles are kept out of a hull (labels wrapper, as in z_1100_s041). Only the own export
#    (arcade_002_projectile). ArcadeVirusChroma uses the same type id.
#
# 2. 268 ArcadePiano, attack 6000562 (Arcade003Projectile): the note stream's beams fade along their length (coverage 30-125
#    above the water, >= 128 only near the notes). The label flickered with the fade: the beam column was labelled at t319-321
#    and cut to the notes at t323 (AI flag: attack 90 t323, 'too small / cut off'). Fix: the type's clip body from coverage 32
#    (the same FADE_CUT rule as Beamer's super 6001299) so the beam column stays labelled while the object is alive. The soft
#    glow is not body (class glow, unchanged). Side effect (checked on attack 0 t263-275): the row's strands are labelled more
#    completely; no sparks or impact flashes are taken in.
#
# 3. 268 super music box (spawnable, DamageBooster did 16000170): NOT changed here (needs_user). The key
#    outline pixels just outside its label are drawn in the blue spawn ring's draw (63), so a grow into the
#    model draw changes nothing at ticks 389 / 411 and a grow into draw 63 would take in the ring.

# 4. 300 ElectroSniperPlanthunter super, Belle003UltiProjectile (6000922): the crystal ring with its cyan streak is labelled
#    as a ring with a large hole (AI flags: super 45 t345, t347, t349). The reviewer's note: one convex hull (splintered /
#    streak attack of one object = one hull per shot). ONE_LABEL 'hull' for this type only. The particles around the ring are
#    not in the label (hull is built from the label's own pixels).
#
# 5. 382 ElectroSniperSCmake super, Belle005UltiProjectile (6000928): a ring with its dark centre outlined inside the label
#    (AI flag: super 45 t349, 'holes', 'should be solid'). The label is filled solid: holes of its own body are filled (the
#    outer outline does not change). Only this type; the attack (6000926, z_1026_belle382.py) is untouched.

import numpy as np

# ---- 1. 245: Arcade002 tile chain, one hull per projectile object -----------------------------------------------------
ARCT_TILE_SHAPE = 1                            # clip shape of a tile (its body; normal blend)
ARCT_TYPES = {6000560: 'arcade_002_projectile'}  # type -> its own export name (own shots)
ARCT_TILES = {}                                # object id -> its own tiles (half resolution), set by POST_BODY


def _arct_tiles(o, cam):
    """(540, 1200)/2 bool: the projectile's own tiles from the game's geometry (shape 1, alpha >= 1/2 of the tile's own alpha)"""
    import masks
    S = PD.sc(o['file']); m = np.zeros((masks.H_FULL, masks.W_FULL), bool)
    for q in o['parts'].get('clip', []):
        if q['shape'] != ARCT_TILE_SHAPE or q.get('blend') != 'normal': continue
        a = float(q['mul'][3]) / 255.0
        if a <= 0: continue
        r = masks.raster_poly(S, q, cam)
        if r is None: continue
        x0, y0, C, T = r
        mm = (1.0 - T[..., 0]) / a >= 0.5
        m[y0:y0 + mm.shape[0], x0:x0 + mm.shape[1]] |= mm
    return m[::2, ::2]


def _arct_body(body, ctx):
    o = ctx['o']
    if o.get('export') != ARCT_TYPES.get(o['t']): return body     # another clip / another type: left as it is
    m = _arct_tiles(o, ctx['cam'])
    m = m | (ctx['mine'] & _dilate(m, 2))                           # drawn tile pixels of the same projectile at the edges
    ARCT_TILES[o['id']] = m
    return m


for _t in ARCT_TYPES:
    POST_BODY[_t] = _arct_body
    ONE_LABEL[_t] = 'hull'                                           # one convex hull per projectile object

_arct_prev_labels = labels


def labels(objs, cam, R, state=None, ended=(), img=None):
    import bulletlabels as _BLa
    from scipy.ndimage import binary_dilation as _bd
    ARCT_TILES.clear()
    P2 = _arct_prev_labels(objs, cam, R, state=state, ended=ended, img=img)
    ids = P2['ids']; share = P2['share']; lids = P2['lids']
    lid_obj = {_BLa._lid(o): o['id'] for o in objs if o['t'] in ARCT_TYPES}
    T_of = {}
    for k, l in enumerate(lids):
        if l[2] == 'body' and lid_obj.get(l[0]) in ARCT_TILES:
            T_of[k] = ARCT_TILES[lid_obj[l[0]]]
    if len(T_of) > 1:
        for k, T in T_of.items():
            mine = ids == k
            if not mine.any(): continue
            others = np.zeros(mine.shape, bool)
            for k2, T2 in T_of.items():
                if k2 != k: others |= T2
            if not others.any(): continue
            ring = _bd(others, structure=np.ones((3, 3), bool))
            cut = mine & ring & ~T                     # this hull's fill beside another projectile's tiles: kept 1 px away
            if cut.any(): ids[cut] = -1; share[cut] = 0
    P2['ids'] = ids; P2['share'] = share
    return P2


# ---- 2. 268: Arcade003 stream, translucent beams counted as body --------------------------------------------------------
FADE_CUT.update({6000562: 32})


# ---- 4. 300: Belle003 super, one hull per shot (crystal ring + streak) ---------------------------------------------------
ONE_LABEL.update({6000922: 'hull'})


# ---- 5. 382: Belle005 super, solid label (holes of its own body filled) ------------------------------------------------
BELLE_SUPER_FILL = {6000928}
_bsf_prev_labels = labels


def labels(objs, cam, R, state=None, ended=(), img=None):
    from scipy.ndimage import binary_fill_holes as _bfh
    P2 = _bsf_prev_labels(objs, cam, R, state=state, ended=ended, img=img)
    ids = P2['ids']; lids = P2['lids']
    for o in objs:
        if o['t'] not in BELLE_SUPER_FILL: continue
        for k, l in enumerate(lids):
            if l[0] != o['id'] or l[2] != 'body': continue
            m = ids == k
            if not m.any(): continue
            f = _bfh(m) & ~m & (ids < 0)                 # enclosed holes of the label, not taken by another label
            if f.any(): ids[f] = k
    P2['ids'] = ids
    return P2
