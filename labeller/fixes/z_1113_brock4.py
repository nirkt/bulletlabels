# brock4 round-4 fixes (owner 2026-11-06 re-check). All global names start with _br4_.
# Pages: 175 Zeus 02 (super t363), 214 RankedProgression Lvl3 01 / 02 (zone, super rockets). 184 Ranger red 'missing label' (super t345 / 347 /
# 363 / 377): NOT reproducible on a fresh capture (every rocket of the volley has a label at those ticks, incl. the ones merged into a
# neighbour's label by z_1110_merge), no change made (see the report).
#
# 1. 175 ZEUS super (type 6000066, skin 175 only), super 45 tick 363 "holes / ragged". z_1110_zeus keeps, of the trail-sprite pixels of a
#    rocket label, only those within 2 px of the thick part of the sprite (opening by a 3.2-px disc) so the thin lightning forks of the
#    long bolt (3-6 px wide: the left fork and the zigzag branch that the white bolt sprite is drawn with) were cut away and the label
#    ended up a lopsided, notched half of the bolt. Now a connected piece of the label's trail-sprite pixels that is BIG (>= _BR4_BIG px,
#    i.e. the bolt body: ~2000 px, a spark squiggle is < 150 px) is kept whole (forks included); the zeus rule (thick part / near the
#    crystal only) still applies to all the small pieces, so squiggles stay out.
import numpy as _br4_np
from scipy.ndimage import binary_fill_holes as _br4_fill, label as _br4_cc

_BR4_T175 = 6000066
_BR4_BIG = 400
_BR4_CV = 20
_br4_zeus = POST_BODY.get(_BR4_T175)      # installed chain (z_1110_zeus on top of the older fixes)


def _br4_bolt(body, ctx):
    out = _br4_zeus(body, ctx) if _br4_zeus is not None else body
    if not out.any(): return out
    tr = _s175_trail_px(ctx) & (ctx['cv'] >= _BR4_CV)      # the type's own trail-clip sprite pixels (bolt + squiggles)
    if not tr.any(): return out
    L, n = _br4_cc(tr, structure=_br4_np.ones((3, 3)))
    if n == 0: return out
    sz = _br4_np.bincount(L.ravel())
    keep = [k for k in range(1, n + 1) if sz[k] >= _BR4_BIG and (out & (L == k)).any()]
    big = _br4_np.isin(L, keep)
    if not big.any(): return out
    res = out | big
    c = _close(res, 3) & (res | _dilate(res, 3))          # the lightning forks enclose a dark gap: close the neck, fill it
    return _br4_fill(c) | res


POST_BODY[_BR4_T175] = _br4_bolt


# 2. 214 RocketGirlRankedProgrsesionSkinLevel3 (Brock 012) burn zone BurnMainAttackPropass (17000188; no other skin uses it), page
#    214_01 marked bad by the owner (no frame flagged): "ragged / holes" look of the zone label. Measured on the fresh capture: the zone
#    was the only zone of the Brock skins that z_1107_brock does NOT cut with the map's solid blocks (its did is missing from
#    BROCK_ZONE_DID): a3_agent3 cut it by DEPTH instead, which gave the stepped, jagged edge along the crates and a detached
#    scribble inside the barrel (the zone's own beam sprite is drawn over the barrel, so those pixels stayed in: 150-300 px
#    crumb, ticks 283 / 295 / 311 / 315 / 321). Now it is handled like every other Brock burn zone: the disc (1.27 x the logic radius:
#    the drawn pink rim, measured by a3_agent3) minus the silhouettes of the solid map tiles (z_1107_brock, owner rule: cut only by
#    walls / crates / barrels), minus other labels. Names are read at call time by z_1107_brock.
BROCK_ZONE_DID.add(17000188)
BZ_ZONE_SCALE[17000188] = 1.27

# 3. 214 SUPER rockets (type 6000084 Brock012UltiProjectile = skin 214's super only; page 214_02 super 45 ticks 345 / 347 / 377 "holes / ragged",
#    page _01 marked bad). Each rocket is drawn as a grey cylinder with a blue band, a white star flash at its tail and a blue glass orb /
#    glow: the label took the sprite pixels as they come (star points as spikes, bites between the flash and the cylinder, holes). Rule
#    (owner: one clean piece in the object's own outline, filled solid, smooth edges): the label is closed (3 px: the bites), holes are
#    filled, then smoothed (Gaussian sigma 1.5 at 0.5: spikes rounded, notches filled) and never grown more than 2 px beyond the closed
#    label nor beyond the predicted region; pieces smaller than 10 % of the main piece (specks) are dropped. A rocket that is
#    partly hidden keeps what is visible (no new pixels are invented). Keyed to 6000084 only (a3_agent3 lists 6000086 = the enemy
#    colour of the same rocket in its mesh list, not touched here).
from scipy.ndimage import gaussian_filter as _br4_gauss, binary_opening as _br4_open
_BR4_T214 = 6000084
_BR4_CLOSE = 4
_BR4_SIG = 2.0
_BR4_GROW = 0
_BR4_NECK = 2
_br4_prev214 = POST_BODY.get(_BR4_T214)


def _br4_rocket214(body, ctx):
    if _br4_prev214 is not None: body = _br4_prev214(body, ctx)
    if not body.any(): return body
    c = _br4_fill(_close(body, _BR4_CLOSE))
    g = _br4_gauss(_br4_np.pad(c, 6).astype(_br4_np.float32), _BR4_SIG)[6:-6, 6:-6] >= 0.5
    out = _br4_fill(g) & (_dilate(c, _BR4_GROW) if _BR4_GROW else c)
    k = _BR4_NECK; yy, xx = _br4_np.ogrid[-k:k + 1, -k:k + 1]
    op = _br4_open(_br4_np.pad(out, k + 1), structure=(xx * xx + yy * yy) <= k * k + 1e-6)[k + 1:-(k + 1), k + 1:-(k + 1)]      # thin necks / threads (< 2 k px) go
    out = (op | (out & body)) if op.any() else out
    L, n = _br4_cc(out, structure=_br4_np.ones((3, 3)))
    if n > 1:
        sz = _br4_np.bincount(L.ravel())[1:]
        out = _br4_np.isin(L, 1 + _br4_np.nonzero(sz >= 0.1 * sz.max())[0])
    return out if out.sum() >= 0.8 * body.sum() else body


POST_BODY[_BR4_T214] = _br4_rocket214


# 4. same zone, ticks where its ring is NOT drawn: the logic zone is born at the rocket's landing (273: the ice burst is still drawn, no ring
#    sprite yet: 0 px of the zone's own sprite in the disc until 279) and lives on ~10 ticks after its ring has faded out (the ring's own
#    sprite pixels, coverage >= 20, fall from ~500 px at 315 to 9 px at 321 and 0 afterwards, while the logic zone is listed until 331): the
#    label sat on empty ground / on the hit burst. Owner rule (11-03): zones = the VISIBLE circle. So the zone's label is dropped while
#    fewer than _BR4_ZMIN px of its own ring / beam sprites (sprite_pixels of the zone, coverage >= _BR4_ZCV) are drawn inside its disc;
#    data-derived per zone item, so a second zone (the second shot) is judged on its own sprites.
_BR4_ZID = 17000188
_BR4_ZMIN = 0   # coordinator 11-06: owner rule = zone labelled as the full logic circle from birth (drop-while-not-drawn OFF)
_BR4_ZCV = 20
_br4_zl_prev = zone_labels


def zone_labels(state, cam, R, taken, own_team=None):
    out = _br4_zl_prev(state, cam, R, taken, own_team)
    for it in state.get('items', []):
        if int(it[1]) != _BR4_ZID or int(it[0]) not in out: continue
        m = out[int(it[0])][0]
        own = True if own_team is None else own_team(int(it[2]))
        C = sprite_pixels(('zone', _BR4_ZID), own, R)
        if int(((C > 0) & m & (R['cv'][::2, ::2] >= _BR4_ZCV)).sum()) < _BR4_ZMIN: out.pop(int(it[0]), None)
    return out
