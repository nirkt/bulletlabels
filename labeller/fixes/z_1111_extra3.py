# extra3 fix (round 3, 2026-11-05): pages the OWNER marked bad: 429 BarkeepBrawlAcademy, 444 InsectManBrawlentine, 452 Mender199Gem.
# Exec'd in idlabels' namespace (IDLABELS_OVERRIDE). Unique prefix _e3_.
import numpy as _e3_np
from scipy.ndimage import label as _e3_cc, binary_dilation as _e3_dil

# ---- 429 BarkeepBrawlAcademy (Barley009, super bottles = Barley009UltiProjectile 6000265, only this skin uses it) ----
# Owner (page 429_02, super 45 deg tick 369): "some extra things in there". The bottles' body label also takes pieces of the
# zone's own artwork drawn from the same atlas (the eye emblem / pupil, rune sparks, glass bits: 7-90 px pieces, none of them
# touching the bottle). They are small outlines inside the puddle zone, and they punch holes in the zone label (zone = full circle).
# Fix: body pieces that do not touch the object's predicted bottle body (grown 3 px) are dropped; the zone circle then takes
# those pixels back (zone labels are built after the body labels, with 'taken').
_E3_T429 = 6000265
_e3_prev_429 = POST_BODY.get(_E3_T429)


def _e3_429(body, ctx):
    b = _e3_prev_429(body, ctx) if _e3_prev_429 else body
    if not b.any(): return b
    pb = ctx['pred_body']
    if not pb.any(): return b
    L, n = _e3_cc(b)
    if n <= 1: return b
    keep = _e3_np.unique(L[_e3_dil(pb, iterations=3) & b]); keep = keep[keep > 0]
    return _e3_np.isin(L, keep)


POST_BODY[_E3_T429] = _e3_429


# ---- 444 InsectManBrawlentine (Angelo004Projectile 6001436, the heart arrow; only this skin's type) ----
# Owner (page 444_01): "some effects are labeled". At the shot's first ticks (attack 0: t265; attack 90: t305, also t303-307) the
# arrow's body label also covers the brawler's CAST effect: the two big white-blue wings (sprite shape 19 of the arrow clip, the
# same atlas shape as the yellow lobe beside the stem) and the white heart between them (a hole in that label). They are not the
# arrow (the arrow = fletching heart + stem + head heart). The wing pieces are separate pieces of the body that do not touch the
# arrow's predicted body, so: body pieces with under 30 % of their pixels inside the object's predicted body are dropped.
_E3_T444 = 6001436
_e3_prev_444 = POST_BODY.get(_E3_T444)


def _e3_444(body, ctx):
    b = _e3_prev_444(body, ctx) if _e3_prev_444 else body
    if not b.any(): return b
    pb = ctx['pred_body']
    if not pb.any(): return b
    L, n = _e3_cc(b)
    if n <= 1: return b
    sz = _e3_np.bincount(L.ravel(), minlength=n + 1); inpb = _e3_np.bincount(L[pb].ravel(), minlength=n + 1)
    keep = [i for i in range(1, n + 1) if inpb[i] >= 0.3 * sz[i]]   # the wings: under 5 % inside the prediction
    big = max([sz[i] for i in keep] or [0])
    keep = _e3_np.array([i for i in keep if sz[i] >= 40 or sz[i] * 10 >= big], int)   # specks (< 40 px, under a tenth of the main piece)
    return _e3_np.isin(L, keep)


POST_BODY[_E3_T444] = _e3_444


# ---- 452 Mender199Gem (Mender002Projectile 6001758, the water jet; only this skin uses it) ----
# Owner (page 452_01): "attacks wrong". Checked on the capture (both attack rows), what was wrong:
#  (1) the two attack rows were labelled differently: attack 0 (a flat, horizontal shot) got the installed z_1104 treatment (dust
#      pixels near the jet join it, one hull per shot) but attack 90 (a vertical shot) did not (its _horizontal() test failed), so
#      the same jet was one hull in one row and a jet outline plus separate outlines on every spray droplet in the other;
#  (2) the spray left after the jet is gone was still labelled, as a few drops at the wall / the ground (attack 0 t273-275, attack 90
#      t313-317, incl. the splash 'flower' on the ground at maximum range): those are the end animation (rule: ended attacks and
#      their leftovers are not labelled), they only look like a bullet because the jet's dust sprites are reused for the splash.
# Fix: (1) _horizontal() is made always true for this type's helpers (the check only meant 'the beam, not the other shots'; the beam
# test itself is the dust-pixel count _MC452_BEAM, orientation-free), so every shot is one hull of its jet + spray, as attack 0 was
# (the hull still never takes the brawler / world); (2) the jet label lives only while the jet is flying: the object's age (ticks
# since its first frame) up to _E3_452_LIFE ticks (jet + spray of the shot: attack 0 t265-271, attack 90 t305-311); the leftovers
# after that carry no label. Age = the tracker's path length - 2 (one path point per tick, independent of which frames are labelled).
_E3_T452 = 6001758
_E3_452_LIFE = 6


# (1) NOT INSTALLED (coordinator 11-05): the _horizontal() override (one big hull for attack 90 too) waits for the owner's answer.


_e3_prev_452 = POST_BODY.get(_E3_T452)


def _e3_452(body, ctx):
    age = len(ctx['o'].get('path') or []) - 2        # ticks since the shot's first frame (the tracker's path: one point per tick)
    if age > _E3_452_LIFE: return _e3_np.zeros_like(body)
    return _e3_prev_452(body, ctx) if _e3_prev_452 else body


POST_BODY[_E3_T452] = _e3_452


# ---- 452 Mender199Gem super (Mender002UltiProjectile 6001759, only this skin) ----
# Owner (page 452_01): "attacks wrong". Super, t355: the label of the second shard (the tick its logic projectile has just landed,
# o['finished'] > 0, while the first shard's landing wave is drawn next to it) reaches up through that wave (the big white-blue wave,
# a different effect drawn from the same atlas shapes 20 / 26 as the shard's glow): a zig-zag strip about 30 px above the shard.
# Fix: for a landed shard, body pixels more than _E3_452U_NEAR px from its predicted body are dropped (the label then follows the
# shard: it matches the predicted body within a few px on landed shards), the rest is opened by 2 px (the neck to the dropped
# part goes) and kept. Flying shards are not touched (their streak lies up to 20 px behind the predicted body).
from scipy.ndimage import distance_transform_edt as _e3_edt, binary_opening as _e3_open
_E3_T452U = 6001759
_E3_452U_NEAR = 6
_e3_prev_452u = POST_BODY.get(_E3_T452U)


def _e3_452u(body, ctx):
    b = _e3_prev_452u(body, ctx) if _e3_prev_452u else body
    pb = ctx['pred_body']
    if not b.any() or not pb.any() or not (ctx['o'].get('finished') or 0) > 0: return b
    far = _e3_edt(~pb) > _E3_452U_NEAR
    if not (b & far).any(): return b
    c = b & ~far
    o2 = _e3_open(c, iterations=2)
    return c & _e3_dil(o2, iterations=2) if o2.any() else c


POST_BODY[_E3_T452U] = _e3_452u
