# installed 11-02 from out/sweep2/haikutasks/siteai4/misc/fix.py (owner rulings: 215 propeller+bowl pod, 545 crest shape 61 in, 572 orbs only / no chain; single-skin types)
# misc owner ruling (2026-11-02, RULINGS_1102.md, wally_bowl) -- skin 215 WallyWindstock, super 45: Sprout007UltiProjectile
# (type 6000765). The green glass bowl under the propeller IS part of the object: the label is the propeller plus the bowl as
# ONE solid pod (no glow halo outside the bowl's rim, no holes). Exec'd in idlabels' namespace after the installed fixes
# (IDLABELS_OVERRIDE); the installed propeller hook (z_1103_misc_a closing) runs first and is kept; z_1104 is NOT re-enabled.
# Measured on sw2_215 ticks 343-361: the bowl's rim is clip shapes 1 (blue bottom) and 2 (cyan ring) of
# sprout_007_ulti_projectile (effects_brawler_sprout.sc), translucent (coverage 32-127). Their pixels (coverage >= 32, of this
# object's own draw set) fit one circle: centre moves with the flight, radius 19.3 px on every tick (median residual 0.1 px),
# matching the outer rim. The bowl is therefore the filled circle of that fit (radius + 1 px for the rim's outer edge);
# the propeller's own label is unioned with it, closed and hole-filled, largest piece kept.
# Shape 8 (the glass and the blades' green) and shape 14 (the blades) are NOT used for the fit: they are shared with the
# propeller and the bowl's interior is translucent. Only one object of this type is drawn per shot (no other skin uses it).
import numpy as _np_b
from scipy.ndimage import label as _cc_b, binary_fill_holes as _bfh_b, binary_closing as _bc_b

_BOWL_TYPE = 6000765
_BOWL_CLIP_FILE = 'sc/effects_brawler_sprout.sc'
_BOWL_CLIPS = ('sprout_007_ulti_projectile', 'sprout_007_ulti_projectile_red')
_BOWL_RIM_SHAPES = (1, 2)       # clip shapes of the bowl's rim (blue bottom, cyan ring); +1 in the shape map
_BOWL_RIM_COV = 32              # coverage of the translucent rim pixels
_BOWL_EDGE = 1.0                # px added to the fitted rim centre-line radius (the outer edge of the rim)
_BOWL_MIN_RIM = 40              # rim pixels needed for a fit
_prev_bowl_post = POST_BODY.get(_BOWL_TYPE)


def _bowl_fit_circle(x, y):
    """least-squares (Kasa) circle through the points: cx, cy, r"""
    A = _np_b.c_[x, y, _np_b.ones(len(x))]; b = x * x + y * y
    a, bb, c = _np_b.linalg.lstsq(A, b, rcond=None)[0]
    cx, cy = a / 2.0, bb / 2.0
    return cx, cy, float(_np_b.sqrt(max(c + cx * cx + cy * cy, 0.0)))


def _fix_6000765_bowl(body, ctx):
    if _prev_bowl_post is not None: body = _prev_bowl_post(body, ctx)
    R = ctx['R']; cv = ctx['cv']; region = ctx['region']
    SP = _a4_shape_pixels(R, _BOWL_CLIP_FILE, _BOWL_CLIPS)          # shape id + 1 per pixel, 0: none
    rim = _np_b.isin(SP, [s + 1 for s in _BOWL_RIM_SHAPES]) & (cv >= _BOWL_RIM_COV) & region
    ys, xs = _np_b.nonzero(rim)
    if len(xs) < _BOWL_MIN_RIM: return body                          # no bowl in view: the propeller only
    cx, cy, r = _bowl_fit_circle(xs.astype(float), ys.astype(float))
    res = _np_b.hypot(xs - cx, ys - cy) - r
    inl = _np_b.abs(res) < 2.0                                        # refit on the rim's own inliers
    if inl.sum() < _BOWL_MIN_RIM: return body
    cx, cy, r = _bowl_fit_circle(xs[inl].astype(float), ys[inl].astype(float))
    if not (12.0 <= r <= 26.0): return body                          # not the bowl's size: unchanged
    H, W = body.shape
    yy, xx = _np_b.mgrid[0:H, 0:W]
    disc = _np_b.hypot(xx - cx, yy - cy) <= r + _BOWL_EDGE
    m = body | disc
    m = _bc_b(_np_b.pad(m, 2), structure=_np_b.ones((3, 3), bool))[2:-2, 2:-2] | m   # gaps between blades and bowl
    m = _bfh_b(m)
    L, n = _cc_b(m, structure=_np_b.ones((3, 3), bool))
    if n > 1:
        sz = _np_b.bincount(L.ravel()); sz[0] = 0; m = L == sz.argmax()
    return m


POST_BODY[_BOWL_TYPE] = _fix_6000765_bowl


# ---- owner ruling maisie_crest (2026-11-02): skin 545 MaisieWasteland, attack 90: Maisie003Projectile (6001332), the crystal.
# The bright cyan crest / spike above the crystal IS part of the crystal: its label takes the crest. Checked on sw2_545 ticks
# 309-317: the crest is clip shape 61 of maisie_003_atk_projectile (cyan, coverage ~120-250, 78-175 px). The vertical beam
# above it is NOT shape 61 (it is a trail sprite, no label, unchanged); the soft blue haze (shape 0) stays out. The crest
# pixels are added where they lie within 5 px of the crystal's label (their own parts, not the beam), then closed by 3 px
# (one piece), holes filled, largest piece.
_MAISIE003 = 6001332
_M3_FILE = 'sc/effects_brawler.sc'
_M3_CLIPS = ('maisie_003_atk_projectile', 'maisie_003_atk_projectile_red')
_M3_CREST = 61                  # clip shape of the bright cyan crest (+1 in the shape map)
_M3_CREST_COV = 32              # coverage of the crest pixels (the crest is partly translucent at its tips)
_M3_CREST_MIN = 10              # px: crest parts smaller than this are specks / nubs (the crest's tip at tick 309 is 23 px)
_prev_m3 = POST_BODY.get(_MAISIE003)


def _fix_6001332_crest(body, ctx):
    if _prev_m3 is not None: body = _prev_m3(body, ctx)
    if not body.any(): return body
    SP = _a4_shape_pixels(ctx['R'], _M3_FILE, _M3_CLIPS)
    crest = (SP == _M3_CREST + 1) & (ctx['cv'] >= _M3_CREST_COV) & ctx['region']
    if not crest.any(): return body
    L, n = _cc_b(crest, structure=_np_b.ones((3, 3), bool))
    near = _dilate(body, 5)                                          # the crest's tip may sit 4-5 px above the crystal's label
    keep = _np_b.unique(L[near & crest]); keep = keep[keep > 0]      # crest parts next to the crystal's label
    sz = _np_b.bincount(L.ravel()); keep = keep[sz[keep] >= _M3_CREST_MIN]   # small bits (specks, nubs) are not the crest
    add = crest & _np_b.isin(L, keep)
    if not add.any(): return body
    m = _close(body | add, 3)                                        # the gap between the spike and the crystal: one piece
    m = _bfh_b(m | body)
    L2, n2 = _cc_b(m, structure=_np_b.ones((3, 3), bool))
    if n2 > 1:
        sz = _np_b.bincount(L2.ravel()); sz[0] = 0; m = L2 == sz.argmax()
    return m


POST_BODY[_MAISIE003] = _fix_6001332_crest


# ---- owner ruling cooker_chain (2026-11-02): skin 572 CookerOnceupon, Pearl003Projectile (6001367), the pearl orbs. NO chain
# label: the orbs only. Measured on sw2_572 (attack 90 ticks 305-317, attack 0 ticks 263-281): the orbs are clip shapes 47, 61,
# 140 and 19846 of pearl_003_atk_projectile (bright cyan discs with an orange rim, coverage 190-250). The chain links and the
# purple halo around each orb are shape 0 (translucent, coverage ~55), the magenta specks are shape 3688 (sparkles in the halo):
# none of these is labelled. The label is the orbs' own pixels (opaque), holes filled, one piece per orb; pieces under 12 px
# are dropped (specks). Same type on the attack 0 row (the chain of orbs there too): no chain label on either row.
# Type 6001367 is keyed here (no other skin's owner is checked; any other Cooker skin that draws Pearl003 gets the same rule).
_PEARL003 = 6001367
_P3_FILE = 'sc/effects_brawler.sc'
_P3_CLIPS = ('pearl_003_atk_projectile', 'pearl_003_atk_projectile_red')
_P3_ORB = (47, 61, 140, 19846)   # the orb's own clip shapes (+1 in the shape map)
_P3_MIN = 12                     # px: pieces smaller than this are specks
_prev_p3 = POST_BODY.get(_PEARL003)


def _fix_6001367_orbs(body, ctx):
    if _prev_p3 is not None: body = _prev_p3(body, ctx)
    R = ctx['R']; cv = ctx['cv']
    SP = _a4_shape_pixels(R, _P3_FILE, _P3_CLIPS)
    orb = _np_b.isin(SP, [s + 1 for s in _P3_ORB]) & (cv >= CUT) & ctx['region']
    if not orb.any(): return body & False
    m = _bfh_b(_bc_b(_np_b.pad(orb, 1), structure=_np_b.ones((3, 3), bool))[1:-1, 1:-1] | orb)
    L, n = _cc_b(m, structure=_np_b.ones((3, 3), bool))
    if n == 0: return body
    sz = _np_b.bincount(L.ravel())[1:]
    return _np_b.isin(L, [k + 1 for k in range(n) if sz[k] >= _P3_MIN])


POST_BODY[_PEARL003] = _fix_6001367_orbs
