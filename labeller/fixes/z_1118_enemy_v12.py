# 2026-11-08 enemy-label fix, group v12 (skins 529 530 532 545 585 625). Prefix _f12_. Installed as py/fixes/z_1118_enemy_v12.py.
# Own (blue) labels are never touched: every rule below is enemy-only (red clip shape ids / ctx['own'] False).
import numpy as _f12_np

# ---------------------------------------------------------------------------------------------------------------------------
# (585 CannonGirlChristmas) Bonnie004Projectile 6001185: z_1024_bonnie585 keeps the bell by the BLUE clip's shape ids (bell 11680,
# clapper 11677) and drops the ribbon tail (47 / 63 / 11657 / 11653). The red clip bonnie_004_atk_projectile_red is the same layout
# with other shape ids (bell 11679, clapper 11675, ribbon 11652): for an enemy shot no bell was found and the ribbon (11652) was not
# dropped, so the label ran up the ribbon. The shape-pixel map of the two Bonnie clips reports the red ids as their blue
# counterparts (blue ids are never touched: own labels pixel-identical).
_f12_MAP = {11675: 11677, 11679: 11680, 11652: 11657}
_f12_CLIPS = ('bonnie_004_atk_projectile_red',)
_f12_LUT = _f12_np.arange(65536, dtype=_f12_np.int32)
for _r, _b in _f12_MAP.items(): _f12_LUT[_r + 1] = _b + 1      # maps are shape id + 1
_f12_prev_sp = _a4_shape_pixels


def _a4_shape_pixels(R, swf, names):
    out = _f12_prev_sp(R, swf, names)
    if any(n in _f12_CLIPS for n in names if isinstance(n, str)) and out.size:
        out = _f12_LUT[out]
    return out


# ---------------------------------------------------------------------------------------------------------------------------
# (530 DoorManDetective) DoorMan002Projectile 6001326 (attack 0 / 90, first tick of a shot): the enemy shot has a big pink puff at the
# shooter's muzzle (an additive launch burst of the red run, ~350-750 px) that is drawn with the same atlas shape as the clip's own tiny
# round part (clip shape 19). The sprite-code map therefore calls its pixels body (code 1, inside the type's search region) and the label
# stretched from the bullet back over the whole puff (label area 2-3x the blue label). The blue run has no such puff in the label (its
# cyan puffs are other sprites). The tracker's predicted bullet (ctx['pred_body']) lines up with the drawn bullet in both runs (the
# blue body is always inside its 4 px dilation), so for an enemy shot the body is cut to the 5 px dilated convex hull of the prediction.
# Enemy objects only (ctx['own'] False), type 6001326 only.
from scipy.ndimage import binary_dilation as _f12_dil
_f12_DM = 6001326
_f12_DM_K = 5


def _f12_dm_make(prev):
    def post(body, ctx):
        if not ctx.get('own') and body.any():
            pb = ctx['pred_body']
            if pb.any():
                body = body & _f12_dil(hull_mask(pb), iterations=_f12_DM_K)
        return prev(body, ctx) if prev is not None else body
    return post


if not getattr(POST_BODY.get(_f12_DM), '_f12', False):
    _f12_f = _f12_dm_make(POST_BODY.get(_f12_DM)); _f12_f._f12 = True; POST_BODY[_f12_DM] = _f12_f


# ---------------------------------------------------------------------------------------------------------------------------
# (532 SniperMariposa) Piper006Projectile 6000379 (butterfly, attack 0 / 90): the clip's soft glow ellipse (clip shape 0, sprite code 1) is
# drawn about twice as opaque in the red art (ID-pass coverage median 90-140, vs 40-50 in the blue run), so a large part of it reaches the
# CUT (128) and the label became a rounded blob 2-3x the blue label (wings + red halo); in the blue run only ~20-40 px of it (the core under
# the butterfly) pass. For an enemy shot the glow shape's pixels are taken out of the body (the butterfly's own shapes 8016-8019 stay); the ones
# the butterfly encloses (hidden under the glow in the ID pass) come back as holes of the rest. Enemy objects only, type 6000379 only.
from scipy.ndimage import binary_fill_holes as _f12_fill
_f12_PI = 6000379
_f12_PI_FILE = 'sc/effects_brawler.sc'
_f12_PI_CLIPS = ('piper_006_atk_projectile', 'piper_006_atk_projectile_red')
_f12_PI_GLOW = 0                       # clip shape id of the soft glow ellipse (+1 in the shape map)


def _f12_pi_make(prev):
    def post(body, ctx):
        if not ctx.get('own') and body.any():
            try: SPm = _a4_shape_pixels(ctx['R'], _f12_PI_FILE, _f12_PI_CLIPS)
            except Exception: SPm = None
            if SPm is not None and SPm.shape == body.shape:
                keep = body & (SPm != _f12_PI_GLOW + 1)
                if keep.sum() >= 0.3 * body.sum():
                    body = keep | (_f12_fill(keep) & body)
        return prev(body, ctx) if prev is not None else body
    return post


if not getattr(POST_BODY.get(_f12_PI), '_f12', False):
    _f12_f = _f12_pi_make(POST_BODY.get(_f12_PI)); _f12_f._f12 = True; POST_BODY[_f12_PI] = _f12_f


# ---------------------------------------------------------------------------------------------------------------------------
# (545 MaisieWasteland) Maisie003Projectile 6001332 (crystal fist, attack 0 / 90, first drawn tick of a shot): the RED shot is covered by a
# big additive muzzle burst (draws of no sprite of the fist: the same launch burst as 051 / 466 / 603). The ID-pass body keeps only two
# exposed slivers (88-103 px, 8-30 % of the predicted fist, vs 240-270 px in the blue run, where the prediction is 96 % the label). Same
# hidden-part rule as z_1109_enemy_pl (enemy only): when under half of the predicted body is in the body, the predicted body is added (cut
# to the region); the installed crest rule (z_1106_misc) then runs on the completed body. Own labels and later ticks are untouched.
_f12_MA = 6001332


def _f12_ma_make(prev):
    def post(body, ctx):
        if not ctx.get('own'):
            pb = ctx['pred_body']
            if pb is not None and pb.any() and (body & pb).sum() < 0.5 * pb.sum():
                body = body | (pb & ctx['region'])
        return prev(body, ctx) if prev is not None else body
    return post


if not getattr(POST_BODY.get(_f12_MA), '_f12', False):
    _f12_f = _f12_ma_make(POST_BODY.get(_f12_MA)); _f12_f._f12 = True; POST_BODY[_f12_MA] = _f12_f


# ---------------------------------------------------------------------------------------------------------------------------
# (529 BarrelBotCupcake) Darryl007Projectile 6000433 (the pearl train, attack 0 / 90, first drawn tick(s) of a shot): the RED shot is covered
# by a big opaque white launch burst (draws of no sprite of the pearls). Only slivers of the pearls stay visible: 25-46 % of each pearl's
# predicted body is in the ID-pass body at the first tick (blue run: 78-97 %), so the enemy label lacks the hidden pearls (first tick 442-737 px vs 1008-1099
# in the blue run; the checker's verdict flips between SAME and DIFFERENT with the burst's size from capture to capture). Same hidden-part rule
# as z_1109_enemy_pl / 545 above (enemy only): when under 70 % of an object's predicted body is in the body, the predicted body is added (cut
# to the region), before the installed purple-core / hole rules. Own labels and enemy ticks where the pearls are visible are untouched.
_f12_DA = 6000433


def _f12_da_make(prev):
    def post(body, ctx):
        if not ctx.get('own'):
            pb = ctx['pred_body']
            if pb is not None and pb.any() and (body & pb).sum() < 0.7 * pb.sum():
                body = body | (pb & ctx['region'])
        return prev(body, ctx) if prev is not None else body
    return post


if not getattr(POST_BODY.get(_f12_DA), '_f12', False):
    _f12_f = _f12_da_make(POST_BODY.get(_f12_DA)); _f12_f._f12 = True; POST_BODY[_f12_DA] = _f12_f


# ---------------------------------------------------------------------------------------------------------------------------
# (625 LuchadorMHA) Primo014Projectile 6000410 (flaming fist, attack 0 / 90): z_1116_enemy_grpF's pink test (b >= 120, r >= 150, b >= g - 30) for
# the red art takes in, besides the fist, the pink / magenta / orange stripes of the striped arm (the 3D model is arm + fist) that runs back from
# the fist: the blue test is cyan, which the blue arm's white / blue / yellow stripes never pass. The enemy label was the fist with ragged arm
# pieces behind it (area 1.4-2x the blue label; a second label on the next, hidden fist's arm). The blue rule (z_1027_s625 / z_1104) opens the body
# with a wide disk so the thin stripes go. For an enemy shot the result of the installed chain is opened with a disk of 6 px (stripes are <= ~8 px
# wide, the fist is 25+ px), the pieces under 30 % of the largest kept, the rounded edge restored from the chain's own pixels within 3 px and
# holes filled. A label lying under 70 % inside its object's predicted body (the freshly spawned second fist, hidden behind the first: only arm
# stripes pass the pink test, the blue run has no label there) is dropped. Enemy objects only, type 6000410 only.
from scipy.ndimage import binary_closing as _f12_close, binary_opening as _f12_open, label as _f12_cc
_f12_PR = 6000410
_f12_PR_OPEN = 6


def _f12_disk(k):
    yy, xx = _f12_np.mgrid[-k:k + 1, -k:k + 1]
    return (xx ** 2 + yy ** 2) <= k * k


def _f12_pr_make(prev):
    def post(body, ctx):
        res = prev(body, ctx) if prev is not None else body
        if ctx.get('own') or not res.any(): return res
        p = 12
        B = _f12_np.pad(res, p)
        core = _f12_open(B, structure=_f12_disk(_f12_PR_OPEN))
        if core.sum() < 0.3 * res.sum(): return res
        L, n = _f12_cc(core)
        if n > 1:
            sz = _f12_np.bincount(L.ravel())[1:]; core = _f12_np.isin(L, 1 + _f12_np.nonzero(sz >= 0.3 * sz.max())[0])
        core = core | (B & _f12_dil(core, structure=_f12_disk(3)) & _f12_open(B, structure=_f12_disk(2)))
        core = _f12_fill(_f12_close(core, structure=_f12_disk(3)))
        out = core[p:-p, p:-p] & ctx['region']
        # a freshly spawned second fist (hidden behind the first, predicted body 1/6 of a normal one): the pink test only finds the first fist's arm
        # stripes (under 70 % of the label inside the predicted body; every real fist label is 99 % inside it): no fist, no label (the blue run has none)
        pb = ctx['pred_body']
        if pb is not None and pb.any() and out.any() and (out & pb).sum() < 0.7 * out.sum(): return _f12_np.zeros_like(out)
        return out
    return post


if not getattr(POST_BODY.get(_f12_PR), '_f12', False):
    _f12_f = _f12_pr_make(POST_BODY.get(_f12_PR)); _f12_f._f12 = True; POST_BODY[_f12_PR] = _f12_f
