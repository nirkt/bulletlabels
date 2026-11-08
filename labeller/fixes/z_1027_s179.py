# sweep 10-27 fix for skin 179 (Sonnet revision, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s179/)
# Skin 179 RocketGirlBrown (sweep 2026-10-27, findings 3120 = tick 363, 3121 = tick 377; 2nd attempt after the reviewer
# rejected fix.py: its 3 px prediction window left smoke specks / spurs at ticks 365, 367, 373, 377).
# Brock008UltiProjectile (type 6000068, Brock's super rockets; clip brock_008_ulti_projectile of sc/effects_brawler.sc).
# The rocket's own sprites are a fixed set of shapes (the flying chick rocket: 146, 635, 2883, 14791, 14792, 14794 and the
# small 1); the explosion smoke / fire puffs of the rockets that already hit are drawn with the same clip's shape 63 (big puff),
# shape 0 (ember) and 47 (spark), and the rocket's label took them as body (up to 1000 px of smoke around a 150-400 px rocket).
# Rules 3/4: smoke, explosions and embers are never labelled. Fix (per object, POST_BODY), shape-based, not prediction-based:
#  1. core = body pixels of the rocket shapes; its connected parts (bridged 2 px) are the rocket (several parts when smoke
#     or a crate hides the middle of it). Only parts within ULTI2_NEAR px of the logic prediction (ctx['pred_body']) are this
#     rocket's (the body can hold a neighbouring rocket's pixels); when no part of at least ULTI2_MIN px is that near (the
#     prediction lags the drawn rocket, tick 375) all parts count. Parts under ULTI2_MIN px or under ULTI2_FRAC of the
#     largest are dropped (debris crumbs).
#  2. the label = body within ULTI2_RIM px of the kept core (sprite edges, the few shape-63 / none pixels of the rocket's own
#     exhaust), closed and filled solid, nothing else: smoke further away is out, no spurs.
#  3. if the body has no core at all (nothing but smoke / ember), the label is empty (the rocket is gone, only its end effect).
ULTI2_SHAPES = {6000068: dict(file='sc/effects_brawler.sc',
                              clips=('brock_008_ulti_projectile', 'brock_008_ulti_projectile_red'),
                              core=(1, 146, 635, 2883, 14791, 14792, 14794))}
ULTI2_FRAC = 0.3
ULTI2_NEAR = 10   # px: a core part touching the prediction's neighbourhood belongs to this rocket
ULTI2_RIM = 2
ULTI2_MIN = 8


def _brock_ulti2(body, ctx):
    from scipy.ndimage import label as _cc, binary_fill_holes as _fill
    if not body.any(): return body
    p = ULTI2_SHAPES[ctx['t']]
    SP = _a4_shape_pixels(ctx['R'], p['file'], p['clips']) - 1       # shape id, -1: none
    core = body & np.isin(SP, p['core'])
    if not core.any(): return body & False
    L, n = _cc(_dilate(core, 2))
    sz = np.array([(core & (L == k)).sum() for k in range(1, n + 1)])
    pb = ctx['pred_body']                  # which parts are THIS rocket (neighbouring rockets share the body's draws)
    near = np.zeros(n, bool)
    if pb.any():
        win = _dilate(pb, ULTI2_NEAR)
        near = np.array([(core & (L == k) & win).any() for k in range(1, n + 1)])
    if not (near & (sz >= ULTI2_MIN)).any(): near[:] = True   # prediction lags / rocket hidden (tick 375): any part
    sz = np.where(near, sz, 0)
    keep = (sz >= ULTI2_MIN) & (sz >= ULTI2_FRAC * sz.max())
    kc = core & np.isin(L, 1 + np.nonzero(keep)[0])
    b = body & _dilate(kc, ULTI2_RIM)
    b = b | (_close(b, 2) & _dilate(kc, 3))                        # solid: gaps between the parts of one rocket
    b = _fill(b) | b
    L8, m8 = _cc(b, structure=np.ones((3, 3)))                      # no specks: a part under 6 px beside a bigger one goes
    if m8 > 1:
        z8 = np.bincount(L8.ravel())[1:]
        b = np.isin(L8, 1 + np.nonzero((z8 >= 6) | (z8 == z8.max()))[0])
    return b


for _t in ULTI2_SHAPES: POST_BODY[_t] = _brock_ulti2
