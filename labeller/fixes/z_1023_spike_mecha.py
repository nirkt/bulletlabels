# Spike Mecha 156 attack: ball only, no chain (user Q9, 2026-10-23; trial fix out/sweep2/bugtrial/opus_override.py)
# Spike003Projectile (6000117, Cactusmecha's attack): a mace ball on a chain. Its clip (sc/effects_brawler.sc
# spike_003_atk1_projectile / _red) draws, all as body layers:
#   shape 58990  the ball (texture 0, 4096x4096)
#   shape 58409  the chain's rod (texture 0: batched into the SAME draw call as the ball)
#   shape 58408  the chain links (texture 1, 2560x4064, three links; the trail clips repeat links)
# so the label took the chain. The user (Q9, 2026-10-23): ball only. Per pixel the shape the front-most draw shows
# (uv in the atlas, _a4_shape_pixels from fixes/a4): the chain shapes are dropped; pixels whose uv falls in no shape (sprite
# edges, filtering) are kept only within EDGE px of the ball sprite; parts without ball pixels dropped; holes filled.
SPIKE_MECHA = {6000117: dict(file='sc/effects_brawler.sc', clips=('spike_003_atk1_projectile', 'spike_003_atk1_projectile_red'),
                             ball=(58990,), chain=(58408, 58409), edge=2)}


def _spike_mecha_ball(body, ctx):
    if not body.any(): return body
    from scipy.ndimage import binary_fill_holes, label as cc
    p = SPIKE_MECHA[ctx['t']]
    SP = _a4_shape_pixels(ctx['R'], p['file'], p['clips'])   # shape id + 1, 0: none of the clip's shapes
    ball = body & np.isin(SP, [s + 1 for s in p['ball']])
    if not ball.any(): return body & ~np.isin(SP, [s + 1 for s in p['chain']])
    other = body & (SP == 0) & _dilate(ball, p['edge'])   # edge pixels next to the ball
    m = ball | other
    L, n = cc(m); keep = np.unique(L[ball]); m = np.isin(L, keep[keep > 0])
    return binary_fill_holes(m)


for _t in SPIKE_MECHA: POST_BODY[_t] = _spike_mecha_ball
