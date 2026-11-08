# Label-fix agent 7 overrides (2026-10-20): KickerDude, Silencer, Duelist, Chronomancer, SpawnerDude, Knight (Ash),
# StickyBomb, SnakeOil, Roller skins. Exec'd in idlabels' namespace after load.

# Fang 003 (Octo, page 524): the crystal body is drawn translucent (coverage spread 32..255 over the sprite); with CUT 128
# its label had a big notch in the visibly drawn crystal
FADE_CUT[6001107] = 32
# Fang 005 (BaseballRed, page 561): the whole boot is drawn translucent (coverage at most ~190, mostly 32..190); with CUT
# 128 only its bright sole crescent and a sliver were labelled
FADE_CUT[6001111] = 32


def _a7_keep_near_pred(body, ctx, r=3, frac=0.0, draw_in=0.0):
    """the label's connected parts that touch the predicted body (grown r px); parts elsewhere in its search region are
    other draws sharing its sprites (a neighbour projectile's hit / end animation). frac > 0: also parts smaller than
    frac of the largest kept part are dropped (detached flame flicks / sparks). draw_in > 0: first, draws (sprites) of
    which less than this share lies in the predicted body are dropped (an ended shot's hit sprite touching it)"""
    from scipy.ndimage import label as cc
    pb = ctx['pred_body']
    if not pb.any() or not body.any(): return body
    if draw_in > 0:
        ih = ctx["ids_h"]
        for d in np.unique(ih[body]):
            m = body & (ih == d)
            if (m & pb).sum() < draw_in * m.sum(): body = body & ~m
        if not body.any(): return body
    L, n = cc(body, structure=np.ones((3, 3), bool)); keep = np.unique(L[_dilate(pb, r) & body]); keep = keep[keep > 0]
    if frac > 0 and keep.size:
        sz = np.bincount(L.ravel()); keep = keep[sz[keep] >= frac * sz[keep].max()]
    return np.isin(L, keep)


# Otis (Silencer) attack, every skin with its hit sprites in the projectile's atlas: the hit burst / splash of a
# neighbouring shot that has ended (Otis006 Strikers page 458: the cyan burst at the wall, tick 271; splash bits on the
# water, tick 319) fell in the live shot's search region and joined its label
for _t in (6001211, 6001205):   # Otis006 (458), Otis004 Worldfinals (345: splash on the water, tick 319)
    POST_BODY[_t] = _a7_keep_near_pred
# Otis005 (Demon, page 386): as above (the blue-flame splash on the water joined the last shot's label, tick 319), and its
# detached little flame flicks beside the fireball (sparks, ticks 269, 309: parts under 40% of the largest) and the hit
# ring-eye of an ended shot touching it (tick 319: its draws lie outside the predicted body) dropped
POST_BODY[6001208] = lambda body, ctx: _a7_keep_near_pred(body, ctx, frac=0.4, draw_in=0.15)
# Cordelius003 (Duelist Wasteland, page 335): the hit burst (cyan flames) of the ended shot shares the attack's atlas and
# joined the next shot's label (ticks 271-273 at the dummy, 311-313 below the shot)
POST_BODY[6001346] = lambda body, ctx: _a7_keep_near_pred(body, ctx, draw_in=0.15)
# Cordelius005 (Duelist Toon, page 384): the ended shot's leftover (its blue hit blob, clip sprites) joined the next
# shot's label (tick 311)
POST_BODY[6001350] = lambda body, ctx: _a7_keep_near_pred(body, ctx, draw_in=0.15)


def _a7_cordelius002_atk(body, ctx):
    """Cordelius002 (Duelist Brain, page 332) attack: a neuron (bright core, thin arms, end nodes; all drawn at full
    brightness) with a dim DNA helix behind it (one translucent clip sprite, frame brightness 200-248) and a dark spot
    under it (~100). The helix is the shot's trail look (rule 3) and the dark spot a shadow: draws whose median frame
    brightness (max of R, G, B) is below 250 are dropped; then the parts touching the prediction. The arms and nodes are
    small separate draws whose sprites are not in the clip's shape set: added when near the body (an arm's node is
    ~11 px from the core; the trail dots ~40 px behind). Then ONE_LABEL hull"""
    if ctx['img'] is None or not body.any(): return body
    from scipy.ndimage import label as cc
    ih = ctx['ids_h']; v = ctx['img'].max(-1); drop = []
    for d in np.unique(ih[body]):
        m = body & (ih == d)
        if np.median(v[m]) < 250: body = body & ~m; drop.append(d)
    body = _a7_keep_near_pred(body, ctx, draw_in=0.15)
    if body.any():
        # the arms / nodes: small opaque draws (<= 150 px each in the frame; sprites outside the clip's shape set) of
        # bright cyan (median green and blue >= 200) within 14 px of the body
        img = ctx['img'].astype(np.int16); cvh = ctx['cv']; near = _dilate(body, 14) & ctx['region']
        tot = np.bincount(ih.ravel()); own = set(np.unique(ih[body]).tolist())
        for d in np.unique(ih[near & ~body]):
            if d <= 0 or d in own or d in drop or tot[d] > 150: continue
            m = near & (ih == d)
            if np.median(cvh[ih == d]) >= 128 and np.median(img[..., 1][m]) >= 200 and np.median(img[..., 2][m]) >= 200:
                body = body | m
    return body


POST_BODY[6001344] = _a7_cordelius002_atk
# Cordelius002 super: an atom, thin orbit rings around a core: one hull label (rule 5) over the rings and core only (the
# twinkles and dots beside it dropped first: parts not touching the prediction, draws mostly outside it)
POST_BODY[6001345] = lambda body, ctx: _a7_keep_near_pred(body, ctx, draw_in=0.15)
ONE_LABEL.update({6001344: 'hull', 6001345: 'hull'})
# Cordelius002 attack: its arms and nodes are drawn translucent (coverage down to ~20) as well
FADE_CUT[6001344] = 16
# Chronomancer003 (StarrGirls, page 211) secondary shard: specks of the ended shots' shattering crystals (hit effect,
# same atlas) joined its label (tick 273)
POST_BODY[6001615] = lambda body, ctx: _a7_keep_near_pred(body, ctx, draw_in=0.15)
