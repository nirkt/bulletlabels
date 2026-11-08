# away_1104 / brock (final-review round, 2026-11-04): RocketGirl (Brock) skins 175 179 180 184 186 187 188 205. Owner complaints, binding.
# Names are prefixed _ab4_brock_. Runs after z_1107_brock.py (it replaces nothing there, it changes its data / wraps its hooks).
#
# 1. ZONE TOO SMALL (owner: 175 Zeus 'zone circle too small', 179 Brown, 180 Virus; 175 page 2 / Rangers: the zone again).
#    The burn zone of every Brock skin is drawn with the same ring effect (brock_def_sp_fire_ground, measured by z_1027_s175 at 1.13
#    to 1.17 logic radii; the Ranger skins already use 1.15, owner 11-02 'just label the visible circle'). z_1107_brock draws the
#    plain 1.0 circle for 175 / 179 / 180 / 205: the label ends inside the ring. Now every Brock zone is 1.15 x the logic radius
#    (z_1107_brock's BZ_ZONE_SCALE is read at call time): still cut by walls / crates / barrels exactly as there.
_AB4B_ZONES = (17000179, 17000180, 17000181, 17000182, 17000183, 17000184, 17000185, 17000186, 17000187)
for _d in _AB4B_ZONES: BZ_ZONE_SCALE[_d] = 1.15

# 2. SUPER ROCKETS CUT (owner, 'missing label' at super ticks 347 / 363 / 377 of 184 / 186 / 187 / 188 / 205, 'label the super
#    bolts' on 175). The rocket label only holds the clip sprites the ID pass shows, the long white-green cylinder / the blue body of
#    the rocket is a 3D / translucent part and was left out (measured: label 251 px for a predicted body of 545 px). Hidden-parts rule:
#    the label keeps the full shape. The predicted body (aligned to the drawn sprites, as the idlabels body does) is added where
#    nothing opaque is in front of it (no 3D character model in front), when the prediction agrees with what is labelled (>= 70 % of the label
#    inside it) and the label holds less than 75 % of the predicted body (rockets that are already whole stay as they are). Then gaps closed and holes filled. Types: 6000066 (175), 6000070 (180), 6000076-80 (184-188), 6000082 (205).
_AB4B_ROCKETS = (6000066, 6000070, 6000076, 6000077, 6000078, 6000079, 6000080, 6000082)


def _ab4_brock_rocket(prev):
    def f(body, ctx):
        from scipy.ndimage import binary_fill_holes as _fill
        if prev is not None: body = prev(body, ctx)
        pb = ctx['pred_body']
        if not pb.any(): return body
        import os as _os
        _dbg = _os.environ.get('AB4DBG')
        o = ctx['o']; R = ctx['R']; cam = ctx['cam']; C = ctx['C']
        mine = body | ctx['mine']
        q = align(pb, mine, C, ctx['region']) if mine.any() else pb
        if body.any() and (body & q).sum() < 0.7 * body.sum(): return body
        if (body & q).sum() >= 0.75 * q.sum(): return body       # the label already holds (nearly) the whole predicted body
        # not where a 3D character model (the brawler, the rangers) is the front-most draw: those hide the rocket for real
        ids_h = ctx['ids_h']
        mdl = [d for d, dr in R['draws'].items() if dr.get('th') == MODEL_TEX]
        vis = ~np.isin(ids_h, mdl) if mdl else np.ones(body.shape, bool)
        if _dbg: print('AB4', ctx['o']['id'], 'body', int(body.sum()), 'q', int(q.sum()), 'body&q', int((body & q).sum()))
        add = q & vis & ~body
        if not add.any(): return body
        out = body | add
        out = _fill(_close(out, 2)) & (out | _dilate(q, 2))
        return out | body
    return f


for _t in _AB4B_ROCKETS:
    POST_BODY[_t] = _ab4_brock_rocket(POST_BODY.get(_t))


# 3. 205 SteamPunk ATTACK (type 6000081, attack 0 deg, ticks 265 / 267 / 269 / 271: owner 'holes / ragged'). The rocket's label has a
#    notched, spiky flame, a thin gap between the brass collar and the cylinder (two pieces) and chewed edges. Rule: one clean solid
#    piece in the rocket's own outline. The body is closed by 3 px (the gap between its parts), holes filled, smoothed (Gaussian
#    sigma 1.5, threshold 0.5: notches filled, spurs rounded), and never grown more than 3 px beyond the original label nor beyond
#    the predicted region. Keyed to the attack type 6000081 only.
_AB4B_ATT205 = 6000081


def _ab4_brock_smooth(prev):
    def f(body, ctx):
        from scipy.ndimage import binary_fill_holes as _fill, gaussian_filter as _gauss
        if prev is not None: body = prev(body, ctx)
        if not body.any(): return body
        reg = _dilate(ctx['pred_any'], 4) if ctx['pred_any'].any() else _dilate(body, 4)
        c = _fill(_close(body, 3))
        g = _gauss(np.pad(c, 5).astype(np.float32), 1.5)[5:-5, 5:-5] >= 0.5
        out = _fill(g) & _dilate(body, 3) & reg
        return out if out.any() else body
    return f


POST_BODY[_AB4B_ATT205] = _ab4_brock_smooth(POST_BODY.get(_AB4B_ATT205))


# 4. 175 ZEUS SUPER BOLTS (type 6000066; owner: 'label the super bolts', page 2 'super bolt body still NOT labelled'). The super is a
#    volley of lightning shards (cyan blade with a yellow core) and one long white bolt that strikes down to the impact. The label only
#    held the yellow cores and the top of the long bolt: the blades are translucent trail-body sprites (sprite code 3, coverage 26-44,
#    below the 128 cut) and the long bolt's body is trail-over sprites (code 4) farther than 10 px from the prediction (measured at
#    t363: 737 of 2191 bolt pixels labelled). Rule: the pixels of the type's own sprites (codes 3 and 4, coverage >= 20) that touch
#    the label (connected through 2 px gaps) are part of the bolt / shard, and so are the opaque pixels of a small 128 x 128 sprite draw
#    touching the label (the shard's blade, no sprite code) in the same connected piece; isolated spark squiggles that touch no label stay out. The
#    impact flash and the blue ring are other draws (code 0 / 1 / 2) and stay out. Then holes filled.
_AB4B_BOLT = 6000066
_AB4B_BOLT_CV = 20


def _ab4_brock_bolt(prev):
    def f(body, ctx):
        from scipy.ndimage import label as _cc, binary_fill_holes as _fill, binary_opening as _opening
        if prev is not None: body = prev(body, ctx)
        if not body.any(): return body
        C = ctx['C']; cv = ctx['cv']; R = ctx['R']; ids = ctx['ids_h']; img = ctx['img']
        cand = ((C == 3) | (C == 4)) & (cv >= _AB4B_BOLT_CV)
        # the blade of a shard is its own small sprite draw (128 x 128 texture, drawn opaque, no sprite code): a draw that touches
        # the label and is only a few hundred px in the whole frame is part of the shard (never the ground / world draws, which are huge)
        own_d = [int(d) for d in np.unique(ids[_dilate(body, 1) & (ids > 0)])]
        for d in own_d:
            dr = R['draws'].get(d)
            if dr is None or dr.get('tw') != 128 or dr.get('tht') != 128: continue
            m = ids == d
            if m.sum() > 2500: continue
            cand = cand | (m & (cv >= CUT) & (C == 0))
        L, n = _cc(_close(cand | body, 2) & (cand | body | _dilate(body, 2)), np.ones((3, 3)))
        keep = np.unique(L[body & (L > 0)]); keep = keep[keep > 0]
        add = np.isin(L, keep) & cand
        add = _opening(add, np.ones((3, 3), bool))                       # thin necks (<= 2 px) to detached spark squiggles go
        L2, n2 = _cc(add | _dilate(body, 1), np.ones((3, 3)))
        keep2 = np.unique(L2[body & (L2 > 0)]); keep2 = keep2[keep2 > 0]
        add = add & np.isin(L2, keep2)
        out = body | add
        return _fill(out) | out
    return f


POST_BODY[_AB4B_BOLT] = _ab4_brock_bolt(POST_BODY.get(_AB4B_BOLT))
