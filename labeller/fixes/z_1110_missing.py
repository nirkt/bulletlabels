# away_1105 / group "missing" (final-review round 2, 2026-11-05): owner-marked BAD pages with missing / too small labels.
# One file, one section per page. Exec'd after all installed fixes (IDLABELS_OVERRIDE). Prefix of every global: _ms_
import numpy as _ms_np
import bulletlabels as _ms_BL


# ===== page 015 BanditGirlDefault, row "super 45 deg", ticks 349 / 351 "too small / cut off", 353 "missing label" =====
# Type 6000001 (ShotgunGirlUltiProjectile, Shelly's default super; every skin that uses the default super projectile takes it).
# Measured on the fresh capture hull_missing_015 (spawnfix host): one shot = 9 objects x 4 ball slots, every slot a visible ball.
# The installed z_1109_misc (_aw4m_s015_body) puts a disc (radius S017_R) on every slot of a live object. Two things are left:
#  (a) 349 / 351: some discs are bitten (obj 30 slots -1 / 2 / 3: 10-68 % covered). The cause is its depth test
#      "wd >= dz - S017_DEPTH": the depth buffer holds a plateau 0.16-0.20 nearer than the ball (value -0.95 vs ball -0.794) under the
#      lower-left balls of the shot although nothing stands in front of them (the front-most draw there is the ball's own sprite,
#      dmask 0). Rule now: the depth test only removes pixels whose FRONT-MOST draw writes depth (dmask 1: world geometry, a wall /
#      crate drawn in front); effect draws (smoke, the ball's own sprite, dmask 0) never cut a ball.
#  (b) 353: objects whose logic is 'finished' (hit the crate / wall) but which the tracker still draws (not hidden: their balls
#      are still flying and fully visible, the owner flags them as missing) get no label because of the 'finished' rule. Rule now:
#      an object keeps its discs for as long as its prediction exists (= the tracker still draws it); once the tracker hides it
#      (355+: only the splash is left) it has no prediction and no label, as before.
_ms_S015 = {6000001}
_ms_dm_cache = {}


def _ms_worldpx(R):
    k = id(R['id'])
    if k in _ms_dm_cache: return _ms_dm_cache[k]
    ids = R['id'][::2, ::2]
    dm = _ms_np.zeros(int(ids.max()) + 1, bool)
    for d, r in R['draws'].items():
        if d < len(dm): dm[d] = bool(r.get('dmask', 0))
    _ms_dm_cache.clear(); _ms_dm_cache[k] = dm[ids]
    return _ms_dm_cache[k]


def _ms_s015_body(body, ctx):
    o = ctx['o']; R = ctx['R']; cam = ctx['cam']
    if not ctx['pred_body'].any(): return body
    Q = _ms_BL.labels([o], cam, cut=0.5)
    qids, qlids = Q['ids'], Q['lids']
    H, W = body.shape
    yy, xx = _ms_np.mgrid[0:H, 0:W]
    out = _ms_np.zeros((H, W), bool); n = 0
    for k, l in enumerate(qlids):
        if l[0] != o['id'] or l[2] != 'body': continue
        M = qids == k
        if int(M.sum()) < S017_MIN: continue
        ys, xs = _ms_np.nonzero(M)
        out |= (xx - xs.mean()) ** 2 + (yy - ys.mean()) ** 2 <= S017_R ** 2; n += 1
    if not n: return body
    wd = R.get('wd')
    if wd is not None and o.get('anchor') is not None:
        dz = _eye_depth(cam, o['anchor'], R['meta'].get('wref', 0.0))[0]
        out &= ~((wd < dz - S017_DEPTH) & _ms_worldpx(R))
    return out


for _t in _ms_S015:
    POST_BODY[_t] = _ms_s015_body


# ===== page 205 RocketGirlSteamPunk, row "super 45 deg", ticks 347 / 363 / 377 "missing label" (owner: "the head of the rocket isnt labeled") =====
# Type 6000082 (Brock011UltiProjectile = skin 205's super; no other skin uses it).
# Measured on the fresh capture hull_missing_205: a super rocket = the star flash and olive barrel (clip sprites, labelled) plus a 3D
# mesh of three parts drawn as three opaque draws of one 128 x 128 texture (prog 123, th 26c53c1e): the back part, the white cylinder
# and the NOSE CONE (the 'head'). All rockets of a shot share these three draws. The installed rocket fix (z_1109_brock) adds the
# predicted body, but the prediction of the mesh ends 6-10 px short of the real nose, so the nose cone (draw 12: 486 px at 353, 986 px
# at 363 = about 200 px per rocket) is outside every label, and parts of the cylinder as well.
# Rule: the mesh texture is found from the data (the 128 x 128 non-character texture of the opaque draws that cover the object's own
# prediction); every pixel whose front-most draw uses that texture and which lies within _ms_R205 px of the object's label or prediction
# (so it belongs to THIS rocket, not to another rocket of the shot drawn elsewhere) is added to the label; pixels that another rocket's
# prediction covers and this one's does not are left to that rocket. Then gaps closed and holes filled, so one solid rocket.
from scipy.ndimage import gaussian_filter as _ms_gauss, binary_fill_holes as _ms_fill, label as _ms_cc, distance_transform_edt as _ms_edt
_ms_T205 = 6000082
_ms_R205 = 5


def _ms_rocket(prev):
    def f(body, ctx):
        if prev is not None: body = prev(body, ctx)
        pb = ctx['pred_body']; R = ctx['R']; ids = ctx['ids_h']
        if not pb.any() and not body.any(): return body
        ref = pb | body
        near = _ms_np.unique(ids[ref & (ids > 0)])
        ths = set()
        for d in near:
            dr = R['draws'].get(int(d))
            if dr is None or dr.get('tw') != 128 or dr.get('tht') != 128: continue
            th = dr.get('th') or ''
            if not th or th == '0' * 16: continue
            if int((pb & (ids == d)).sum()) >= 20: ths.add(th)
        if not ths: return body
        ds = [d for d, dr in R['draws'].items() if dr.get('th') in ths and dr.get('tw') == 128 and dr.get('tht') == 128]
        cand = _ms_np.isin(ids, ds)
        if not cand.any(): return body
        dist = _ms_edt(~(ref | body))
        cand &= dist <= _ms_R205
        # pieces of the mesh: keep the connected pieces that touch this rocket's label / prediction
        L, n = _ms_cc(cand | body, _ms_np.ones((3, 3)))
        keep = _ms_np.unique(L[ref & cand & (L > 0)]) if (ref & cand).any() else []
        keep = [k for k in keep if k > 0]
        add = _ms_np.isin(L, keep) & cand & ~body
        out = body | add
        out = _ms_fill(_close(out, 2)) & (out | _dilate(out, 2))
        return out | body
    return f


POST_BODY[_ms_T205] = _ms_rocket(POST_BODY.get(_ms_T205))


# ===== page 285 CrowSuper, row "attack 0 deg", tick 269 "missing label" =====
# Type 6000319 (Crow004Projectile, the 3-blade throw; other Crow skins with this projectile share it).
# Fresh captures (hull_missing_285, twice): all three blades of every shot are labelled at 263-279 (one label per blade, tick 269: 268 / 286 /
# 357 px = the aligned predicted blade), so the page's defect (its middle blade only partly labelled, the cyan skull particles drawn over
# the blades) does not appear any more with the installed fixes / global fill step. Defensive hidden-parts rule (owner rule: when an effect
# covers part of a projectile the label keeps the full shape): the aligned predicted blade (align(), the same step the body uses) is added
# wherever the front-most draw is not world geometry (dmask 1) or a character model, when the label already agrees with it (>= 50 % of
# the predicted blade is labelled) and the label holds less than 90 % of it. On the fresh captures this adds nothing.
_ms_T285 = 6000319


def _ms_blade(prev):
    def f(body, ctx):
        if prev is not None: body = prev(body, ctx)
        pb = ctx['pred_body']; R = ctx['R']; ids = ctx['ids_h']
        if not pb.any() or not body.any(): return body
        mine = body | ctx['mine']
        q = align(pb, mine, ctx['C'], ctx['region'])
        n = int(q.sum())
        if n < 40: return body
        inter = int((body & q).sum())
        if inter < 0.5 * n or inter >= 0.9 * n: return body
        bad = [d for d, dr in R['draws'].items() if dr.get('dmask', 0) or dr.get('th') == MODEL_TEX]
        add = q & ~body & ~_ms_np.isin(ids, bad)
        if not add.any(): return body
        out = _ms_fill(_close(body | add, 1)) & (body | add | _dilate(q, 1))
        return out | body
    return f


POST_BODY[_ms_T285] = _ms_blade(POST_BODY.get(_ms_T285))


# ===== page 346 DeadMariachiorigins, rows "attack 0 deg" tick 265 and "attack 90 deg" tick 305 "too small / cut off" =====
# Type 6000347 (Poco007Projectile, the sound-wave attack; skin 346's own, other Poco skins use other ids). Style kept: one label per shot,
# the shot's waves labelled solid in their own outline (open owner question Q3 about one label per shot / per strand is NOT touched).
# Measured on the fresh capture hull_missing_346 (tick 265: 1925 clip-body px, label 2147 px): the label is built from the wave's clip pixels
# with coverage >= CUT (128). The thin strands of a wave fade out towards their free end (coverage 20-127), so the label stops 10-15 px
# short of where the strands visibly end (305: the top of every strand, 265: the left ends of the strands and the tail of the last clef),
# which is the 'cut off' the owner sees. The label is also limited to the predicted region (hull of the path), which ends before the drawn
# strands do (flat cut at the top at 305, vertical cut at the left end at 265). Rule (like idlabels' EXTEND_BODY): clip-body pixels (sprite
# code 1) of the same draws as the label's own, coverage >= _ms_CV346, wherever they lie, that touch the
# existing label (connected through <= 2 px gaps) are added; the label is then closed (radius 2) and its holes filled, so each wave stays
# one solid piece. The soft glow (trail-body sprites, code 3, and everything of code 0 / 2) is never added.
_ms_T346 = 6000347
_ms_CV346 = 24
_ms_G346 = 150    # green channel of a bright strand line (the soft glow between lines is G < 130)
_ms_CL346 = 3


def _ms_wave346(prev):
    def f(body, ctx):
        if prev is not None: body = prev(body, ctx)
        if not body.any(): return body
        C, cv, img = ctx['C'], ctx['cv'], ctx['img']
        if img is None: return body
        g = img[..., 1].astype(_ms_np.int16)
        strand = ((C == 1) & (cv >= _ms_CV346)) | ((C == 3) & (cv >= _ms_CV346) & (g >= _ms_G346))
        cand = strand & ~body
        if not cand.any(): return body
        L, n = _ms_cc(_dilate(body, 2) | cand, _ms_np.ones((3, 3)))
        keep = [k for k in _ms_np.unique(L[body]) if k > 0]
        add = cand & _ms_np.isin(L, keep)
        out = body | add
        out = _ms_fill(_close(out, _ms_CL346)) & (out | _dilate(out, _ms_CL346))
        # smooth the serrated strand tips: blur the solid label and re-threshold (never more than 2 px outside it)
        sm = _ms_gauss(_ms_np.pad(out | body, 5).astype(_ms_np.float32), 1.6)[5:-5, 5:-5] >= 0.5
        out = _ms_fill(sm) & _dilate(out | body, 2)
        return out | body
    return f


POST_BODY[_ms_T346] = _ms_wave346(POST_BODY.get(_ms_T346))


# ===== pages 360 HookDudeMake and 387 HookDudeAngel (Gene), row "super 45 deg", ticks 345-353 "missing label" =====
# Types 6000511 (Gene005UltiProjectile, the super's black-hole vortex) and 6000512 (Gene005UltiProjectile2, its follow-up object on the new
# host, not seen in the fresh captures; the same rule); no installed fix touches them. Skins sharing the type: every Gene skin with the
# default super projectile (360 and 387 at least).
# Measured on hull_missing_360: the vortex is a swirl of opaque purple arms (labelled, 2450-2600 px) around a dark funnel, with more arm /
# funnel layers drawn as translucent 'glow-class' clip layers (sprite code 2, 3400-3500 px, coverage mostly 40-160), so the label is a
# broken ring of the brightest arms with the left swirl, the lower funnel and the centre outside it; the owner sees the vortex mostly unlabelled.
# Rule: code-2 pixels of the type's clip with coverage >= _ms_CV360 that touch the label are body (the same hidden / translucent-parts rule as
# the owner's 'keep the full shape'); the label is closed (radius _ms_CL360), its holes (the funnel) filled, and smoothed. The soft outer halo
# (coverage < _ms_CV360) stays out.
_ms_T360 = (6000511, 6000512)
_ms_CV360 = 64
_ms_CL360 = 4


def _ms_vortex(prev):
    def f(body, ctx):
        if prev is not None: body = prev(body, ctx)
        if not body.any(): return body
        C, cv = ctx['C'], ctx['cv']
        cand = (C == 2) & (cv >= _ms_CV360) & ~body
        L, n = _ms_cc(_dilate(body, 2) | cand, _ms_np.ones((3, 3)))
        keep = [k for k in _ms_np.unique(L[body]) if k > 0]
        add = cand & _ms_np.isin(L, keep)
        out = body | add
        out = _ms_fill(_close(out, _ms_CL360)) & (out | _dilate(out, _ms_CL360))
        sm = _ms_gauss(_ms_np.pad(out, 5).astype(_ms_np.float32), 1.6)[5:-5, 5:-5] >= 0.5
        out = _ms_fill(sm) & _dilate(out, 2)
        return out | body
    return f


for _t in _ms_T360:
    POST_BODY[_t] = _ms_vortex(POST_BODY.get(_t))


# ===== page 387 HookDudeAngel (Gene), row "super 45 deg", ticks 345-353 "missing label" =====
# Types 6000515 (Gene006UltiProjectile, skin 387's super: an orb with an eye, two cyan rings and yellow tips) and 6000516 (its
# Projectile2 follow-up on the new host, not seen in the fresh capture; same rule). No installed fix touches them; other skins using
# these ids share it.
# Measured on hull_missing_387 (tick 349: label 1910 px): the label holds the eye / lower half and parts of the rings; the upper ring arcs and
# the side parts are outside (the orb's sprites are code 5 'shared with the trail' / code 0, covered by a purple glow disc of the same
# sprite family, so the clip-body rule only keeps coverage >= 128 and the opaque core). The bright body is clearly separable from the glow
# disc: orb pixels of the type's sprites (C > 0) with a green channel >= _ms_G387 (cyan ring 200+, yellow 230, white eye 240; the glow disc
# is G 60-140) and coverage >= 64. Rule: those pixels that touch the label are body; the label is closed (radius 3), holes filled, smoothed.
# The star at the top right (a separate sprite, not connected) and the dark blue / purple glow stay out.
_ms_T387 = (6000515, 6000516)
_ms_G387 = 170


def _ms_orb(prev):
    def f(body, ctx):
        if prev is not None: body = prev(body, ctx)
        if not body.any() or ctx['img'] is None: return body
        C, cv = ctx['C'], ctx['cv']
        g = ctx['img'][..., 1].astype(_ms_np.int16)
        cand = (C > 0) & (cv >= 64) & (g >= _ms_G387) & ~body
        L, n = _ms_cc(_dilate(body, 2) | cand, _ms_np.ones((3, 3)))
        keep = [k for k in _ms_np.unique(L[body]) if k > 0]
        add = cand & _ms_np.isin(L, keep)
        out = body | add
        out = _ms_fill(_close(out, 3)) & (out | _dilate(out, 3))
        sm = _ms_gauss(_ms_np.pad(out, 5).astype(_ms_np.float32), 1.2)[5:-5, 5:-5] >= 0.5
        out = _ms_fill(sm) & _dilate(out, 2)
        return out | body
    return f


for _t in _ms_T387:
    POST_BODY[_t] = _ms_orb(POST_BODY.get(_t))


# ===== page 390 DeadMariachiGoodRandoms (Poco keyboards), row "super 45 deg", tick 355 "missing label" =====
# Type 6000352 (Poco009UltiProjectile, the keyboard super of skin 390; other Poco skins with this projectile share it; the other Poco ids
# are untouched). Measured on hull_missing_390: the three keyboard members have logic state 'finished' = 1 from tick 353 (range reached);
# the client draws both keyboards at FULL size at 353, 355 and 357 and dissolves them only at 359-361 (gone at 363). z_1108_poco_bullets
# keeps such a finished member labelled while its cloud is intact, but only if the tracker still reports it as drawn; at 355 / 357 the
# tracker has hidden the members (drawn False) although the keyboards are intact on screen, so the keyboards lose their labels two ticks
# too early (owner: 355 'missing label'). Rule: a finished (finished == 1) member of this type that the tracker no longer draws counts as
# drawn and running for the labelling step while the keyboards are still intact: the white piano-key pixels (all channels >= 170) over the type's sprites
# (keys: 2840 px at 349, 1731 at 357, 70 at 359 when they dissolve) must be at least _ms_KEEP390 of the group's maximum; once they dissolve it is unlabelled as before.
_ms_T390 = {6000352}
_ms_KEEP390 = 0.5      # a finished member's keyboards must still show this share of the most white piano-key pixels the group had
_ms_mem390 = {'tick': -1, 'max': {}}
_ms_prev_labels390 = labels


def labels(objs, cam, R, state=None, ended=(), img=None):
    tick = (state or {}).get('tick')
    mine = [o for o in objs if o['t'] in _ms_T390 and not o.get('ended')]
    if tick is None or not mine: return _ms_prev_labels390(objs, cam, R, state=state, ended=ended, img=img)
    if tick < _ms_mem390['tick']: _ms_mem390['max'].clear()
    _ms_mem390['tick'] = tick
    try:
        C = None
        for own in {o['own'] for o in mine}:
            c = sprite_pixels(mine[0]['t'], own, R)
            C = c if C is None else _ms_np.where(C > 0, C, c)
        im_ = img.astype(_ms_np.int16) if img is not None else None
        area = int(((_ms_np.minimum(_ms_np.minimum(im_[..., 0], im_[..., 1]), im_[..., 2]) >= 170) & ((C == 1) | (C == 3))).sum()) if im_ is not None else 0   # white piano-key pixels over the keyboards
        gids = {o.get('label_id', o['id']) for o in mine}
        out = []
        for o in objs:
            g = o.get('label_id', o['id'])
            if o in mine and g in gids and o.get('finished', 0) == 1 and not o.get('drawn', True):
                mx = _ms_mem390['max'].get(g, 0)
                if mx > 0 and area >= _ms_KEEP390 * mx: o = dict(o, drawn=True, finished=0, _ms_rev=True)
            out.append(o)
        for o in out:
            g = o.get('label_id', o['id'])
            if o in mine or o.get('_ms_rev'):
                if o.get('drawn', True) and o.get('finished', 0) == 0 and not o.get('_ms_rev'):
                    _ms_mem390['max'][g] = max(_ms_mem390['max'].get(g, 0), area)
        objs = out
    except Exception:
        pass
    return _ms_prev_labels390(objs, cam, R, state=state, ended=ended, img=img)


# The ID pass cannot show the keyboards at these ticks: from 355 the client draws the big translucent trail circle OVER them (its draw is the
# front-most one, the piano keys' own small draws of 353 are hidden under it), so the sprite codes give nothing but the circle. For a revived
# member the label is therefore taken from the picture inside that circle (the type's own trail-body sprite pixels, which lie over the keyboard): the bright blue / white keyboard pixels (blue channel high,
# blue clearly above red), closed (radius 2), holes filled, parts under _ms_MIN390 px dropped (notes, sparks) -- the keyboard in its own
# outline. The soft glow around it (blue channel low) stays out.
_ms_MIN390 = 300


def _ms_keyboard(prev):
    def f(body, ctx):
        if prev is not None: body = prev(body, ctx)
        if not ctx['o'].get('_ms_rev') or ctx['img'] is None: return body
        im = ctx['img'].astype(_ms_np.int16); r, g, b = im[..., 0], im[..., 1], im[..., 2]
        keys = (_ms_np.minimum(_ms_np.minimum(r, g), b) >= 170)                       # white piano keys
        inner = (b >= 200) & (r < 130) & (g < 135)                                    # translucent blue body
        line = (g >= 215) & (b >= 235) & (r < 160)                                    # cyan frame lines
        m = (keys | inner | line) & ((ctx['C'] == 3) | (ctx['C'] == 1))
        m = _ms_fill(_close(m, 3))
        L, n = _ms_cc(m, _ms_np.ones((3, 3)))
        if n == 0: return body
        sz = _ms_np.bincount(L.ravel())
        keep = [k for k in range(1, n + 1) if sz[k] >= _ms_MIN390]
        if not keep: return body
        out = _ms_np.zeros(body.shape, bool)
        for k in keep: out |= hull_mask(L == k)     # a keyboard is a convex curved slab: its piece's hull
        return out
    return f


for _t in _ms_T390:
    POST_BODY[_t] = _ms_keyboard(POST_BODY.get(_t))
