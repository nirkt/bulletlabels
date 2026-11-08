# sticky5 fix file (round 5, 2026-11-06 evening) -> W/py/fixes/z_1115_sticky5.py. Every global name here has the prefix _s5_.
#
# FIX 1  Grom (CrossBomber) super, landing frame t365. Owner on 148 CrossBomberDragon super 45 tick 365: "effect or box in the middle shouldnt
# be labeled". When the cube lands it spawns the burst pieces (child type = parent type + 1: the four spiky heads that fly out, 6000981 on 148)
# and the landed cube itself (type 6000980, the box in the middle, drawn with the landing glow) is just the end of the flight: the owner's rule
# "landing / end effects are never labelled". The parent super object is therefore NOT labelled once one of its own children (same src,
# type + 1) is labelled with >= 100 px (the pieces are drawn over the cube; 061: the pieces are still hidden at t365, the ball keeps its label). The children keep their labels (hidden-parts fill of z_1114_cut5 only fills inside each child's own predicted body).
# Types: 6000958 (061 Default, + overcharged copy 6000960), 6000964 (126 Easter), 6000968 (130 Franken), 6000972 (135 Hindu), 6000976 (142
# SupercellMake), 6000980 (148 Dragon); each parent type is used by one skin only (the overcharged one by 061).
_s5_GROM_PARENT = {6000958: 6000959, 6000960: 6000961, 6000964: 6000965, 6000968: 6000969, 6000972: 6000973, 6000976: 6000977, 6000980: 6000981}
import bulletlabels as _s5_BL
from scipy.ndimage import binary_fill_holes as _s5_fill
_s5_prev_labels = labels


def _s5_px(P, o):
    lid = (_s5_BL._lid(o), -1, 'body')
    if lid not in P['lids']: return None, 0
    k = P['lids'].index(lid)
    return k, int((P['ids'] == k).sum())


def labels(objs, cam, R, state=None, ended=(), img=None):
    P = _s5_prev_labels(objs, cam, R, state=state, ended=ended, img=img)
    kids = {}
    for o in objs: kids.setdefault((o.get('src'), o['t']), []).append(o)
    for o in objs:
        if o['t'] not in _s5_GROM_PARENT: continue
        ch = kids.get((o.get('src'), _s5_GROM_PARENT[o['t']]), [])
        # only when the burst pieces are really drawn (labelled) over the landed cube; while they are still hidden the cube keeps its label
        if any(_s5_px(P, c)[1] >= 100 for c in ch):
            k, n = _s5_px(P, o)
            if n: P['ids'][P['ids'] == k] = -1
    return P


# FIX 1b  same frame: the burst pieces themselves carry strays in the middle (a 180 px blob of the landing glow / cube that one piece's
# label picked up next to the box, and a 17 px speck on another): owner "effect or box in the middle shouldnt be labeled". Each Grom burst piece
# (children 6000959 / 61 / 65 / 69 / 73 / 77 / 81) keeps only the connected parts of its label that lie on its own predicted body (pred_body, dilated 4 px),
# i.e. the head itself; parts that are not on the piece (>= half of the part outside) are dropped. Runs after the installed hooks (cut5 fill).
from scipy.ndimage import label as _s5_cc


def _s5_wrap_child(t):
    prev = POST_BODY.get(t)

    def f(body, ctx):
        if prev is not None: body = prev(body, ctx)
        pb = ctx['pred_body']
        if ctx['o']['id'] in _s5_ring:
            # FIX 1c: the tick after the landing (t367) the white landing flash / dust cloud covers the burst pieces. A piece whose prediction is
            # missing or of which < 25 % is visible as clip body is not visible: no label (explosions are never labelled; 135 t367: the label
            # was the white flash with holes). Same visibility test as the StickyBomb bubbles (FIX 2).
            vis = _s5_fill((ctx['C'] == 1) & (ctx['cv'] >= 32))
            if not pb.any() or (vis & pb).sum() < 0.25 * pb.sum(): return np.zeros_like(body)
        if not body.any() or not pb.any(): return body
        if ctx['o']['id'] in _s5_ring2 and pb.sum() >= 200:
            # FIX 1d: two ticks later (t369) the pieces are flung out and visible, but the dust cloud / the landed body / the flash around them
            # shares their clip sprites (135: the label was the dust puffs + the white body, 1700-2400 px for a 420 px piece). The piece keeps
            # what lies within 8 px of its own predicted body.
            body = body & _dilate(pb, 8)
            if not body.any(): return body
        L, n = _s5_cc(body, structure=np.ones((3, 3), bool))
        if n < 2: return body
        near = _dilate(pb, 4)
        out = np.zeros_like(body)
        for j in range(1, n + 1):
            m = L == j
            if (m & near).sum() >= 0.5 * m.sum(): out |= m
        return out if out.any() else body
    POST_BODY[t] = f


for _s5_t in _s5_GROM_PARENT.values():
    _s5_wrap_child(_s5_t)


# ---------------------------------------------------------------------------------------------------------------------------------
# FIX 2  StickyBomb (Dynamike) super child bubbles, type 6000934 StickyBombUltiProjectile2 = child of the super of EVERY StickyBomb skin
# (060 Default 6000933, 281 / 286 / 292 / 312 / 333 / 403 / 445 via their Squeak00x ultis; z_1113_sticky4 knew only 281 / 286 / 445).
# Owner (round 5): 060 super 45 tick 367 "should not be labelled"; 292 tick 367 "some extra effects in the middle are labeled".
# At the landing (t365-367) the explosion (brown dirt cloud, white flash) is drawn over the young bubbles; what was labelled there are scraps of
# that effect. Data-derived rule instead of per-skin ids: a bubble that is young (<= 2 ticks since it spawned, same age rule as sticky4) and whose
# predicted body (game state, full size >= 300 px) is mostly (< 25 % visible, 060: 2-12 %, 292 visible ring: 47-81 %) covered by an effect draw (only < 25 % of its filled clip-body pixels are visible
# inside the prediction) is not visible as a bubble -> no label (the landing explosion is never labelled). Older bubbles (t369 on, the dirt only
# partly covers them and the clip-body measure is unreliable there) and young bubbles that are visible (292: the ring of bubbles) keep the label.
_s5_prev_child = POST_BODY.get(6000934)   # = z_1113_sticky4._st4_child (kept as the fallback when the prediction is missing)
_s5_seen = {}
_s5_last = [0]
_s5_young = set()
_s5_ring = set()
_s5_ring2 = set()
_s5_GROM_KIDS = set(_s5_GROM_PARENT.values())


def _s5_child(body, ctx):
    pb = ctx['pred_body']
    if _s5_prev_child is not None: body = _s5_prev_child(body, ctx)
    oid = ctx['o']['id']
    if oid not in _s5_young or not pb.any(): return body
    cand = _s5_fill((ctx['C'] == 1) & (ctx['cv'] >= 32))
    if pb.sum() >= 300 and (cand & pb).sum() < 0.25 * pb.sum(): return np.zeros_like(body)
    if oid not in _s5_ring: return body
    # FIX 2b (292 t367 "extra effects in the middle"): sticky4 took the WHOLE connected clip-body blob that touches the prediction, so a bubble
    # touching its neighbour (or the glow in the middle of the ring) pulled the neighbour / the middle into its label (292 t367: the left-bottom
    # bubbles carried a neck + an inner loop through the middle). In the tight ring (the tick after the spawn) a bubble is one round blob: its
    # clip-body pixels (closed by 2 px, holes = ghost face / glints filled) within 5 px of its predicted body, the parts that lie on the
    # prediction, as ONE convex hull cut to 5 px around the prediction. Nothing of the glow in the middle is taken.
    sil = _s5_fill(_close(cand, 2) & (ctx['region'] | cand)) & _dilate(pb, 5)
    L, n = _s5_cc(sil, structure=np.ones((3, 3), bool))
    sel = np.zeros(body.shape, bool)
    for k in range(1, n + 1):
        m = L == k
        if (m & pb).sum() >= 0.15 * pb.sum() or ((m & pb).sum() >= 0.5 * m.sum() and m.sum() >= 30): sel |= m
    if not sel.any(): return body
    return hull_mask(sel) & _dilate(pb, 5)


POST_BODY[6000934] = _s5_child
_s5_prev_labels2 = labels


def labels(objs, cam, R, state=None, ended=(), img=None):
    tick = int(state.get('tick', 0)) if state else 0
    if tick < _s5_last[0]: _s5_seen.clear()
    _s5_last[0] = tick
    _s5_young.clear(); _s5_ring.clear(); _s5_ring2.clear()
    for o in objs:
        if o['t'] == 6000934 or o['t'] in _s5_GROM_KIDS:
            _s5_seen.setdefault(o['id'], tick)
            age = tick - _s5_seen[o['id']]
            if age <= 2: _s5_young.add(o['id'])
            if 1 <= age <= 2: _s5_ring.add(o['id'])
            if 3 <= age <= 4: _s5_ring2.add(o['id'])
    return _s5_prev_labels2(objs, cam, R, state=state, ended=ended, img=img)
