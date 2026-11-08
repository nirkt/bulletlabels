# brock5 round-5 fixes (owner 2026-11-06 evening). All global names start with _b5_.
#
# 1. 175 ZEUS super (type 6000066 Brock007UltiProjectile; skin 175 only), super 45 t363 / t377 / t393: "the labels for the lighting isnt
#    perfect and the big spark in the middle shouldnt be labeled". The lightning bolt is the long trail-clip sprite (zigzag + forks, ~2100 px)
#    that hangs under a rocket's crystal. z_1107_brock (bolt = code-4 pixels 4-connected to the body inside the prediction reach) and
#    z_1113_brock4 (big trail piece kept whole IF it overlaps the label) both miss the bolt when the crystal's label and the bolt sprite only
#    TOUCH (1 px gap, as on the fresh t363 capture): the bolt got no label at all. Rule now: a connected piece (8-conn) of the type's own
#    trail-clip sprite pixels (coverage >= 20) that is BIG (>= _B5_BIG px: the bolt, a spark squiggle is < 200) and lies within _B5_TOUCH px of
#    the label or of the object's predicted region is added whole (forks included), the dark gap between the forks closed and filled. The
#    impact flash / spark at the foot of the bolt is another draw (not the trail sprite): never added.
import numpy as _b5_np
from scipy.ndimage import binary_fill_holes as _b5_fill, label as _b5_cc

_B5_T175 = 6000066
_B5_BIG = 400
_B5_CV = 20
_B5_TOUCH = 3
_B5_YR = 170           # yellow patch: red channel >= this and blue <= green - _B5_YB
_B5_YB = 30
_B5_BRIDGE = 9
_b5_prev175 = POST_BODY.get(_B5_T175)      # installed chain: zeus + brock4 (+ older)


def _b5_bolt(body, ctx):
    out = _b5_prev175(body, ctx) if _b5_prev175 is not None else body
    pa = ctx['pred_any']
    anchor = out | pa
    if not anchor.any(): return out
    tr = _s175_trail_px(ctx) & (ctx['cv'] >= _B5_CV)
    if not tr.any(): return out
    L, n = _b5_cc(tr, structure=_b5_np.ones((3, 3)))
    if n == 0: return out
    sz = _b5_np.bincount(L.ravel())
    near = _dilate(anchor, _B5_TOUCH)
    keep = [k for k in range(1, n + 1) if sz[k] >= _B5_BIG and (near & (L == k)).any()]
    res = out
    if keep:
        res = out | _b5_np.isin(L, keep)
        res = _b5_fill(res) | res      # only the real holes of the bolt sprite (lens between its forks) are filled, no closing: a closing
                                       # would fill the V under the forks, which is the white impact flash
    # the pale-yellow lightning patch drawn on a crystal (the same trail sprite): the zeus rule cut its spikes / lobes off (t377 / t393). The
    # YELLOW pixels of the trail sprite that touch the label are added whole; the cyan squiggles beside the crystal (sparks) stay out.
    img = ctx['img']
    if img is not None and out.any():
        im = img.astype(_b5_np.int16)
        yel = tr & (im[..., 0] >= _B5_YR) & (im[..., 2] <= im[..., 1] - _B5_YB)
        if yel.any():
            Ly, ny = _b5_cc(yel, structure=_b5_np.ones((3, 3)))
            ky = _b5_np.unique(Ly[_dilate(out, 1) & (Ly > 0)]); ky = ky[ky > 0]
            if ky.size:
                add = _b5_np.isin(Ly, ky)
                res = res | add
                res = (_close(res, 2) & (res | _dilate(add, 2))) | res
    # a thin tail of the crystal that sticks out below an object drawn over it (t393: the tip behind the orange puff) is a separate crumb: the
    # hidden part between it and the crystal is filled (hidden-parts rule), inside the predicted body only.
    L2, n2 = _b5_cc(res, structure=_b5_np.ones((3, 3)))
    if n2 > 1 and not keep:
        sz2 = _b5_np.bincount(L2.ravel()); main = int(sz2[1:].argmax()) + 1
        for k in range(1, n2 + 1):
            if k == main or sz2[k] >= 0.15 * sz2[main]: continue
            pk = L2 == k; zone = _dilate(pk, _B5_BRIDGE) & (L2 == main)
            if zone.any():
                res = res | (hull_mask(pk | zone) & _dilate(ctx['pred_any'], 3))
    return res


POST_BODY[_B5_T175] = _b5_bolt


# 2. 214 RocketGirlRankedProgrsesionSkinLevel3 super rockets (type 6000084 Brock012UltiProjectile = skin 214's super only), pages 214_01 / _02.
#    Owner: _02 t377 "some extra stuff is labeled" (first fix), after the mesh-only version (v2: labels were partial cylinders): _01 t345 /
#    t347 / t363 and _02 t379 "too small / cut off", _02 t395 "should not be labelled". Coordinator 11-06: the bright star and the blue glow at
#    the rocket's front are the rocket's HEAD: they are part of the label (v2 removed). Rule now (v1 + one more):
#    a) the rocket as it leaves the gun: the muzzle burst is drawn over it and its effect-sprite pixels beyond the PREDICTED BODY (pred_body +
#       _B5_PB px) make up >= _B5_EXCESS x the predicted-body size. Then only those burst pixels are cut (owner 11-06, page 214_01 t347:
#       "keep the whole rocket"): pixels with an atlas shape outside the predicted body go, the rocket's own mesh pixels (code 0) stay
#       wherever they are; closed and filled. Ordinary rockets (star head and glow inside ~20 % of the predicted body) are untouched.
#       The smoke (trail shape 63) is not added by this file.
#    b) a rocket whose logic has FINISHED (o['finished']: it landed) and of which no mesh is visible any more (fewer than _B5_MESH pixels of
#       the label with no atlas shape, code 0): what is drawn is the landing burst / the tail flare behind a crate (t377: the flare at the
#       crate, t395: a blob inside the ice dome), not the bullet -> no label. Rockets still in flight (the birth frames at the gun included)
#       are never dropped.
_B5_T214 = 6000084
_B5_PB = 3
_B5_MESH = 25
_B5_EXCESS = 0.35
_B5_COVER = 0.7
_B5_YOUNG = 6          # ticks of logic path: a rocket that has just left the gun (the muzzle burst is drawn over it)
_b5_prev214 = POST_BODY.get(_B5_T214)


def _b5_rocket214(body, ctx):
    out = _b5_prev214(body, ctx) if _b5_prev214 is not None else body
    if not out.any(): return out
    if ctx['o'].get('finished') and int((out & (ctx['C'] == 0) & (ctx['cv'] >= CUT)).sum()) < _B5_MESH:
        return _b5_np.zeros_like(out)                                                                           # b)
    pb = ctx['pred_body']
    if not pb.any(): return out
    lim = _dilate(pb, _B5_PB)
    fx = (ctx['C'] != 0)                                   # pixels of effect sprites (clip / trail shapes); mesh pixels = code 0
    res = out
    if int((out & ~lim & fx).sum()) >= _B5_EXCESS * int(pb.sum()):                                              # a) muzzle burst
        keep = out & (lim | ~fx)                           # only the burst sprites beyond the predicted body go; the rocket's own mesh stays
        if keep.sum() >= 0.3 * out.sum():
            keep = (_close(keep, 2) & (lim | ~fx | keep)) | keep
            keep = _b5_fill(keep) & (lim | keep | ~fx)
            L, n = _b5_cc(keep, structure=_b5_np.ones((3, 3)))
            if n > 1:
                sz = _b5_np.bincount(L.ravel())[1:]
                keep = _b5_np.isin(L, 1 + _b5_np.nonzero(sz >= 0.1 * sz.max())[0])
            res = keep
    # c) hidden parts (owner rule): where the burst (a blended effect sprite) is drawn over the rocket, the ID pass cannot see its body. A
    #    rocket whose label holds less than _B5_COVER of its predicted body gets the predicted-body pixels that lie under blended sprites back
    #    (never pixels under opaque objects: the brawler, walls), when they touch / lie within 2 px of the label; then closed and filled.
    #    Only for a rocket that has just left the gun (logic path <= _B5_YOUNG points): later a foreign blended sprite over it (the ice burst) is
    #    not the muzzle burst and must not enlarge the label.
    young = len(ctx['o'].get('path') or ()) <= _B5_YOUNG
    if young and int((res & pb).sum()) < _B5_COVER * int(pb.sum()):
        hid = pb & blended_sprite(ctx['R'], ctx['ids_h']) & ~res
        if hid.any():
            L, n = _b5_cc(hid | res, structure=_b5_np.ones((3, 3)))
            ks = _b5_np.unique(L[res & (L > 0)]); ks = ks[ks > 0]
            add = _b5_np.isin(L, ks) & hid
            if add.any():
                r2 = res | add
                r2 = (_close(r2, 2) & (r2 | _dilate(add, 1) | lim)) | r2
                res = _b5_fill(r2) | r2
    return res


POST_BODY[_B5_T214] = _b5_rocket214


# 3. 214 super rockets, page 214_02 t395 "should not be labelled" (and the same blob at t379 on a fresh capture). Cause: z_1113_ended4 (global
#    "ended-but-still-visible bullets are labelled") labels the tail-flare sprites of a rocket that has already LANDED (logic finished, no longer
#    in the live object list) inside the landing burst / ice dome: a small blob that is not the bullet (the rocket's mesh is gone, only the
#    effect sprite of the same atlas is drawn; ended4 learned that shape from the live labels, which now - correctly, owner 11-06 - include the
#    rocket's star head). Rule: labels of an ENDED rocket of type 6000084 whose object is not live in this frame are removed (the live pipeline
#    still labels a landed rocket while its body is drawn: o['drawn']). Keyed to the type; data: the `ended` argument + the live objects.
# 4. 214 page _02 t377 "should be split / merged": two rockets in the upper middle touch each other. Owner standing rule (223, 11-06): touching
#    or overlapping bullets = ONE merged label without seams or holes, bullets that do not touch stay separate. Rule: two labels of type 6000084
#    whose masks come within _B5_TOUCH px of each other are merged into one label (the lower id keeps it, the seam closed by 2 px and holes filled).
_B5_ENDED_T = 6000084
_B5_GAP = 2
_b5_prev_labels = labels


def _b5_merge_touching(P, ks):
    ids = P['ids'].copy(); share = P['share'].copy()
    masks = {k: ids == k for k in ks}
    par = {k: k for k in ks}
    def find(x):
        while par[x] != x: x = par[x]
        return x
    for i, a in enumerate(ks):
        if not masks[a].any(): continue
        da = _dilate(masks[a], _B5_GAP)
        for b in ks[i + 1:]:
            if masks[b].any() and (da & masks[b]).any(): par[find(b)] = find(a)
    groups = {}
    for k in ks:
        if masks[k].any(): groups.setdefault(find(k), []).append(k)
    changed = False
    for r, g in groups.items():
        if len(g) < 2: continue
        u = _b5_np.zeros(ids.shape, bool)
        for k in g: u |= masks[k]
        u2 = _b5_fill(_close(u, _B5_GAP)) & ((ids < 0) | u)           # the seam between them: only free pixels are added
        add = u2 & ~u
        for k in g: ids[masks[k]] = r
        ids[add] = r; share[add] = 0.5; changed = True
    if not changed: return P
    P2 = dict(P); P2.update(ids=ids, share=share)
    return P2


def labels(objs, cam, R, state=None, ended=(), img=None):
    P = _b5_prev_labels(objs, cam, R, state=state, ended=ended, img=img)
    try:
        live = {o['id'] for o in objs}
        gone = {e['id'] for e in (ended or ()) if e.get('t') == _B5_ENDED_T and e['id'] not in live}
        if gone:
            kill = [k for k, lid in enumerate(P['lids']) if lid[0] in gone]
            if kill:
                ids = P['ids'].copy(); share = P['share'].copy()
                for k in kill:
                    m = ids == k; ids[m] = -1; share[m] = 0
                P = dict(P); P.update(ids=ids, share=share)
        mine = {BL._lid(o) for o in objs if o['t'] == _B5_ENDED_T}
        ks = [k for k, lid in enumerate(P['lids']) if lid[0] in mine and lid[2] == 'body']
        if len(ks) > 1: P = _b5_merge_touching(P, ks)
        return P
    except Exception:
        return P
