# installed 11-02 from out/sweep2/haikutasks/siteai5/ended_visible/fix.py (owner rulings: ended-but-visible attacks stay labelled for the named Ash / Mortis / Bull Lion types; Bull Demon 603 one outline per bubble)
# owner rulings 2026-11-02 (RULINGS_1102.md, second batch; task out/sweep2/haikutasks/siteai5/ended_visible/task.md), exec'd last
# (IDLABELS_OVERRIDE) in idlabels' namespace, after every installed fix.
#
# 1. ended_visible: an attack that is still CLEARLY VISIBLE after the game says its logic ended stays labelled until it disappears.
#    Named cases only, keyed to these type ids (the global 10-23 'ended attacks unlabelled' rule stays for every other type):
#      Ash (Knight attacks):      6001043 (129 KnightPizza), 6001046 (131 KnightPinata), 6001055 (150 KnightAdidas)
#      Mortis (Undertaker supers): 6000299 (026 default), 6000300-6000307 (482, 526, 546, 578, 195, 598)
#      Bull Lion puff:            6000047 (516, t313 puff with finished = 1 still drawn)
#    Where the labels were dropped after the logic end (checked on the code, not only on the pictures):
#      - finished = 1 drops: z_1103_undertaker._mort_fin (6000300-07), z_1100_s026._is_finished_drop (6000299),
#        z_1102_s195._s195_finished (6000304), and the 'finished' cut in z_1101_s129knight's hull hook (the three Ash types);
#      - the wave rule in idlabels.labels: a synthesized object of a wave attack that left the display (o['ended'], Tracker.gone)
#        is dropped when no member of its group is alive (trail_body types: the Ash types and the Mortis supers).
#    For the listed types those drops are switched off, and the object's own body decides: its sprite pixels as before the end
#    (clip body, and the trail body of the attack's trail sprites, the same shape rules), then the same post-processing. No drawn
#    body = no label (an end animation, hit flash or explosion is not its sprite, so it is not labelled; unchanged).
#    z_1027_s526 (6000301: the electric ring that replaces the blue sphere 2 ticks after the impact is an end animation) keeps its
#    age rule: it is not part of this rule. Bull Lion 516 needed no drop removed: its finished puff still drawn is labelled already
#    (checked on sw2_516 t313: object 2000016, finished 1, drawn 1, 513 px labelled); no change for that type.
#
# 2. bull603 (Bull Demon 603, Bull011Projectile 6000051, sw2_603): the splintered bubble column gets ONE OUTLINE PER BUBBLE, loose
#    small pieces dropped, no hull. Each object is one bubble. Measured on sw2_603 (ticks 267-313): every bubble had one main piece
#    plus loose fragments: 1-30 px specks and 2-50 px bits (e.g. 2000011 t271: 381 + 31 + 28 + 1 + 1 + 1 px; 2000016 t307: 552 + 45 +
#    11 + 7 + 7 + 3), and the splinters of one bubble lie 2-14 px from its main piece (2000012 t267: 254 + 142 px at 12 px; t269:
#    387 + 183 px at 14 px). Rule per object, after the installed hooks of 6000051 (flash cut, specks, holes):
#      a. pieces under BB_SPECK px are dropped (the largest always stays);
#      b. the rest is closed by BB_JOIN px (the splinters of one bubble within about 2 * BB_JOIN px join it), holes filled,
#         clipped to the bubble's own region (the prediction's neighbourhood), and only its largest piece is kept: one outline.
#    No convex hull (the hull of 2000011 et al. took in the gaps of the column, earlier attempt).

# ---- 1. ended_visible ------------------------------------------------------------------------------------------------
EV_TYPES = {6001043, 6001046, 6001055,                                  # Ash attacks (129 / 131 / 150)
            6000299, 6000300, 6000301, 6000302, 6000303, 6000304, 6000305, 6000306, 6000307,   # Mortis supers (026, 482, 526, 546, 578, 195, 598)
            6000047}                                                    # Bull Lion puff (516)
EV_KNIGHT = (6001043, 6001046, 6001055)


def _ev_pred(name):
    """the drop predicate `name` of an installed fix file, without the ended_visible types (looked up at call time there)"""
    prev = globals().get(name)
    if prev is None or getattr(prev, '_ev', False): return
    def pred(o, _prev=prev):
        return bool(_prev(o)) and o['t'] not in EV_TYPES
    pred._ev = True
    globals()[name] = pred


for _name in ('_mort_fin', '_is_finished_drop', '_s195_finished'):
    _ev_pred(_name)


EV_HOLE = 60   # px (half resolution): an enclosed hole of the Ash hull up to this size is filled (the depth cut of solid
               # geometry left a 41 px hole inside the hull of 6001043 at sw2_129 t277: a hole, not a gap in the cloud)


def _ev_unfinished(hook):
    """the knight hull hook cuts a finished object (its logic ended): for the ended_visible types the hook sees the object as
    not finished, so its body (the sprites still drawn) decides. Enclosed holes of up to EV_HOLE px are filled."""
    def fn(body, ctx):
        o = ctx['o']
        if o.get('finished'):
            ctx = dict(ctx, o=dict(o, finished=0))
        out = hook(body, ctx)
        if out.any():
            out = out | small_holes(out, EV_HOLE)
        return out
    fn._ev = True
    return fn


for _t in EV_KNIGHT:
    if _t in POST_BODY and not getattr(POST_BODY[_t], '_ev', False):
        POST_BODY[_t] = _ev_unfinished(POST_BODY[_t])


_ev_prev_labels = labels


def labels(objs, cam, R, state=None, ended=(), img=None):
    """the wave rule (o['ended'] of a synthesized attack object that left the display) is not applied to the ended_visible types"""
    objs = [dict(o, ended=False) if (o.get('ended') and o['t'] in EV_TYPES) else o for o in objs]
    return _ev_prev_labels(objs, cam, R, state=state, ended=ended, img=img)


# ---- 2. bull603: one outline per bubble ---------------------------------------------------------------------------
BB_TYPE = 6000051      # Bull011Projectile, BullDudeDemon 603 (bull_011_atk_projectile)
BB_SPECK = 40          # px: pieces of a bubble's label under this size are loose fragments (dropped; the largest always stays)
BB_JOIN = 8            # px: closing radius: the splinters of one bubble (gaps up to about 2 * BB_JOIN px) join its outline


def _bubble_one(body, ctx):
    """one outline per bubble: loose small pieces dropped, the rest closed, hole-filled and clipped to the bubble's region;
    only its largest piece is kept (no hull)"""
    from scipy.ndimage import binary_fill_holes, label as _bb_cc
    if not body.any():
        return body
    L, n = _bb_cc(body, structure=np.ones((3, 3), bool))
    if n > 1:
        sz = np.bincount(L.ravel()); keep = sz >= BB_SPECK; keep[0] = False
        if not keep.any(): keep[int(np.argmax(sz[1:])) + 1] = True
        body = keep[L]
    if not body.any():
        return body
    m = (binary_fill_holes(_close(body, BB_JOIN)) & ctx['region']) | body
    L, n = _bb_cc(m, structure=np.ones((3, 3), bool))
    if n > 1:
        sz = np.bincount(L.ravel()); m = L == (int(np.argmax(sz[1:])) + 1)
    return m


_bb_prev = POST_BODY.get(BB_TYPE)
if not getattr(_bb_prev, '_bb', False):
    def _bb_post(body, ctx, _p=_bb_prev):
        if _p is not None: body = _p(body, ctx)
        return _bubble_one(body, ctx)
    _bb_post._bb = True
    POST_BODY[BB_TYPE] = _bb_post
