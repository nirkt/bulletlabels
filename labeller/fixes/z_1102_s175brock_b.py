# hosted review 2026-11-01 fix (Haiku fix, reviewed by the coordinator, installed 11-01 LOCAL ONLY; source out/sweep2/haikutasks/site1101/g_brock2/fix2.py (exec of fix.py removed); runs after z_1101_*)
# g_brock2 fix2 (coordinator review 2026-11-01, supersedes the super-rocket trim of fix.py; zones of fix.py kept).
# Runs fix.py first (zones + far-pixel trim), then:
#
# 1. RANGER SUPER ROCKETS 184-188 (types 6000076-6000080; user: 185-188 t377 'too big', 184 t377 holes / ragged, t393 cut off):
#    the ring and smoke next to the missile are atlas shape 0 of the rocket's own clip (measured on 185 / 184 captures at
#    ticks 363-393: shape 0 is the blue ground ring / trail, the missile body is 'no shape' (mesh) and its head flare is
#    shapes 61 (cyan) and 63 (yellow)). Label pixels of shape 0 are dropped; the rest of the label is kept, holes filled so
#    the missile is one solid outline. Keyed to the five types only.
#
# 2. 175 SPRITE BOLT SUPER ROCKET (type 6000066; user: the cyan bolt body is unlabelled, only its yellow head is outlined at
#    t377 / t393): the bolt's trail-over sprites (code 4, the cyan body drawn over the clip) that are 4-connected to the labelled
#    head and lie within G2B_BOLT_NEAR half-res px of the predicted body are added; holes filled. Nothing else is added.
#
# 3. RADIUS of the Ranger burn zones 17000182-17000186: the drawn ring's rim lies at 1.14-1.16 logic radii (a3_agent3 measured
#    it on page 186; the user: 'just label the visible circle'). Those zones use 1.15 x the logic radius; the other zones keep
#    the radius of fix.py (1.0).

# (fix.py of the same task is installed just before this file as z_1102_s175brock_a.py)

G2B_RANGER_TYPES = (6000076, 6000077, 6000078, 6000079, 6000080)
G2B_BOLT_TYPE = 6000066
G2B_BOLT_NEAR = 10
G2B_RANGER_ZONE_SCALE = {17000182: 1.15, 17000183: 1.15, 17000184: 1.15, 17000185: 1.15, 17000186: 1.15}


def _g2b_shape0(R, t, own=True):
    import projtab
    from scipy.ndimage import binary_fill_holes as _fill
    T = projtab.table()[t]
    names = tuple(sorted({T['blue'], T['red']}))
    return _a4_shape_pixels(R, T['file'], names) - 1      # shape id, -1: none


def _g2b_ranger(body, ctx):
    from scipy.ndimage import binary_fill_holes as _fill
    if not body.any(): return body
    SP = _g2b_shape0(ctx['R'], ctx['t'])
    out = body & (SP != 0)
    if not out.any(): return body
    return _fill(out) | out


def _g2b_bolt(body, ctx):
    from scipy.ndimage import label as _cc, binary_fill_holes as _fill
    if not body.any(): return body
    pb = ctx['pred_body']; C = ctx['C']
    if not pb.any(): return body
    near = _dilate(pb, G2B_BOLT_NEAR)
    add = (C == 4) & near & ~body
    if not add.any(): return body
    L, n = _cc(add | body)
    keep = body | np.isin(L, np.unique(L[body & (L > 0)]))
    keep = keep & (body | add)
    return _fill(keep) | keep


for _t in G2B_RANGER_TYPES:
    _g2b_prev = POST_BODY.get(_t)
    POST_BODY[_t] = (lambda p: (lambda b, c: _g2b_ranger(p(b, c), c)))(_g2b_prev) if _g2b_prev else _g2b_ranger
_g2b_prev = POST_BODY.get(G2B_BOLT_TYPE)
POST_BODY[G2B_BOLT_TYPE] = (lambda p: (lambda b, c: _g2b_bolt(p(b, c), c)))(_g2b_prev) if _g2b_prev else _g2b_bolt

# zone radius for the Ranger burn zones: the fix.py zone wrapper runs first, then the 1.15 circles replace its label for these ids.
_g2b_zone_prev = zone_labels


def zone_labels(state, cam, R, taken, own_team=None):
    import owners
    out = _g2b_zone_prev(state, cam, R, taken, own_team)
    items = [it for it in state.get('items', []) if int(it[1]) in G2B_RANGER_ZONE_SCALE]
    if not items: return out
    H, W = R['id'][::2, ::2].shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    for it in items:
        iid, did = int(it[0]), int(it[1])
        Z = owners.zone(did)
        if not Z or owners.area(did)['kind'] != 'zone': continue
        rz = float(Z['logic_row'].get('Radius') or 300) * G2B_RANGER_ZONE_SCALE[did]
        x, y = int(it[3]), int(it[4])
        cx, cy = _proj(cam, x, y); dx = abs(_proj(cam, x + rz, y)[0] - cx); dy = abs(_proj(cam, x, y + rz)[1] - cy)
        if dx < 1 or dy < 1: continue
        part = (((xx - cx) / dx) ** 2 + ((yy - cy) / dy) ** 2 <= 1) & ~taken
        if part.any(): out[iid] = (part, did, int(it[2]))
        else: out.pop(iid, None)
    return out
