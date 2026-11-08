# away_1105 round 4, group "ended4" (2026-11-06): GLOBAL rule "ended-but-still-visible pellets are labelled" (owner 11-06, binding:
# shown on 015 BanditGirlDefault super t353 / 390 super t355; "every brawler"). Installed as py/fixes/z_1113_ended4.py. Prefix of every global: _en4_
# Switch: env IDLABELS_ENDED_VISIBLE (default 1 = on; 0 = this file does nothing). Read at every labels() call.
#
# What it does. The tracker hides a projectile the moment its logic ends (state 'finished', hit, range) or is dropped, and lists it in
# tr.ended (the `ended` argument of labels(): id, type, own, age in ticks, last logic x / y / z, src, owner). The client, however,
# keeps drawing the projectile's own sprite for a tick or a few (the stars of a Prereg Shelly shot sit at their end point and fade out
# while the hit puffs play), and the earlier global rule (10-23, "ended attacks unlabelled") left them without a label. Now: a flying-
# bullet sprite that is still the front-most draw after its logic ended IS labelled, in its own outline, class / owner of its shot
# (label id = the ended projectile's id, same meta as a live one), until it is no longer drawn.
#
# Rule (data-derived, per ended entry of the tracker (age <= track.END_TICKS) whose shot was labelled while it flew, type t):
#   candidate pixels = the front-most draw shows a shape of the projectile's OWN clip body (idlabels.sprite_pixels code 1), at least half
#     opaque (CUT), carrying no label yet, and
#       - NOT a shape of the effects the type plays when it ends (end_sprites: HitEffectEnv / HitEffectChar / MaxRangeReachedEffect /
#         CancelEffect clips + their emitters): hit flashes, explosions, splashes, debris, smoke and every end animation are never
#         labelled, even where an effect reuses a bullet atlas region (the shape is then ambiguous and stays out);
#       - a shape that the type's FLYING bullets (finished == 0, labelled live) showed in an earlier frame (memory per type);
#   a candidate part (8-connected, >= _EN4_MIN px, mostly not a 1-px seam between live labels) belongs to the nearest ended entry of its
#     type whose logic point, projected at z = 0 .. 450, lies within _EN4_NEAR px of the part's centroid (the bullet is drawn where it ended);
#   the parts of one entry together must be between _EN4_SHRINK and _EN4_GROWTH times the area of that shot's last live label (a fading
#     bullet keeps its size; fragments of smoke / debris sprites that reuse a bullet shape are far smaller or bigger): else nothing is added;
#   the parts' mean colour must be within _EN4_COLOR (RGB distance) of the mean colour of the shot's last live label (measured: fading
#     bullets 3-57; fire / smoke / debris sprites that reuse a bullet shape, Brock 018: 158-267): an effect in another colour is not the bullet;
#   each part is grown by EDGE_GROW px over the opaque pixels of its own draws (sprite edges, as live bodies) and holes are filled (the same
#     steps as the live body in idlabels.labels); its label id is that of the shot's pellet slot whose last live label lay nearest
#     (id (shot, slot, 'body'), same meta class / owner as the live labels of the shot);
#   BODY_ONLY types (Gale snowballs, Rosa Toon donuts): pieces under BODY_ONLY_FRAC of the entry's largest are dropped as for live ones.
# Skipped types: wave / trail-body types (their own ended rules: z_1107_ended_visible), zones, anything not drawn by a clip sprite.
# Skipped entries: shots still drawn live (their own label pipeline decides), shots that never had a label.
# Known limit: a type whose bullet sprite shape is ALSO used by its end effect (e.g. CrossBomber 148 bombs) is ambiguous by shape and gets
# no label after its end.
# Live labels are never touched: only pixels that carry no label are added (ids >= 0 of the previous result stay, same ids).
import os as _en4_os
import sys as _en4_sys
import numpy as _en4_np
from scipy.ndimage import label as _en4_cc, binary_fill_holes as _en4_fill

_EN4_MIN = 6          # px (half resolution): smaller parts are specks
_EN4_CORE = 0.5      # share of a part that must lie more than 1 px away from every existing label (else it is a seam between live labels)
_EN4_NEAR = 45        # px (half resolution): max distance of a part's centroid from the ended entry's logic point projected at z = 0 .. 450
_EN4_SHAPE_SHARE = 0.2   # a clip shape counts as the flying bullet's own when it makes up this share of the shot's last live label
_EN4_SHAPE_IN = 0.8      # share of a part's pixels that must be of such shapes
_EN4_COLOR = 90          # max RGB distance between the mean colour of the ended parts and of the shot's last live label
_EN4_SHRINK = 0.4     # ... and at least this share of it (a few fragments of a smoke / debris sprite are far smaller than the bullet that just ended)
_EN4_GROWTH = 2.5     # the parts of one entry together may be at most this many times the area of its last live label (a bullet fading out does not grow; debris / smoke does)
_EN4_MEM = {}         # label id (shot, pellet slot, 'body') -> dict(oid, tick, ys, xs, vel (dy, dx per tick) or None, area): its last live label
_EN4_LAST = [-1]
_EN4_SHAPES = {}      # (type, own) -> set of clip-body shape keys seen under LIVE (flying, finished == 0) labels of that type
_EN4_WARNED = [False]
_en4_prev_labels = labels


def _en4_on():
    return _en4_os.environ.get('IDLABELS_ENDED_VISIBLE', '1') not in ('0', '', 'off', 'false')


def _en4_kmap(t, own, R):
    """(H, W) int32: key of the clip-BODY atlas shape (code 1 of sprite_pixels) the front-most draw shows, 0 elsewhere (same texture / atlas
    matching as idlabels.sprite_pixels; keys are stable per (type, own): texture order + polygon index)"""
    ids = R['id'][::2, ::2]; uv = R['uv']; H, W = uv.shape[:2]; out = _en4_np.zeros((H, W), _en4_np.int32)
    S = sprites(t, own)
    if not S: return out
    draws = R['draws']; dz = _en4_np.unique(ids[ids > 0])
    size_of = {int(i): (draws[int(i)]['tw'], draws[int(i)]['tht']) for i in dz if int(i) in draws}
    base = 0
    for (file, tex), polys in S.items():
        b0 = base; base += len(polys) + 1
        tw, th = tex_size(file, tex)
        sel = [i for i, s_ in size_of.items() if s_ == (tw, th)]
        if not sel: continue
        byth = {}
        for i in sel: byth.setdefault(draws[i].get('th'), []).append(i)
        sel = [i for h_, L_ in byth.items() if atlas_ok(R, h_, L_, file, tex) for i in L_]
        if not sel: continue
        m = _en4_np.isin(ids, sel) & (uv[..., 0] >= 0)
        if not m.any(): continue
        rm = region_map(t, own, file, tex); Hh, Ww = rm.shape
        ys, xs = _en4_np.nonzero(m); u = _en4_np.clip((uv[ys, xs, 0] * Ww).astype(int), 0, Ww - 1); v = _en4_np.clip((uv[ys, xs, 1] * Hh).astype(int), 0, Hh - 1)
        k = rm[v, u]; hit = k > 0
        if not hit.any(): continue
        body = _en4_np.array([CODE[(p_, c_)] == 1 for _, p_, c_ in polys], bool)
        hit &= body[_en4_np.maximum(k, 1) - 1]
        out[ys[hit], xs[hit]] = b0 + k[hit]
    return out


def _en4_remember(P, objs, tick, R, img=None):
    """per label id (shot id, pellet slot, 'body') of every LIVE object: its last label mask and velocity; per type: the clip shapes a flying bullet showed"""
    if tick is None: return
    if tick < _EN4_LAST[0]: _EN4_MEM.clear(); _EN4_SHAPES.clear()
    _EN4_LAST[0] = tick
    ids = P['ids']; lids = P['lids']; kc = {}
    for o in objs:
        if not o.get('drawn', True) or o.get('ended'): continue
        own_ids = {o['id'], o.get('label_id', o['id'])}
        for k, lid in enumerate(lids):
            if lid[0] not in own_ids or lid[2] != 'body': continue
            m = ids == k
            if not m.any(): continue
            kh = None
            if not o.get('finished', 0) and o['t'] not in ZONE and not trail_body(o['t']):
                key = (o['t'], o['own'])
                try:
                    if key not in kc: kc[key] = _en4_kmap(o['t'], o['own'], R)
                    u, c = _en4_np.unique(kc[key][m & (kc[key] > 0)], return_counts=True)
                    _EN4_SHAPES.setdefault(key, set()).update(u.tolist()); kh = dict(zip(u.tolist(), c.tolist()))
                except Exception: pass
            ys, xs = _en4_np.nonzero(m); old = _EN4_MEM.get(lid)
            vel = None
            if old is not None and tick > old['tick']:
                vel = ((ys.mean() - old['ys'].mean()) / (tick - old['tick']), (xs.mean() - old['xs'].mean()) / (tick - old['tick']))
            elif old is not None: vel = old['vel']
            if kh is None and old is not None: kh = old.get('kh')
            col = img[ys, xs].astype(_en4_np.float64).mean(0) if img is not None else None
            _EN4_MEM[lid] = dict(oid=o['id'], tick=tick, ys=ys, xs=xs, vel=vel, area=len(ys), kh=kh, col=col)


def _en4_add(P, objs, cam, R, state, ended, img):
    import projtab as _pt, owners as _ow
    ids = P['ids'].copy(); share = P['share'].copy(); lids = list(P['lids']); taken = ids >= 0
    H, W = ids.shape; cv = R['cv'][::2, ::2]; ids_h = R['id'][::2, ::2]
    live = {o['id'] for o in objs if o.get('drawn', True) and not o.get('ended')}
    T = _pt.table(); groups = {}
    for e in ended:
        t = e['t']
        if e['id'] in live or t not in T or trail_body(t) or t in ZONE: continue
        mine = [lid for lid, m in _EN4_MEM.items() if m['oid'] == e['id']]
        if mine: groups.setdefault((t, e['own']), []).append((e, mine))
    dbg = bool(_en4_os.environ.get('IDLABELS_EN4_DEBUG')); tick = (state or {}).get('tick')
    added = {}; new_taken = _en4_np.zeros((H, W), bool)
    for (t, own), es in groups.items():
        try:
            if not has_clip(t, own): continue
            C = sprite_pixels(t, own, R)
        except Exception:
            continue
        if not (C == 1).any(): continue
        try: E = sprite_pixels(('end', t), own, R) > 0
        except Exception: E = _en4_np.zeros((H, W), bool)
        try: K = _en4_kmap(t, own, R)
        except Exception: continue
        seen = _EN4_SHAPES.get((t, own))
        if not seen: continue
        cand = (C == 1) & (cv >= CUT) & ~taken & ~E & _en4_np.isin(K, list(seen))   # only shapes the type's FLYING bullet showed
        if not cand.any(): continue
        near_live = _dilate(taken, 1)
        allowed = {}   # ended entry id -> the clip shapes that make up its last live label (>= _EN4_SHAPE_SHARE of its pixels)
        for e, mine in es:
            tot = {}
            for lid in mine:
                for kk, c in (_EN4_MEM[lid].get('kh') or {}).items(): tot[kk] = tot.get(kk, 0) + c
            n_ = float(sum(tot.values())) or 1.0
            allowed[e['id']] = {kk for kk, c in tot.items() if c >= _EN4_SHAPE_SHARE * n_}
        pts = [(e, [_proj(cam, e['x'], e['y'], z) for z in (0, 150, 300, 450)]) for e, _ in es]
        L, n = _en4_cc(cand, structure=_en4_np.ones((3, 3)))
        sz = _en4_np.bincount(L.ravel())
        parts = {}   # ended entry id -> [part mask, ...]
        for k in range(1, n + 1):
            if sz[k] < _EN4_MIN: continue
            q = L == k
            if (q & ~near_live).sum() < _EN4_CORE * sz[k]: continue   # seams / slivers between live labels (1-px gaps), not a bullet
            ys, xs = _en4_np.nonzero(q); cx, cy = xs.mean(), ys.mean(); best = None
            for e, pp in pts:
                d = min(_en4_np.hypot(cx - a, cy - b) for a, b in pp)
                if d <= _EN4_NEAR and (best is None or d < best[0]): best = (d, e['id'])
            if dbg: print('EN4c', tick, 'comp', int(sz[k]), 'cen', int(cx), int(cy), 'best', best, file=_en4_sys.stderr)
            if best is None: continue
            if (_en4_np.isin(K[q], list(allowed[best[1]])).sum() < _EN4_SHAPE_IN * sz[k]): continue   # not made of the shapes the flying bullet was
            parts.setdefault(best[1], []).append(q)
        for e, mine in es:
            ms = parts.get(e['id'])
            if not ms: continue
            if t in BODY_ONLY and len(ms) > 1:
                big = max(int(m.sum()) for m in ms); ms = [m for m in ms if m.sum() >= BODY_ONLY_FRAC * big]
            tot = sum(int(m.sum()) for m in ms); ar = sum(_EN4_MEM[lid]['area'] for lid in mine)
            cl = [(_EN4_MEM[lid]['area'], _EN4_MEM[lid]['col']) for lid in mine if _EN4_MEM[lid].get('col') is not None]
            if img is not None and cl:
                live_col = sum(a_ * c_ for a_, c_ in cl) / float(sum(a_ for a_, _ in cl))
                pm = _en4_np.zeros(img.shape[:2], bool)
                for m in ms: pm |= m
                cd = float(_en4_np.linalg.norm(img[pm].astype(_en4_np.float64).mean(0) - live_col))
                if dbg: print('EN4col', tick, e['id'], 'age', e['age'], 'dist %.0f' % cd, file=_en4_sys.stderr)
                if cd > _EN4_COLOR: continue   # another colour than the bullet that just ended: an effect that reuses its shapes
            if dbg: print('EN4', tick, e['id'], 'age', e['age'], 'parts', len(ms), 'area', tot, 'last', ar, 'ratio %.2f' % (tot / ar), file=_en4_sys.stderr)
            if not (_EN4_SHRINK * ar <= tot <= _EN4_GROWTH * ar): continue   # debris / smoke rather than the fading bullet
            for m in ms:
                own_d = _en4_np.unique(ids_h[m]); own_d = own_d[own_d > 0]
                b = m.copy()
                ok = _en4_np.isin(ids_h, own_d) & (cv >= CUT) & ((C == 0) | (C == 1)) & ~taken & ~E & ~new_taken
                for _ in range(EDGE_GROW): b |= _dilate(b, 1) & ok
                b |= _en4_fill(b) & ~taken & ~new_taken
                # label id: the pellet slot whose last label was nearest (shots label per slot), else the entry's first
                ys, xs = _en4_np.nonzero(m)
                lid = min(mine, key=lambda l: (_EN4_MEM[l]['xs'].mean() - xs.mean()) ** 2 + (_EN4_MEM[l]['ys'].mean() - ys.mean()) ** 2)
                prev = added.get(lid)
                added[lid] = ((b | prev[0]) if prev else b, e); new_taken |= b
    if not added: return P
    for lid, (body, e0) in added.items():
        if lid not in lids: lids.append(lid)
        k = lids.index(lid)
        m = body & (ids < 0)
        ids[m] = k; share[m] = _en4_np.maximum(cv[m] / 255.0, 0.5)
    P2 = dict(P); P2.update(ids=ids, share=share, lids=lids)
    hits = dict(P.get('hits') or {})
    for lid, (body, e0) in added.items(): hits.setdefault(lid[0], e0)
    P2['hits'] = hits
    mt = P.get('meta')
    if mt is not None and state is not None:
        mt = dict(mt)
        did_of = {int(c[0]): int(c[1]) for c in state['chars']}
        hero_of = {int(c[2]): int(c[1]) for c in state['chars'] if _ow.character(int(c[1]))['kind'] == 'hero'}
        for lid, (body, e0) in added.items():
            if lid[0] not in mt: mt[lid[0]] = _ow.runtime(e0['t'], did_of.get(e0['src']), hero_of.get(e0['owner']))
        P2['meta'] = mt
    return P2


def labels(objs, cam, R, state=None, ended=(), img=None):
    P = _en4_prev_labels(objs, cam, R, state=state, ended=ended, img=img)
    if not _en4_on(): return P
    tick = (state or {}).get('tick')
    try: _en4_remember(P, objs, tick, R, img)
    except Exception:
        if _en4_os.environ.get('IDLABELS_EN4_DEBUG'):
            import traceback; traceback.print_exc(file=_en4_sys.stderr)
    if not ended or tick is None: return P
    try:
        return _en4_add(P, objs, cam, R, state, list(ended), img)
    except Exception as ex:   # never lose the frame's labels
        if not _en4_WARNED[0]:
            _en4_WARNED[0] = True; import traceback; traceback.print_exc(file=_en4_sys.stderr)
        return P
