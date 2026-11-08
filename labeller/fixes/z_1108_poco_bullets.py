# Poco (DeadMariachi) attack + super, plain bullet rule (task poco_bullets, owner 2026-11-03): label the drawn cloud parts of a LIVE shot
# that are left unlabelled. Types: PB_TYPES = POCO_ATK | POCO_SUPER (6000335..6000356, 6001521).
# Cause (diagnosed on capture 028, ticks 273-277 of the attack 0 row): the installed rule z_1028_poco "a member with finished != 0 has ended
# its logic" drops a member as soon as its state row says finished = 1 (natural end of range, a wall hit is finished = 4 and not drawn).
# But the client still draws that member's cloud at FULL size on that tick (finished = 1 and drawn = True; the cloud only dissolves into the
# end splash 1-2 ticks later), so a whole drawn cloud (the lower / upper cloud of tick 275) was left outside the label.
# Fix: for the labelling step such a member counts as running while its drawn cloud is still intact: finished == 1, drawn, and the
# type's cloud pixels nearest to it (cells of the drawn members, projected anchors) cover at least PB_KEEP of the largest cloud area this
# member has had so far (memory per process, keyed by member id) -- or, with no memory, at least PB_ABS half-res pixels. Once the cloud has
# dissolved (splash, small area) the member is finished as before, so end animations stay unlabelled. Everything else (the per-skin
# pipelines before this file and the global fill/smooth after it) runs unchanged on the objects whose finished flag is cleared for the step.
import numpy as _pbn

PB_TYPES = {6000335, 6000337, 6000339, 6000341, 6000343, 6000345, 6000347, 6000349, 6000351, 6000353, 6000355,
            6000336, 6000338, 6000340, 6000342, 6000344, 6000346, 6000348, 6000350, 6000352, 6000354, 6000356, 6001521}
PB_KEEP = 0.6      # a finished member's cloud must still have this share of its largest area so far
PB_R = 45          # half-res px: radius around a member's projected anchor in which its cloud is measured
PB_OPEN = 2        # px: radius of the opening that tidies the added part
PB_ABS = 500       # half-res pixels: area threshold when the member has no memory
_pb_mem = {'tick': -1, 'max': {}}
_pb_prev_labels = labels


def _pb_areas(objs, cam, R, types):
    """{member id: pixels of the type's cloud sprites in its cell} (cells: nearest drawn member of the same type, projected anchor)"""
    out = {}
    for t in types:
        mem = [o for o in objs if o['t'] == t and o.get('drawn', True) and not o.get('ended')]
        if not mem: continue
        C = None
        for own in {o['own'] for o in mem}:
            c = sprite_pixels(t, own, R) > 0
            C = c if C is None else (C | c)
        if C is None or not C.any(): continue
        ys, xs = _pbn.nonzero(C)
        best = _pbn.full(len(ys), _pbn.inf); who = _pbn.full(len(ys), -1)
        for k, m in enumerate(mem):
            px, py = _proj(cam, float(m['anchor'][0]), -float(m['anchor'][1]), 0.0)
            d = (xs - px) ** 2 + (ys - py) ** 2; u = d < best; best[u] = d[u]; who[u] = k
        near = best <= PB_R * PB_R   # only the cloud around the anchor counts (a lone survivor must not own the whole frame)
        for k, m in enumerate(mem): out[m['id']] = int(((who == k) & near).sum())
    return out


def labels(objs, cam, R, state=None, ended=(), img=None):
    tick = (state or {}).get('tick')
    if tick is None or not any(o['t'] in PB_TYPES for o in objs): return _pb_prev_labels(objs, cam, R, state=state, ended=ended, img=img)
    if tick < _pb_mem['tick']: _pb_mem['max'].clear()
    _pb_mem['tick'] = tick
    try:
        types = {o['t'] for o in objs if o['t'] in PB_TYPES}
        A = _pb_areas(objs, cam, R, types)
    except Exception:
        A = {}
    out = []; flipped = False
    for o in objs:
        if o['t'] in PB_TYPES and o['id'] in A:
            mx = _pb_mem['max'].get(o['id'], 0); a = A[o['id']]
            if o.get('finished', 0) == 1 and o.get('drawn', True) and not o.get('ended'):
                if (mx > 0 and a >= PB_KEEP * mx) or (mx == 0 and a >= PB_ABS): o = dict(o, finished=0); flipped = True
            if o.get('finished', 0) == 0: _pb_mem['max'][o['id']] = max(mx, a)
        out.append(o)
    Pn = _pb_prev_labels(out, cam, R, state=state, ended=ended, img=img)
    if not flipped: return Pn
    # only what this fix ADDED is tidied: strips thinner than ~2 * PB_OPEN px (a note sprite drawn across a turret, ...) are cut off the
    # added part (opening, then 1 px of reconstruction); the label that was there before is not touched
    try:
        from scipy.ndimage import binary_opening, binary_dilation
        Po = _pb_prev_labels(objs, cam, R, state=state, ended=ended, img=img)
        add = (Pn['ids'] >= 0) & ~(Po['ids'] >= 0)
        if add.any():
            yy, xx = _pbn.mgrid[-PB_OPEN:PB_OPEN + 1, -PB_OPEN:PB_OPEN + 1]; dk = (yy * yy + xx * xx) <= PB_OPEN * PB_OPEN
            keep = binary_dilation(binary_opening(add, dk), _pbn.ones((3, 3), bool)) & add
            cut = add & ~keep
            Pn['ids'][cut] = -1; Pn['share'][cut] = 0
    except Exception:
        pass
    return Pn
