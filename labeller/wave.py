"""Wave attacks (glow.type_kind 'wave': Frank, Ash, Buzz006 / 008, Rosa007, Lola003, ... LABEL_SPEC 2a): ONE body label for
the whole attack (the user's decision, round 2): all wave projectiles one brawler fires on one tick (Frank's fan of 4,
Ash's 3) or in one burst (Buzz, Rosa, Lola) are one attack with one instance id, and its region is the convex hull of their covered paths grown by the
hitbox radius (attack_tris), so it covers the fan between them too.

What the client draws for such a projectile is not its clip (an empty or faint clip, e.g. `gen_particle_empty`) but its
TrailEffect (effects.csv): particle emitters that spawn ground cracks, slashes, sprinkles... where the projectile passes.
The particles do not move (velocity ~0, followEmitter false), are flat on the ground (GroundBasis, layers GroundLow /
GroundHigh) and live a fixed time; where each one lands and which crack sprite it shows is libc rand() (spawn rate, size
and angle jitter), so single particles are not reproducible from the state. Their envelope is:
    the ground band the wave swept, as wide as its hitbox (logic Radius), over the last `span` seconds,
    span = the longest emitterDelay + duration + lifetime + lifetimeJitter of the trail's emitters,
and it stays (shrinking from the start) for `span` after the projectile ends, as the cracks do.
  span(t)                     seconds a point of the path stays covered
  swept_tris(points, r)       (m, 3, 3) world render-coordinate triangles (z 0) of the union of discs of radius r around
                              the path points and the strips between them"""
import csv, functools, json, os
import numpy as np
import rc

_DIR = os.path.dirname(os.path.abspath(__file__))
EMITTERS = os.path.join(_DIR, '..', 'data', 'trail_emitters_all.json')   # every emitter (tools/decode_emitters.py)
EMITTERS_R1 = os.path.join(_DIR, '..', 'data', 'trail_emitters.json')     # round 1: the first emitter of each trail
DEFAULT_SPAN = 0.7   # s, used only when no emitter of the trail is decoded


@functools.lru_cache(maxsize=None)
def _effects():
    """effect name -> its particle emitter names (effects.csv: an effect is its named row plus the unnamed rows below)"""
    R = list(csv.reader(open(os.path.join(rc.ASSETS, 'csv_client', 'effects.csv'), encoding='utf8')))
    i = R[0].index('ParticleEmitterName'); out = {}; cur = None
    for r in R[2:]:
        if r[0]: cur = r[0]; out[cur] = []
        if cur is not None and len(r) > i and r[i]: out[cur].append(r[i])
    return out


@functools.lru_cache(maxsize=None)
def _emitters():
    E = {}
    for p in (EMITTERS_R1, EMITTERS):
        if os.path.exists(p): E.update({k: v for k, v in json.load(open(p)).items() if v})
    return E


def _num(x):
    """a scalar emitter parameter, or a curve [h1, h2, h3, h4, start, end] (two bezier handles, then the end values):
    its largest end value"""
    if isinstance(x, (int, float)): return float(x)
    if isinstance(x, list) and len(x) >= 6: return max(float(v) for v in x[4:] if isinstance(v, (int, float)))
    return 0.0


LINGER = os.path.join(_DIR, '..', 'data', 'wave_linger.json')   # measured on the client's draws (tools/wave_linger.py)


@functools.lru_cache(maxsize=None)
def span(t):
    """seconds one point of a wave's path stays covered: measured for the type on isolated captures (the median time
    its drawing stays visible after its projectiles end, tools/wave_linger.py), else from its trail emitters"""
    if os.path.exists(LINGER):
        m = json.load(open(LINGER)).get(str(t))
        if m: return max(m['median'], 0.1)
    return emitter_span(t)


@functools.lru_cache(maxsize=None)
def emitter_span(t):
    """seconds from the trail emitters: the longest emitterDelay + duration + lifetime (+ jitters) (an upper bound: faint
    emitters fade out long before their lifetime ends)"""
    import projtab
    T = projtab.table()[t]; E = _emitters(); best = 0.0
    for eff in (T['trail'], T['special_trail']):
        for em in _effects().get(eff, []) if eff else []:
            e = (E.get(em) or {}).get('emitter')
            if not e: continue
            dur = max((_num(c.get('duration', 0.0)) + _num(c.get('durationJitter', 0.0)) for c in e.get('components', [])
                       if c.get('__class') == 'SpawnRate' and not c.get('infinite')), default=0.0)
            best = max(best, _num(e.get('emitterDelay', 0.0)) + dur + _num(e.get('lifetime', 0.0)) + _num(e.get('lifetimeJitter', 0.0)))
    return best if best > 0 else DEFAULT_SPAN


def swept_tris(points, r, nseg=16):
    """points (n, 2) logic x, y (the drawn path) -> triangles (m, 3, 3) render coords (x, -y, 0): a disc of radius r
    around every point and a 2r-wide strip between consecutive points"""
    P = np.asarray(points, np.float64).reshape(-1, 2)
    if not len(P): return np.zeros((0, 3, 3))
    tris = []
    a = np.linspace(0, 2 * np.pi, nseg + 1)
    ring = np.stack([np.cos(a), np.sin(a)], 1) * r
    for p in P:
        c = ring + p
        for k in range(nseg): tris.append([p, c[k], c[k + 1]])
    for p, q in zip(P[:-1], P[1:]):
        d = q - p; L = np.hypot(*d)
        if L < 1e-6: continue
        nrm = np.array([-d[1], d[0]]) / L * r
        tris.append([p + nrm, q + nrm, q - nrm]); tris.append([p + nrm, q - nrm, p - nrm])
    T = np.asarray(tris, np.float64)
    W = np.zeros(T.shape[:2] + (3,)); W[..., 0] = T[..., 0]; W[..., 1] = -T[..., 1]
    return W


@functools.lru_cache(maxsize=None)
def follows(t):
    """the wave's drawing travels with the projectile: its trail has emitters with followEmitter (Gale005 super: the
    wave body / head sprites; Gene006 super 2, Rosa007, the Mummy gadget); the static emitters of such a trail are
    drops (fish, starfish, twinkles) left behind, not the wave. Others (Frank, Ash, Buzz, Lola...) leave every
    particle on the ground where it passes."""
    import projtab
    T = projtab.table()[t]; E = _emitters()
    return any((E.get(em) or {}).get('emitter', {}).get('followEmitter')
               for eff in (T['trail'], T['special_trail']) if eff for em in _effects().get(eff, []))


def path_points(e, dt):
    """the drawn ground positions of tracker entry e still covered: those emitted within span(t) seconds of now
    (e['hist'] holds one control per update; e.get('gone', 0) updates have passed since the projectile ended).
    A following wave (follows(t)) covers only where the projectile is now, and its last position for span(t) after it ends."""
    h = e['hist']; n = len(h); g = e.get('gone', 0); now = n - 1 + g; s = span(e['t'])
    if follows(e['t']): return [h[-1]['pos'][:2]] if h and g * dt <= s + 1e-9 else []
    return [c['pos'][:2] for k, c in enumerate(h) if (now - k) * dt <= s + 1e-9]


def hull(P):
    """convex hull (counter-clockwise, Andrew's monotone chain) of points (n, 2)"""
    P = sorted(set(map(tuple, np.round(np.asarray(P, np.float64), 3))))
    if len(P) <= 2: return np.array(P)
    def cross(o, a, b): return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo, up = [], []
    for p in P:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0: lo.pop()
        lo.append(p)
    for p in reversed(P):
        while len(up) >= 2 and cross(up[-2], up[-1], p) <= 0: up.pop()
        up.append(p)
    return np.array(lo[:-1] + up[:-1])


def attack_tris(points, r, nseg=16):
    """ONE region for a whole wave attack: the convex hull of all its projectiles' covered path points, grown by the
    hitbox radius r (hull + disc: the hull's fan, a disc at each hull vertex, a strip along each hull edge) ->
    (m, 3, 3) render coords (x, -y, 0)"""
    H = hull(points)
    if not len(H): return np.zeros((0, 3, 3))
    tris = list(swept_tris(np.r_[H, H[:1]], r, nseg)) if len(H) > 1 else list(swept_tris(H, r, nseg))
    for k in range(1, len(H) - 1):   # the hull's interior (fan from vertex 0; convex)
        T = np.zeros((3, 3)); T[:, 0] = [H[0][0], H[k][0], H[k + 1][0]]; T[:, 1] = [-H[0][1], -H[k][1], -H[k + 1][1]]; tris.append(T)
    return np.asarray(tris, np.float64)


BURST_GAP = 10   # frames (0.5 s): a wave projectile starting this soon after the brawler's previous one belongs to the same attack


def groups(entries):
    """wave projectiles of one attack form one label: fired by one brawler on one tick (Frank: 4 in a fan; Ash: types 1-3
    at once) or in a burst, each within BURST_GAP frames of the previous (Buzz006 / 008, Rosa007, Lola003)
    -> {tracker id: attack id (the smallest id of the attack)}"""
    by = {}
    for i, e in entries: by.setdefault(int(e['row'][2]), []).append((e['first'], i))
    out = {}
    for owner, L in by.items():
        L.sort(); gid = None; prev = None
        for first, i in L:
            if gid is None or first - prev > BURST_GAP: gid = i
            else: gid = min(gid, i)
            out[i] = gid; prev = first
        # ids are allocated in order, so the first of a chain is its smallest id; fix any out-of-order chain
    E = dict(entries)
    for owner, L in by.items():
        chains = {}
        for first, i in sorted(L): chains.setdefault(out[i], []).append(i)
        for g, mem in chains.items():
            known = [E[i]['attack_id'] for i in mem if 'attack_id' in E[i]]   # an attack keeps its id for its whole life,
            m = min(known) if known else min(mem)                              # also after its first projectile is gone
            for i in mem: out[i] = m; E[i].setdefault('attack_id', m)
    return out
