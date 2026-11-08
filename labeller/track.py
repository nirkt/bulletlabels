"""State-driven projectile tracking: everything the client derives from consecutive game states for its projectile
display objects, without any knowledge of the client's draws (UNDERSTANDING.md 2, 5).
  WF = frame_weights(B)                    the client's interpolation weight per frame (exact where objects move)
  TR = trajectories(B)                     per projectile: sim (start, end) and its getAngle (exact / ambiguous)
  tr = Tracker(TR, rng=random.Random(0))   one display object per projectile the client shows
  for each frame: tr.step(f, state, prev_state, WF[f], team_of(state, viewer), None, viewer)
                  for e in tr.d.values(): tr.predict_parts(e), tr.mesh(e) ...
Things the state does not determine are drawn with rng (Tracker(rng=...)): the RandomStartFrame start frame (the
client's rand()), and the angle when the trajectory leaves two candidates. With rng=None the first candidate / frame
0 is used."""
import collections
import numpy as np
import projdraw as PD, projtab

J = {k: i for i, k in enumerate(["id", "dataId", "owner", "x", "y", "z", "px", "py", "src", "damage", "travel", "travel0",
                                  "finished", "impact", "client", "cx", "cy", "cz"])}
NIL = -2147483648
END_TICKS = 14   # ticks a projectile's hit / end effect is searched for after it ended (0.7 s)


def tdiv(a, b):
    q = abs(int(a)) // b
    return -q if a < 0 else q


def lerp_i(cur, prev, w):
    """the client's snapshot interpolation (0xc210b0): cur * w / 1000 + prev * (1000 - w) / 1000, unless equal"""
    return int(cur) if cur == prev else tdiv(int(cur) * w, 1000) + tdiv(int(prev) * (1000 - w), 1000)


C = {k: i for i, k in enumerate(["id", "dataId", "owner", "x", "y", "z", "hp", "maxHp", "state", "facing", "lastDmgTick", "spawnProt",
                                  "f469", "ammo", "cooldown", "ultiActive", "lastAttacker", "charType", "vtKind", "client", "cx", "cy", "cz"])}


def weight_range(s, sp, viewer=None):
    """integer interpolation weights w (permille) that reproduce the client positions of all moving projectiles and
    characters (not the viewer's own, which the client predicts) from the previous and current state"""
    W = np.arange(1001); ok = np.ones(1001, bool); n = 0
    rows = []
    pp = {int(p[0]): p for p in sp['proj']}
    for p in s['proj']:
        if p[J['client']] == 1 and int(p[0]) in pp:
            q = pp[int(p[0])]; rows += [(p[J[k]], q[J[k]], p[J['c' + k]]) for k in ('x', 'y', 'z')]
    pc = {int(c[0]): c for c in sp['chars']}
    for c in s['chars']:
        if c[C['client']] == 1 and int(c[0]) in pc and (viewer is None or int(c[C['owner']]) != viewer):
            q = pc[int(c[0])]; rows += [(c[C[k]], q[C[k]], c[C['c' + k]]) for k in ('x', 'y', 'z')]
    for cur, prev, c in rows:
        cur, prev, c = int(cur), int(prev), int(c)
        if cur == prev or c == NIL: continue
        a = cur * W; b = prev * (1000 - W)
        o = (np.where(a < 0, -(np.abs(a) // 1000), a // 1000) + np.where(b < 0, -(np.abs(b) // 1000), b // 1000)) == c
        if o.any(): ok &= o; n += 1
    w = np.nonzero(ok)[0]
    return w if n else None


def frame_weight(s, sp, prev_w=-1, viewer=None):
    """the frame's interpolation weight: consistent with every interpolated object; among several, the one nearest the
    previous frame's (it changes slowly); -1 when nothing constrains it"""
    if sp is None: return prev_w
    w = weight_range(s, sp, viewer)
    if w is None or not len(w): return prev_w
    if prev_w >= 0: return int(w[np.argmin(np.abs(w - prev_w))])
    return int(np.median(w))


def frame_weights(B):
    """interpolation weight of every frame: exact where moving objects constrain it (nearest the previous frame's
    when several fit), else the nearest constrained frame's value (the weight drifts slowly)"""
    fr = sorted(B.meta); rng = {}; sp = None
    for f in fr:
        s = B.state(f)
        if s is not None and sp is not None:
            r = weight_range(s, sp, int(B.meta[f].get('player', 0)))
            if r is not None and len(r): rng[f] = r
        sp = s
    out = {}; prev = -1
    for f in fr:
        if f in rng:
            r = rng[f]; prev = int(r[np.argmin(np.abs(r - prev))]) if prev >= 0 else int(np.median(r)); out[f] = prev
    known = sorted(out)
    for f in fr:
        if f in out or not known: continue
        i = np.searchsorted(known, f)
        cands = [known[j] for j in (i - 1, i) if 0 <= j < len(known)]
        out[f] = out[min(cands, key=lambda g: abs(g - f))]
    return out


def _tdiv_arr(a, T):
    return np.where(a < 0, -(np.abs(a) // T), a // T)


def _lerp_solutions(ticks, v, T, t0, W=48):
    """ALL integer (start, end) with tdiv((T - t) start, T) + tdiv(t end, T) == v for every sample (sim 0xb76488: one
    C division per term), searched in a window of +-W around the least-squares solution -> list of (start, end)"""
    t = ticks - t0
    if (t < 0).any() or (t > T).any(): return []
    A = np.c_[(T - t) / T, t / T]
    c, *_ = np.linalg.lstsq(A, v.astype(float), rcond=None)
    S = np.arange(int(round(c[0])) - W, int(round(c[0])) + W + 1)
    E = np.arange(int(round(c[1])) - W, int(round(c[1])) + W + 1)
    ok = np.ones((len(S), len(E)), bool)
    for ti, vi in zip(t.tolist(), v.tolist()):
        ok &= (_tdiv_arr((T - ti) * S, T)[:, None] + _tdiv_arr(ti * E, T)[None, :]) == vi
        if not ok.any(): return []
    i, j = np.nonzero(ok)
    return list(zip(S[i].tolist(), E[j].tolist()))


def trajectories(B):
    """per projectile id: [start x, start y, end x, end y, how, angles] of its flight as the simulation stores them; the
    logic angle the client gets is getAngle(end - start), set once at creation (0xb6ff00 in 0xb6f958; the tick 0xb70408
    rewrites it only for curving and homing shots).
    Exact way: the sim places a straight projectile at tdiv((T - t) start, T) + tdiv(t end, T) (0xb76488) for t ticks
    out of T, and adds 1000 / T to travel permille each tick (so T is known from the per-tick travel); every integer
    (start, end) that reproduces all observed positions (whole life, plus the previous position the state carries) is
    enumerated: 'exact' when all give one getAngle, 'ambiguous' with the candidate angles otherwise (the state does not
    hold the angle: near a step of the atan table two angles fit the same positions). Fallback: line fit against travel."""
    pts = {}; tr0 = {}
    for f in sorted(B.meta):
        s = B.state(f)
        if s is None: continue
        for p in s['proj']:
            i = int(p[0])
            x, y, px, py = int(p[J['x']]), int(p[J['y']]), int(p[J['px']]), int(p[J['py']])
            if i not in pts and (px or py) and (px, py) != (x, y) and int(p[J['travel0']]) > 0:   # the sim's previous position
                pts[i] = [(int(p[J['travel']]) - int(p[J['travel0']]), px, py, 0, int(s['tick']) - 1)]
            pts.setdefault(i, []).append((int(p[J['travel']]), x, y, int(p[J['finished']]), int(s['tick'])))
            if int(p[J['travel0']]) > 0: tr0[i] = int(p[J['travel0']])
    out = {}
    for i, L in pts.items():
        A = np.array([q for k, q in enumerate(L) if not q[3] or k == 0], float)
        A = A[np.unique(A[:, 4], return_index=True)[1]]
        res = None
        if len(A) >= 2:
            ticks = A[:, 4].astype(np.int64); xs = A[:, 1].astype(np.int64); ys = A[:, 2].astype(np.int64)
            d0 = tr0.get(i, 0)
            Ts = [T for T in range(2, 400) if 1000 // T == d0] if d0 > 0 else list(range(max(2, int(ticks[-1] - ticks[0]) + 1), 64))
            k0 = int(ticks[0]) - (int(A[0, 0]) // d0 if d0 > 0 else 0)
            angs = set(); sol = None
            for T in Ts:
                for t0 in (range(k0 - 2, k0 + 3) if d0 > 0 else range(int(ticks[0]) - 8, int(ticks[0]))):
                    sx = _lerp_solutions(ticks, xs, T, t0)
                    if not sx: continue
                    sy = _lerp_solutions(ticks, ys, T, t0)
                    if not sy: continue
                    for a_ in {(ex - sx_, ey - sy_) for sx_, ex in sx for sy_, ey in sy}:
                        angs.add(PD.iatan2(*a_))
                    sol = (sx[0][0], sy[0][0], sx[0][1], sy[0][1])
            if len(angs) == 1: res = list(sol) + ['exact', sorted(angs)]
            elif len(angs) > 1: res = list(sol) + ['ambiguous', sorted(angs)]
        if res is None:
            A2 = A[np.unique(A[:, 0], return_index=True)[1]]
            if len(A2) >= 2 and np.ptp(A2[:, 0]) > 0:
                X = np.c_[np.ones(len(A2)), A2[:, 0]]
                cx, *_ = np.linalg.lstsq(X, A2[:, 1], rcond=None); cy, *_ = np.linalg.lstsq(X, A2[:, 2], rcond=None)
                res = [round(cx[0]), round(cy[0]), round(cx[0] + 1000 * cx[1]), round(cy[0] + 1000 * cy[1]), 'fit']
            elif len(L) >= 2:
                res = [L[0][1], L[0][2], L[-1][1], L[-1][2], 'ends']
            if res is not None: res.append([PD.iatan2(res[2] - res[0], res[3] - res[1])])
        if res is not None: out[i] = res
    return out


def team_of(s, player):
    for pl in s['players']:
        if int(pl[0]) == int(player): return int(pl[1])
    return None




class Tracker:
    """per projectile id: display object, control history, logic angle (getAngle of the flight vector).
    fit_starts=True leaves the display of RandomStartFrame types unbuilt (verify.py fits the start frame to the draws);
    otherwise the start frame is drawn with rng (the client's rand() % n)."""
    def __init__(self, traj=None, rng=None, fit_starts=False):
        self.d = {}; self.traj = traj or {}; self.rng = rng; self.fit_starts = fit_starts
        self.pool = {}   # the client's pools of shadow / glow clips (projdraw.GroundParts)
        self.gone = {}   # wave attacks whose projectile ended, while their trail is still on the ground (wave.py)
        self.ended = {}  # projectiles that ended in the last END_TICKS ticks: their hit / end effects (idlabels.end_labels)

    def step(self, f, s, sp, w, viewer_team, types, viewer=0):
        T = projtab.table(); pp = {int(q[0]): q for q in sp['proj']} if sp is not None else {}
        pc = {int(c[0]): c for c in sp['chars']} if sp is not None else {}
        seen = set()
        live = {int(p[0]) for p in s['proj'] if p[J['client']] == 1 and int(p[J['dataId']]) in T and (not types or int(p[J['dataId']]) in types)}
        import wave
        for i in list(self.gone):   # ended wave attacks: one more update; gone once no path point is still covered
            g = self.gone[i]; g['gone'] = g.get('gone', 0) + 1
            if not wave.path_points(g, PD.DT): del self.gone[i]
        for i in list(self.ended):
            self.ended[i]['age'] += 1
            if self.ended[i]['age'] > END_TICKS: del self.ended[i]
        for i in list(self.d):   # objects the client dropped: their ground clips go back to the pool first
            if i not in live:
                e = self.d[i]; self._drop(i)
                if e.get('row') is not None and i not in self.ended:
                    r = e['row']; self.ended[i] = dict(id=i, t=e['t'], own=e['own'], age=0, x=int(r[J['x']]), y=int(r[J['y']]),
                                                      z=int(r[J['z']]), src=int(r[J['src']]), owner=int(r[J['owner']]))
                if e['hist'] and _is_wave(e): e['gone'] = 0; self.gone[i] = e
        for p in s['proj']:
            i = int(p[0]); t = int(p[J['dataId']])
            if p[J['client']] != 1 or t not in T or (types and t not in types): continue
            seen.add(i)
            e = self.d.get(i)
            if e is None:
                own = team_of(s, p[J['owner']]) == viewer_team
                e = self.d[i] = dict(t=t, own=own, first=f, spawn=(int(p[J['x']]), int(p[J['y']])), hist=[], disp=None, start=None,
                                     lead=(0, 0), zoff=T[t]['zoffset'], ground=PD.GroundParts(t, own, self.pool))
                if i in pp: e['spawn'] = (int(pp[i][J['x']]), int(pp[i][J['y']]))
                # own-shot lead (0x53a9a8): the viewer's character's drawn (predicted) position minus its interpolated
                # server position, fixed on the projectile's first update
                if own and T[t]['shot_by_hero'] and int(p[J['owner']]) == viewer and w >= 0:
                    for c in s['chars']:
                        if int(c[C['owner']]) == viewer and int(c[C['vtKind']]) == 0 and int(c[0]) in pc:
                            q = pc[int(c[0])]
                            e['lead'] = (int(c[C['cx']]) - lerp_i(int(c[C['x']]), int(q[C['x']]), w),
                                         int(c[C['cy']]) - lerp_i(int(c[C['y']]), int(q[C['y']]), w))
            if i in self.traj and (self.traj[i][2] != self.traj[i][0] or self.traj[i][3] != self.traj[i][1]):
                e['acands'] = self.traj[i][5]; e['astat'] = {'exact': 0, 'ambiguous': 1}.get(self.traj[i][4], 2)
                if 'angle_fit' not in e and len(e['acands']) > 1 and self.rng is not None: e['angle_fit'] = self.rng.choice(e['acands'])
                e['angle'] = e.get('angle_fit', e['acands'][0])
            else:
                x0, y0 = e['spawn']; dx, dy = int(p[J['x']]) - x0, int(p[J['y']]) - y0
                if dx or dy: e['angle'] = PD.logic_angle_of(dx, dy)
                e['astat'] = 3
            tr = int(p[J['travel']])
            if w >= 0 and i in pp: tr = lerp_i(tr, int(pp[i][J['travel']]), w)
            lead_now = PD.lead_offset(e['lead'], tr)
            pos = (int(p[J['cx']]) + lead_now[0], int(p[J['cy']]) + lead_now[1], int(p[J['cz']]) + e['zoff'])
            ctrl = dict(angle=e.get('angle', 0), travel=tr, finished=int(p[J['finished']]), pos=pos)
            e['hist'].append(ctrl); e['row'] = p; e['ctrl'] = ctrl; e['f'] = f
            if int(p[J['finished']]) and i not in self.ended:   # it hit / reached its range: its end effect starts
                self.ended[i] = dict(id=i, t=t, own=e['own'], age=0, x=int(p[J['x']]), y=int(p[J['y']]), z=int(p[J['z']]),
                                     src=int(p[J['src']]), owner=int(p[J['owner']]))
            e['lead_now'] = lead_now
            e['zoff_now'] = e['zoff']; e['zoff'] = PD.zoffset_next(e['zoff'])
            if e['disp'] is None and not (self.fit_starts and T[t]['random_start'] in (1, 2, 3, 4)):
                e['start'] = self.start_frame(e) if T[t]['random_start'] in (1, 2, 3, 4) else 0
                e['disp'] = PD.Display(t, e['own'], e['start'], ground=False)
                for c in e['hist']: e['disp'].update(**c)
            elif e['disp'] is not None:
                for _, d in e.get('cands') or [(None, e['disp'])]: d.update(**ctrl)
            e['ground'].update(self._hidden(e, ctrl))

    def fill_gone(self, n):
        for g in self.gone.values(): g['gone'] = g.get('gone', 0) + n
        for i in list(self.ended):
            self.ended[i]['age'] += n
            if self.ended[i]['age'] > END_TICKS: del self.ended[i]

    def _hidden(self, e, ctrl):
        """the object's visibility this frame (the display's, or computed alike while the display waits for a fit)"""
        if e['disp'] is not None: return e['disp'].hidden
        T = projtab.table()[e['t']]; n = len(e['hist']); age = n * PD.DT
        fin = sum(PD.DT for c in e['hist'] if c['finished'] in (1, 3)) if not T['hide_faster'] else 0.0
        if any(c['finished'] and (c['finished'] not in (1, 3) or T['hide_faster']) for c in e['hist']): fin = 1.0
        return age < T['hide_time'] * 0.001 or (fin > 0.1 and not T['is_beam'])

    def _drop(self, i):
        e = self.d.pop(i)
        if e.get('ground') is not None: e['ground'].release(self.pool)

    def fill(self, n):
        self.fill_gone(n)
        self._fill(n)

    def _fill(self, n):
        """n frames the recording does not hold (a gap in the ticks): the client updated every display object once per
        frame anyway (dt each), so the clips advance n times; travel and position continue at the last per-tick step
        (approximate between the stored ticks; the objects are marked gap_filled)"""
        for e in self.d.values():
            if e.get('disp') is None or not e['hist']: continue
            c = dict(e['hist'][-1]); d0 = int(e['row'][J['travel0']]) if int(e['row'][J['travel0']]) > 0 else 0
            for _ in range(n):
                c = dict(c, travel=min(1000, c['travel'] + d0))
                e['disp'].update(**c); e['hist'].append(c)
            e['gap_filled'] = True

    def proj(self, e):
        p = e['row']; c = e['ctrl']
        return PD.Proj(e['t'], e['own'], float(p[J['cx']]), float(p[J['cy']]), float(p[J['cz']]), c['angle'], e['f'] - e['first'],
                       travel=c['travel'], finished=c['finished'], oid=int(p[0]), own_offset=e.get('lead_now', (0, 0)),
                       zoffset=e.get('zoff_now', 0))

    def predict_parts(self, e):
        """{part: placed polygons} of everything the display object draws (main clip, shadow / glow)"""
        if e['disp'] is None or e['disp'].hidden: return {}
        P = self.proj(e); pp = dict(e['ground'].polys()); pp.update(e['disp'].part_polys())
        return {k: PD.place(P, v, part=k) for k, v in pp.items()}

    def start_frame(self, e):
        """the client's rand() % n for RandomStartFrame (any value in projdraw.start_range)"""
        if self.rng is None: return 0
        T = projtab.table()[e['t']]; S = PD.sc(T['file']) if T['file'] else None
        nm = T['blue'] if e['own'] else T['red']
        if S is None or S.exports.get(nm) not in S.clips: return 0
        n = PD.start_range(T, S, S.exports[nm])
        return tuple(self.rng.randrange(k) for k in n) if isinstance(n, tuple) else self.rng.randrange(n)

    def mesh(self, e):
        """4 x 4 transform of the type's 3D object (None without one / while hidden)"""
        T = projtab.table()[e['t']]
        if not (T['blue_scw'] if e['own'] else T['red_scw']) or e['disp'] is None or e['disp'].hidden: return None
        return PD.mesh_matrix(self.proj(e), e['disp'])

    def predict(self, e):
        return PD.place(self.proj(e), e['disp'].polys()) if e['disp'] is not None else None


def _is_wave(e):
    """wave types and the trail-body types (idlabels.trail_body): kept in Tracker.gone while their trail is drawn"""
    import idlabels
    return idlabels.trail_body(e['t'])
