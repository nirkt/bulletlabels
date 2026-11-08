"""What the client draws for one projectile: its display object built and updated like the client's projectile
object (ctor 0x539f28, update 0x53a938), flattened from the game's sprite files (sc6.py), placed in the world like the
renderer does. See UNDERSTANDING.md.
  P = Proj(t, own, x, y, z, angle=..., age=..., travel_permille=..., ...)   (client logic values, world units)
  polys = draw(P, cam_forward) -> [dict(path, shape, cmd, tex, uv (n,2) u16, world (n,3) render coords (x, -y, z),
                                         local (n,2) root clip px, mul (4,), add (3,), blend ('normal'|'add'|..),
                                         part ('clip'|'shadow'|'glow'), pellet (index of the root child slot or -1))]"""
import math, os
import numpy as np
import rc, projtab

import rc as _rc
LIB = _rc.LIBG
_lib = open(LIB, 'rb').read()
SIN = np.frombuffer(_lib, '<i2', 91, 0x1ff68e).astype(np.int64)   # LogicMath sin table (0..90 deg, x1024), 0xd005e0
ATAN = np.frombuffer(_lib, 'i1', 129, 0x1ff60c).astype(np.int64)  # LogicMath atan table (128 steps), 0xd00960
STAGE = 48.0                    # stage px per tile horizontally (0x1fb8e4), vertically 36 (0x1fb8e8): 0x544c34
PX = 300.0 / STAGE              # world units per stage px in the drawn plane (6.25) [V data]
HEIGHT = 0.6                    # drawn height / logic height (global 0.6 at 0x1f94f4, used at 0x53c258 for meshes) [V]
BLEND = {0: 'normal', 3: 'multiply', 4: 'screen', 8: 'add', 12: 'erase', 15: 'erase2'}
_SC = {}


def sc(name):
    import sc6
    if name not in _SC: _SC[name] = sc6.SC6(os.path.join(rc.ASSETS, name))
    return _SC[name]


def isin(deg):
    """LogicMath::sin(deg) x1024 (0xd005e0)"""
    u = deg % 360
    v = u - 180 if u >= 180 else u
    w = v if v < 91 else 180 - v
    s = int(SIN[w])
    return s if u < 180 else -s


def iatan2(x, y):
    """LogicMath::getAngle(x, y) in degrees 0..359 (0xd00960)"""
    x = int(x); y = int(y)
    if x == 0 and y == 0: return 0
    def q(n, d): return int(ATAN[int((n << 7) / d)]) if d else 0   # C int division (truncation)
    if x > 0 and y >= 0:
        return int(ATAN[(y << 7) // x]) if y < x else 90 - int(ATAN[(x << 7) // y])
    ax = abs(x)
    if x < 1 and y > 0:
        return int(ATAN[(ax << 7) // y]) + 90 if ax < y else 180 - int(ATAN[(y << 7) // ax])
    ay = abs(y)
    if x >= 0 or y > 0:   # x >= 0, y < 0
        if ax < ay: return int(ATAN[(ax << 7) // ay]) + 270
        if x != 0:
            r = 360 - int(ATAN[(ay << 7) // ax])
            if r > 359: r = -int(ATAN[(ay << 7) // ax])
            return r + 360 if r < 0 else r
        return x
    if ay < ax: return int(ATAN[(ay << 7) // ax]) + 180
    return 270 - int(ATAN[(ax << 7) // ay]) if y != 0 else 0


def _div1024(v):
    """(v + (v < 0 ? 1023 : 0)) >> 10 like the compiled /1024"""
    return (v + 1023 if v < 0 else v) >> 10


def stage_angle(logic_angle, cam_yaw_deg=0, flipped=False):
    """the clip's rotation in degrees (display +0x204, 0x53af9c..0x53b030): screen angle of the flight direction in
    the 48 x 36 px stage, turned by 180"""
    c = _div1024(isin(logic_angle + 90) * 1000)
    s = _div1024(isin(logic_angle) * 1000)
    a = iatan2(c, int(np.float32(np.float32(s) * np.float32(36.0)) / np.float32(48.0)))
    a = a + (-cam_yaw_deg if flipped else cam_yaw_deg) + (180 if flipped else 0)   # 0x96bd80
    return a + (-180 if a >= 180 else 180)


def logic_angle_of(dx, dy):
    """the flight direction as the logic stores it (integer degrees)"""
    return iatan2(dx, dy)


class Proj:
    def __init__(self, t, own, x, y, z, angle, age, travel=0, finished=0, finish_age=-1, start_frame=0, special_state=0,
                 own_offset=(0.0, 0.0), zoffset=None, scale_mod=0, prev=None, oid=None):
        """t type id; own: the viewer's team (blue clips); x, y, z: the client logic position (state cx, cy, cz);
        angle: logic angle (deg); age: display updates since the client created the object (0 on its first frame);
        travel: logic travel permille; finished: logic finish kind (0, 1 range, 2, 3 hit, 4 wall, 5 cancel);
        finish_age: client updates (1/60 s steps) since finishing; start_frame: the client's random start frame
        (RandomStartFrame 1 / 2), not determined by the state; special_state: logic +0x1d8; own_offset: the own-shot
        lead (display +0x188); prev: (x, y, z) of the previous frame (UseZFrames)"""
        self.t, self.own, self.x, self.y, self.z = t, own, x, y, z
        self.angle, self.age, self.travel, self.finished, self.finish_age = angle, age, travel, finished, finish_age
        self.start_frame, self.special_state, self.own_offset, self.zoffset = start_frame, special_state, own_offset, zoffset
        self.scale_mod, self.prev, self.oid = scale_mod, prev, oid


def lead_offset(lead, travel):
    """the own-shot lead applied this update (0x53aabc..0x53ab14): lead x (1 - 5 min(0.2, travel x 0.001)), float32,
    fused multiply-add, truncated to int"""
    f32 = np.float32
    s9 = f32(f32(travel) * f32(0.001)); s0 = f32(0.2) if s9 > f32(0.2) else s9
    fac = f32(np.float64(s0) * -5.0 + 1.0)
    return tuple(int(np.trunc(f32(f32(v) * fac))) for v in lead)


def zoffset_next(z):
    """Zoffset (logic, display +0x1d8) used this update; it drops by 50 per update while > 0 (0x53ab18..)"""
    return int(np.float32(z) - np.float32(50.0)) if z > 0 else z


def scale_of(T, travel, mod=0):
    """Scale x interpolated ScaleStart..ScaleEnd x 1e-4 (0x53b1f0..; 0xc6ec1c), integer arithmetic like the client"""
    s0, s1 = T['scale_start'], T['scale_end']
    if s0 == 100 and s1 == 100: v = 100
    else: v = int(s1 * travel / 1000) + int(s0 * (1000 - travel) / 1000)
    if mod: v = int((mod + 100) * v / 100)
    return float(np.float32(v * T['scale']) * np.float32(0.0001))


def visible(T, P):
    """the display object's visibility (0x53aa54..0x53aab8): hidden before HideTime, and from 0.1 s after the logic
    finished the projectile (at once unless it finished by range (1) or a hit (3), or HideFaster)"""
    if P.age * 0.05 < T['hide_time'] * 0.001: return False   # age in 1/20 s frames; +0x1e8 counts render time
    if P.finished:
        if T['is_beam']: return True
        if P.finished not in (1, 3) or T['hide_faster']: return False
        return P.finish_age * np.float32(1 / 60) <= 0.1 + 1e-7 if P.finish_age >= 0 else True
    return True


def named_children(root):
    """slots of the root's children named normal / special / special2 (ctor 0x539f28 looks them up by instance name
    when the root has 2 or 3 frames; both normal and special must exist)"""
    if root['nframes'] not in (2, 3) or not root.get('child_names'): return {}
    nm = {n: i for i, n in enumerate(root['child_names']) if n in ('normal', 'special', 'special2')}
    if 'normal' not in nm or 'special' not in nm: return {}
    if root['nframes'] != 3: nm.pop('special2', None)
    return nm


def _travel_frame(travel, n):
    """(int)(travel * 0.001 * n) clamped to [0, n - 1] in float32 like 0x53b2f0"""
    v = int(np.float32(np.float32(travel) * np.float32(0.001)) * np.float32(n))
    return 0 if v < 1 else min(n - 1, v)


DT = 0.05 + 1e-6   # display update / render dt in the capture environment: 50 ms plus the microseconds between the
                   # env setting the app clock and the game reading it (expertbot loader_v7/env.c cl_frame_start)


class Display:
    """the client's display object of one projectile (ctor 0x539f28, update 0x53a938), its clips simulated exactly
    (clipsim.py): the main clip ('clip', +0x28 holder) and, on the ground, the shadow ('shadow', ShadowExportName,
    separate object +0x1a8) or else the team's ground glow ('glow', Blue/RedGroundGlowExportName, +0xd8 holder).
    Create when the client first shows the projectile, call update() once per frame."""
    def __init__(self, t, own, start_frame=0, ground=True):
        import clipsim
        self.T = T = projtab.table()[t]; self.t = t; self.own = own; self.start_frame = start_frame
        self.S = sc(T['file']) if T['file'] else None
        name = T['blue'] if own else T['red']
        self.cid = self.S.exports.get(name) if (self.S is not None and name) else None
        if self.cid is not None and self.cid not in self.S.clips: self.cid = None
        self.age_time = 0.0; self.finish_time = 0.0; self.updates = 0; self.hidden = False
        # holder stage position (+0x1ec..: 48 x, 36 (y - z/2), 48 z; -1 = none yet) and its step (+0x1f8..): the ctor
        # sets the step to 9999 x 3, so before the second update the mesh pitch is atan(1/sqrt 2) = -35.26 deg [V vmesh.py]
        self.stage = None; self.vel = (np.float32(9999.0),) * 3
        # ground parts (shadow / glow): fresh here; track.Tracker builds them separately from the client's clip pool
        # (GroundParts) and passes ground=False
        self.parts = GroundParts(t, own).clips if ground else {}
        if self.cid is None: self.sim = None; return
        self.sim = sim = clipsim.ClipSim(self.S, self.cid)
        root = self.S.clips[self.cid]; n = root['nframes']; rs = T['random_start']
        self.named = named = named_children(root)
        if T['special_visual']: sim.goto_stop((), 0)
        if not named:
            if rs == 1: sim.goto_play((), start_frame % max(1, n))
            elif rs == 2: sim.goto_stop((), start_frame % max(1, n))
        else:   # named children: one rand() per child, in the order normal, special, special2 (0x63a2..); start_frame
            sim.goto_stop((), 0)   # may be a tuple of them (a single value is used for every child)
            for k, nm in enumerate(('normal', 'special', 'special2')):
                if nm not in named: continue
                slot = named[nm]; kn = self.S.clips[root['children'][slot]]['nframes']
                sf = start_frame[k] if isinstance(start_frame, (tuple, list)) else start_frame
                if rs == 1: sim.goto_play((slot,), sf % max(1, kn))
                elif rs == 2: sim.goto_stop((slot,), sf % max(1, kn))

    def update(self, angle=0, travel=0, finished=0, special_state=0, dt=DT, render=True, pos=None):
        """one frame: the projectile update (root control, visibility), then the render traversal.
        pos: the holder position in logic units (x + lead, y + lead, z + Zoffset); needed by UseZFrames and the mesh pitch"""
        T = self.T
        self.age_time += dt; self.updates += 1
        cur = stage_of(pos) if pos is not None else None
        moved = False
        if cur is not None and self.stage is not None and not finished:   # 0x53ab7c..0x53ac9c: step since the last update
            dx = np.float32(cur[0] - self.stage[0]); dy = np.float32(cur[1] - self.stage[1])
            self.vel = (dx, dy, np.float32(cur[2] - self.stage[2]))
            moved = float(dx * dx + dy * dy) > 3.0
        if finished:
            if finished in (1, 3) and not T['hide_faster']: self.finish_time += dt
            else: self.finish_time = 1.0
        hidden = (self.age_time < T['hide_time'] * 0.001) or (self.finish_time > 0.1 and not T['is_beam'])
        self.hidden = hidden
        if render and not hidden:
            for p in self.parts.values(): p.render(dt)
        if self.sim is None:
            if cur is not None: self.stage = cur
            return
        root = self.S.clips[self.cid]; n = root['nframes']; sim = self.sim; rs = T['random_start']
        if T['rendering'] == 1:
            v = stage_angle(angle) * n; sim.goto_stop((), 0 if v < 360 else min(n - 1, v // 360))
        if T['anim_by_dist']:
            if self.named:
                sel = self.named['normal']
                if T['special_visual'] and special_state == 1 and 'special' in self.named: sel = self.named['special']
                if T['special_visual'] and special_state == 2 and 'special2' in self.named: sel = self.named['special2']
                kn = self.S.clips[root['children'][sel]]['nframes']; sim.goto_stop((sel,), _travel_frame(travel, kn))
            else:
                sim.goto_stop((), _travel_frame(travel, n))
        elif T['special_visual']:
            k = special_state
            if rs in (3, 4): k = self.start_frame + special_state * max(1, n // (2 if rs == 3 else 3))
            sim.goto_stop((), k)
        if T['rendering'] == 2 and moved:   # UseZFrames (0x53b0a0..0x53b13c): frame by the vertical flight angle
            sim.goto_stop((), zframe(self.vel, n))
        if cur is not None: self.stage = cur
        if render and not hidden: sim.render(dt)

    def polys(self):
        """main clip polygons (root clip px)"""
        return [] if (self.sim is None or self.hidden) else self.sim.polys()

    def part_polys(self):
        """{'clip' | 'shadow' | 'glow': polygons}: everything the object draws this frame"""
        if self.hidden: return {}
        out = {k: p.polys() for k, p in self.parts.items()}
        if self.sim is not None: out['clip'] = self.sim.polys()
        return out


class GroundParts:
    """the shadow (ShadowExportName) or, without one, the team's ground glow of a projectile object. The client takes
    these clips from a per-clip pool (0xc8989c -> 0xcd9798 -> 0xcdc52c): a LIFO stack of released instances. A reused
    instance gets gotoAndPlay(0) on its root (0xcb45e4: its time is kept) and its own matrix / colour / visibility
    reset; its nested clips keep their timelines. Released instances are pushed back (0xcdc6ec) when the projectile
    object goes [V Dynamike's spinning TNT shadow, pose_p1: every shot's phase].
    pool: {(file, export): [ClipSim, ...]} shared by all projectiles of a battle (None: always fresh)"""
    def __init__(self, t, own, pool=None):
        import clipsim
        T = projtab.table()[t]; self.clips = {}; self.keys = {}
        S = sc(T['file']) if T['file'] else None
        if S is None: return
        gname = T['shadow'] or (T['blue_glow'] if own else T['red_glow'])
        gid = S.exports.get(gname) if gname else None
        if gid not in S.clips: return
        part = 'shadow' if T['shadow'] else 'glow'; key = (T['file'], gname)
        st = pool.get(key) if pool is not None else None
        if st:
            c = st.pop(); c.goto_play((), 0)
        else:
            c = clipsim.ClipSim(S, gid)
        self.clips[part] = c; self.keys[part] = key

    def update(self, hidden, dt=None):
        if not hidden:
            for c in self.clips.values(): c.render(DT if dt is None else dt)

    def polys(self):
        return {k: c.polys() for k, c in self.clips.items()}

    def release(self, pool):
        for part, c in self.clips.items(): pool.setdefault(self.keys[part], []).append(c)
        self.clips = {}


def stage_of(pos):
    """holder stage position like the client (0x5348ac: tiles = v x 0.0033333334 in float32; 0x544c34: sx = 48 x,
    sy = 36 (y - z / 2)) plus 48 z (0x53ab8c, the vertical term of the flight step)"""
    f = np.float32; t = [f(f(v) * f(0.0033333334)) for v in pos]
    return (f(f(48.0) * t[0]), f(f(36.0) * f(t[1] + t[2] * f(-0.5))), f(f(48.0) * t[2]))


def flight_pitch(vel):
    """atan2(-max(0.01, |step xy|), -step z) in degrees + 90: 0 for level flight (0x53c1f0)"""
    h = max(0.01, float(np.sqrt(np.float32(vel[0] * vel[0] + vel[1] * vel[1]))))
    return float(np.float32(math.atan2(-h, -float(vel[2])))) * 57.295776 + 90.0


def zframe(vel, n):
    """UseZFrames frame: (int)(atan2(-max(0.01, |step xy|), -step z) deg + 180) x n / 180, in [0, n - 1]"""
    h = max(0.01, float(np.sqrt(np.float32(vel[0] * vel[0] + vel[1] * vel[1]))))
    v = int(float(np.float32(math.atan2(-h, -float(vel[2])))) * 57.295776 + 180.0) * n
    return 0 if v < 180 else min(n - 1, v // 180)


def mesh_matrix(P, disp):
    """the 3D object (BlueSCW / RedSCW) transform (0x53c13c..0x53c28c): scale Scale x ScaleStart..End % x 0.03,
    yaw 90 - logic angle, pitch from the flight step, position (x, -y, 0.6 z) of the holder (x + lead, y + lead,
    z + Zoffset) -> 4 x 4 (render coords); the mesh's own node transforms come after it"""
    T = projtab.table()[P.t]
    s_ = scale_of(T, P.travel, P.scale_mod) * 1e4 * 0.03   # (Scale x percent) x 0.03
    yaw = math.radians(90.0 - P.angle)
    pit = math.radians(flight_pitch(disp.vel)) if disp is not None else 0.0
    cz, sz = math.cos(yaw), math.sin(yaw); cx, sx = math.cos(pit), math.sin(pit)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]); Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    M = np.eye(4); M[:3, :3] = Rz @ Rx * s_
    ox, oy = P.own_offset
    M[:3, 3] = [P.x + ox, -(P.y + oy), HEIGHT * (P.z + (P.zoffset or 0))]
    return M


def start_range(T, S, cid):
    """how many start frames the client's rand() can give (ctor 0x63a2.. / 0x539f28): RandomStartFrame 1 / 2 ->
    rand % n of the root, or with named children one rand() % n_child per child (a tuple of ranges); 3 -> rand % (n / 2) and
    4 -> rand % (n / 3) of the root (an offset for the special visual states) [V Rico010: wider ranges pick a
    look-alike after-bounce variant]"""
    root = S.clips[cid]; named = named_children(root); rs = T['random_start']
    if rs == 3: return max(1, root['nframes'] // 2)
    if rs == 4: return max(1, root['nframes'] // 3)
    if named: return tuple(max(1, S.clips[root['children'][named[nm]]]['nframes']) for nm in ('normal', 'special', 'special2') if nm in named)
    return max(1, root['nframes'])


def pellet_slots(S, cid):
    """root slots that are pellets: the root clip repeats one child clip >= 2 times (Shelly's / Bull's balls: one
    projectile drawn as several equal sub-clips); otherwise none (the projectile is one bullet)"""
    import collections
    kids = S.clips[cid]['children']; cnt = collections.Counter(k for k in kids if k in S.clips)
    if not cnt: return set()
    k, n = cnt.most_common(1)[0]
    return {i for i, c in enumerate(kids) if c == k} if n >= 2 else set()


def _ground_basis():
    return np.array([1.0, 0.0, 0.0]), np.array([0.0, -1.0, 0.0])


def place(P, polys, cam_forward=(0.0, 0.587785, -0.809017), part='clip'):
    """polygons of one part -> world (render coords x, -y, z) like the client.
    clip:   anchor (x + lead, y + lead, 0.6 (z + Zoffset)) in the plane facing the camera (GroundBasis: the horizontal
            plane), rotation = stage angle (unless Use360Frames / DoNotRotateClip), scale, 6.25 world units per stage px
    shadow: on the ground (z 0) under the anchor, scale min(sqrt(z_tiles 0.7) + 0.2, 1.5), not rotated (0x53b4f0)
    glow:   on the ground under the anchor, the main clip's matrix x (sqrt(z_tiles 0.6) + 1) (0x53b468)"""
    T = projtab.table()[P.t]
    s = scale_of(T, P.travel, P.scale_mod)
    rot = 0.0 if T['rendering'] in (1, 3) else float(stage_angle(P.angle))
    sy = s
    if T['rendering'] == 4: sy = s if (stage_angle(P.angle) - 91) % (1 << 32) > 178 else -s
    ca, sa = math.cos(math.radians(rot)), math.sin(math.radians(rot))
    M = np.array([[ca * s, -sa * sy], [sa * s, ca * sy]])
    ox, oy = P.own_offset
    zt = (P.z + (P.zoffset or 0)) / 300.0
    if part == 'clip' and T['ground_basis']:   # GroundBasis (data +0xcc -> display +0x240, read by vtable slot 0x38
        # 0x53c804): the clip lies flat, in the ground basis, at the clip height, with the same stage-angle matrix
        # [V Lola's attack, cap_b2: 282 frames, p90 0.001; the logic angle instead: errors ~370]
        u, v = _ground_basis(); O = np.array([P.x + ox, -(P.y + oy), HEIGHT * (P.z + (P.zoffset or 0))])
    elif part == 'clip':
        f = np.asarray(cam_forward, float); f = f / np.linalg.norm(f)
        u = np.array([1.0, 0.0, 0.0]); v = np.cross(f, u); v /= np.linalg.norm(v)
        O = np.array([P.x + ox, -(P.y + oy), HEIGHT * (P.z + (P.zoffset or 0))])
    else:
        u, v = _ground_basis(); O = np.array([P.x + ox, -(P.y + oy), 0.0])
        if part == 'shadow':
            k = float(np.minimum(np.float32(math.sqrt(max(zt, 0.0) * 0.7)) + np.float32(0.2), np.float32(1.5)))
            M = np.eye(2) * k
        else:
            M = M * float(np.float32(math.sqrt(max(zt, 0.0) * 0.6)) + np.float32(1.0))
    pel = None
    out = []
    for path, sid, ci, tex, uv, xy, mul, add, bl in polys:
        st = xy @ M.T
        W = O + PX * (st[:, :1] * u + st[:, 1:2] * v)
        out.append(dict(path=path, shape=sid, cmd=ci, tex=tex, uv=uv, world=W, local=xy, mul=mul, add=add,
                        blend=BLEND.get(bl, bl), part=part, pellet=-1))
    if part == 'clip' and out:
        S = sc(T['file']); cid = S.exports.get(T['blue'] if P.own else T['red'])
        ps = pellet_slots(S, cid) if cid in S.clips else set()
        for q in out: q['pellet'] = q['path'][0] if (q['path'] and q['path'][0] in ps) else -1
    return out


def place_all(P, disp, cam_forward=(0.0, 0.587785, -0.809017)):
    """every part the display object draws this frame, placed: list of polygon dicts (key 'part'), in the client's
    draw order of the parts (ground parts first)"""
    pp = disp.part_polys(); out = []
    for part in ('shadow', 'glow', 'clip'):
        if part in pp: out += place(P, pp[part], cam_forward, part)
    return out
