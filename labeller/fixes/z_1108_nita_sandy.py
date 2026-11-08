# Predictable hulls for Nita (Shaman) and Sandy (Sandstorm) attacks, every skin (2026-11-03, hull-design agent nita_sandy).
# Exec'd in idlabels' namespace AFTER all installed fixes (IDLABELS_OVERRIDE), wraps labels(); needs the Frank template helpers
# (py/fixes/z_1027_frank_template.py: _ft42_hull, _ft42_proj, _ft42_unproj, _ft42_stop, _ft42_blocking, _ft42_bushes, _ft42_world,
# FT42_* constants) and z_1023_zone_solid.py (_zs_grid, _zs_solid, _ZS_IMG).
#
# What the hull is (owner 11-03: "an uniform, predictable hull (like frank)"; Nita: "the hull should be based on the hitbox, since
# the ball sometimes sends huge sparks"):
#   NO pixels. A solid cylinder standing on the projectile's hitbox disc, projected with the frame's camera:
#     centre   C  = the projectile's client (drawn) position (state row cx, cy = the tracker's path[-1]); that is the logic position
#                   a fraction LAG (about 0.42 update) behind. Fallback: the logic position.
#     footprint   the disc of the projectile's hitbox (projtab radius: Shaman 250, Sandy 200) times a per-skin scale `r`,
#     visual height  from z0 to z1 (logic z; the ball floats over the ground), `fu` shifts the disc forward along the aim.
#   The label is the screen hull of the disc at z0 and at z1 (convex, mirror-symmetric about the aim axis in the ground), minus the
#   part of the disc in / behind a projectile-blocking tile as seen from the shot's origin (rays from O through the capture's tile
#   map, as Frank: _ft42_stop). Nothing drawn in front (brawlers, crates, bushes, walls) carves holes in it: only the map cuts.
#   Lifetime: while the tracker draws the projectile (obj 'drawn', from the shot's first client tick to the last drawn tick; Sandy: while
#   any cloud of the shot is drawn).
#   O (ray origin of the wall cut): the shooter's position (state chars row of the projectile's src), else the first drawn position
#   minus a constant back along the aim.
#   Same inputs -> same shape; sparks, glow, trail and the hit / end animation never change it.
#
# Per brawler / attack:
#  SHAMAN (Nita) attack, 13 types (ShamanProjectile 6000222 and the skin balls 6000226 .. 6000246, the overcharged balls 6000224 /
#  6000244): ONE hull per ball = the hitbox disc (projtab radius 250) x per-skin scale r, standing from z0 to z1, centred on the ball's
#  drawn position plus fu along the aim. Start-up growth: disc and height are g0 x their size on the ball's first drawn update and
#  reach 1 x after gt updates (age = len(path) - 1). Params NS_SHAMAN[type] (r, z0, z1, fu, g0, gt), default NS_SHAMAN_DEFAULT.
#  SANDY (Sandstorm) attack, 7 types (6000666 .. 6000678): ONE hull per shot (all clouds of the shot: label_id group) = the hitbox
#  discs (radius 200 x r) at the last K drawn positions of every member, raised z0 .. z1. Params NS_SANDY[type] (K, r, z0, z1).
#  Not touched: ShamanStarPowerProjectile 6000599 (radius 0, thrown by the bear, not Nita's attack), SandstormModifierProjectile
#  6001860 (no visual), every super / gadget / star power / spawnable label.
# Installed fixes whose per-pixel treatment of these TYPES is replaced (the old label of the type is removed first, the hull put in
# its place; the fixes stay installed and are harmless): z_1027_s232 (6000226), a4_agent4 + z_1103_bo_ruffs_shaman (6000228),
# z_1104_bo_ruffs_shaman (6000232), z_1027_s449 (6000246), z_1101_s098 (6000244), z_1027_s255 (6000668), z_1027_s270 (6000670),
# z_1027_s296 + z_1103_spawner_sandy / z_1104_sandy_spawner (6000672, 6000666 part), z_1027_s359 + z_1104 (6000674),
# z_1027_s363 + z_1104 (6000676), z_1027_s414 + z_1103_spawner_sandy (6000678), a2_agent2 _sandy_end (Sandy end-burst drop).
# (Their other types, e.g. the supers 6000667 / 6000669 ..., are not touched.)
import numpy as _np
from PIL import Image as _Im, ImageDraw as _Dr

# ---- Shaman (Nita) attack: one hull per ball, per-skin footprint scale r, heights z0 / z1, forward shift fu -----------------
NS_R_HIT = 250.0   # projtab radius of every Shaman attack projectile (the hitbox)
NS_SHAMAN_DEFAULT = dict(r=0.9, z0=200.0, z1=500.0, fu=0.0, g0=1.0, gt=1.0)   # g0, gt: the drawing grows from g0 x to 1 x over gt updates
NS_SHAMAN = {   # type id -> per-skin parameters (fitted on the capture's current labels, see report.md)
    6000222: dict(r=1.00, z0=100, z1=500, fu=0),   # 023 ShamanDefault: cov 0.93 (min 0.86) extra 0.60, 17 frames
    6000242: dict(r=0.90, z0=100, z1=400, fu=0),   # 146 ShamanStrangeOvercharge: cov 0.93 (min 0.88) extra 0.44, 17 frames
    6000234: dict(r=1.00, z0=300, z1=1000, fu=0),   # 227 ShamanReindeer: cov 0.94 (min 0.86) extra 0.73, 17 frames
    6000226: dict(r=0.80, z0=300, z1=700, fu=0),   # 232 ShamanShiba: cov 0.93 (min 0.87) extra 0.66, 17 frames
    6000228: dict(r=0.80, z0=0, z1=700, fu=0, g0=0.7, gt=6),   # 276 ShamanSummer: cov 0.93 (min 0.81) extra 0.32, 17 frames
    6000230: dict(r=0.70, z0=300, z1=700, fu=0, g0=0.7, gt=3),   # 294 ShamanDragon: cov 0.95 (min 0.83) extra 0.41, 17 frames
    6000232: dict(r=0.80, z0=0, z1=300, fu=0, g0=0.7, gt=3),   # 318 ShamanGummybear: cov 0.94 (min 0.82) extra 0.49, 17 frames
    6000236: dict(r=0.80, z0=0, z1=300, fu=0),   # 342 ShamanDarkcircus: cov 0.98 (min 0.97) extra 0.10, 17 frames
    6000238: dict(r=0.70, z0=0, z1=300, fu=0, g0=0.7, gt=3),   # 358 ShamanGodzilla: cov 0.93 (min 0.80) extra 0.33, 17 frames
    6000240: dict(r=0.90, z0=0, z1=300, fu=0),   # 393 ShamanStarrForce: cov 0.94 (min 0.91) extra 0.90, 17 frames
    6000246: dict(r=0.80, z0=100, z1=300, fu=0, g0=0.4, gt=6),   # 449 ShamanAprilFools: cov 0.94 (min 0.84) extra 0.42, 17 frames
    6000244: dict(r=0.90, z0=100, z1=400, fu=0),   # Strange overcharged ball: not captured (needs the overcharge), the 146 base ball's parameters
    # 6000224 (default overcharged ball): not captured, NS_SHAMAN_DEFAULT
}
NS_SHAMAN_TYPES = {6000222, 6000224, 6000226, 6000228, 6000230, 6000232, 6000234, 6000236, 6000238, 6000240, 6000242, 6000244, 6000246}
NS_SHAMAN_BULLET = {6000226, 6000238, 6000240}   # owner 11-03: Shiba 232, Godzilla 358, StarrForce 393 keep the plain bullet rule (no hull)
NS_SHAMAN_TYPES -= NS_SHAMAN_BULLET
NS_ORIGIN_BACK = 261.0   # units: the ball's first drawn position is this far ahead of the shooter (every Shaman skin)


def _ns_param(t):
    p = dict(NS_SHAMAN_DEFAULT); p.update(NS_SHAMAN.get(t, {})); return p


def _ns_centre(o):
    """drawn (client) position of a projectile: the tracker's last drawn ground position, else its logic position"""
    if o.get('path'): return _np.array(o['path'][-1], float)
    return _np.array(o['logic'], float)


def _ns_disc_mask(cam, O, th, ring, z0, z1, grid=None, shape=(540, 1200)):
    """(H, W) bool: the solid prism (footprint polygon ring, convex hull taken, heights z0 .. z1) as drawn, minus the part of the disc in or beyond a
    projectile-blocking tile seen from O (map only; no cut by what is drawn in front).
    Same construction as the Frank template's _ft42_mask (rays from O, per-pixel column test), with a floor height z0."""
    H, Wd = shape
    ring = _ft42_hull(ring)   # the footprint polygon (world x, y)
    C = ring.mean(0)
    S = _np.concatenate([_ft42_proj(cam, ring[:, 0], ring[:, 1], z0), _ft42_proj(cam, ring[:, 0], ring[:, 1], z1)])
    Sh = _ft42_hull(S)
    I = _Im.new('L', (Wd, H), 0); _Dr.Draw(I).polygon([tuple(p) for p in Sh], fill=1, outline=1)
    Cm = _np.asarray(I, bool)
    if not Cm.any(): return Cm
    c, s = _np.cos(_np.radians(th)), _np.sin(_np.radians(th))
    # local raster of the disc (u along the aim from O, v across), minus the wall part
    g = FT42_CELL
    rel = ring - O; Fl = _np.c_[rel[:, 0] * c + rel[:, 1] * s, -rel[:, 0] * s + rel[:, 1] * c]
    u0, v0 = Fl.min(0) - 2 * g; u1, v1 = Fl.max(0) + 2 * g
    nu, nv = int(_np.ceil((u1 - u0) / g)) + 1, int(_np.ceil((v1 - v0) / g)) + 1
    J = _Im.new('L', (nu, nv), 0); _Dr.Draw(J).polygon([((p[0] - u0) / g, (p[1] - v0) / g) for p in Fl], fill=1, outline=1)
    keep = _np.asarray(J, bool)
    if grid is not None:
        vv, uu = _np.nonzero(keep)
        Pw = _ft42_world(O, th, _np.c_[u0 + uu * g, v0 + vv * g])
        al = _np.degrees(_np.arctan2(Pw[:, 1] - O[1], Pw[:, 0] - O[0])); rr = _np.hypot(Pw[:, 0] - O[0], Pw[:, 1] - O[1])
        ref = al[0]; rl = (al - ref + 180) % 360 - 180
        k = _np.round((rl - rl.min()) / FT42_RAY_DA).astype(int)
        rays = ref + rl.min() + _np.arange(k.max() + 1) * FT42_RAY_DA
        d = _ft42_stop(grid, O, rays, rr.max())
        keep[vv[rr >= d[k]], uu[rr >= d[k]]] = False
    ys, xs = _np.nonzero(Cm)
    nz = int(_np.ceil(abs(_ft42_proj(cam, C[0], C[1], z1)[1] - _ft42_proj(cam, C[0], C[1], z0)[1]) / 0.7)) + 2
    ztop = _np.full(len(ys), -1.0)
    for z in _np.linspace(z0, z1, nz):
        x, y = _ft42_unproj(cam, xs + 0.5, ys + 0.5, z)
        u = (x - O[0]) * c + (y - O[1]) * s; v = -(x - O[0]) * s + (y - O[1]) * c
        iu = _np.round((u - u0) / g).astype(int); iv = _np.round((v - v0) / g).astype(int)
        inb = (iu >= 0) & (iu < nu) & (iv >= 0) & (iv < nv)
        hit = _np.zeros(len(iu), bool); hit[inb] = keep[iv[inb], iu[inb]]
        ztop[hit] = z
    ok = ztop >= 0
    M = _np.zeros(shape, bool); M[ys[ok], xs[ok]] = True
    return M


def _ns_owner_pos(o, state):
    """the shooter's current position (state chars row of the projectile's source), or None"""
    if state is not None:
        for c in state.get('chars', []):
            if int(c[0]) == int(o.get('src', -1)): return _np.array([float(c[3]), float(c[4])])
    return None


def _ns_origin(o, state, aim_u, first, back):
    p = _ns_owner_pos(o, state)
    return p if p is not None else first - back * aim_u


def _ns_ring(c, rad, n=28):
    a = _np.linspace(0, 2 * _np.pi, n, endpoint=False)
    return _np.c_[c[0] + rad * _np.cos(a), c[1] + rad * _np.sin(a)]


def _ns_shaman_mask(o, cam, grid, state, shape):
    p = _ns_param(o['t']); C = _ns_centre(o)
    th = float(o['angle']); u = _np.array([_np.cos(_np.radians(th)), _np.sin(_np.radians(th))])
    C = C + p['fu'] * u
    first = _np.array(o['path'][0], float) if o.get('path') else C
    O = _ns_origin(o, state, u, first, NS_ORIGIN_BACK)
    age = len(o['path']) - 1   # updates since the ball's first drawn update
    g = p['g0'] + (1.0 - p['g0']) * min(1.0, age / float(p['gt']))   # start-up growth of the drawing (disc and height)
    return _ns_disc_mask(cam, O, th, _ns_ring(C, NS_R_HIT * p['r'] * g), p['z0'], p['z0'] + (p['z1'] - p['z0']) * g, grid=grid, shape=shape)


# ---- Sandy (Sandstorm) attack: ONE hull per shot (her three sand clouds fly together), every skin ---------------------------
# footprint = the hitbox discs (projtab radius 200, x per-skin scale r) at the last K drawn (client) positions of every cloud of the
# shot (the cloud is the puff the projectile drags along its last K updates), all in one convex hull, raised z0 .. z1.
# Members: the shot's clouds that are drawn, plus the ones that reached their range (state finished 1: their cloud stays where it
# ended while the rest of the shot still flies); clouds that hit a wall at once (finished 4, never drawn) are the hit animation.
# Lifetime: while at least one cloud of the shot is drawn.
NS_R_SANDY = 200.0
NS_SANDY_DEFAULT = dict(K=3, r=1.0, z0=300.0, z1=500.0)
NS_SANDY = {   # type id -> per-skin parameters; r is the scale of the hitbox radius 200
    6000666: dict(K=3, r=1.00, z0=300, z1=500),   # 042 SandstormDefault: cov 0.94 (min 0.80) extra 0.33, 12 frames
    6000668: dict(K=2, r=1.25, z0=300, z1=500),   # 255 SandstormGiftShop: cov 0.94 (min 0.81) extra 0.31, 12 frames
    6000670: dict(K=1, r=0.88, z0=300, z1=500),   # 270 SandstormLantern: cov 0.94 (min 0.90) extra 0.53, 12 frames
    6000672: dict(K=2, r=1.25, z0=300, z1=500),   # 296 SandstormKoya: cov 0.94 (min 0.84) extra 0.33, 12 frames
    6000674: dict(K=3, r=1.25, z0=200, z1=600),   # 359 SandstormMake: cov 0.94 (min 0.86) extra 0.40, 12 frames
    6000676: dict(K=1, r=1.50, z0=300, z1=700),   # 363 SandstormDeepsea (3D sheet mesh, the poorest fit): cov 0.92 (min 0.81) extra 1.02, 12 frames
    6000678: dict(K=1, r=1.25, z0=200, z1=500),   # 414 SandstormGreekGod: cov 0.94 (min 0.86) extra 0.27, 12 frames
}
NS_SANDY_TYPES = {6000666, 6000668, 6000670, 6000672, 6000674, 6000676, 6000678}


def _ns_sandy_param(t):
    p = dict(NS_SANDY_DEFAULT); p.update(NS_SANDY.get(t, {})); return p


def _ns_sandy_mask(mem, cam, grid, state, shape):
    t = mem[0]['t']; p = _ns_sandy_param(t)
    pts = []
    for o in mem:
        for q in o['path'][-int(p['K']):]: pts.append(_ns_ring(_np.array(q, float), NS_R_SANDY * p['r'], 20))
    ring = _np.concatenate(pts)
    th = float(_np.median([o['angle'] for o in mem])); u = _np.array([_np.cos(_np.radians(th)), _np.sin(_np.radians(th))])
    first = _np.mean([o['path'][0] for o in mem], 0)
    O = _ns_origin(mem[0], state, u, _np.array(first, float), 700.0)
    return _ns_disc_mask(cam, O, th, ring, p['z0'], p['z1'], grid=grid, shape=shape)


_ns_prev = labels


def _ns_clear(P, objs_):
    ids, share, lids = P['ids'], P['share'], P['lids']
    keys = set()
    for o in objs_: keys |= {(o['id'], -1, 'body'), (o.get('label_id', o['id']), -1, 'body')}
    ks = [lids.index(l) for l in keys if l in lids]
    if ks:
        old = _np.isin(ids, ks); ids[old] = -1; share[old] = 0


def _ns_put(P, lid, M, cv):
    import bulletlabels as BL
    ids, share, lids = P['ids'], P['share'], P['lids']
    if lid not in lids: lids.append(lid)
    k = lids.index(lid); m = M & (ids < 0)
    ids[m] = k; share[m] = _np.maximum(cv[m] / 255.0, BL.CUT)


def labels(objs, cam, R, state=None, ended=(), img=None):
    P = _ns_prev(objs, cam, R, state=state, ended=ended, img=img)
    mine = [o for o in objs if o['t'] in NS_SHAMAN_TYPES or o['t'] in NS_SANDY_TYPES]
    if not mine: return P
    cv = R['cv'][::2, ::2]; grid = None; ids_shape = P['ids'].shape
    shots = {}
    for o in mine:
        if o['t'] in NS_SANDY_TYPES: shots.setdefault((o['t'], o.get('label_id', o['id'])), []).append(o)
    for o in mine:
        if o['t'] in NS_SHAMAN_TYPES:
            _ns_clear(P, [o])
            if o.get('ended') or not o.get('drawn') or not o.get('path'): continue
            if grid is None: grid = _zs_grid(R)
            M = _ns_shaman_mask(o, cam, grid, state, ids_shape)
            if M.any(): _ns_put(P, (o['id'], -1, 'body'), M, cv)
    for (t, gid), allm in shots.items():
        _ns_clear(P, allm)
        mem = [o for o in allm if not o.get('ended') and o.get('path') and (o.get('drawn') or o.get('finished') == 1)]
        if not any(o.get('drawn') for o in mem): continue
        if grid is None: grid = _zs_grid(R)
        M = _ns_sandy_mask(mem, cam, grid, state, ids_shape)
        if M.any(): _ns_put(P, (gid, -1, 'body'), M, cv)
    return P
