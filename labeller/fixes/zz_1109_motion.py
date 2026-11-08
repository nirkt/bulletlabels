# Motion outputs of the labeler (owner, 2026-11-04): per bullet label its DIRECTION, WHERE its attack will END and WHEN it ends.
# v3 (2026-11-04), accuracy pass, each point from the game / client behaviour (details at the places they are used):
#  * the TRAVEL OFFSET of a projectile (travel one tick ahead of its position) is decided by what the client saw: a projectile created in
#    the tick it is first updated is listed one state later (state 'client' = 0 on its first row) and its first client row carries
#    travel = (EarlyTicks + 2) steps, else EarlyTicks steps. With the object's path length (rows shown so far) the offset is exact:
#    measured 2,700 projectiles, no exception. Lobbed end tick causal 71 % -> 100 % exact, arc path p90 10.9 -> 0.8 px.
#  * Dynamike's bounce: xy decoded from the client's copy of the sim (position function 0xb76488, branch BouncePercent > 0): full speed to
#    (1 - b/100) of the lerp, then HALF speed, end extended by b/150; the bounce is exactly b/100 of the lerp. (z keeps measured constants.)
#  * walls and characters share one measured collision margin of 20 units (every hit <= 19.8, every miss >= 23.3): wall end tick 94 -> 97 %,
#    hit end tick 76 -> 96 % exact. Characters with a velocity (from the previous state seen) are tracked tick by tick (not validated: the
#    sweep characters stand still).
#  * Targeted skills (Crab super, Bronson): the aim point is not in the state; the lerp length is T x Speed / 20 (T = round(D / (Speed / 20))).
#  * PiercesEnvironment: the FIRST bullet is stopped by an intact destructible tile like any other shot and destroys it; later bullets fly
#    through (measured on every Colt / Shelly / Gunslinger super). Destroyed tiles are remembered per map while frames come in order.
# v2 (2026-11-04): adds the predicted PATH (IDLABELS_MOTION_TRAJ) and the uncut range end. Drop-in replacement of v1: install as
# py/fixes/zz_1109_motion.py (after zz_1108_fill_smooth.py), NEVER next to v1 (both wrap labels()). Wraps labels(); no mask / id / share is changed.
#
# OUTPUT. P['motion'] = {label id: dict(
#     dir=(dx, dy)            unit vector of the flight in world (logic) coordinates (x right, y up)
#     angle_deg               its angle (degrees, 0 = +x, counter-clockwise, 0..360)
#     dir_screen=(du, dv)     unit vector on the label frame (u right, v DOWN), the projection of dir on the ground
#     end_world=(x, y)        logic point where the attack will end (ground)
#     end_screen=(u, v)       the same point in label pixels (1200 x 540)
#     end_reason              'range' (reaches its range), 'wall' (stops at a projectile-blocking tile or the map border),
#                             'target' (a lobbed / thrown projectile's landing point), 'hit' (ends on a character that stands
#                             in its way, characters taken to stand still; also an already ended hit),
#                             'unknown' (see UNRELIABLE below: the numbers are a best guess)
#     end_tick                absolute tick of the state in which the projectile is finished (state 'finished' != 0)
#     ticks_left              end_tick - the current tick (0: it ends in this very state)
#     walls                   False when no map was found (the wall raycast was skipped), else True
#     range_end_world=(x, y)  the end of the UNCUT path: the bullet's full range / landing point (Dynamike: where the bounce comes to rest),
#     range_end_screen=(u, v) with walls, the map border and characters IGNORED (end_* above keep the wall / hit logic; END toggle)
#     range_end_tick          absolute tick it would reach range_end_world (ENDTIME toggle)
#     traj_kind               'straight' | 'arc' | 'bounce' (TRAJ toggle)
#     traj                    [(tick, x, y, z, u, v), ...] one point per tick from now (point 0 = where the projectile is DRAWN now) to
#                             range_end_tick: world x, y, height z (logic units; the client draws at 0.6 z), screen u, v in label pixels.
#                             ALWAYS the full range: no wall raycast, no border, no character hit stop. The owner's wall model is expected
#                             to cut it. 'straight' bullets: flat, every tick, start -> max range. 'arc': lobbed / thrown (Barley, Tick, Penny,
#                             Brock super ...): straight ground line, height parabola z(t) = z0 (1 - t/T) + A t (T - t) (z0 = the source's
#                             ProjectileStartZ, A solved from the height now). 'bounce': Dynamike's TNT: that arc to the landing, then a
#                             half-speed bounce (xy x 0.5, gravity / 4, start height / speed measured as fractions of the landing speed)
#                             until it lands again. Curved TravelType, boomerang and bouncing-type bullets are 'straight': the point the
#                             client DRAWS them at is on the straight sim line (<= 2 units, all 233 tracks of 7 TravelTypes + boomerangs
#                             + bouncers; their wiggle is inside the sprite), and the sim state shows NO return leg of a boomerang (it
#                             ends at its range) nor a reflected path of a bouncer (it ends at the first wall); no 'curve' / 'return' kind.
#                             The points are what the client draws (state interpolated by the frame's weight), not the raw state.
#     mode                    'causal' | 'target' (see below)
#   ) for every label id of an attack / super / spawnable projectile (P['meta'] kind), i.e. the keys of P['meta'] that are
#   projectile labels. A wave / multi-projectile shot is ONE label: dir is the mean direction of its members, end_world /
#   end_tick are those of its member whose end is farthest from the shooter. Zone, hit and spawn labels have no entry.
#
# TOGGLES (environment, read at every call; all default OFF, so nothing changes unless asked; independent of each other):
#     IDLABELS_MOTION_DIR=1      dir, angle_deg, dir_screen
#     IDLABELS_MOTION_END=1      end_world, end_screen, end_reason, walls, range_end_world, range_end_screen
#     IDLABELS_MOTION_ENDTIME=1  end_tick, ticks_left, range_end_tick
#     IDLABELS_MOTION_TRAJ=1     traj, traj_kind (the path to the full range; the owner's wall model cuts it)
#   Only the keys of the switched-on groups are in an entry (an entry exists when at least one group is on). Further
#   IDLABELS_MOTION_CAUSAL=1: ignore the objects' 'target' (the recorded flight end of track.trajectories, which looks ahead
#   in the recorded states) and predict from the current state only (the validated 'causal' mode, see below).
#   The wall raycast needs the capture's map: IDLABELS_MAP=<dir>/map.npy, else map.npy found above R['dir'] (as z_1023_zone_solid).
#
# WHAT THE GAME DOES (every number below is checked against the sweep states; out/sweep2/haikutasks/away_1104/motion/report.md)
#  * The simulation moves every projectile on ONE integer lerp start -> end over T ticks (0xb76488), whatever its look: curved
#    (TravelType), wave, boomerang, bouncing and lobbed types all have a straight constant-step path in the state (the wiggle /
#    arc is drawn by the client). So direction = the state's step (x - px, y - py) = getAngle(end - start).
#  * end (range) = shooter + CastingRange * 100 (skills.csv) along the aim, start = shooter + 50 along it: T = round((R - 50) /
#    (Speed / 20)) ticks, step = (R - 50) / T. travel permille grows by 1000 // T per tick: T = 1000 // travel0 (unique for travel0
#    >= 32; below that the candidate nearest to (R - 50) / step is taken). The projectile is finished on the tick t = T.
#  * travel is one tick AHEAD of the position on a projectile whose first state row has px = x (created in the very tick it is
#    updated: 55 % of the projectiles): t = travel // travel0 - 1; else t = travel // travel0. Later rows do not tell the two
#    apart; the shooter's position decides when it is still near its start (|pos - (shooter + 50 dir)| / step is within 0.3 of
#    an integer on one side), else the more frequent case is taken (MO_PRIOR_H).
#  * walls: finished kind 4 is set on the tick whose step would enter a projectile-blocking tile (tiles.csv BlocksProjectiles; a
#    margin of MO_WALL_MARGIN units beyond the step is already a hit, measured), the position is the boundary point minus 1;
#    kind 2 = map border. PassesEnvironment types fly through tiles, PiercesEnvironment types through destructible ones only
#    (IsDestructible), IgnoreLevelBorder types through the border. Lobbed (Indirect) types ignore walls and land on their target.
#  * TriggerWithDelayMs types (Dynamike, cluster bombs) finish a fixed number of ticks after creation.
# NOT PREDICTED: characters that move into / out of the path, and walls destroyed during the match (the map is the start map).
#   Characters are hit within MO_CHAR_R (120) + the projectile's Radius of their centre (measured on every kind-3 end).
# UNRELIABLE (end_reason 'unknown'): types with
#   IsHomingMissile / SteerStrength (flight bends toward a target), IsBeam (none of them occur in the sweep states: untested).
#   Tick-exact only when the object's path length is given (the labeler does); without it the older first-row / shooter tests are used.
import os as _mo_os, math as _mo_math, csv as _mo_csv, functools as _mo_ft
import numpy as _mo_np

MO_WALL_MARGIN = 20.0     # units: a wall closer than one step plus this is hit on the next tick (measured: every hit <= 19.8, every miss >= 23.3 on 421 rows)
MO_PRIOR_H = 1            # travel-ahead case taken when the shooter does not decide (the more frequent one)
MO_TICK_MS = 50
MO_SPAWN_OFFSET = 50.0    # the lerp starts this far ahead of the shooter (R - 50 is the lerp length)
MO_NIL = -2147483648
MO_KIND_REASON = {1: 'range', 2: 'wall', 3: 'hit', 4: 'wall'}
import rc as _mo_rc
MO_RC = _mo_rc.ASSETS   # the game assets (labeller/rc.py)


def _mo_assets():
    try: return rc.ASSETS
    except NameError: return MO_RC


@_mo_ft.lru_cache(maxsize=None)
def _mo_tiles():
    """(set of tile rows with BlocksProjectiles, set of those that are IsDestructible)"""
    rows = list(_mo_csv.reader(open(_mo_os.path.join(_mo_assets(), 'csv_logic', 'tiles.csv'), encoding='utf8')))
    h = rows[0]; bi, di = h.index('BlocksProjectiles'), h.index('IsDestructible')
    blk, dst = set(), set()
    for k, r in enumerate(rows[2:]):
        if len(r) > bi and r[bi].strip().lower() == 'true':
            blk.add(k)
            if len(r) > di and r[di].strip().lower() == 'true': dst.add(k)
    return frozenset(blk), frozenset(dst)


@_mo_ft.lru_cache(maxsize=None)
def _mo_logic_rows():
    rows = list(_mo_csv.reader(open(_mo_os.path.join(_mo_assets(), 'csv_logic', 'projectiles_logic.csv'), encoding='utf8')))
    return {r[0]: dict(zip(rows[0], r)) for r in rows[2:]}


@_mo_ft.lru_cache(maxsize=None)
def _mo_range_of():
    """{projectile skin name: (CastingRange * 100, Targeted)} from the skill that lists the projectile (skin projectiles: their
    parent's). Targeted skills end at the aim point (nearer than the range): the state does not hold it"""
    import owners
    S = owners._tables(); out = {}
    for s in S['skills'].values():
        for col in ('Projectiles', 'OverchargedProjectiles'):
            for p in owners._split(s.get(col)):
                try: out.setdefault(p, (int(s.get('CastingRange') or 0) * 100, str(s.get('Targeted')).strip().lower() == 'true'))
                except ValueError: pass
    idx = {r['Name']: r for r in S['pskin']}
    for nm, r0 in list(idx.items()):
        base, r, seen = nm, r0, set()
        while base not in out and base not in seen and r.get('ParentProjectileForSkin'):
            seen.add(base); base = r['ParentProjectileForSkin']; r = idx.get(base, {})
        if base in out: out.setdefault(nm, out[base])
    return out


@_mo_ft.lru_cache(maxsize=None)
def _mo_info(t):
    """what the game tables say about projectile type t (data id 6000000 + row of projectiles_skin)"""
    import projtab
    T = projtab.table().get(t)
    if T is None: return None
    L = _mo_logic_rows().get(T['logic'], {})
    flag = lambda k: str(L.get(k, '')).strip().lower() == 'true'
    num = lambda k: int(L.get(k) or 0) if str(L.get(k) or '0').lstrip('-').isdigit() else 0
    info = dict(name=T['name'], speed=num('Speed'), radius=num('Radius'), lob=flag('Indirect'), const_fly=flag('ConstantFlyTime'),
                bounce_pct=num('BouncePercent'), gravity=num('Gravity'), beam=flag('IsBeam'), homing=flag('IsHomingMissile') or flag('IsFriendlyHomingMissile'),
                steer=num('SteerStrength') > 0, pierce_env=flag('PiercesEnvironment'), pass_env=flag('PassesEnvironment'),
                ignore_border=flag('IgnoreLevelBorder'), delay_ms=num('TriggerWithDelayMs'), early=num('EarlyTicks'),
                boomerang=num('BoomerangType') > 0, bouncing=flag('IsBouncing'), pierce_char=flag('PiercesCharacters'), travel_type=num('TravelType'),
                chain=bool(L.get('ChainBullets')), spawn=bool(L.get('SpawnCharacter')))
    rg = _mo_range_of().get(T['name'])
    info['range'] = rg[0] if rg else None; info['targeted'] = bool(rg and rg[1])
    return info


def mo_category(t):
    """the category a type is reported under (validation): the first rule that holds"""
    I = _mo_info(t)
    if I is None: return 'unknown'
    if I['beam']: return 'beam'
    if I['homing'] or I['steer']: return 'homing'
    if I['lob'] and I['bounce_pct'] > 0: return 'lobbed-bounce'
    if I['lob']: return 'lobbed'
    if I['boomerang']: return 'boomerang'
    if I['bouncing']: return 'bouncing'
    if I['spawn']: return 'spawner'
    if I['chain']: return 'chained'
    if I['travel_type']: return 'curved'
    return 'straight'


def _mo_unreliable(info):
    return info['beam'] or info['homing'] or info['steer']


def _mo_wall_dist(grid, x, y, ux, uy, rmax, info, dead=()):
    """distance along (ux, uy) from (x, y) to the first tile the projectile cannot enter (0: it stands in one), inf when
    none within rmax; reason 'wall' | 'border'. grid: (60, 60) tile rows (map value - 27000000)"""
    blk, dst = _mo_tiles()
    H, W = grid.shape
    # PiercesEnvironment shots are stopped by an INTACT destructible tile like any other shot (measured: the first bullet of a Colt / Shelly
    # super stops at the first crate or fence, the next ones fly through it): the tile is destroyed by that hit; `dead` holds the tiles seen
    # destroyed this way. PassesEnvironment types fly through everything.
    block = frozenset() if info['pass_env'] else blk
    tx, ty = int(_mo_math.floor(x / 300.0)), int(_mo_math.floor(y / 300.0))

    def hit(a, b):
        if a < 0 or a >= W or b < 0 or b >= H: return 'border'
        return 'wall' if (int(grid[b, a]) in block and (a, b) not in dead) else None
    h0 = hit(tx, ty)
    if h0 and not (h0 == 'border' and info['ignore_border']): return 0.0, h0
    sx = 1 if ux > 0 else -1; sy = 1 if uy > 0 else -1
    tmx = (((tx + (1 if ux > 0 else 0)) * 300.0 - x) / ux) if abs(ux) > 1e-12 else _mo_math.inf
    tmy = (((ty + (1 if uy > 0 else 0)) * 300.0 - y) / uy) if abs(uy) > 1e-12 else _mo_math.inf
    tdx = 300.0 / abs(ux) if abs(ux) > 1e-12 else _mo_math.inf
    tdy = 300.0 / abs(uy) if abs(uy) > 1e-12 else _mo_math.inf
    while True:
        if tmx < tmy: d = tmx; tx += sx; tmx += tdx
        else: d = tmy; ty += sy; tmy += tdy
        if d > rmax: return _mo_math.inf, None
        h = hit(tx, ty)
        if h == 'border' and info['ignore_border']: continue
        if h: return d, h


def _mo_crosses_destructible(grid, x, y, ux, uy, dist):
    """True when the segment crosses a destructible projectile-blocking tile (a PiercesEnvironment shot may or may not get through)"""
    blk, dst = _mo_tiles(); both = blk & dst
    for s_ in _mo_np.arange(0.0, dist + 25.0, 25.0):
        tx, ty = int(_mo_math.floor((x + ux * s_) / 300.0)), int(_mo_math.floor((y + uy * s_) / 300.0))
        if 0 <= tx < grid.shape[1] and 0 <= ty < grid.shape[0] and int(grid[ty, tx]) in both: return True
    return False


MO_CHAR_R = 120.0      # a character is hit when the projectile centre is within this + the projectile's Radius (measured: 120.0-121.0)
MO_HIT_MARGIN = 20.0


def _mo_hit_dist(x, y, ux, uy, rmax, others, owner, radius, L=None):
    """distance along (ux, uy) at which the path first enters a character's hit circle (others: (x, y, owner player, hp[, vx, vy]) of the
    characters, vx / vy in units per tick when known from the previous state; the projectile's own player's are skipped), inf when none
    within rmax. A character without a velocity is taken to stand still (analytic circle); with one, the projectile (one step L per tick)
    and the character (one step v per tick) are compared tick by tick."""
    best = _mo_math.inf; rr = MO_CHAR_R + radius
    for ch in others:
        cx, cy, co, hp = ch[:4]
        if co == owner or hp <= 0: continue
        vx, vy = (ch[4], ch[5]) if len(ch) >= 6 else (0.0, 0.0)
        if (vx == 0.0 and vy == 0.0) or not L:
            vxr, vyr = cx - x, cy - y
            al = vxr * ux + vyr * uy; pe = abs(vxr * uy - vyr * ux)
            if pe > rr: continue
            s_in = al - _mo_math.sqrt(rr * rr - pe * pe)
            if al < -rr or s_in > rmax: continue
            best = min(best, max(0.0, s_in))
        else:
            for j in range(1, int(rmax / L) + 3):
                px_, py_ = x + ux * L * j, y + uy * L * j; qx, qy = cx + vx * j, cy + vy * j
                if _mo_math.hypot(px_ - qx, py_ - qy) <= rr:
                    best = min(best, L * j); break
    return best


def _mo_candidates_T(d):
    return [T for T in range(1, 1001) if 1000 // T == d]



# ---------------------------------------------------------------- arcs and bounces of lobbed projectiles
# z (logic height) of a lobbed projectile is a parabola in the ticks t since creation: z(t) = z0 (1 - t/T) + A t (T - t) (it starts at the
# source character's ProjectileStartZ (characters.csv) and is on the ground at the end of the lerp, t = T; measured on 281 lobbed types:
# fit residual of z(0) = StartZ - 5, z(T) = 0 +- 3). A (half the gravity per tick^2) is not a table value (it follows Gravity, T and
# the range in a way not decoded), so it is solved from the projectile's own z now.
# BouncePercent lobs (Dynamike): the same parabola with A = MO_BOUNCE_A[Gravity] (measured: TNT 26.01 / 2 per tick^2... see below), the
# projectile lands (z = 0) before the lerp ends, and from then on ALL motion runs at half speed: xy steps x 0.5, gravity / 4,
# the bounce starts at z = MO_BOUNCE_Z0 |v| with upward velocity MO_BOUNCE_V |v| (|v| = vertical speed at the landing; 0.1225 = 0.35^2;
# measured on both Dynamike projectiles). It ends at the delay: TriggerWithDelayMs ticks after creation (+1).
MO_BOUNCE_A = {430: 26.01, 330: 23.01}      # |a| (units / tick^2) of the pre-landing parabola, by Gravity (measured)
MO_BOUNCE_A_DEFAULT = 0.0605                # |a| = this x Gravity when the Gravity is not in the table
MO_BOUNCE_V = 0.1225
MO_BOUNCE_Z0 = 0.0775
MO_BOUNCE_SLOW = 0.5


@_mo_ft.lru_cache(maxsize=None)
def _mo_start_z(did):
    """ProjectileStartZ of character data id did (characters.csv) or None"""
    import owners
    try:
        i = int(did) - 16000000; c = owners._tables()['chars'][i]
        return float(c.get('ProjectileStartZ') or 0) or None
    except Exception:
        return None


def _mo_arc_A(z0, z_now, t_now, T_):
    """A of z(t) = z0 (1 - t/T) + A t (T - t) from the height now; None when t or T - t < 1"""
    if t_now < 1 or T_ - t_now < 1: return None
    return (z_now - z0 * (1.0 - t_now / float(T_))) / (t_now * (T_ - t_now))


def _mo_bounce_profile(I, z0, z_now, t_now):
    """(a, v0, t_L, v_L) of a BouncePercent lob: z(t) = z0 + v0 t + a t^2 until it lands at t_L (fractional ticks since creation) with the
    vertical speed v_L (< 0); None when t_now < 1"""
    if t_now < 1: return None
    a = -MO_BOUNCE_A.get(I['gravity'], MO_BOUNCE_A_DEFAULT * I['gravity'])
    v0 = (z_now - z0 - a * t_now * t_now) / t_now
    disc = v0 * v0 - 4 * a * z0
    t_L = (-v0 - _mo_math.sqrt(max(0.0, disc))) / (2 * a)
    return a, v0, t_L, v0 + 2 * a * t_L


def mo_predict(row, tick, grid=None, angle=None, shooter=None, target=None, others=None, src_did=None, n_seen=None, vel=None, dead=None):
    """Where and when one projectile ends, from its state row (18 ints: id, dataId, owner, x, y, z, px, py, src, damage, travel,
    travel0, finished, impact, client, cx, cy, cz) in the state of tick `tick`.
      grid: (60, 60) tile rows or None (no wall raycast)   angle: its logic angle in degrees when known (the object's)
      shooter: (x, y) of the projectile's source character in this state or None
      target: (x, y) recorded flight end (track.trajectories) or None -> mode 'causal'
      n_seen: how many state rows the client has shown this projectile so far (its path length): decides the travel offset exactly (below)
      dead: set of (tile x, tile y) of destructible tiles already destroyed by a piercing shot (they no longer stop anything) or None
      src_did: data id of the source character (ProjectileStartZ of the lob arc) or None
      others: [(x, y, owner player, hp)] of the state's characters (hits on characters, taken to stand still) or None
    -> dict(dir, angle_deg, end_world, end_reason, end_tick, ticks_left, walls, mode, category) or None (no such type)"""
    t = int(row[1]); I = _mo_info(t)
    if I is None: return None
    if I['delay_ms']: target = None      # a delayed type's recorded end is its (never reached) lerp end
    x, y, px, py = float(row[3]), float(row[4]), float(row[6]), float(row[7])
    travel, d, fin = int(row[10]), int(row[11]), int(row[12])
    has_prev = not (px == 0 and py == 0) and (x != px or y != py)    # lobbed projectiles never carry a previous position
    first_row = (px == x and py == y)                                   # created in the tick it is updated
    Lm = _mo_math.hypot(x - px, y - py) if has_prev else 0.0
    R = I['range']; Rl = (R - MO_SPAWN_OFFSET) if R else None          # length of the lerp from the game tables
    # ---- direction: the recorded end, else the state's step, else the angle, else shooter -> projectile
    if target is not None and _mo_math.hypot(target[0] - x, target[1] - y) > 25:
        dd = _mo_math.hypot(target[0] - x, target[1] - y); u = ((target[0] - x) / dd, (target[1] - y) / dd)
    elif has_prev and Lm >= 60 or (has_prev and angle is None):
        u = ((x - px) / Lm, (y - py) / Lm)
    elif angle is not None:
        u = (_mo_math.cos(_mo_math.radians(angle)), _mo_math.sin(_mo_math.radians(angle)))
    elif shooter is not None and _mo_math.hypot(x - shooter[0], y - shooter[1]) > 1:
        dd = _mo_math.hypot(x - shooter[0], y - shooter[1]); u = ((x - shooter[0]) / dd, (y - shooter[1]) / dd)
    else:
        u = (1.0, 0.0)
    ang = _mo_math.degrees(_mo_math.atan2(u[1], u[0])) % 360.0
    out = dict(dir=u, angle_deg=ang, category=mo_category(t), walls=grid is not None, mode='causal')
    unrel = _mo_unreliable(I)
    if fin:   # ended already: it ends here, now
        reason = MO_KIND_REASON.get(fin, 'unknown')
        if reason == 'range' and I['lob']: reason = 'target'
        out.update(end_world=(x, y), end_reason=reason, end_tick=tick, ticks_left=0, range_end=(x, y), range_ticks=0)
        return out
    if d <= 0:
        out.update(end_world=(x, y), end_reason='unknown', end_tick=tick, ticks_left=0, range_end=(x, y), range_ticks=0); return out
    # ---- T: ticks of the lerp (travel permille grows 1000 // T per tick)
    Tc = _mo_candidates_T(d) or [max(1, 1000 // d)]
    if len(Tc) > 1:
        if R and has_prev: Tref = Rl / Lm
        elif R and I['speed']: Tref = Rl / (I['speed'] / 20.0)
        else: Tref = None
        Tc = [min(Tc, key=lambda T_: abs(T_ - Tref))] if Tref else [Tc[len(Tc) // 2]]
    T_ = Tc[0]
    T_d = T_
    if I['delay_ms']: T_ = max(1, int(round(I['delay_ms'] / float(MO_TICK_MS))))   # delayed types: the lerp lasts the delay (travel counts slower)
    # ---- t: ticks since creation. travel // d, minus 1 when travel runs a tick ahead of the position
    kk = travel // d
    S = (shooter[0] + MO_SPAWN_OFFSET * u[0], shooter[1] + MO_SPAWN_OFFSET * u[1]) if shooter is not None else None
    L = Lm if has_prev else None
    h = 1 if first_row else None; hsrc = 'row' if first_row else None
    if n_seen is not None and n_seen >= 1 and travel < 1000:
        # the client lists a projectile one state AFTER it was created when it was created in the tick it is first updated; then its first
        # client row carries travel = (EarlyTicks + 2) steps (else EarlyTicks steps) and travel runs one tick ahead of the position (h = 1).
        # Measured on 2,700 projectiles of the sweep (state 'client' = 0 on that first row): kk_first_client - EarlyTicks >= 1 <=> h = 1, no exception.
        kk_c = kk - (n_seen - 1)
        h = 1 if kk_c - I['early'] >= 1 else 0; hsrc = 'client'
    if h is None and S is not None and L:
        tp = ((x - S[0]) * u[0] + (y - S[1]) * u[1]) / L
        e0, e1 = abs(tp - kk), abs(tp - (kk - 1))
        if min(e0, e1) < 0.3 and max(e0, e1) > 0.6: h = 0 if e0 < e1 else 1; hsrc = 'shooter'
    if h is None and not has_prev and S is not None and Rl and kk >= 2 and not (I['lob'] and I['bounce_pct'] > 0):   # lobbed: the aim lies within the range
        D = [_mo_math.hypot(x - S[0], y - S[1]) * T_ / max(1, kk - hh) for hh in (0, 1)]
        ok = [hh for hh in (0, 1) if D[hh] <= Rl + 15]
        if len(ok) == 1: h = ok[0]; hsrc = 'range'
        elif not ok: h = 0 if abs(D[0] - Rl) < abs(D[1] - Rl) else 1; hsrc = 'range'
    if h is None and not has_prev and I['lob'] and target is None and kk >= 2 and I['gravity']:   # lobbed: the height decides (A is near Gravity / 5.6
        z0_ = _mo_start_z(src_did)                                                                    # for ConstantFlyTime types, Gravity / max(T, 20) else)
        if z0_:
            Ap = I['gravity'] / (5.6 if I['const_fly'] else float(max(T_d, 20)))
            sc_ = []
            for hh in (0, 1):
                A_ = _mo_arc_A(z0_, float(row[5]), kk - hh, T_d)
                sc_.append(abs(_mo_math.log(max(1e-3, A_ / Ap))) if A_ and A_ > 0 else 9.0)
            if abs(sc_[0] - sc_[1]) > 0.25: h = 0 if sc_[0] < sc_[1] else 1; hsrc = 'height'
    if h is None: h = MO_PRIOR_H; hsrc = 'prior'
    t_now = max(0, kk - h)
    # ---- the end of the lerp
    E = None
    if target is not None: E = (float(target[0]), float(target[1])); out['mode'] = 'target'
    elif not has_prev:
        if S is not None and t_now >= 1: E = (S[0] + (x - S[0]) * T_ / t_now, S[1] + (y - S[1]) * T_ / t_now)
        elif S is not None and Rl: E = (S[0] + u[0] * Rl, S[1] + u[1] * Rl)
    if L is None and I['targeted'] and I['speed'] and not I['lob'] and target is None:
        L = I['speed'] / 20.0     # aimed shots: the lerp length is the aim distance = T x one step (T = round(D / (Speed / 20)): within half a step)
    if L is None:   # one step: the lerp length over T
        if E is not None and S is not None: L = _mo_math.hypot(E[0] - S[0], E[1] - S[1]) / T_
        elif Rl: L = Rl / T_
        else: L = (I['speed'] / 20.0) if I['speed'] else 100.0
    if E is not None and (target is not None or not has_prev):
        n_r = max(0, int(round(_mo_math.hypot(E[0] - x, E[1] - y) / max(L, 1.0))))
        if target is None: n_r = max(0, T_ - t_now)
        elif not has_prev and not I['delay_ms'] and hsrc != 'client': t_now = max(0, T_ - n_r)    # the recorded end fixes the tick count (no first-row / shooter guess)
        if hsrc == 'client' and not I['delay_ms']: n_r = max(0, T_ - t_now)
    else:
        n_r = max(0, T_ - t_now)
        E = (x + u[0] * L * n_r, y + u[1] * L * n_r)
    if I['delay_ms']: n_r = max(0, int(round(I['delay_ms'] / float(MO_TICK_MS))) - t_now)
    bnc = None
    if I['lob'] and I['bounce_pct'] > 0 and I['delay_ms'] and S is not None and t_now >= 1:
        # xy: decoded from the client's copy of the sim (position function 0xb76488, the branch for BouncePercent b = +0x280 > 0):
        #   with p = 1000 t / T permille of the lerp (T = end tick - start tick = 1000 // travel0), V = E' - S, E' = E + b / 150 (E - S):
        #   p <= 1000 - 10 b : S + V p / 1000           (full speed to the landing at (1 - b / 100) T)
        #   p >  1000 - 10 b : S + V (1 - 10 b / 1000 + 0.5 (p - (1000 - 10 b)) / 1000)     (the bounce at HALF speed)
        # so the bounce lasts b / 100 of the lerp, ends at S + V (1 - 5 b / 1000) and the projectile ends earlier at its delay tick.
        z0 = _mo_start_z(src_did) or 400.0
        pr = _mo_bounce_profile(I, z0, float(row[5]), t_now)
        if pr is not None:
            bp = float(I['bounce_pct']); p1 = 1000.0 - 10.0 * bp
            p_now = min(1000.0, 1000.0 * t_now / T_d)
            f_now = (p_now if p_now <= p1 else p1 + 0.5 * (p_now - p1)) / 1000.0
            V = ((x - S[0]) / f_now, (y - S[1]) / f_now)
            def xyfn(tt, S=S, V=V, T_d=T_d, p1=p1):
                pp = min(1000.0, 1000.0 * tt / T_d)
                f = pp / 1000.0 if pp <= p1 else (p1 + 0.5 * (pp - p1)) / 1000.0
                return S[0] + V[0] * f, S[1] + V[1] * f
            n_tot = int(round(I['delay_ms'] / float(MO_TICK_MS)))
            E = xyfn(float(n_tot)); n_r = max(0, n_tot - t_now)
            vb = MO_BOUNCE_V * abs(pr[3]); zb = MO_BOUNCE_Z0 * abs(pr[3]); aq = abs(pr[0]) / 4.0
            u_end = (vb + _mo_math.sqrt(vb * vb + 4 * aq * zb)) / (2 * aq)
            bnc = dict(a=pr[0], v0=pr[1], t_L=min(max(pr[2], 0.0), float(n_tot)), v_L=pr[3], z0=z0, xyfn=xyfn, n_tot=n_tot, u_end=u_end)
    reason = 'target' if I['lob'] else 'range'
    out['range_end'] = E; out['range_ticks'] = n_r     # the uncut end: walls, border and characters ignored (traj)
    n_end, end_xy = n_r, E
    if grid is not None and not I['lob'] and n_r > 0:
        rmax = max(L * n_r, _mo_math.hypot(E[0] - x, E[1] - y)) + MO_WALL_MARGIN + 5
        sd, why = _mo_wall_dist(grid, x, y, u[0], u[1], rmax, I, dead or ())
        if sd < _mo_math.inf:
            n_w = max(1, int(_mo_math.ceil((sd - MO_WALL_MARGIN) / max(L, 1.0) - 1e-9)))
            if n_w <= n_r:
                n_end = n_w; end_xy = (x + u[0] * max(0.0, sd - 1.0), y + u[1] * max(0.0, sd - 1.0)); reason = 'wall'
                if why == 'border':   # the map border clamps each coordinate on its own: the other one keeps its full step
                    end_xy = (min(17999.0, max(1.0, x + u[0] * L * n_w)), min(17999.0, max(1.0, y + u[1] * L * n_w)))
    if I['targeted'] and not has_prev and target is None and not I['lob']: out['aim_estimated'] = True     # the aim point is not in the state: T x Speed / 20
    if others and not I['lob'] and not I['pierce_char'] and n_end > 0:
        rmax = max(L * n_end, _mo_math.hypot(end_xy[0] - x, end_xy[1] - y)) + 5
        sh = _mo_hit_dist(x, y, u[0], u[1], rmax, others, int(row[2]), I['radius'], L)
        if sh < _mo_math.inf:
            n_h = max(1, int(_mo_math.ceil((sh - MO_HIT_MARGIN) / max(L, 1.0) - 1e-9)))
            if n_h <= n_end:
                n_end = n_h; end_xy = (x + u[0] * sh, y + u[1] * sh); reason = 'hit'
    out.update(end_world=end_xy, end_reason=('unknown' if unrel else reason), end_tick=tick + n_end, ticks_left=n_end)
    out['tick_certain'] = hsrc in ('row', 'shooter', 'height', 'client') or target is not None
    out.update(t_now=t_now, T=T_, Td=T_d, L=L, S=S, bounce=bnc, z_now=float(row[5]), src_did=src_did)
    out['reason_raw'] = reason; out['h'] = h; out['hsrc'] = hsrc; out['T'] = T_; out['kk'] = kk
    return out


# ---------------------------------------------------------------- the predicted path
def mo_traj_points(pred, row, I, drawn=None):
    """[(tick, x, y, z)] logic points of the projectile's future path, one per tick from the current tick to the END OF ITS FULL RANGE
    (or landing point; for Dynamike the end of the bounce), IGNORING walls, the map border and characters (the owner's wall model cuts it),
    and the kind: 'arc' (lobbed: height parabola over a straight ground line), 'bounce' (Dynamike: arc, landing, half-speed bounce),
    'straight' (every other type: flat, constant height). The DRAWN position of a curved TravelType, boomerang or bouncing-type
    projectile is a straight line too, measured on every one of them (their wiggle is inside the sprite), and the sim state of boomerang
    types shows no return leg: they end at their range.
    drawn: the client's drawn (x, y, z) now (obj['anchor']) -> every point is shifted by (drawn - logic now), so point 0 is where the
    projectile is drawn right now"""
    x, y = pred['range_end']; tick = pred['end_tick'] - pred['ticks_left']; n = pred['range_ticks']
    rowx, rowy, rowz = float(row[3]), float(row[4]), float(row[5])
    t_now = pred.get('t_now'); T_ = pred.get('T'); bn = pred.get('bounce')
    if t_now is None or n <= 0:
        kind = 'arc' if I['lob'] else 'straight'
        return kind, [(tick, drawn[0] if drawn else rowx, drawn[1] if drawn else rowy, drawn[2] if drawn else rowz)]
    if not I['lob']:
        kind = 'straight'
        fpt = lambda k: (rowx + (x - rowx) * k / float(n), rowy + (y - rowy) * k / float(n), rowz)
    elif bn is not None:
        kind = 'bounce'
        a, v0, t_L, vL, z0 = bn['a'], bn['v0'], bn['t_L'], bn['v_L'], bn['z0']
        def fpt(k):
            tt = t_now + k
            if tt <= t_L: z = z0 + v0 * tt + a * tt * tt
            else:
                uu = min(tt - t_L, bn['u_end']); z = MO_BOUNCE_Z0 * abs(vL) + MO_BOUNCE_V * abs(vL) * uu + (a / 4.0) * uu * uu
            xx, yy = bn['xyfn'](tt)
            return (xx, yy, max(0.0, z))
    else:
        kind = 'arc'
        z0 = _mo_start_z(pred.get('src_did')) or max(rowz, 300.0)
        Tz = pred.get('Td', T_)    # the height parabola lasts 1000 // travel0 ticks, also when the xy lerp is cut short by a delay
        A = _mo_arc_A(z0, rowz, t_now, Tz)
        if A is None: A = I['gravity'] / float(max(Tz, 17)) if I['gravity'] else 10.0
        def fpt(k):
            tt = min(max(t_now + k, 0), Tz)
            return (rowx + (x - rowx) * k / float(n), rowy + (y - rowy) * k / float(n), max(0.0, z0 * (1.0 - tt / float(Tz)) + A * tt * (Tz - tt)))
    # the client draws between the previous and the current state: drawn(k) = state(k) - lag x (state(k) - state(k - 1)); the lag fraction
    # follows from where the projectile is drawn now (drawn) against its state now
    lag = 0.0
    if drawn is not None:
        p0, pm = fpt(0), fpt(-1); sx, sy = p0[0] - pm[0], p0[1] - pm[1]; sn = sx * sx + sy * sy
        if sn > 25.0: lag = min(1.2, max(0.0, ((rowx - drawn[0]) * sx + (rowy - drawn[1]) * sy) / sn))
    pts = []
    for k in range(n + 1):
        c, pv = fpt(k), fpt(k - 1)
        pts.append((tick + k, c[0] - lag * (c[0] - pv[0]), c[1] - lag * (c[1] - pv[1]), c[2] - lag * (c[2] - pv[2])))
    return kind, pts


# ---------------------------------------------------------------- the labels() wrapper
def _mo_flags():
    e = _mo_os.environ
    return (e.get('IDLABELS_MOTION_DIR') == '1', e.get('IDLABELS_MOTION_END') == '1', e.get('IDLABELS_MOTION_ENDTIME') == '1',
            e.get('IDLABELS_MOTION_TRAJ') == '1')


def _mo_grid_path(R):
    p = _mo_os.environ.get('IDLABELS_MAP', '')
    if not p and R is not None and R.get('dir'):
        d = _mo_os.path.abspath(R['dir'])
        for _ in range(4):
            if _mo_os.path.exists(_mo_os.path.join(d, 'map.npy')): return _mo_os.path.join(d, 'map.npy')
            d = _mo_os.path.dirname(d)
    return p


def _mo_grid(R):
    p = _mo_os.environ.get('IDLABELS_MAP', '')
    if not p and R is not None and R.get('dir'):
        d = _mo_os.path.abspath(R['dir'])
        for _ in range(4):
            if _mo_os.path.exists(_mo_os.path.join(d, 'map.npy')): p = _mo_os.path.join(d, 'map.npy'); break
            d = _mo_os.path.dirname(d)
    if not p or not _mo_os.path.exists(p): return None
    return _mo_np.load(p).reshape(60, 60).astype(_mo_np.int64) - 27000000


def _mo_screen(cam, x, y, z=0.0):
    q = _mo_np.asarray(cam, float) @ _mo_np.array([x, -y, 0.6 * z, 1.0])
    return (q[0] / q[3] * 0.5 + 0.5) * 1200, (1 - (q[1] / q[3] * 0.5 + 0.5)) * 540


def mo_group(preds, org=None):
    """one shot = one label: preds [(point (x, y), prediction dict of mo_predict)] of its member projectiles -> (mean unit direction,
    the member prediction whose end is farthest from the shooter org (default: the members' mean position))"""
    ux = sum(p['dir'][0] for _, p in preds); uy = sum(p['dir'][1] for _, p in preds); n = _mo_math.hypot(ux, uy)
    u = (ux / n, uy / n) if n > 1e-9 else preds[0][1]['dir']
    if org is None: org = (sum(q[0] for q, _ in preds) / len(preds), sum(q[1] for q, _ in preds) / len(preds))
    far = max(preds, key=lambda qp: _mo_math.hypot(qp[1]['end_world'][0] - org[0], qp[1]['end_world'][1] - org[1]))[1]
    return u, far


def _mo_mark_dead(state, grid, dead):
    """destructible tiles a piercing shot has just stopped at are destroyed from now on (they no longer stop any shot)"""
    blk, dst = _mo_tiles()
    for r in state['proj']:
        if int(r[12]) != 4: continue
        I = _mo_info(int(r[1]))
        if I is None or not I['pierce_env']: continue
        dx, dy = float(r[3]) - float(r[6]), float(r[4]) - float(r[7]); n = _mo_math.hypot(dx, dy)
        if n < 1: continue
        tx, ty = int((float(r[3]) + dx / n * 4) // 300), int((float(r[4]) + dy / n * 4) // 300)
        if 0 <= tx < grid.shape[1] and 0 <= ty < grid.shape[0] and int(grid[ty, tx]) in (blk & dst): dead.add((tx, ty))


_MO_DEAD = {}     # map path -> set of destroyed tiles
_MO_PREV = {}     # tick -> {character id: (x, y)} of the last states seen (frames come in order; a new capture clears it)


def _mo_chars_with_velocity(state, tick):
    """[(x, y, owner player, hp, vx, vy)] of the state's characters; the velocity (units per tick) from the nearest earlier state seen
    at most 3 ticks ago, 0 when there is none (then characters count as standing still)"""
    if _MO_PREV and tick < max(_MO_PREV): _MO_PREV.clear(); _MO_DEAD.clear()
    cur = {int(c[0]): (float(c[3]), float(c[4])) for c in state['chars']}
    prev = next((_MO_PREV[t_] for t_ in range(tick - 1, tick - 4, -1) if t_ in _MO_PREV), None); pt = next((t_ for t_ in range(tick - 1, tick - 4, -1) if t_ in _MO_PREV), tick - 1)
    out = []
    for c in state['chars']:
        cid = int(c[0]); x, y = cur[cid]; vx = vy = 0.0
        if prev is not None and cid in prev: vx, vy = (x - prev[cid][0]) / (tick - pt), (y - prev[cid][1]) / (tick - pt)
        out.append((x, y, int(c[2]), int(c[6]), vx, vy))
    _MO_PREV[tick] = cur
    for t_ in [t_ for t_ in _MO_PREV if t_ < tick - 6]: del _MO_PREV[t_]
    return out


def mo_motion(objs, cam, state, grid=None, flags=(True, True, True, False), causal=False, dead=None):
    """{label id: motion dict} for the objects of a frame (see the header)"""
    import bulletlabels as BL
    rows = {int(r[0]): r for r in state['proj']}
    chars = {int(c[0]): c for c in state['chars']}
    tick = int(state['tick'])
    others = _mo_chars_with_velocity(state, tick)
    if dead is not None and grid is not None: _mo_mark_dead(state, grid, dead)
    groups = {}
    for o in objs:
        groups.setdefault(BL._lid(o), []).append(o)
    out = {}
    for lid, mem in groups.items():
        ids = set()
        for o in mem:
            ids.add(o['id'])
            for m in o.get('attack_members') or (): ids.add(m)
        preds = []; org = None
        for i in sorted(ids):
            r = rows.get(i)
            if r is None: continue
            o = next((q for q in mem if q['id'] == i), mem[0])
            c = chars.get(int(r[8])); sh = (float(c[3]), float(c[4])) if c is not None else None
            if sh is not None and org is None: org = sh
            sdid = int(c[1]) if c is not None else None
            P_ = mo_predict(r, tick, grid, angle=o.get('angle') if i == o['id'] else None, shooter=sh, src_did=sdid, dead=dead, n_seen=(len(o['path']) if (i == o['id'] and o.get('path')) else None),
                            target=None if causal else (o.get('target') if i == o['id'] else None), others=others)
            if P_ is not None:
                P_['_row'] = r; P_['_obj'] = o
                preds.append(((float(r[3]), float(r[4])), P_))
        if not preds: continue
        u, far = mo_group(preds, org)
        e = {}
        if flags[0]:
            a = _mo_math.degrees(_mo_math.atan2(u[1], u[0])) % 360.0
            px0 = preds[0][0]
            s0 = _mo_screen(cam, px0[0], px0[1]); s1 = _mo_screen(cam, px0[0] + u[0] * 100.0, px0[1] + u[1] * 100.0)
            dv = (s1[0] - s0[0], s1[1] - s0[1]); dn = _mo_math.hypot(*dv) or 1.0
            e.update(dir=(u[0], u[1]), angle_deg=a, dir_screen=(dv[0] / dn, dv[1] / dn))
        if flags[1]:
            g = far['end_world']
            e.update(end_world=(float(g[0]), float(g[1])), end_screen=_mo_screen(cam, g[0], g[1]), end_reason=far['end_reason'],
                     walls=far['walls'])
        farr = max(preds, key=lambda qp: _mo_math.hypot(qp[1]['range_end'][0] - (org or qp[0])[0], qp[1]['range_end'][1] - (org or qp[0])[1]))[1]
        if flags[1]:
            g = farr['range_end']; e.update(range_end_world=(float(g[0]), float(g[1])), range_end_screen=_mo_screen(cam, g[0], g[1]))
        if flags[2]:
            e.update(end_tick=int(far['end_tick']), ticks_left=int(far['ticks_left']), range_end_tick=int(farr['end_tick'] - farr['ticks_left'] + farr['range_ticks']))
        if len(flags) > 3 and flags[3]:
            I = _mo_info(int(farr['_row'][1])); an = farr['_obj'].get('anchor')
            drawn = (float(an[0]), -float(an[1]), float(an[2]) / 0.6) if an is not None else None
            kind, pts = mo_traj_points(farr, farr['_row'], I, drawn)
            e['traj_kind'] = kind
            e['traj'] = [(int(tk), float(px), float(py), float(pz), *_mo_screen(cam, px, py, pz)) for tk, px, py, pz in pts]
        e['mode'] = far['mode']; e['category'] = far['category']; e['members'] = len(preds)
        out[lid] = e
    return out


try: _mo_prev_labels = labels
except NameError: _mo_prev_labels = None

if _mo_prev_labels is not None:
    def labels(objs, cam, R, state=None, ended=(), img=None):
        P = _mo_prev_labels(objs, cam, R, state=state, ended=ended, img=img)
        fl = _mo_flags()
        if not any(fl) or state is None: return P
        P2 = dict(P)
        meta = P.get('meta') or {}
        proj_lids = {lid[0] for lid in P['lids'] if lid[2] == 'body'}
        keep = [o for o in objs if _mo_lid(o) in proj_lids and (meta.get(_mo_lid(o)) or {}).get('kind') in ('attack', 'super', 'spawnable')]
        grid = _mo_grid(R); dead = _MO_DEAD.setdefault(_mo_grid_path(R), set())
        P2['motion'] = mo_motion(keep, cam, state, grid, fl, causal=_mo_os.environ.get('IDLABELS_MOTION_CAUSAL') == '1', dead=dead)
        return P2


def _mo_lid(o):
    return o.get('label_id', o['id'])
