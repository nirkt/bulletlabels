"""Labels from the client's own draws (round 3, the user's decision 2026-10-19): the ID pass (idbuf.py) gives, per
pixel, the front-most non-additive draw, its coverage and the sprite-atlas position it shows. A pixel belongs to a
projectile when that atlas position lies inside one of the projectile's sprite shapes (its clip, or for TRAIL_BODY types
the sprites of its trail particles) and the pixel is near the projectile (its predicted label, or its trail region).
Walls, bushes, UI and other objects drawn in front are therefore never labelled, random start frames and random
particles are exactly what was drawn.
  sprites(t, own) -> SpriteSet of a type: per (file, texture) the atlas polygons of every shape its clip can show
  attribute(objs, R, cam, P) -> {object id: dict(body (H, W) bool, glow (H, W) bool)} at half resolution
Objects with no sprite pixels (3D meshes: no atlas position) keep the predicted label, minus pixels whose front-most
draw is not the mesh's draw (occluded)."""
import functools, os
import numpy as np
from PIL import Image, ImageDraw
import rc, projtab, projdraw as PD, sctex, glow, wave

CUT = 128          # coverage (0..255) from which a pixel is the projectile's (the fragment at least half opaque)
NEAR = 24          # px (half resolution) around the predicted label in which the projectile's sprites are searched
SCALE = 1          # atlas region maps at full texture resolution (neighbouring sprites in an atlas must not merge)
SWATCH = 3         # texels: an atlas polygon thinner than this is a colour-fill swatch (region_map)
# types whose trail particles are part of the projectile (labelled as body): every wave type, plus
# Mortis's super, every skin (the bat swarm is the attack: the user's 'missing' findings, 2026-10-19), and Sandy's attack
# (the sand cloud and its stars are one label: the user, 2026-10-19)
# Sandy's attack and Poco's attack, every skin, and Poco's super, every skin (the green cloud with its notes: the user's
# 'bad' mark on Poco's default super page, 2026-10-20): zones (ZONE)
SANDY_ATK = {6000666, 6000668, 6000670, 6000672, 6000674, 6000676, 6000678}
POCO_ATK = {6000335, 6000337, 6000339, 6000341, 6000343, 6000345, 6000347, 6000349, 6000351, 6000353, 6000355}
POCO_SUPER = {6000336, 6000338, 6000340, 6000342, 6000344, 6000346, 6000348, 6000350, 6000352, 6000354, 6000356, 6001521}
TRAIL_BODY_EXTRA = {6000299, 6000300, 6000301, 6000302, 6000303, 6000304, 6000305, 6000306, 6000307, 6001497, 6001498,
                    6000666, 6000335} | SANDY_ATK | POCO_ATK | POCO_SUPER
# zone attacks: one solid label over the whole drawn cloud (no separate parts): Sandy's attack (the sand cloud, its stars
# not separately) and Poco's attack (its puffs), the user's decisions 2026-10-19
ZONE = SANDY_ATK | POCO_ATK | POCO_SUPER
# Gale's attack, every skin: only the snowballs; its snow drops, splashes and other snow effects are ignored (the user,
# 2026-10-19): a projectile is one snowball, so parts of its label smaller than BODY_ONLY_FRAC of its largest part (the
# drops beside it) are dropped; no hit / end effect label
BODY_ONLY = {6000731, 6000733, 6000735, 6000737, 6000739, 6000741,
             6000590}   # Rosa Toon's donuts: the sprinkles flying off them are not the projectile (2026-10-20)
# solid labels (holes filled, like Bibi's bubble in bulletlabels.FILL_HOLES): Rosa Toon's donut with its sprinkles on it
SOLID = {6000590}
SOLID_OPEN = 2      # px (half resolution): parts thinner than about twice this are cut off a SOLID label
SOLID_SLIVER = 5    # px: pieces the opening cuts off smaller than this are edge slivers (kept)
# thin parts sticking out of the body cut off (the opening of SOLID, without the hole fill): Bulletstorm's 199-gem
# skin attack, its clip's motion-streak strips above the vial (Haiku's 'bad' flag, 2026-10-20)
THIN_OFF = {6001721: 3, 6001722: 3}   # type -> opening px
# trail clips of trail-body types that are not the projectile (name prefixes): Rosa Toon's flying sprinkles
TRAIL_SKIP = {6000590: ('rosa_007_atk_sprinkle',)}
# ground effects in the trails of trail-body types (wave attacks: Frank, Ash, ...): cracks, ground fragments, smoke,
# soot, dust, debris, stones, dust clouds (Ash) are not the projectile (the user's 'bad' marks on Frank's and Ash's pages, 2026-10-20);
# clip names containing one of these (lower case). Not for ZONE types (the whole cloud is their label)
TRAIL_FX = ('crack', 'groundfrag', 'ground_smoke', 'soot', 'dust', 'debris', 'smoke', 'stone', 'cloud',
            # ground decals left behind (Ash Ninja's slash marks, 2026-10-20 Haiku 'bad'; the same kind elsewhere)
            'ground_slash', 'footprint', 'ground_cheese', 'ground_ketchup', 'ground_fire', 'iceground', 'ground_elec',
            'particle_mini', 'particle_leaf')
# types whose ground cracks are drawn with the same atlas sprite as the projectile (Frank's default attack / super:
# one white triangle, tinted blue-white for the crystals and dark brown for the cracks): labelled draws whose median
# brightness (max of R, G, B in the frame) is below DARK_V are dropped (cracks 60-100, crystals 140+, measured)
DARK_DROP = {6000475, 6000476, 6001470, 6001764,
             6001040, 6001041, 6001042}   # Ash 002 (Ninja): its slash marks on the ground are the slash sprite tinted dark
DARK_V = 120
# trail particles touching the body are the projectile (connected, at least half opaque): Surge's default attack, its
# water-splash blob around the core (the user, 2026-10-20: 'whole cyan blob'; detached dots and bubbles stay out)
TRAIL_ATTACH = {6000787, 6000788, 6000789, 6000790, 6000791, 6000792, 6000793, 6000795}
# bright additive layers just outside the body are the projectile (invisible to the ID pass: from the frame): type ->
# min(R, G, B) from which a pixel is the bright layer; grown from the body at most RIM_R px (soft halos stay out).
# Leon's blade edges (the user, 2026-10-20: 'include bright parts')
BRIGHT_RIM = {6000572: 120}
RIM_R = 4
# soft translucent streaks with no opaque sprite: the pixels of its own sprites (body or glow, any coverage) whose
# frame colour is the streak's (blue - green >= threshold), gaps closed. Lily's attack (the user, 2026-10-20: the
# visible purple streak)
STREAK = {6001452: 60}
# attacks drawn only additively (no clip; their trail sprites are additive: the ID pass sees nothing of them): body =
# the frame's bright pixels (max of R, G, B >= threshold) in the projectile's region and near its trail sprites'
# smoke, main cluster only. Yeti's claw streaks (Bronson003, Haiku 'bad', 2026-10-20)
BRIGHT_BODY = {6001224: 200, 6001228: 200}   # + Bronson004 (Darkcircus): its translucent swoosh
# energy attacks and big attacks drawn as many thin lines: ONE solid label over the whole attack (the user, 2026-10-20):
# type -> px (half resolution) of the gaps closed between its lines; holes filled
ONE_LABEL = {6001040: 'hull', 6001041: 'hull', 6001042: 'hull',   # Ash 002 (Ninja): its slashes
             6001224: 'hull'}   # Yeti's claw streaks
# types whose clip body is drawn translucent (fades along its length, or stacked translucent layers; still clearly
# drawn): coverage from which their drawn body pixels are body (instead of CUT). Beamer's super: the beam's start fades (coverage 30-110)
# types whose clip body reaches beyond its predicted region (a long stretched sprite): connected clip-body pixels of
# its own draws are body wherever they reach (opt-in: on other types it grew good labels along their trails)
EXTEND_BODY = {6001299}
FADE_CUT = {6001299: 32,
            6001376: 32}   # Willow's attack: the orb is stacked translucent layers (top coverage mostly 32-127)
BODY_ONLY_FRAC = 0.3
ZONE_CLOSE = 3       # px: gaps between the cloud's puffs closed
ZONE_MIN = 0.15      # parts smaller than this share of the largest are dropped (detached stars, stray puffs)


def _clip_shapes(S, cid, seen=None, out=None):
    """every (shape id, command) a clip can show, over all frames of it and its sub-clips"""
    if seen is None: seen, out = set(), set()
    if cid in seen: return out
    seen.add(cid); c = S.clips[cid]
    for f in range(max(1, c['nframes'])):
        for slot, child, cm, cc, bl in S.frame(cid, f):
            if child in S.shapes: out.update((child, k) for k in range(len(S.shapes[child])))
            elif child in S.clips: _clip_shapes(S, child, seen, out)
    return out


@functools.lru_cache(maxsize=None)
def _layer_classes(t, own):
    """{(shape, cmd): 'body' | 'glow' | 'wave'} of a type's clip layers (glow.layer_class, the same rules as the predicted
    labels; a shape used by several layers takes 'body' if any is body)"""
    T = projtab.table()[t]; exp = T['blue'] if own else T['red']; out = {}
    if not (T['file'] and exp): return out
    try: kind, cl = glow.clip_layers(T['file'], exp)
    except Exception: cl = {}
    for (path, sid, cmd, bl), c in cl.items():
        if out.get((sid, cmd)) != 'body': out[(sid, cmd)] = c
    return out


def _trail_clips(t):
    T = projtab.table()[t]; E = wave._emitters(); out = []
    for eff in (T['trail'], T['special_trail'], T.get('special_trail2', '')):
        for em in wave._effects().get(eff, []) if eff else []:
            for mc in (E.get(em) or {}).get('movieclip', []) or []:
                out.append((mc['swf'], mc['name']))
    return out


def trail_body(t):
    return glow.type_kind(t, True) == 'wave' or t in TRAIL_BODY_EXTRA


@functools.lru_cache(maxsize=None)
def sprites(t, own):
    """{(file, texture index): [(polygon (n, 2) atlas px, part 'clip' | 'trail', class)]}; t = ('zone', area effect
    data id) for a zone (zone_sprites)"""
    if isinstance(t, tuple): return zone_sprites(t[1], own) if t[0] == 'zone' else end_sprites(t[1], own)
    T = projtab.table()[t]; exp = T['blue'] if own else T['red']; out = {}
    tb = trail_body(t); trail = []    # atlas boxes of its trail sprites: (file, tex, u0, v0, u1, v1)
    for swf, name in _trail_clips(t):
        cid = PD.sc(swf).exports.get(name)
        if cid is None: continue
        for sid, k in _clip_shapes(PD.sc(swf), cid):
            tex, uv, xy = PD.sc(swf).shapes[sid][k]; a = np.asarray(uv, np.float64)
            trail.append((swf, tex) + tuple(np.r_[a.min(0), a.max(0)]))

    def shared(file, tex, uv):   # the same atlas region as one of its trail sprites (a copy under another shape id)
        a = np.asarray(uv, np.float64); b = np.r_[a.min(0), a.max(0)]; ar = max(1e-9, (b[2] - b[0]) * (b[3] - b[1]))
        for f2, t2, *c in trail:
            if (f2, t2) != (file, tex): continue
            ix = max(0, min(b[2], c[2]) - max(b[0], c[0])) * max(0, min(b[3], c[3]) - max(b[1], c[1]))
            if ix >= 0.9 * ar and ix >= 0.9 * max(1e-9, (c[2] - c[0]) * (c[3] - c[1])): return True
        return False
    if T['file'] and exp:
        S = PD.sc(T['file']); cid = S.exports.get(exp); lc = _layer_classes(t, own)
        if cid is not None:
            for sid, k in _clip_shapes(S, cid):
                tex, uv, xy = S.shapes[sid][k]
                cls = 'body' if t in glow.UNFADE and sid in glow.UNFADE[t] else lc.get((sid, k))
                if cls is None: cls = 'body' if glow.paints(T['file'], dict(shape=sid, cmd=k, blend=0, mul=np.array([255, 255, 255, 255]))) else 'glow'
                if not tb and shared(T['file'], tex, uv):   # a sprite its trail particles show too (Nita's super):
                    out.setdefault((T['file'], tex), []).append((np.asarray(uv, np.float64) / 65535.0, 'clip', 'shared')); continue
                out.setdefault((T['file'], tex), []).append((np.asarray(uv, np.float64) / 65535.0, 'clip', 'body' if cls == 'wave' else cls))
    # trail sprites: the body of trail-body types; for the others only 'over' (drawn over the projectile's body: the
    # predicted body stays there, the trail itself is not labelled)
    def skipped(name):
        return (name.startswith(TRAIL_SKIP[t]) if t in TRAIL_SKIP else False) or (t not in ZONE and any(k in name.lower() for k in TRAIL_FX))
    for swf, name in _trail_clips(t):
        S = PD.sc(swf); cid = S.exports.get(name)
        if cid is None: continue
        sk = skipped(name)
        for sid, k in _clip_shapes(S, cid):
            tex, uv, xy = S.shapes[sid][k]
            out.setdefault((swf, tex), []).append((np.asarray(uv, np.float64) / 65535.0, 'trail', 'body' if tb and not sk else 'over'))
    return out


# sprite codes of sprite_pixels and their priority where atlas regions overlap (a sprite shared by several parts)
CODE = {('clip', 'body'): 1, ('clip', 'glow'): 2, ('trail', 'body'): 3, ('trail', 'over'): 4, ('clip', 'shared'): 5}
PRIO = {1: 4, 3: 3, 2: 2, 5: 1, 4: 0}   # 5: a clip sprite its trail shows too (resolved per draw in labels())


ZONE_EXPORTS = ('OwnExportName', 'BlueExportName', 'RedExportName', 'NeutralExportName', 'ExportNameTop', 'ExportNameTopRed',
                'ExportNameObject')


@functools.lru_cache(maxsize=None)
def emitter_clips():
    """{emitter name: [{swf, name}]} of every particle emitter (tools/decode_emitter_clips.py), plus the trail emitters"""
    import json
    out = {k: v.get('movieclip') or [] for k, v in wave._emitters().items() if isinstance(v, dict)}
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'emitter_clips.json')
    if os.path.exists(p): out.update(json.load(open(p)))
    return out


@functools.lru_cache(maxsize=None)
def _effect_rows():
    """effect name -> its rows (effects.csv: an effect is its named row plus the unnamed rows below), as dicts"""
    import csv
    R = list(csv.reader(open(os.path.join(rc.ASSETS, 'csv_client', 'effects.csv'), encoding='utf8'))); h = R[0]; out = {}; cur = None
    for r in R[2:]:
        if r and r[0]: cur = r[0]; out[cur] = []
        if cur is not None: out[cur].append(dict(zip(h, r)))
    return out


@functools.lru_cache(maxsize=None)
def effect_clips(name, depth=0):
    """[(swf, export)] every sprite clip an effect can show: its SWF rows (FileName / ExportName), its particle
    emitters' clips, its nested effects (Effect column), and the red variant (the column after Layer's group)"""
    out = []
    if not name or depth > 4: return out
    EC = emitter_clips()
    for r in _effect_rows().get(name, []):
        if r.get('FileName') and r.get('ExportName'): out.append((r['FileName'], r['ExportName']))
        if r.get('ParticleEmitterName'): out += [(mc['swf'], mc['name']) for mc in EC.get(r['ParticleEmitterName'], [])]
        for c in ('Effect', 'EnemyVersion'):
            if r.get(c) and r[c] != name: out += effect_clips(r[c], depth + 1)
    return sorted(set(out))


def _clips_sprites(clips):
    out = {}
    for swf, name in set(clips):
        try: S = PD.sc(swf)
        except Exception: continue
        cid = S.exports.get(name)
        if cid is None: continue
        for sid, k in _clip_shapes(S, cid):
            tex, uv, xy = S.shapes[sid][k]
            out.setdefault((swf, tex), []).append((np.asarray(uv, np.float64) / 65535.0, 'clip', 'body'))
    return out


def zone_sprites(did, own):
    """sprite set of a lingering damage zone (area_effects_skin row): its clips (every export column: own / blue / red /
    neutral / top / object) and the clips of its effects' emitters (Effect, LoopingEffect); all 'clip' body"""
    import owners
    Z = owners.zone(did); out = {}
    if not Z: return out
    clips = [(Z.get('FileName'), Z.get(c)) for c in ZONE_EXPORTS if Z.get(c) and Z.get('FileName')]
    for eff in (Z.get('Effect'), Z.get('LoopingEffect')): clips += effect_clips(eff)
    return _clips_sprites(clips)


END_EFFECTS = ('HitEffectEnv', 'HitEffectChar', 'MaxRangeReachedEffect', 'CancelEffect')


def end_sprites(t, own):
    """sprite set of the effects a projectile type plays when it ends (hit a wall / a character, max range, cancel)"""
    import owners
    row = owners._tables()['pskin'][t - 6000000]; clips = []
    for c in END_EFFECTS: clips += effect_clips(row.get(c))
    return _clips_sprites(clips)


END_R = 350   # logic units around an ended projectile's last position in which its end effect is searched
END_TOP = 450


# GLOBAL (the user, 2026-10-23): once an attack has ended, its leftover effects are unlabelled, for every attack. Wave
# groups: Tracker.gone + the group filter in labels(). Other projectiles leave the state when they end, but a LIVE
# projectile's label could take in the hit burst / end stars of one that ended nearby (same atlas shapes within NEAR; agent 1:
# Colt / Rico bursts at a crate). drop_ended_parts removes such parts. Unit = one connected part of one draw (batched draws
# hold several sprites: a draw-level cut could drop the live sprite). A part is dropped when < ENDED_OWN of it lies within
# ENDED_GROW px of this projectile's predicted label and >= ENDED_AT of it lies around a projectile that ended in the last
# END_TICKS ticks. IDLABELS_NO_ENDED=1 turns it off (before / after comparisons).
ENDED_DROP = not os.environ.get('IDLABELS_NO_ENDED')
ENDED_GROW = 12; ENDED_OWN = 0.25; ENDED_AT = 0.5; ENDED_R = 150   # px; fractions; logic units around the end point + radius
# (10-24: own = within ENDED_GROW px of the projectile's whole predicted label (pred_any), not its predicted body: a bent
# Darkcircus trail's head lay 78 % within 12 px of pred_any but only 30 % within 4 px of pred_body and was dropped, sw2_005 t273)
ENDED_KEEP = set()   # types exempt (per-type exceptions)


def _ended_region(ended, cam, shape, skip):
    H, W = shape; yy, xx = np.mgrid[0:H, 0:W]; region = np.zeros((H, W), bool)
    for e in ended:
        if e['id'] in skip: continue
        x, y = e['x'], e['y']; r = ENDED_R + projtab.table()[e['t']]['radius']
        cx, cy = _proj(cam, x, y); rx = abs(_proj(cam, x + r, y)[0] - cx); ry = abs(_proj(cam, x, y + r)[1] - cy)
        if rx < 1 or ry < 1: continue
        for zz in range(0, END_TOP + 1, 150):
            hx, hy = _proj(cam, x, y, zz); region |= ((xx - hx) / rx) ** 2 + ((yy - hy) / ry) ** 2 <= 1
    return region


def drop_ended_parts(body, o, cam, ended, ids_h, pred_body, pred_any):
    """body without the parts that belong to an ended projectile's leftovers (see ENDED_DROP)"""
    if not body.any() or not pred_any.any(): return body
    skip = {o['id'], o.get('label_id', o['id'])} | set(o.get('attack_members') or [])
    at = _ended_region([e for e in ended if e['id'] not in skip], cam, body.shape, skip)
    if not (at & body).any(): return body
    from scipy.ndimage import label as cc
    own = _dilate(pred_any, ENDED_GROW); pa = own; kept = np.zeros_like(body); cand = []; best = (0, None)
    for d in np.unique(ids_h[body]):
        L, n = cc(body & (ids_h == d), structure=np.ones((3, 3)))
        for k in range(1, n + 1):
            m = L == k; c = m.sum(); ov = (m & pa).sum()
            if ov > best[0]: best = (ov, m)
            if (m & own).sum() < ENDED_OWN * c and (m & at).sum() >= ENDED_AT * c: cand.append(m)
            else: kept |= m
    # a leftover candidate touching the kept body stays (e.g. a splash particle on the live core)
    near = _dilate(kept, 1); out = kept.copy()
    for m in cand:
        if (m & near).any(): out |= m
    if not out.any() and best[1] is not None: out = best[1]   # offset prediction: keep its best part
    return out


def end_labels(ended, cam, R, taken):
    """{projectile id: (H, W) bool}: the pixels near an ended projectile (Tracker.ended) where one of its end effects'
    sprites is the front-most draw, at least half opaque (like the projectile's body)"""
    ids_h = R['id'][::2, ::2]; H, W = ids_h.shape; out = {}; yy, xx = np.mgrid[0:H, 0:W]
    for e in ended:
        if e['t'] in BODY_ONLY or not sprites(('end', e['t']), e['own']): continue
        x, y = e['x'], e['y']; r = END_R + projtab.table()[e['t']]['radius']
        cx, cy = _proj(cam, x, y); rx = abs(_proj(cam, x + r, y)[0] - cx); ry = abs(_proj(cam, x, y + r)[1] - cy)
        if rx < 1 or ry < 1: continue
        region = np.zeros((H, W), bool)
        for zz in range(0, END_TOP + 1, 150):
            hx, hy = _proj(cam, x, y, zz); region |= ((xx - hx) / rx) ** 2 + ((yy - hy) / ry) ** 2 <= 1
        C = sprite_pixels(('end', e['t']), e['own'], R)
        m = region & (C > 0) & (R['cv'][::2, ::2] >= CUT) & ~taken   # explosions are projectiles: at least half opaque
        if m.any(): out[e['id']] = m | (small_holes(m, SPECK) & (C > 0) & ~taken)   # drawn pixels, own-sprite specks filled
    return out


def effect_sprite(R, ids_h, t, own):
    """(H, W) bool: the front-most draw is a blended sprite from one of type t's atlases (by size and accepted hash)"""
    sizes = {tex_size(f, x): (f, x) for f, x in sprites(t, own)}
    sel = [d for d, dr in R['draws'].items() if dr.get('blend') and (dr.get('tw'), dr.get('tht')) in sizes
           and _ATLAS_N.get((dr.get('th'),) + sizes[(dr.get('tw'), dr.get('tht'))], [0, 0])[0] >= ATLAS_MIN_N
           and (dr.get('th'),) + sizes[(dr.get('tw'), dr.get('tht'))] not in REJECTED]
    return np.isin(ids_h, sel) & (R['cv'][::2, ::2] >= CUT) if sel else np.zeros(ids_h.shape, bool)


SPECK = 12
SHARED_NEAR = 4    # px: a shared sprite's pixels within this of the aligned prediction are its body
POST_BODY = {}     # type -> fn(body, ctx) -> body, applied after the per-type rules, before ONE_LABEL (override hooks)
EDGE_GROW = 2


def small_holes(m, n):
    from scipy.ndimage import binary_fill_holes, label as cc
    h = binary_fill_holes(m) & ~m
    if not h.any(): return h
    L, k = cc(h); sz = np.bincount(L.ravel())
    return np.isin(L, np.nonzero((sz <= n) & (np.arange(len(sz)) > 0))[0])


def blended_sprite(R, ids_h):
    sel = [d for d, dr in R['draws'].items() if dr.get('blend') and dr.get('tw', 0) > 0]
    return np.isin(ids_h, sel) & (R['uv'][..., 0] >= 0) if sel else np.zeros(ids_h.shape, bool)


def has_mesh(t, own):
    T = projtab.table().get(t) or {}
    return bool(T.get('blue_scw') if own else T.get('red_scw'))


def has_clip(t, own):
    """the type is drawn from an sc clip (a sprite projectile, not a mesh)"""
    T = projtab.table()[t]; exp = T['blue'] if own else T['red']
    return bool(T['file'] and exp and PD.sc(T['file']).exports.get(exp) is not None)


@functools.lru_cache(maxsize=None)
def tex_size(file, tex):
    h, w = sctex.texture(PD.sc(file), tex).shape[:2]; return w, h


@functools.lru_cache(maxsize=8)
def region_map(t, own, file, tex):
    """atlas map at 1/SCALE: 0 = none, 1 + i = sprite polygon i of sprites(t, own)[(file, tex)]"""
    w, h = tex_size(file, tex); W, H = (w + SCALE - 1) // SCALE, (h + SCALE - 1) // SCALE
    I = Image.new('I', (W, H), 0); d = ImageDraw.Draw(I)
    polys = sprites(t, own)[(file, tex)]
    for i in sorted(range(len(polys)), key=lambda i: PRIO[CODE[polys[i][1:]]]):   # higher priority drawn last, on top
        uv = polys[i][0]; p = uv * [W, H]   # a shape command's points are a triangle STRIP (t, t+1, t+2), not an outline (LABEL_SPEC 4)
        for k in range(len(p) - 2): d.polygon([tuple(p[k]), tuple(p[k + 1]), tuple(p[k + 2])], fill=i + 1)
        # vector colour fills (Rosa Toon's donut dough, cream): every point on one swatch texel (or a thin swatch strip),
        # a polygon of no area that rasterises to nothing; the texels around it (drawn uv is filtered by about 1 texel)
        lo, hi = p.min(0), p.max(0)
        if (hi - lo).min() < SWATCH:
            d.rectangle([int(lo[0]) - 1, int(lo[1]) - 1, int(hi[0]) + 1, int(hi[1]) + 1], fill=i + 1)
    return np.asarray(I, np.int32).astype(np.int16)   # int16: memory (16 maps of up to 4096 x 4096 cached)


@functools.lru_cache(maxsize=8)
def tex_alpha(file, tex):
    return np.ascontiguousarray(sctex.texture(PD.sc(file), tex)[..., 3])


@functools.lru_cache(maxsize=8)
def tex_alpha3(file, tex):
    """atlas alpha, max over 3x3 texels (sprites drawn small sample blurred mip levels: thin parts miss exact texels)"""
    from scipy.ndimage import maximum_filter
    return maximum_filter(tex_alpha(file, tex), size=3)


# which drawn texture is which sc atlas: the draw log names a texture only by its size and a hash of its upload, and
# atlases of the same size (UI, fonts, other effect files) exist. A texture is the atlas when the drawn pixels that are at
# least half opaque show (3x3) atlas alpha >= 64 at their uv. Measured pooled over 4 captures: the right atlases 0.99,
# the others <= 0.65. Counts are pooled over the frames of a run; no decision before ATLAS_MIN_N pixels (size match only)
ATLAS_MIN_N = 300; ATLAS_MIN_FRAC = 0.85
REJECTED = {}      # (texture hash, file, tex) -> (n, frac) of textures refused, for the comparison report
_ATLAS_N = {}      # (texture hash, file, tex) -> [pixels, matching pixels], pooled


def atlas_ok(R, th, draw_ids, file, tex):
    key = (th, file, tex); acc = _ATLAS_N.setdefault(key, [0, 0])
    if acc[0] < 20 * ATLAS_MIN_N:   # keep sampling until the share is well established
        ids = R['id'][::2, ::2]; uv = R['uv']; cv = R['cv'][::2, ::2]
        m = np.isin(ids, draw_ids) & (cv >= CUT) & (uv[..., 0] >= 0)
        if m.any():
            a = tex_alpha3(file, tex); h, w = a.shape; ys, xs = np.nonzero(m)
            u = np.clip((uv[ys, xs, 0] * w).astype(int), 0, w - 1); v = np.clip((uv[ys, xs, 1] * h).astype(int), 0, h - 1)
            acc[0] += len(ys); acc[1] += int((a[v, u] >= 64).sum())
    if acc[0] < ATLAS_MIN_N: return True   # too few pixels to tell: size match only (no evidence against it)
    fr = acc[1] / acc[0]; ok = fr >= ATLAS_MIN_FRAC
    if not ok: REJECTED[key] = (acc[0], round(fr, 2))
    else: REJECTED.pop(key, None)
    return ok


def sprite_pixels(t, own, R):
    """(H, W) int8 at half resolution: 0 none, 1 clip body, 2 clip glow, 3 trail body, 4 trail over the body (CODE); where
    the front-most draw shows one of type t's sprites"""
    ids = R['id'][::2, ::2]; uv = R['uv']; H, W = uv.shape[:2]; out = np.zeros((H, W), np.int8)
    S = sprites(t, own)
    if not S: return out
    draws = R['draws']; dz = np.unique(ids[ids > 0])
    size_of = {int(i): (draws[int(i)]['tw'], draws[int(i)]['tht']) for i in dz if int(i) in draws}
    for (file, tex), polys in S.items():
        tw, th = tex_size(file, tex)
        sel = [i for i, s in size_of.items() if s == (tw, th)]
        if not sel: continue
        byth = {}
        for i in sel: byth.setdefault(draws[i].get('th'), []).append(i)
        sel = [i for h, L in byth.items() if atlas_ok(R, h, L, file, tex) for i in L]
        if not sel: continue
        m = np.isin(ids, sel) & (uv[..., 0] >= 0)
        if not m.any(): continue
        rm = region_map(t, own, file, tex); Hh, Ww = rm.shape
        ys, xs = np.nonzero(m); u = np.clip((uv[ys, xs, 0] * Ww).astype(int), 0, Ww - 1); v = np.clip((uv[ys, xs, 1] * Hh).astype(int), 0, Hh - 1)
        k = rm[v, u]; hit = k > 0
        if not hit.any(): continue
        code = np.array([CODE[(p, c)] for _, p, c in polys], np.int8)
        cur = out[ys[hit], xs[hit]]; new = code[k[hit] - 1]; pr = np.array([0, 4, 2, 3, 0, 1], np.int8)   # PRIO by code
        better = (cur == 0) | (pr[new] > pr[cur])
        out[ys[hit][better], xs[hit][better]] = new[better]
    # sprite edges: the half-resolution uv map leaves some pixels of a draw without a texcoord (-1) while its id and
    # coverage are there; such a pixel takes the code of a neighbour (8-connected) drawn by the same draw
    miss = (out == 0) & (ids > 0) & (uv[..., 0] < 0)
    if miss.any() and out.any():
        fill = np.zeros_like(out); P = np.pad(out, 1); I = np.pad(ids, 1)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if not (dy or dx): continue
                c = P[1 + dy:1 + dy + H, 1 + dx:1 + dx + W]; same = I[1 + dy:1 + dy + H, 1 + dx:1 + dx + W] == ids
                fill = np.where(miss & same & (c > 0) & (fill == 0), c, fill)
        out = np.where(fill > 0, fill, out)
    return out


# ---- labels: the drawn pixels where the ID pass identifies the projectile, the prediction elsewhere (meshes, unknown
# atlases), never over something drawn in front of the projectile
TRAIL_NEAR = 40    # px (half resolution) around a trail-body attack's predicted region
DEPTH_TOL = 0.05   # eye depth (1000 units) by which a draw must be nearer the camera than the projectile to hide it


def _dilate(m, r):
    from scipy.ndimage import binary_dilation
    return binary_dilation(m, iterations=int(r)) if m.any() else m


def _close(m, r):
    from scipy.ndimage import binary_closing
    return binary_closing(np.pad(m, r), iterations=r)[r:-r, r:-r] if m.any() else m


def _eye_depth(cam, P, wref):
    """the ID pass's depth of world points (render coords x, -y, 0.6 z): (w - wref) / 1000"""
    P = np.asarray(P, np.float64).reshape(-1, 3); w = cam[3, :3] @ P.T + cam[3, 3]
    return (w - wref) / 1000.0


OWN_FRAC = 0.5     # a draw is the projectile's own (a mesh) when at least this share of its pixels lies in the prediction


def own_draw_pixels(R, ids_h, pred_any, pred_body, region, sprite=None, owner=None, me=None, mesh_only=False):
    """pixels (half resolution) of the draws that are the projectile's own: draws with at least OWN_FRAC of their pixels
    inside the predicted labels (pred_any: of all projectiles, neighbours' predictions may be offset onto it) and covering
    part of the predicted body, most of all projectiles' bodies (owner: draw -> object, me: this object); their drawn pixels
    (coverage >= CUT) in the region, as connected parts touching the predicted body. None when no such draw (the projectile
    is hidden or additive)."""
    from scipy.ndimage import label as cc
    under = ids_h[pred_body & (ids_h > 0)]
    if not under.size: return None
    skip = set(np.unique(ids_h[sprite]).tolist()) if sprite is not None and sprite.any() else set()   # its sprite draws (trail)
    own = []
    for d, n in zip(*np.unique(under, return_counts=True)):
        tot = (ids_h == d).sum(); inside = ((ids_h == d) & pred_any).sum()
        if inside >= OWN_FRAC * tot and n >= 4 and int(d) not in skip and (owner is None or owner.get(int(d)) == me):
            if mesh_only and (R['uv'][..., 0][ids_h == d] >= 0).mean() >= 0.5: continue   # a sprite draw (atlas uv), not a mesh
            own.append(int(d))
    if not own: return None
    m = np.isin(ids_h, own) & region & (R['cv'][::2, ::2] >= CUT)
    L, n = cc(m)
    keep = np.unique(L[pred_body & m]); keep = keep[keep > 0]
    return np.isin(L, keep) if keep.size else None


def hull_mask(m):
    """the convex hull of a mask's pixels, as a mask"""
    from scipy.spatial import ConvexHull
    ys, xs = np.nonzero(m)
    if len(ys) < 3: return m
    pts = np.c_[xs, ys].astype(float)
    try: h = ConvexHull(pts)
    except Exception: return m
    I = Image.new('L', (m.shape[1], m.shape[0]), 0); ImageDraw.Draw(I).polygon([tuple(pts[k]) for k in h.vertices], fill=1, outline=1)
    return np.asarray(I, bool) | m


def zone(m):
    """one solid region: gaps closed, holes filled, small detached parts dropped"""
    from scipy.ndimage import binary_closing, binary_fill_holes, label as cc
    z = binary_fill_holes(binary_closing(np.pad(m, ZONE_CLOSE), iterations=ZONE_CLOSE)[ZONE_CLOSE:-ZONE_CLOSE, ZONE_CLOSE:-ZONE_CLOSE] | m)
    L, n = cc(z)
    if n > 1:
        sz = np.bincount(L.ravel())[1:]; z = np.isin(L, 1 + np.nonzero(sz >= ZONE_MIN * sz.max())[0])
    return z


ALIGN_R = 12       # px: largest shift of the prediction onto the drawn sprites


def align(pred_body, mine, C, region):
    """pred_body shifted (integer px, |d| <= ALIGN_R) to agree best with the drawn sprites: score = predicted pixels on
    its drawn body (mine) - predicted pixels on no sprite of its own. Unshifted unless a shift is clearly better (labels
    that already agree keep the prediction as is)"""
    if mine.sum() < 30 or not pred_body.any(): return pred_body
    ys, xs = np.nonzero(region); y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    P = pred_body[y0:y1, x0:x1]; M = mine[y0:y1, x0:x1]; N = (C[y0:y1, x0:x1] == 0)
    def shift(a, dy, dx):
        o = np.zeros_like(a); h, w = a.shape
        o[max(0, dy):h + min(0, dy), max(0, dx):w + min(0, dx)] = a[max(0, -dy):h - max(0, dy), max(0, -dx):w - max(0, dx)]
        return o
    def score(dy, dx):
        q = shift(P, dy, dx); return int((q & M).sum()) - int((q & N).sum())
    s0 = score(0, 0); best = (s0, 0, 0)
    for dy in range(-ALIGN_R, ALIGN_R + 1, 2):        # coarse, then refine around the best
        for dx in range(-ALIGN_R, ALIGN_R + 1, 2):
            sc = score(dy, dx)
            if sc > best[0]: best = (sc, dy, dx)
    for dy in range(best[1] - 1, best[1] + 2):
        for dx in range(best[2] - 1, best[2] + 2):
            sc = score(dy, dx)
            if sc > best[0]: best = (sc, dy, dx)
    if best[0] <= s0 + max(10, 0.1 * abs(s0)): return pred_body
    out = np.zeros_like(pred_body); out[y0:y1, x0:x1] = shift(P, best[1], best[2])
    return out


def _proj(cam, x, y, z=0.0):
    """screen position (half resolution, 1200 x 540) of logic point x, y, z (render coords x, -y, 0.6 z)"""
    q = cam @ np.array([x, -y, 0.6 * z, 1.0])
    return (q[0] / q[3] * 0.5 + 0.5) * 1200, (1 - (q[1] / q[3] * 0.5 + 0.5)) * 540


MODEL_TEX = '0' * 16   # characters are drawn as 3D models from a runtime texture (no upload hash): programs 100 / 104
CHAR_NEAR = 80         # px: a model draw belongs to the character nearest to it within this
CHAR_TOP = 450         # logic z of a character's top (the model spans ground .. top on screen)


def spawn_labels(state, cam, R, taken):
    """{character id: (H, W) bool} of the spawnables (characters.csv Type Minion*, owned by a player): the pixels of
    the 3D model draws nearest to them (each model draw to the nearest character of all on screen), at least half
    opaque, not already a projectile's (taken)"""
    import owners
    ids_h = R['id'][::2, ::2]; cv = R['cv'][::2, ::2]; out = {}
    chars = []
    for c in state['chars']:
        x, y, z = int(c[3]), int(c[4]), int(c[5])
        a = np.array(_proj(cam, x, y, z)); b = np.array(_proj(cam, x, y, z + CHAR_TOP)); chars.append((int(c[0]), int(c[1]), int(c[2]), a, b))
    if not any(owners.character(did)['kind'] == 'spawnable' and own >= 0 for _, did, own, _, _ in chars): return out
    models = [d for d, dr in R['draws'].items() if dr.get('th') == MODEL_TEX]
    for d in models:
        m = ids_h == d
        if not m.any(): continue
        ys, xs = np.nonzero(m); p = np.array([xs.mean(), ys.mean()]); best = (CHAR_NEAR, None)
        for cid, did, own, a, b in chars:
            ab = b - a; t = np.clip(np.dot(p - a, ab) / max(1e-6, np.dot(ab, ab)), 0, 1); dist = np.linalg.norm(a + t * ab - p)
            if dist < best[0]: best = (dist, (cid, did, own))
        if best[1] is None: continue
        cid, did, own = best[1]
        if owners.character(did)['kind'] != 'spawnable' or own < 0: continue
        px = m & (cv >= CUT) & ~taken
        if px.any(): out[cid] = out.get(cid, np.zeros_like(m)) | px
    for cid in out: out[cid] |= small_holes(out[cid], SPECK) & ~taken
    return out


ZONE_TOP = 600   # logic z up to which a zone's effects are searched
# zones drawn additively (invisible to the ID pass: only stray opaque bits of them are found): labelled as their logic
# disc on the ground, minus whatever is drawn in front of the ground there (characters, bushes, UI). Penny's burn
# (BurnUlti: blue flames, the user's 'bad' mark on Penny's super page, 2026-10-20)
ZONE_DISC = {'BurnUlti', 'penny007BurnUlti', 'Penny008BurnUlti', 'Penny009BurnUlti',
             # Chronomancer's time-bend ring: two big translucent disc sprites, one raised and faint (the label reached far
             # above the ring); the logic disc is the ring exactly (2026-10-20)
             'ChronomancerUltiTimebend', 'ChronomancerOverchargedUltiTimebend', 'Chronomancer002UltiTimebend',
             'Chronomancer003UltiTimebend'}


# zones that open with an additive dome: the logic disc (as ZONE_DISC) while the zone is at most this many ticks old,
# plus its drawn sprites (the puddle) all along. Voodoo's attack areas (the user, 2026-10-20: 'dome disc, then
# puddle'; the dome is visible at ages 2-8 ticks, gone by 10)
ZONE_DOME = {n: 8 for n in ('VoodooAttackEarth', 'VoodooAttackForest', 'VoodooAttackWater', 'VoodooAttackAll',
                            'Voodoo002AttackEarth', 'Voodoo002AttackForest', 'Voodoo002AttackWater',
                            'Voodoo003AttackEarth', 'Voodoo003AttackForest', 'Voodoo003AttackWater')}
_ZONE_SEEN = {}   # item id -> first tick seen (frames come in order; cleared when the tick goes back: a new capture)


def ground_depth(cam, x, y, rad, wref, H=540, W=1200):
    """(H, W) eye depth (as R['wd']) of the ground plane z = 0 at every pixel (1 / w is affine in screen space on a plane),
    fitted on points around logic x, y"""
    P = [(x - rad, y - rad), (x + rad, y - rad), (x, y + rad), (x + rad, y + rad)]
    S = np.array([_proj(cam, a, b) for a, b in P]); w = np.array([cam[3] @ np.array([a, -b, 0, 1.0]) for a, b in P])
    c = np.linalg.lstsq(np.c_[S, np.ones(len(S))], 1.0 / w, rcond=None)[0]
    yy, xx = np.mgrid[0:H, 0:W]
    return (1.0 / (c[0] * xx + c[1] * yy + c[2]) - wref) / 1000.0
LABEL_HITS = False   # hit / explosion animations of projectiles: not labelled (the user, 2026-10-19)


def zone_labels(state, cam, R, taken, own_team=None):
    """{item id: (H, W) bool} of the lingering damage zones (state items, area effects): the pixels where one of the
    zone's sprites is the front-most draw (any coverage: zones are translucent), within its radius, small gaps closed"""
    import owners
    ids_h = R['id'][::2, ::2]; H, W = ids_h.shape; out = {}
    yy, xx = np.mgrid[0:H, 0:W]
    tick = int(state.get('tick', 0))
    if _ZONE_SEEN and tick < _ZONE_SEEN.get('_last', 0): _ZONE_SEEN.clear()
    _ZONE_SEEN['_last'] = tick
    for it in state.get('items', []):
        did = int(it[1]); Z = owners.zone(did)
        age = tick - _ZONE_SEEN.setdefault(int(it[0]), tick)
        if not Z or owners.area(did)['kind'] != 'zone': continue   # only lingering zones; short areas are explosions (not labelled)
        x, y = int(it[3]), int(it[4]); rad = float(Z['logic_row'].get('Radius') or 300) * 1.15 + 150
        cx, cy = _proj(cam, x, y); rx = abs(_proj(cam, x + rad, y)[0] - cx); ry = abs(_proj(cam, x, y + rad)[1] - cy)
        if rx < 1 or ry < 1: continue
        region = np.zeros((H, W), bool)
        for zz in range(0, ZONE_TOP + 1, 150):   # the zone's circle from the ground up (splashes and clouds rise)
            hx, hy = _proj(cam, x, y, zz); region |= ((xx - hx) / rx) ** 2 + ((yy - hy) / ry) ** 2 <= 1
        if not region.any(): continue
        disc_m = None
        if (Z.get('Name') in ZONE_DISC or age <= ZONE_DOME.get(Z.get('Name'), -1)) and R.get('wd') is not None:
            rz = float(Z['logic_row'].get('Radius') or 300)
            dx = abs(_proj(cam, x + rz, y)[0] - cx); dy = abs(_proj(cam, x, y + rz)[1] - cy)
            disc = ((xx - cx) / dx) ** 2 + ((yy - cy) / dy) ** 2 <= 1
            disc_m = disc & (R['wd'] >= ground_depth(cam, x, y, rz, R['meta'].get('wref', 0.0)) - DEPTH_TOL) & ~taken
            if Z.get('Name') in ZONE_DISC:
                if disc_m.any(): out[int(it[0])] = (disc_m, did, int(it[2]))
                continue
        own = True if own_team is None else own_team(int(it[2]))
        C = sprite_pixels(('zone', did), own, R)
        m = region & (C > 0) & ~taken
        if m.any(): m = _close(m, 2) & region & ~taken
        if disc_m is not None: m = m | disc_m
        if m.any(): out[int(it[0])] = (m, did, int(it[2]))
    return out


def meta(objs, state, P):
    """{label id: dict(kind attack | super | spawnable | zone | gadget | starpower, brawler, skin, name)} of a labels()
    result with state (the user's classes, 2026-10-19; owners.py)"""
    import owners, bulletlabels as BL
    did_of = {int(c[0]): int(c[1]) for c in state['chars']}
    hero_of = {int(c[2]): int(c[1]) for c in state['chars'] if owners.character(int(c[1]))['kind'] == 'hero'}
    out = {}
    for o in objs:
        r = owners.runtime(o['t'], did_of.get(o.get('src')), hero_of.get(o.get('owner')))
        out[BL._lid(o)] = r; out.setdefault(o['id'], r)
    for lid in P['lids']:
        if lid[2] == 'spawn':
            did = did_of.get(lid[0]); c = owners.character(did) if did else {}
            out[lid[0]] = dict(kind='spawnable', brawler=owners.spawnable_owner(c.get('name')), skin=None, name=c.get('name'))
    for pid, e in P.get('hits', {}).items():
        out.setdefault(pid, owners.runtime(e['t'], did_of.get(e['src']), hero_of.get(e['owner'])))
    for k, (did, own) in P.get('zones', {}).items():   # area effects: lingering zones, or explosions (the spawner's kind)
        a = owners.area(did)
        out[k] = dict(kind=a['kind'], brawler=a['brawler'] or (owners.character(hero_of[own])['name'] if own in hero_of else None),
                      skin=None, name=a['name'])
    return out


def labels(objs, cam, R, state=None, ended=(), img=None):
    """like bulletlabels.labels (same keys) with the body / ground part from the client's draws (see the module doc).
    R: idbuf.load of the frame; img: the frame (540, 1200, 3), for DARK_DROP. Glow map: the prediction, unchanged. With the frame's state, also the spawnables
    (label (character id, -1, 'spawn')) and the area effects ((item id, -1, 'area'): lingering zones, or explosions
    with their spawner's kind), and 'meta': per label
    id its kind (attack / super / spawnable / zone / gadget / starpower), brawler and skin."""
    import bulletlabels as BL
    from scipy.ndimage import binary_fill_holes
    # wave / trail-body attacks whose logic projectile has ended (Tracker.gone: kept while their trail is on the ground):
    # not labelled, their leftover particles are an end animation (the user, 2026-10-20: 'unlabel after the end')
    # A wave group's label (its region) lives on its first projectile: that object is dropped only once no member of the
    # group is alive any more (else the whole group lost its label when its first projectile ended: agent 6, Buzz SB)
    alive = {o.get('label_id', o['id']) for o in objs if not o.get('ended')}
    objs = [o for o in objs if not (o.get('ended') and trail_body(o['t']) and o.get('label_id', o['id']) not in alive)]
    P = BL.labels(objs, cam, cut=0.01)                       # the prediction (owner maps at a low cut: regions)
    share, ids, lids = P['share'], P['ids'], P['lids']; H, W = ids.shape
    cv = R['cv'][::2, ::2]; wd = R.get('wd'); wref = R['meta'].get('wref', 0.0)
    codes = {}
    out_share = np.zeros((H, W), np.float32); out_ids = np.full((H, W), -1, np.int32); claim = np.full((H, W), np.inf, np.float32)
    byid = {}
    for k, lid in enumerate(lids): byid.setdefault(lid[0], []).append(k)
    ids_h = R['id'][::2, ::2]; pred = []; owner = {}; pairs = []
    for j, o in enumerate(objs):
        ks = byid.get(BL._lid(o), []) + ([k for k in byid.get(o['id'], [])] if BL._lid(o) != o['id'] else [])
        body_k = [k for k in ks if lids[k][2] == 'body']
        pred_any = np.isin(ids, ks) if ks else np.zeros((H, W), bool)
        pred_body = np.isin(ids, body_k) & (share >= BL.CUT) if body_k else np.zeros((H, W), bool)
        pred.append((ks, pred_any, pred_body))
        pairs += [(int(n), j, int(d)) for d, n in zip(*np.unique(ids_h[pred_body & (ids_h > 0)], return_counts=True))]
    # draws to projectiles by overlap with the predicted bodies (predictions can be offset onto a neighbour): one draw per
    # projectile first, largest overlaps first; the remaining draws to the projectile whose body they cover most
    all_pred = ids >= 0; cand = {}
    for n, j, d in pairs:
        if d not in cand: cand[d] = ((ids_h == d) & all_pred).sum() >= OWN_FRAC * (ids_h == d).sum()
    pairs = [p for p in pairs if cand[p[2]]]
    pairs.sort(reverse=True); has = set()
    for n, j, d in pairs:
        if d not in owner and j not in has: owner[d] = j; has.add(j)
    for n, j, d in pairs:
        if d not in owner: owner[d] = j
    obj_draws = {}
    for j, o in enumerate(objs):
        t, own = o['t'], o['own']; key = (t, own)
        if key not in codes: codes[key] = sprite_pixels(t, own, R)
        C = codes[key]; ks, pred_any, pred_body = pred[j]
        if (C == 5).any():
            # sprites shared by its clip and its trail: body in its clip draws (the draws that show its unshared clip
            # sprites, else those covering most of its predicted body) near its aligned prediction, trail elsewhere
            reg0 = _dilate(pred_any, NEAR); C = C.copy(); c5 = (C == 5) & reg0
            clipd = np.unique(ids_h[reg0 & (C == 1)])
            if not clipd.size and pred_body.any():
                d5, n5 = np.unique(ids_h[c5 & pred_body], return_counts=True); clipd = d5[n5 >= max(4, 0.25 * n5.max())] if d5.size else d5
            pb5 = _dilate(pred_body, SHARED_NEAR)   # not aligned: shared pixels include neighbours and trail copies
            body5 = c5 & np.isin(ids_h, clipd) & pb5
            C[body5] = 1; C[(C == 5)] = 4
        ground_k = [k for k in ks if lids[k][2] == 'ground']
        region = _dilate(pred_any, TRAIL_NEAR if trail_body(t) else NEAR)
        obj_draws[int(o['id'])] = sorted(int(d) for d in np.unique(ids_h[(C > 0) & region]) if d > 0)   # its draws (compositing)
        unf = t in glow.UNFADE
        mine = region & ((C == 1) | (C == 3)) & ((cv >= FADE_CUT.get(t, CUT)) | (unf & (cv > 0)))
        # a 3D body too (AxeJuggler005's ghost: its clip is only the tail glow): its own model draw is drawn body
        if has_mesh(t, own) and (trail_body(t) or (C[region] == 1).any() or (C[region] == 2).any()):
            ob = own_draw_pixels(R, ids_h, all_pred, pred_body, region, C > 0, owner, j, mesh_only=True)
            if ob is not None: mine = mine | ob
        # the prediction aligned to the drawn body (an offset prediction, Voodoo's super, would take glow beside the body)
        pb = align(pred_body, mine, C, region)
        # the projectile's own glow layer drawn over its body (a highlight: Chola's stars, Lola's rings): the body under it
        # is not visible to the ID pass; body where the drawn body encloses it or the aligned prediction says body
        if mine.any(): mine |= binary_fill_holes(mine) & region & (C == 2)
        mine |= region & (C == 2) & pb
        # its own trail drawn over its body (Gale's super puffs over the gust wall): the body under it, from the prediction
        # (not for trail-body types: their 'over' sprites are the ground effects of TRAIL_FX / TRAIL_SKIP, never body)
        if not trail_body(t): mine |= region & (C == 4) & pb
        # visibility of predicted pixels: nothing nearer the camera in front of the projectile
        if wd is not None and o.get('anchor') is not None:
            dz = _eye_depth(cam, o['anchor'], wref)[0]; visible = wd >= dz - DEPTH_TOL
        else: visible = np.ones((H, W), bool)
        cr = C[region]
        if trail_body(t) or (cr == 1).any() or (cr == 2).any():   # its clip drawn / trail attacks: drawn pixels only
            body = mine
        elif has_clip(t, own) and (C[pred_body] == 4).any() and mine.any():   # its clip drawn with sprites its trail shares
            # plus its own non-sprite draw if it has one (Meeple's die: a 3D mesh over a shadow sprite its trail shares)
            ob = own_draw_pixels(R, ids_h, all_pred, pred_body, region, C > 0, owner, j, mesh_only=True)
            body = mine | (ob if ob is not None else False)
        else:   # mesh / unknown atlas: the projectile's own draw if the ID pass shows one, else the prediction, unhidden;
                # plus its own trail drawn over its predicted body (mine)
            ob = own_draw_pixels(R, ids_h, all_pred, pred_body, region, C > 0, owner, j)
            body = (ob if ob is not None else pred_body & visible) | mine
        if body.any() and t in EXTEND_BODY:
            # its clip body drawn beyond the predicted region (Beamer's super: one long stretched beam sprite, the prediction
            # shorter): the connected opaque clip-body pixels of the same draws, wherever they reach
            own1 = np.unique(ids_h[body & (C == 1)]); own1 = own1[own1 > 0]
            if own1.size:
                from scipy.ndimage import label as cc
                cand = np.isin(ids_h, own1) & (C == 1) & (cv >= FADE_CUT.get(t, CUT)); L, n = cc(cand | body)
                keep = np.unique(L[body]); body = body | (np.isin(L, keep[keep > 0]) & cand)
        if body.any() and not trail_body(t):
            # stacked translucent layers of its own: the ID pass's coverage is the top layer's alpha only. Holes in the body
            # where its own sprite is on top are body; edges untouched
            hole = binary_fill_holes(body) & ~body
            body |= hole & (C > 0)
            # enclosed glints / sparkles of its effects drawn over the body (blended sprites from its atlases, shared
            # generic shapes: Voodoo's earth attack); ground showing through a hole (Lola's ring interiors) stays out
            es = (effect_sprite(R, ids_h, t, own) & (C == 0)) | (C == 1)
            if hole.any(): body |= hole & (es | (C > 0))
            # the same in narrow gaps open to the outside (<= 2 px channels between the body's parts), never its glow
            body |= _close(body, 2) & es & region
            # tiny enclosed holes (<= SPECK px) under any blended sprite (sparkles over it), never ground showing through
            body |= small_holes(body, SPECK) & blended_sprite(R, ids_h)
            # sprite edges whose uv falls just outside its shapes (filtering) or is missing: opaque pixels of the body's
            # own draws touching it, at most EDGE_GROW px
            own_d = np.unique(ids_h[body & (C > 0)]); own_d = own_d[own_d > 0]
            if own_d.size:
                ok = np.isin(ids_h, own_d) & (cv >= CUT) & region & ((C == 0) | (C == 1))   # never its glow / trail
                for _ in range(EDGE_GROW): body |= _dilate(body, 1) & ok
        elif body.any() and t not in ZONE:
            # trail-body types: holes in the body where one of its own sprites is on top (layers under half opaque at
            # the top: Frank's crystals), as for the other types; ground showing through stays out
            hole = binary_fill_holes(body) & ~body; body |= hole & (C > 0) & (C != 4)
            body |= small_holes(body, SPECK) & blended_sprite(R, ids_h)
        if (t in BL.FILL_HOLES or t in SOLID or t in THIN_OFF) and body.any(): body = binary_fill_holes(body) & region
        if (t in SOLID or t in THIN_OFF) and body.any():   # thin parts sticking out (sprinkles touching the donut): opened away; slivers
            from scipy.ndimage import binary_opening, label as cc   # the opening cuts off the edge itself stay
            op = binary_opening(body, iterations=THIN_OFF.get(t, SOLID_OPEN)); L, n = cc(body & ~op); sz = np.bincount(L.ravel())
            body = op | np.isin(L, np.nonzero((sz < SOLID_SLIVER) & (np.arange(len(sz)) > 0))[0])
        if t in ZONE and body.any(): body = zone(body) & region
        if t in BODY_ONLY and body.any():
            from scipy.ndimage import label as cc
            L, n = cc(body); sz = np.bincount(L.ravel())[1:]
            keep = [k + 1 for k in range(n) if sz[k] >= BODY_ONLY_FRAC * sz.max()]
            body = np.isin(L, keep)
        if t in TRAIL_ATTACH and body.any():
            from scipy.ndimage import label as cc
            att = region & ((C == 3) | (C == 4)) & (cv >= CUT); L, n = cc(att | body)
            keep = np.unique(L[body]); body = body | (np.isin(L, keep[keep > 0]) & att)
            body |= binary_fill_holes(body) & region   # one blob: enclosed gaps between the splashes too
        if t in BRIGHT_BODY and img is not None:
            from scipy.ndimage import label as cc
            bb = region & (img.max(-1) >= BRIGHT_BODY[t]) & (_dilate(pred_any, 4) | _dilate(C > 0, 4))
            bb = _close(bb, 2); L, n = cc(bb)
            if n:
                sz = np.bincount(L.ravel())[1:]; bb = np.isin(L, 1 + np.nonzero(sz >= 0.15 * sz.max())[0])
                body = body | binary_fill_holes(bb)
        if t in STREAK and img is not None:
            # its sprites' footprint widened (the streak's additive centre line has no sprite code of its own)
            rs = _dilate(pred_any, TRAIL_NEAR)   # the streak's tail reaches past the projectile's region
            im = img.astype(np.int16); st = rs & _dilate((C == 1) | (C == 2), 6) & (im[..., 2] - im[..., 1] >= STREAK[t])
            if st.any(): body = body | (binary_fill_holes(_close(st, 2)) & rs)
        if t in BRIGHT_RIM and img is not None and body.any():
            bright = region & (img.min(-1) >= BRIGHT_RIM[t])
            for _ in range(RIM_R): body = body | (_dilate(body, 1) & bright)
        if t in DARK_DROP and img is not None and body.any():
            v = img.max(-1)
            for d in np.unique(ids_h[body]):
                m = body & (ids_h == d)
                if np.median(v[m]) < DARK_V: body &= ~m
        if t in POST_BODY:   # per-type hooks (fix agents' overrides): fn(body, ctx) -> body
            body = POST_BODY[t](body, dict(o=o, t=t, own=own, C=C, region=region, cv=cv, img=img, ids_h=ids_h, R=R, cam=cam,
                                         pred_any=pred_any, pred_body=pred_body, mine=mine, state=state))
        if ENDED_DROP and ended and t not in ENDED_KEEP and t not in ZONE and not trail_body(t):
            body = drop_ended_parts(body, o, cam, ended, ids_h, pred_body, pred_any)
        if t in ONE_LABEL and body.any():
            r1 = ONE_LABEL[t]
            if r1 == 'hull': body = hull_mask(body)
            else: body = binary_fill_holes(_close(body, r1)) & _dilate(region, r1)
        lid_body = (BL._lid(o), -1, 'body')
        if lid_body not in lids: lids = lids + [lid_body]
        kb = lids.index(lid_body)
        # one owner per pixel: the object whose predicted label is nearest (ties: the first)
        d = np.where(pred_any, 0.0, 1.0).astype(np.float32)
        take = body & (d < claim); out_ids[take] = kb; out_share[take] = np.maximum(cv[take] / 255.0, BL.CUT); claim[take] = d[take]
        if ground_k:
            g = np.isin(ids, ground_k) & (share >= BL.CUT) & visible & (out_ids < 0)
            out_ids[g] = ground_k[0]; out_share[g] = share[g]
    P2 = dict(P); P2.update(share=out_share, ids=out_ids, lids=lids)
    P2['draws'] = obj_draws   # object id -> the ID-pass draw ids of its own sprites near it (GPU compositing; labels unaffected)
    if state is not None:
        taken = out_ids >= 0
        for cid, m in spawn_labels(state, cam, R, taken).items():
            lid = (cid, -1, 'spawn'); lids = lids + [lid]; k = len(lids) - 1
            out_ids[m] = k; out_share[m] = np.maximum(cv[m] / 255.0, BL.CUT); taken |= m
        zones = {}
        for iid, (m, did, own) in zone_labels(state, cam, R, taken).items():
            lid = (iid, -1, 'area'); lids = lids + [lid]; k = len(lids) - 1
            out_ids[m] = k; out_share[m] = BL.CUT; taken |= m; zones[iid] = (did, own)
        hits = {}
        # explosion / hit animations are not labelled (the user, 2026-10-19: "the labels also mark the explode animation
        # as a bullet, this isnt right"); end_labels stays for reference, LABEL_HITS = False
        for pid, m in (end_labels(list(ended), cam, R, taken) if LABEL_HITS else {}).items():
            lid = (pid, -1, 'hit'); lids = lids + [lid]; k = len(lids) - 1
            out_ids[m] = k; out_share[m] = BL.CUT; taken |= m; hits[pid] = [e for e in ended if e['id'] == pid][0]
        P2.update(ids=out_ids, share=out_share, lids=lids, zones=zones, hits=hits)
        P2['meta'] = meta(objs, state, P2)
    return P2


# per-type fixes, executed in this module's namespace after load (they extend the per-type lists ONE_LABEL, BRIGHT_BODY,
# TRAIL_SKIP, POST_BODY ... and may wrap labels / zone_labels / zone_sprites): py/fixes/*.py in name order. a0-a7: the
# parallel fix agents' overrides (2026-10-20, out/sweep2/fixagents/agent_K_report.md); z_*: later decisions.
# IDLABELS_NOFIXES=1 skips them, IDLABELS_SKIP=a.py,b.py skips some; IDLABELS_OVERRIDE=<files joined by os.pathsep> adds more after them (experiments)
import glob as _glob
if not os.environ.get('IDLABELS_NOFIXES'):
    _skip = set(filter(None, os.environ.get('IDLABELS_SKIP', '').split(',')))   # fix files to leave out (testing a replacement)
    for _p in sorted(_glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fixes', '*.py'))):
        if os.path.basename(_p) in _skip: continue
        exec(compile(open(_p).read(), _p, 'exec'), globals())
if os.environ.get('IDLABELS_OVERRIDE'):
    for _p in os.environ['IDLABELS_OVERRIDE'].split(os.pathsep):
        if _p and os.path.exists(_p): exec(compile(open(_p).read(), _p, 'exec'), globals())
