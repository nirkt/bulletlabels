# label-fix agent 3 overrides (2026-10-20): Controller, RocketGirl, Undertaker, Whirlwind, Enrager, BeeSniper, Skater, Mummy.
# exec'd in idlabels' namespace (IDLABELS_OVERRIDE); every rule keyed to explicit type ids.
import numpy as _np


A3_MESH_PROG = {116, 126, 129}   # the client's programs of projectile glb meshes (Brock 003 / 010: 116, Brock 012: 126,
# Mortis 008: 129, measured);
# terrain 62-82, characters 100 / 104


def _a3_mesh_draws(ctx):
    """draws under the projectile's predicted body that are a projectile 3D mesh (no atlas uv, program A3_MESH_PROG)"""
    R, ids, pb = ctx['R'], ctx['ids_h'], ctx['pred_body']
    uvneg = R['uv'][..., 0] < 0; out = []
    for d in _np.unique(ids[pb & (ids > 0)]):
        r = R['draws'].get(int(d), {})
        if r.get('th') == MODEL_TEX or r.get('prog') not in A3_MESH_PROG: continue
        m = ids == d
        if uvneg[m].mean() >= 0.5: out.append(int(d))
    return out


# RocketGirl (Brock) 003 Hotrod: the rocket is a 3D mesh drawn as ONE instanced draw for all rockets in flight (the
# super's 9 rockets): the draw went to one rocket only (idlabels' one-owner-per-draw), the others kept only their tail
# star. Each rocket takes the mesh pixels connected to its predicted body. Its clip's smoke puffs (sprites shared with
# its rocketSmoke trail, code 5, or its trail drawn over it, code 4: draws showing (almost) no clip-only sprite, < 10 %; the tail star's draws show both) are smoke, not the
# rocket (rule 3): dropped.
_A3_BROCK_MESH = {6000057, 6000058, 6001983,
                  # RocketGirlRanger (Brock 010), every colour: the same instanced missile mesh and smoke (page 186)
                  6000071, 6000072, 6000073, 6000074, 6000075, 6000076, 6000077, 6000078, 6000079, 6000080,
                  6001990, 6001991, 6001992, 6001993, 6001994,
                  # RocketGirlRankedProgrsesionSkinLevel3 (Brock 012): its super's missile mesh; its smoke is trail-over
                  # (code 4) sprites taken under the prediction (page 214)
                  6000084, 6000086}


def _a3_brock_mesh(body, ctx):
    from scipy.ndimage import label as cc
    region, pa, pb, cv = ctx['region'], ctx['pred_any'], ctx['pred_body'], ctx['cv']
    ids = ctx['ids_h']; C0 = sprite_pixels(ctx['t'], ctx['own'], ctx['R'])
    for d in _np.unique(ids[body & ((C0 == 5) | (C0 == 4))]):   # draws of shared / trail sprites, (almost) no clip-only one: smoke
        m = ids == d
        if ((C0 == 1) & m).sum() < 0.1 * ((C0 > 0) & m).sum(): body = body & ~m
    md = _a3_mesh_draws(ctx)
    if md:
        m = _np.isin(ctx['ids_h'], md) & (cv >= CUT) & region & _dilate(pa, 2)
        L, n = cc(m); keep = _np.unique(L[pb & m]); keep = keep[keep > 0]
        if keep.size: body = body | _np.isin(L, keep)
    if body.any(): body = body | (small_holes(body, SPECK) & region)   # specks of its smoke drawn over the rocket
    return body


for _t in _A3_BROCK_MESH: POST_BODY[_t] = _a3_brock_mesh


# red1028 / s184 (2026-10-28): RocketGirlRanger (Brock 010) ATTACK, enemy (red clip) shots. Replaces a3_agent3.py (same file
# otherwise: only the block below is new; the own / blue path is untouched).
# Problem: the enemy rocket's label took the big white muzzle flash (ticks 265 / 305), the grey and maroon smoke puffs behind
# the rocket with ragged spurs (267, 315 ...) and separate fragments (269). The red clip is the same clip as the blue one
# (same shapes 63 / 565 / 567, same layers; only the add-colour 255,102,102 instead of 51,154,247) but its flame saturates to
# white and its smoke puffs are light grey / maroon, so the ID pass coverage of those sprites reaches much further than the
# predicted body (flame + rocket, the clip simulation). a3's smoke filter cannot cut them: the flash / puffs share draws with
# the flame's own sprites (shared shape 565, code 5 + code 1 in one draw), so no draw is dropped. Nothing is colour keyed here:
# the own (blue) label of this type lies inside the predicted body dilated by 2 px on every tick of the kept capture (0 px
# outside, 14 ticks checked), the enemy label should too. For the enemy shot the body is cut to the predicted body dilated
# by A3_ENEMY_PB_GROW, the connected parts touching the predicted body are kept, and holes are filled.
# Own: ctx['own'] True is returned unchanged (pixel-identical, see compare_own_check.txt).
A3_ENEMY_ATTACK = {6000071, 6000072, 6000073, 6000074, 6000075}   # Brock010Projectile{Red,Blue,Yellow,Pink,Black}
A3_ENEMY_PB_GROW = 2
A3_ENEMY_CRUMB = 0.1


def _a3_brock010_attack(body, ctx):
    from scipy.ndimage import label as cc, binary_fill_holes
    b = _a3_brock_mesh(body, ctx)
    if ctx['own'] or not b.any(): return b
    pb = ctx['pred_body']
    if not pb.any(): return b
    keep = b & _dilate(pb, A3_ENEMY_PB_GROW)
    if not keep.any(): return b          # never lose a projectile
    L, n = cc(keep)
    hit = _np.unique(L[pb & keep]); hit = hit[hit > 0]
    if hit.size: keep = _np.isin(L, hit)
    L, n = cc(keep)
    if n > 1:   # crumbs of the flame / smoke left inside the predicted body: parts under A3_ENEMY_CRUMB of the main part
        sz = _np.bincount(L.ravel())[1:]; keep = _np.isin(L, 1 + _np.nonzero(sz >= A3_ENEMY_CRUMB * sz.max())[0])
    return keep | (binary_fill_holes(keep) & ctx['region'] & b)


for _t in A3_ENEMY_ATTACK: POST_BODY[_t] = _a3_brock010_attack


# RocketGirl (Brock), every skin: its star power burn area after the attack (BurnMainAttack, effect brock_def_starpower):
# a translucent blue disc with additive flames and a thin opaque rim; the ID pass found only the rim (a crescent label).
# Additive lingering zone: its logic ground disc minus what is in front (rule 10, ZONE_DISC)
# The drawn rim lies outside the logic radius: its outer edge at 1.13-1.16 logic radii (measured on page 155, ticks 281 /
# 301 / 321, the rim's ID-pass pixels and the frame's cyan rim): the disc is drawn at A3_DISC_SCALE x the logic radius.
# RocketGirlRanger (Brock 010), its colour variants: the same burn ring (brock_010_dot_*), rim outer edge at 1.14-1.16
# logic radii (page 186 ticks 281 / 301); its ID-pass label was a much larger faint sprite circle around the ring
_A3_BURN = ('BurnMainAttack', 'BurnMainAttackRangerRed', 'BurnMainAttackRangerBlue', 'BurnMainAttackRangerYellow',
            'BurnMainAttackRangerPink', 'BurnMainAttackRangerBlack',
            'BurnMainAttackPropass')   # RocketGirlRankedProgrsesionSkinLevel3 (Brock 012)
ZONE_DISC.update(_A3_BURN)
A3_DISC_SCALE = {n: 1.15 for n in _A3_BURN}
# Brock 012's ring is wider: its pink outer edge at 1.22-1.29 logic radii (colour profiles up / left / up-left, page 214
# t289 / t301; it pulses)
A3_DISC_SCALE['BurnMainAttackPropass'] = 1.27
A3_FRONT_FRAC = 0.3
A3_FRONT_DRAW = {n for n in _A3_BURN if 'Ranger' in n}
A3_OWN_IN = {'BurnMainAttackPropass'}
_a3_zone_labels0 = zone_labels


def zone_labels(state, cam, R, taken, own_team=None):
    import owners
    out = _a3_zone_labels0(state, cam, R, taken, own_team)
    if R.get('wd') is None: return out
    H, W = R['id'][::2, ::2].shape; yy, xx = _np.mgrid[0:H, 0:W]
    for it in state.get('items', []):
        did = int(it[1]); Z = owners.zone(did)
        if not Z or Z.get('Name') not in A3_DISC_SCALE or owners.area(did)['kind'] != 'zone': continue
        x, y = int(it[3]), int(it[4]); rz = float(Z['logic_row'].get('Radius') or 300) * A3_DISC_SCALE[Z['Name']]
        cx, cy = _proj(cam, x, y); dx = abs(_proj(cam, x + rz, y)[0] - cx); dy = abs(_proj(cam, x, y + rz)[1] - cy)
        if dx < 1 or dy < 1: continue
        disc = ((xx - cx) / dx) ** 2 + ((yy - cy) / dy) ** 2 <= 1
        m = disc & (R['wd'] >= ground_depth(cam, x, y, rz, R['meta'].get('wref', 0.0)) - DEPTH_TOL) & ~taken
        # a draw standing up in front of the disc (the Ranger burn's robot figure) is out whole, its feet too: draws at
        # least A3_FRONT_FRAC in front within the disc (A3_FRONT_DRAW zones only: Brock 012's beam across its disc stays in)
        ids = R['id'][::2, ::2]; fr = disc & ~m
        for d in (_np.unique(ids[fr & (ids > 0)]) if Z['Name'] in A3_FRONT_DRAW else ()):
            dm = disc & (ids == d)
            if (dm & fr).sum() >= A3_FRONT_FRAC * dm.sum(): m &= ~dm
        if Z['Name'] in A3_OWN_IN:   # its own effect sprites drawn over the disc (Brock 012's beam across it) are the zone:
            # blended draws from the atlas of its identified sprites (the beam's core is not in its sprite set)
            own = True if own_team is None else own_team(int(it[2]))
            zc = disc & (sprite_pixels(('zone', did), own, R) > 0)
            ths = {R['draws'][int(d)].get('th') for d in _np.unique(ids[zc & (ids > 0)])}
            dd = [d for d in _np.unique(ids[disc & (ids > 0)]) if R['draws'].get(int(d), {}).get('th') in ths
                  and R['draws'][int(d)].get('blend') == 1]
            m |= disc & (zc | _np.isin(ids, dd)) & ~taken
        if m.any(): out[int(it[0])] = (m, did, int(it[2]))
        else: out.pop(int(it[0]), None)
    return out


# Undertaker (Mortis) 008 Demon: its super's ghost is a 3D mesh (prog 129) with a trail; idlabels' mesh search
# (own_draw_pixels, mesh_only) took the river's water draws (prog 82, no atlas uv) lying inside the long trail region as
# its mesh (the magenta label over the water, page 195 t367). Non-sprite draws that are not a projectile mesh program dropped.
def _a3_mesh_only_proj(body, ctx):
    R, ids = ctx['R'], ctx['ids_h']; uvneg = R['uv'][..., 0] < 0
    for d in _np.unique(ids[body & (ids > 0)]):
        m = ids == d; r = R['draws'].get(int(d), {})
        if uvneg[m].mean() >= 0.5 and r.get('prog') not in A3_MESH_PROG: body = body & ~m
    return body


for _t in (6000304, 6000305, 6000306): POST_BODY[_t] = _a3_mesh_only_proj


# Undertaker (Mortis) 003 Bazaar: its super's tornado clip has translucent glow layers (a soft yellow halo around the
# swirl and a streak back to Mortis); idlabels took them as body (glow over / enclosed by the body). Non-body-code pixels
# kept only where the frame is the tornado's yellow / white (min(R, G) >= A3_M003_V: its swirl, 180-250 measured); the halo
# over the ground (65-95), over the crate (orange, G ~120) and over water (R ~30) dropped (page 482 t349 / t361)
A3_M003_V = 150


def _a3_mortis003(body, ctx):
    from scipy.ndimage import binary_fill_holes, label as cc
    if ctx['img'] is None: return body
    body = body & ~((ctx['C'] != 1) & (ctx['img'][..., :2].min(-1) < A3_M003_V))
    if body.any():
        L, n = cc(body); sz = _np.bincount(L.ravel())[1:]   # one tornado: crumbs cut off the halo dropped
        body = _np.isin(L, 1 + _np.nonzero(sz >= 0.05 * sz.max())[0])
        body = body | (binary_fill_holes(body) & ctx['region'])
    return body


POST_BODY[6000300] = _a3_mortis003


# Undertaker (Mortis) 005 Mecha: its super is a blue energy sphere (one trail sprite, a radial gradient: coverage 150 at
# the core falling to 0); only the core (coverage >= 128, a third of its radius) was labelled. The sphere's visible edge
# (frame green channel dropping from 190 to 100) lies at coverage 70-77 (page 526 t349, radial profile)
FADE_CUT[6000301] = 72
# its trail's other particles are effects, not the sphere / its electric bats (rule 3): Carl's line particles and the
# yellow electric sparks; and its clip's yellow bolts / olive leaf bits flying apart from the sphere: parts of the label
# not within 3 px of its largest part (the sphere with its bats) dropped (page 526 t355 / t361)
TRAIL_SKIP[6000301] = ('carl_atk_line_particle', 'particle_elec_yellow')


def _a3_mortis005(body, ctx):
    from scipy.ndimage import label as cc
    L, n = cc(body)
    if n < 2: return body
    sz = _np.bincount(L.ravel())[1:]; big = L == 1 + int(_np.argmax(sz))
    keep = _np.unique(L[_dilate(big, 3)]); return _np.isin(L, keep[keep > 0])


POST_BODY[6000301] = _a3_mortis005


# Whirlwind (Carl) 010 Carreta: a white launch puff over Carl's head drawn from the projectile's atlas (clip-body code)
# was labelled with it (page 610 t265, touching the crescent's label); the puff lies outside the crescent's predicted
# label: draws whose pixels are mostly (> half) outside the prediction grown 4 px dropped, then parts not within 3 px of
# the largest dropped
def _a3_carl010(body, ctx):
    ids = ctx['ids_h']; near = _dilate(ctx['pred_any'], 4)
    for d in _np.unique(ids[body & (ids > 0)]):
        m = body & (ids == d)
        if (m & ~near).sum() > 0.5 * m.sum(): body = body & ~m
    return _a3_mortis005(body, ctx)


POST_BODY[6000621] = _a3_carl010
