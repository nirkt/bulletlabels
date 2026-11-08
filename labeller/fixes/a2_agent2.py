# label-fix agent 2 overrides (2026-10-20): Sandstorm, BullDude, Sniper, IceDude, ElectroSniper, Cactus, Conductor, Mender, AxeJuggler skins
# exec'd in idlabels' namespace (IDLABELS_OVERRIDE)

# ---- Sandy (Sandstorm), attack, every skin (SANDY_ATK, ZONE): at max range the sand cloud bursts into small scattered
# flakes while the logic projectile still lives for 2-4 ticks (sw2_255 ticks 273-275, 315: largest drawn part 17-74 px
# against a predicted cloud of 12-15k px). That burst is the end animation (rule 2): no label once the cloud's largest
# drawn part is below SANDY_END_FRAC of its predicted body. The intact and the breaking cloud (largest part >= 10 %,
# ticks 265-271) keep their zone label.
SANDY_END_FRAC = 0.03
def _sandy_end(body, ctx):
    from scipy.ndimage import label as cc
    m = ctx['mine']; pb = int(ctx['pred_body'].sum())
    if not m.any() or pb == 0: return body
    L, n = cc(m); mx = int(np.bincount(L.ravel())[1:].max())
    return body if mx >= SANDY_END_FRAC * pb else np.zeros_like(body)
for _t in SANDY_ATK: POST_BODY[_t] = _sandy_end

# ---- helper: drop the body pixels drawn by one of the type's own clip sprites that is a streak / effect, identified by
# its atlas texture size (the clip's other body sprites are on another atlas); pixels of it lying over the kept body
# (the streak's root drawn over the pellet) stay when enclosed by the kept body (gaps <= 2 px closed)
def _drop_tex(sizes):
    def fn(body, ctx):
        from scipy.ndimage import binary_fill_holes
        R = ctx['R']; ids_h = ctx['ids_h']
        sel = [d for d, dr in R['draws'].items() if (dr.get('tw'), dr.get('tht')) in sizes]
        if not sel: return body
        rm = body & np.isin(ids_h, sel)
        if not rm.any(): return body
        keep = body & ~rm
        return keep | (rm & binary_fill_holes(_close(keep, 2)))
    return fn

# Bull Mang (BullDudeMang, Bull005Projectile): each star pellet is drawn with a motion-streak line behind it (clip shape
# 24826 on the 2560x4064 atlas; the star is shape 11557 on the 4096 atlas). The streak is a trail (rule 3, as the
# Bulletstorm 199-gem streak strips): only the stars are labelled (sw2_517 t269/t271/t309-311)
POST_BODY[6000046] = _drop_tex({(2560, 4064)})

# Piper Carnival (SniperCarnival, Piper007Projectile): the bird flies its first ticks as a translucent green bird (its
# clip body sprite drawn at coverage ~40-120, only the eye >= 128: unlabelled but clearly drawn, sw2_552 t265, t305);
# it turns into the opaque yellow bird later. Lower coverage cut, as Willow's orb
FADE_CUT[6000380] = 32

# Belle SCmake (ElectroSniperSCmake, Belle005Projectile 6000926): the bolt is a flame drawn as several thin translucent
# flame strands (its opaque lines label as a mesh of strips with holes, sw2_382 t305-313): rule 5, ONE hull label over
# the strands. Built only from the strand parts overlapping its predicted body (the twinkles / swirl particles flying off
# it, parts with no predicted-body pixel, stay out of the hull)
def _belle005_lines(body, ctx):
    from scipy.ndimage import label as cc
    L, n = cc(body)
    if not n: return body
    keep = np.unique(L[body & ctx['pred_body']]); keep = keep[keep > 0]
    return np.isin(L, keep)
POST_BODY[6000926] = _belle005_lines
ONE_LABEL[6000926] = 'hull'

# Belle Fantasy (ElectroSniperFantasy, Belle006Projectile 6000929): the crystal bolt is drawn under a big translucent
# glow ellipse that shares its atlas (code 5, coverage < 128): the ID pass sees only a few sparks of it, the crystal
# itself was unlabelled (sw2_417 t265, t307, t311). Body = the crystal from the frame: pixels whose min(R, G, B) >= 100
# (crystal 96-255, its cyan glow 48-80, ground ~32, the cyan swirl trail particles and water have a low red channel)
# near its predicted label, the cluster(s) touching its predicted body, their convex hull (the crystal is convex)
BELLE006_MINV = 100
def _belle006_crystal(body, ctx):
    from scipy.ndimage import label as cc, binary_fill_holes
    img = ctx['img']
    if img is None: return body
    bb = ctx['region'] & (img.min(-1) >= BELLE006_MINV) & _dilate(ctx['pred_any'], 4)
    if not bb.any(): return body
    bb = _close(bb, 1); L, n = cc(bb)
    keep = np.unique(L[bb & _dilate(ctx['pred_body'], 2)]); keep = keep[keep > 0]
    if not keep.size: return body
    return hull_mask(np.isin(L, keep))   # the crystal is convex; its darker facets (min < 100) are inside it
POST_BODY[6000929] = _belle006_crystal

# Belle Fantasy super (Belle006UltiProjectile 6000931): the arrowhead is a neon chevron drawn additively over a big glow
# ellipse sharing its atlas; the ID pass labelled only a few inner strips (sw2_417 t345-349). Rule 5 (energy lines): ONE
# hull label over the chevron's lines, taken from the frame: green channel >= 192 (the neon lines 200-255; its interior
# 110-170 and the blue tail streak <= 160 stay out of the seed, the interior is inside the hull) near its predicted body
def _belle006_ulti(body, ctx):
    from scipy.ndimage import label as cc
    img = ctx['img']
    if img is None: return body
    bb = ctx['region'] & (img[..., 1] >= 192) & _dilate(ctx['pred_any'], 4)
    if not bb.any(): return body
    L, n = cc(_close(bb, 1))
    keep = np.unique(L[_dilate(ctx['pred_body'], 3)]); keep = keep[keep > 0]
    if not keep.size: return body
    return np.isin(L, keep)
POST_BODY[6000931] = _belle006_ulti
ONE_LABEL[6000931] = 'hull'

# Mender default (MenderProjectile 6001755): a white light ball inside a cyan additive glow (the glow disc sprite is the
# ID pass's front-most draw, coverage < 128): only a few px at the tether's end were labelled (sw2_111 t265-311).
# Body = the white ball from the frame: min(R, G, B) >= 150 (ball 160-255; cyan glow, teal tether and ground <= 110)
# near the prediction, the clusters touching its predicted body, convex hull. The glow (soft halo) and the tether stay out
def _frame_core(minv, near=4):
    def fn(body, ctx):
        from scipy.ndimage import label as cc
        img = ctx['img']
        if img is None: return body
        bb = ctx['region'] & (img.min(-1) >= minv) & _dilate(ctx['pred_any'], near)
        if not bb.any(): return np.zeros_like(body)
        L, n = cc(_close(bb, 1))
        keep = np.unique(L[_dilate(ctx['pred_body'], 2)]); keep = keep[keep > 0]
        if not keep.size: return np.zeros_like(body)
        return hull_mask(np.isin(L, keep))
    return fn
POST_BODY[6001755] = _frame_core(150)

# Mender 199-gem (Mender199Gem, Mender002Projectile 6001758, a wave type): its water beam is drawn with the trail clip
# 'glowbert_def_atk_dust' (soft blobs on the glowbert atlas, opaque here: coverage >= 128 over 2-3k px), which TRAIL_FX
# ('dust') marks as a ground effect: the beam was unlabelled, only its spark droplets were (sw2_452 t265-269, t305-309).
# For this type the dust sprites are the beam: the draws showing them (>= 30 px of code 4, only that clip is skipped) are
# the beam, their at least half-opaque pixels in its region are body, small holes filled
def _mender002_beam(body, ctx):
    # the beam column and its foam head are drawn in the same draw(s) as the dust blobs (one batched draw, its other
    # sprites outside the dust clip's atlas polygons): every at least half-opaque pixel of those draws in its region
    C = ctx['C']; ids_h = ctx['ids_h']; cv = ctx['cv']
    seed = ctx['region'] & (C == 4) & (cv >= CUT)
    if not seed.any(): return body
    d, n = np.unique(ids_h[seed], return_counts=True); d = d[(n >= 30) & (d > 0)]
    if not d.size: return body
    b = body | (ctx['region'] & np.isin(ids_h, d) & (cv >= CUT))
    return b | small_holes(b, SPECK)
POST_BODY[6001758] = _mender002_beam

# Spike super zones (Cactus*, 'Spike_00N_ulti_explosion' / 'CactusUltiExplosion'): the zone's dark disc has no opaque
# sprite in its middle (the ID pass shows the ground there), so the label was the outer ring band plus the inner spikes
# with the zone's centre a hole (Mecha sw2_156 t371/t389, default sw2_020 t371-411, Pink sw2_153, Log sw2_167).
# ZONE_DISC_UNION: the drawn sprites' label as before PLUS the logic ground disc (as ZONE_DISC: minus anything drawn in
# front by depth; the disc matches the drawn rim), plus the zone's own sprite pixels inside the disc's hull (its spikes
# rise above the ground, the depth test cuts them out) and small enclosed slivers.
# Spike Lord ('Spike_004_ulti_explosion', sw2_162 t371-411) the opposite: its sprite label took the soft purple haze far
# outside the rim; there the disc alone (ZONE_DISC) + its own sprites inside the disc's hull (ZONE_DISC_OWN)
ZONE_DISC_UNION = {'Spike_003_ulti_explosion', 'CactusUltiExplosion',
                   'Spike_002_ulti_explosion', 'Spike_005_ulti_explosion', 'Spike_009_ulti_explosion'}   # + Pink 153, Log 167, Carreta 204 (verified t389)
ZONE_DISC_OWN = {'Spike_004_ulti_explosion'}
ZONE_DISC.update(ZONE_DISC_OWN)
_zone_labels_base = zone_labels
def _ground_disc(it, Z, cam, R, taken):
    x, y = int(it[3]), int(it[4]); rz = float(Z['logic_row'].get('Radius') or 300)
    cx, cy = _proj(cam, x, y); dx = abs(_proj(cam, x + rz, y)[0] - cx); dy = abs(_proj(cam, x, y + rz)[1] - cy)
    H, W = taken.shape; yy, xx = np.mgrid[0:H, 0:W]
    if dx < 1 or dy < 1 or R.get('wd') is None: return np.zeros((H, W), bool)
    disc = ((xx - cx) / dx) ** 2 + ((yy - cy) / dy) ** 2 <= 1
    return disc & (R['wd'] >= ground_depth(cam, x, y, rz, R['meta'].get('wref', 0.0)) - DEPTH_TOL) & ~taken
def zone_labels(state, cam, R, taken, own_team=None):
    import owners
    out = _zone_labels_base(state, cam, R, taken, own_team)
    for it in state.get('items', []):
        iid, did = int(it[0]), int(it[1]); Z = owners.zone(did)
        if not Z or owners.area(did)['kind'] != 'zone': continue
        nm = Z.get('Name')
        if nm not in ZONE_DISC_UNION and nm not in ZONE_DISC_OWN: continue
        m = out[iid][0] if iid in out else np.zeros(taken.shape, bool)
        if nm in ZONE_DISC_UNION: m = m | _ground_disc(it, Z, cam, R, taken)
        if not m.any(): continue
        C = sprite_pixels(('zone', did), True, R)
        m2 = m | (hull_mask(m if nm in ZONE_DISC_OWN else _ground_disc(it, Z, cam, R, taken)) & (C > 0) & ~taken)
        out[iid] = (m2 | (small_holes(m2, 60) & ~taken), did, int(it[2]))   # the spikes' edge slivers (enclosed, small)
    return out
