# label-fix agent 4 overrides (2026-10-20): WeaponThrower (Bronson), Crow, Shaman (Nita), PowerLeveler (Surge),
# Luchador (El Primo), Ambusher (Lily), Samurai (Kenji), Splitter skins. exec'd in idlabels' namespace.


# ---- helpers: which sprite shape of a given set of clips the front-most draw shows, per pixel (half resolution)
@functools.lru_cache(maxsize=None)
def _a4_polys(swf, names):
    """{(swf, tex): [(shape id, uv polygon (n, 2) in 0..1)]} of every shape the named clips of swf can show"""
    S = PD.sc(swf); out = {}
    for name in names:
        cid = S.exports.get(name)
        if cid is None: continue
        for sid, k in sorted(_clip_shapes(S, cid)):
            tex, uv, xy = S.shapes[sid][k]
            out.setdefault((swf, tex), []).append((sid, np.asarray(uv, np.float64) / 65535.0))
    return out


@functools.lru_cache(maxsize=16)
def _a4_rmap(swf, names, tex):
    """atlas map: 0 none, else 1 + index into _a4_polys(swf, names)[(swf, tex)]"""
    w, h = tex_size(swf, tex); I = Image.new('I', (w, h), 0); d = ImageDraw.Draw(I)
    for i, (sid, uv) in enumerate(_a4_polys(swf, names)[(swf, tex)]):
        p = uv * [w, h]
        for k in range(len(p) - 2): d.polygon([tuple(p[k]), tuple(p[k + 1]), tuple(p[k + 2])], fill=i + 1)
        lo, hi = p.min(0), p.max(0)
        if (hi - lo).min() < SWATCH: d.rectangle([int(lo[0]) - 1, int(lo[1]) - 1, int(hi[0]) + 1, int(hi[1]) + 1], fill=i + 1)
    return np.asarray(I, np.int32)


def _a4_shape_pixels(R, swf, names):
    """(H, W) int32 at half resolution: the shape id (+1) of the named clips the front-most draw shows there, else 0"""
    ids = R['id'][::2, ::2]; uv = R['uv']; out = np.zeros(ids.shape, np.int32); draws = R['draws']
    dz = [int(i) for i in np.unique(ids[ids > 0]) if int(i) in draws]
    for (file, tex), polys in _a4_polys(swf, names).items():
        tw, th = tex_size(file, tex); sel = [i for i in dz if (draws[i].get('tw'), draws[i].get('tht')) == (tw, th)]
        byth = {}
        for i in sel: byth.setdefault(draws[i].get('th'), []).append(i)
        sel = [i for hh, L in byth.items() if atlas_ok(R, hh, L, file, tex) for i in L]
        if not sel: continue
        m = np.isin(ids, sel) & (uv[..., 0] >= 0)
        if not m.any(): continue
        rm = _a4_rmap(file, names, tex); Hh, Ww = rm.shape; ys, xs = np.nonzero(m)
        u = np.clip((uv[ys, xs, 0] * Ww).astype(int), 0, Ww - 1); v = np.clip((uv[ys, xs, 1] * Hh).astype(int), 0, Hh - 1)
        k = rm[v, u]; hit = k > 0
        sids = np.array([s for s, _ in polys], np.int32)
        out[ys[hit], xs[hit]] = sids[k[hit] - 1] + 1
    return out


def _a4_lines(ctx, swf, clips, shape_ids, grow=1, min_cv=0):
    """pixels of the given shapes of the given clips (the attack's own line sprites) whose line touches the projectile's
    region (a line reaching out of the region is taken whole)"""
    from scipy.ndimage import label as cc
    SP = _a4_shape_pixels(ctx['R'], swf, tuple(clips))
    m = np.isin(SP, [s + 1 for s in shape_ids]) & (ctx['cv'] >= min_cv)
    if not m.any(): return m
    L, n = cc(_dilate(m, grow)); keep = np.unique(L[ctx['region'] & m]); keep = keep[keep > 0]
    return m & np.isin(L, keep)


A4_DEBUG = os.environ.get('A4_DEBUG')


# WeaponThrower Yeti (Bronson003, skin 133): its attack is drawn only by its trail clip bronson_003_atk_trail (the 3D
# claws are not visible in flight); the user (2026-10-20): "the label needs to cover only the swoosh, without including
# other effects/particles". The swoosh's claw streaks are the clip's stretched bar sprite (shape 666): ONE label = the hull
# of those lines only (ONE_LABEL 'hull'); its glow dots (shape 0), sparkle (7294), soft triangle (50825), snow puffs,
# snowflakes and twinkles stay out. Replaces BRIGHT_BODY (frame brightness took sparks, the snow puff, particles).
BRIGHT_BODY.pop(6001224, None)
def _a4_fix_6001224(body, ctx):
    ln = _a4_lines(ctx, 'sc/effects_brawler.sc', ('bronson_003_atk_trail',), (666,))
    if A4_DEBUG: print('A4 6001224 obj', ctx['o']['id'], 'body in', int(body.sum()), 'lines', int(ln.sum()))
    return ln
POST_BODY[6001224] = _a4_fix_6001224
ONE_LABEL[6001224] = 'hull'


def _a4_bright_clip(ctx, swf, clips, vmin=200, blue=True, grow=0):
    """BRIGHT_BODY restricted to the attack's own sprites: pixels whose front-most draw shows one of the named clips'
    sprites, bright in the frame (max of R, G, B >= vmin), blue >= red (drops orange particles), main parts, holes filled"""
    from scipy.ndimage import label as cc, binary_fill_holes
    img = ctx['img']
    if img is None: return np.zeros_like(ctx['region'])
    SP = _a4_shape_pixels(ctx['R'], swf, tuple(clips)); im = img.astype(np.int16)
    m = (SP > 0) & _dilate(ctx['region'], grow) & (im.max(-1) >= vmin)
    if blue: m &= im[..., 2] >= im[..., 0]
    m = _close(m, 2); L, n = cc(m)
    if not n: return m
    sz = np.bincount(L.ravel())[1:]; m = np.isin(L, 1 + np.nonzero(sz >= 0.15 * sz.max())[0])
    m = binary_fill_holes(m)
    if blue: m &= im[..., 0] <= im[..., 2] + 20   # orange pixels the fill took in
    return m


# WeaponThrower Darkcircus (Bronson004, skin 136): the swoosh is drawn only by its trail clip
# bronson_004_atk_trail_afterimage (translucent; the 3D weapon is not visible in flight). The user (2026-10-20): "the label
# needs to cover only the swoosh, without including other effects/particles": the bright swoosh dome from the clip's
# own sprites only (not the impact clip, not orange particles, not the dim motion band behind it), holes filled.
# Replaces BRIGHT_BODY (frame brightness anywhere near: took the orange particles beside it).
BRIGHT_BODY.pop(6001228, None)
def _a4_fix_6001228(body, ctx):
    b = _a4_bright_clip(ctx, 'sc/effects_brawler.sc', ('bronson_004_atk_trail_afterimage',))
    if A4_DEBUG: print('A4 6001228 obj', ctx['o']['id'], 'body in', int(body.sum()), 'swoosh', int(b.sum()))
    return b
POST_BODY[6001228] = _a4_fix_6001228


# WeaponThrower SupercellMake (Bronson005, skin 141): the thrown ring is a 3D mesh drawn under its own translucent glow
# sprite (bronson_005_atk_projectile_glow, shared with its trail, coverage < 128): the ID pass sees the glow in front and
# only the ring's uncovered rim was labelled (Haiku: 'outline offset from the ring'). Ring = the mesh's predicted
# silhouette where its glow is drawn over it (FADE_CUT lets the glow pixels in), never the glow beyond the silhouette
# (the shared-sprite rule takes it up to SHARED_NEAR px around), plus the mesh's own visible draw.
FADE_CUT[6001232] = 1
def _a4_fix_6001232(body, ctx):
    return body & (ctx['pred_body'] | (ctx['C'] == 0))
POST_BODY[6001232] = _a4_fix_6001232


# PowerLeveler Buzz (Surge008, skin 590): the bolt is a bright blue ellipse drawn by its clip's translucent glow sprite
# (coverage < 128), only the white blobs in it were labelled. Body = the pixels where the front-most draw is one of its own
# sprites and the frame is bright (max of R, G, B >= 210: the ellipse; its soft halo is dimmer), main parts, holes filled.
def _a4_fix_6000806(body, ctx):
    from scipy.ndimage import label as cc, binary_fill_holes
    if ctx['img'] is None: return body
    m = ctx['region'] & (ctx['C'] > 0) & (ctx['img'].max(-1) >= 210)
    m = _close(m | body, 1); L, n = cc(m)
    if not n: return body
    sz = np.bincount(L.ravel())[1:]; keep = 1 + np.nonzero(sz >= 0.15 * sz.max())[0]
    return binary_fill_holes(np.isin(L, keep)) & ctx['region']
POST_BODY[6000806] = _a4_fix_6000806


# Luchador Butler (Primo009, skin 533): its fist is a mesh; its trail's grey smoke puffs and electric sparks drawn over
# it were labelled (the 'trail over the predicted body' rule; Haiku: outlines enclose the smoke clouds). Smoke and sparks
# are never the attack (rule 3): pixels where a smoke puff (primo_009_atk_smoke_*) is in front are dropped. The small
# electric sparks over the fist stay (the fist under them, as before: no holes in the fist).
def _a4_fix_6000405(body, ctx):
    sm = _a4_shape_pixels(ctx['R'], 'sc/effects_brawler.sc', ('primo_009_atk_smoke_01', 'primo_009_atk_smoke_02')) > 0
    return body & ~sm
POST_BODY[6000405] = _a4_fix_6000405


# Luchador Carreta (Primo013, skin 609): the glove is a mesh drawn in front; its trail's big translucent blur disc
# (gen_circle_blur_strong) around it was labelled where it overlaps the predicted body (beyond the glove, and the speed
# strips behind it; Haiku: overextends over the blue trail). Pixels where its trail sprite is the front-most draw are dropped.
def _a4_fix_6000409(body, ctx):   # holes then filled: its own V symbol drawn on the glove (a clip sprite its trail shares)
    from scipy.ndimage import binary_fill_holes
    b = body & (ctx['C'] != 4)
    return binary_fill_holes(b) if b.any() else b
POST_BODY[6000409] = _a4_fix_6000409


# Shaman Summer (Shaman004, skin 276): the water-wave body was labelled with many holes where its trail's water splashes
# (drawn 'over') sit on it (Haiku: fragmented). Enclosed holes are filled (water on water; the splashes outside the body
# stay out).
def _a4_fix_6000228(body, ctx):
    from scipy.ndimage import binary_fill_holes
    return binary_fill_holes(body) if body.any() else body
POST_BODY[6000228] = _a4_fix_6000228


# Crow Harvest (Crow007, skin 375): specks of its fire sparks (sprites shared with its clip) flying off the dagger were
# labelled as separate small parts: one dagger per projectile, parts smaller than BODY_ONLY_FRAC of the largest dropped.
BODY_ONLY.add(6000325)


# Lily Angel (Ambusher002, skin 194; a wave-kind type: its trail particles are its body): the attack is the bright streak
# drawn by its trail clip lily_002_atk_projectile_emitter; the feathers (lily_002_atk_feather_*) and twinkle stars flying
# off it were labelled (Haiku: outlines on star sparkles). Consistent with Lily's default (STREAK: the visible streak
# only): body = its pixels where the front-most draw is the streak emitter's sprite, holes filled. When the streak has
# faded under half coverage (after the throw ends) nothing is left: no label on the leftover feathers / stars.
def _a4_fix_6001455(body, ctx):
    from scipy.ndimage import binary_fill_holes
    em = _a4_shape_pixels(ctx['R'], 'sc/effects_brawler.sc', ('lily_002_atk_projectile_emitter',)) > 0
    b = body & em
    return binary_fill_holes(_close(b, 2)) & ctx['region'] if b.any() else b   # gaps between its layers closed
POST_BODY[6001455] = _a4_fix_6001455


# Splitter Greek (Splitter004, skin 567): its ground shadow (trail clip rt_004_atk_projectile_shadow) is the trident's own
# sprite tinted dark brown; its shaft was labelled where it lies near the predicted body (Haiku's 'bad'). Labelled parts
# whose median frame brightness (max of R, G, B) is under 160 are dropped (shadow parts measured 97-123, the golden
# trident 250+; DARK_DROP's per-draw median at 120 missed them: the shadow can share the trident's draw).
def _a4_fix_6001315(body, ctx):
    from scipy.ndimage import label as cc
    if ctx['img'] is None or not body.any(): return body
    v = ctx['img'].max(-1); L, n = cc(body)
    drop = [k for k in range(1, n + 1) if np.median(v[L == k]) < 160]
    return body & ~np.isin(L, drop)
POST_BODY[6001315] = _a4_fix_6001315


# TODO (NOT VERIFIED, left disabled 2026-10-20): the user's answer on Yeti 133: "it needs to include the whole attack" ->
# the hull must cover the claw lines (shape 666) AND the bright flash / sparkle at the swoosh head (shape 7294 of
# bronson_003_atk_trail), still without the snow puff, snowflakes, twinkles and glow dots. Shape 7294's footprint also
# reaches into the frost / snow puff below the swoosh, so it likely needs a brightness or coverage cut (options being
# compared in agent_4/x133_269.png, x133_309.png: 7294 alone, 7294 & max RGB >= 230, 7294 & cv >= 64). Draft:
# def _a4_fix_6001224(body, ctx):
#     ln = _a4_lines(ctx, 'sc/effects_brawler.sc', ('bronson_003_atk_trail',), (666,))
#     SP = _a4_shape_pixels(ctx['R'], 'sc/effects_brawler.sc', ('bronson_003_atk_trail',))
#     fl = (SP == 7294 + 1) & _dilate(ln, 6) & (ctx['img'].max(-1) >= 230)   # cut to be chosen after checking the PNGs
#     return ln | fl
# POST_BODY[6001224] = _a4_fix_6001224
