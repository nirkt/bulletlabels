# label-fix agent 5 overrides (Barkeep, Ruffs, HammerDude, AssaultShotgun, MinigunDude, Ninja, Painter, Speedy skins)
# exec'd in idlabels' namespace after load

# ---- Barkeep (Barley) zones -------------------------------------------------------------------------------------------
# Barley's lingering puddles (area_effects_skin rows) list, besides the puddle, the bottle's impact splash (barley_def_atk_impact),
# flying glass shards / debris, and gas smoke (particle_gas_seq: the same sprite as the flying bottle's smoke trail).
# Impacts, shards, debris, smoke and sparks are not the zone (BRIEF rules 2, 3): on BarkeepDrunken (Barley007) the
# bottles' smoke trail was labelled as zone (tick 369), on BarkeepMs (Barley005) the opening impact splash drawn over the
# walls was labelled (ticks 289, 329). Those clips are taken out of these zones' sprite sets; the puddle itself is unchanged.
A5_ZONE_CLIP_SKIP = {n: ('impact', 'shard', 'debris', 'particle_gas', 'twinkle', 'fire_spark')
                     for n in ('Barley_004_atk', 'Barley_004_ulti', 'Barley_005_atk', 'Barley_005_ulti',
                               'Barley_007_atk', 'Barley_007_ulti', 'Barley_009_atk', 'Barley_009_ulti')}
_a5_zone_sprites_orig = zone_sprites


def zone_sprites(did, own):
    import owners
    Z = owners.zone(did)
    skip = A5_ZONE_CLIP_SKIP.get((Z or {}).get('Name'))
    if not skip: return _a5_zone_sprites_orig(did, own)
    clips = [(Z.get('FileName'), Z.get(c)) for c in ZONE_EXPORTS if Z.get(c) and Z.get('FileName')]
    for eff in (Z.get('Effect'), Z.get('LoopingEffect')): clips += effect_clips(eff)
    return _clips_sprites([c for c in clips if not any(s in c[1].lower() for s in skip)])


# BarkeepMaple (Barley004) attack / super bottles: the bottle clip's liquid shapes (407, 63) are the same atlas shapes as
# the orange puddle on the ground, so a bottle flying over its puddle took big puddle pieces as its body (super, tick 369:
# 3000 px of puddle labelled magenta), and its maple-leaf trail particles drawn near it. Body pixels of a draw are dropped
# when the draw is a big one lying mostly away from this bottle (a puddle draw: > 2x the bottle's predicted size, under
# 35% of it within 8 px of the prediction), or when the draw is mostly its trail sprites (code 4: the leaves; rule 3).
A5_FOREIGN_DRAW = {6000254, 6000255}


def _a5_foreign_draws(body, ctx):
    ids, pa, C = ctx['ids_h'], ctx['pred_any'], ctx['C']
    if not body.any() or not pa.any(): return body
    near = _dilate(pa, 8); n_pred = int(pa.sum())
    for d in np.unique(ids[body]):
        if d <= 0: continue
        dm = ids == d; tot = int(dm.sum())
        if (tot > 2 * n_pred and (dm & near).sum() < 0.35 * tot) or (C[dm] == 4).mean() > 0.5:
            body = body & ~dm
    return body


for _t in A5_FOREIGN_DRAW: POST_BODY[_t] = _a5_foreign_draws

# BarkeepMaple zones: their 'cloud' clip shows the same maple-leaf shape (58784) as the bottles' leaf trail, so the leaves
# flying from the bottles were labelled as zone (super tick 369). Leaves are particles (rule 3): pixels showing the leaf
# trail sprites (code 4 of the Barley004 bottle types) are taken out of these zones, except where the puddle encloses
# them (the leaves floating on the puddle are its decoration, as before).
A5_ZONE_TRAIL_OUT = {'Barley_004_atk': 6000254, 'Barley_004_ulti': 6000255}
_a5_zone_labels_orig = zone_labels


def zone_labels(state, cam, R, taken, own_team=None):
    import owners
    out = _a5_zone_labels_orig(state, cam, R, taken, own_team)
    for k, (m, did, own) in list(out.items()):
        t = A5_ZONE_TRAIL_OUT.get((owners.zone(did) or {}).get('Name'))
        if t is None: continue
        o = True if own_team is None else own_team(own)
        from scipy.ndimage import binary_fill_holes
        leaf = sprite_pixels(t, o, R) == 4; m2 = m & ~leaf
        m = m2 | (m & leaf & binary_fill_holes(m2))   # leaves lying inside the puddle (enclosed by it) stay zone
        if m.any(): out[k] = (m, did, own)
        else: del out[k]
    return out

# BarkeepWizard (Barley002) zones: blue fire rings drawn additively (the ID pass sees only stray bits: the page showed
# arcs and specks, and nothing on the settled rings). Additive lingering zones are their logic ground disc minus anything
# in front (rule 10, ZONE_DISC)
ZONE_DISC.update({'Barley_002_atk', 'Barley_002_ulti'})

# ---- Ruffs ------------------------------------------------------------------------------------------------------------
# RuffsChimmy (Ruffs004) attack: the bullet (cyan star with a paw on it) is drawn under its own alpha glow sprite, so the
# ID pass gives the glow (code 2) over most of the bright star arms and the paw; only the middle was labelled (ticks 267,
# 309). The bright cyan pixels (frame green >= A5_CYAN_G) of its own draws and sprites, connected to the body (gaps of
# 1 px closed), are body; the soft halo (green <= ~140) stays out, the additive trail stars (no draw) too.
A5_CYAN_G = {6000894: 160}


def _a5_cyan_body(body, ctx):
    from scipy.ndimage import label as cc, binary_fill_holes
    img, ids, C, region = ctx['img'], ctx['ids_h'], ctx['C'], ctx['region']
    if img is None or not body.any(): return body
    own_d = np.unique(ids[body]); own_d = own_d[own_d > 0]
    cand = region & np.isin(ids, own_d) & (C > 0) & (C != 4) & (img[..., 1] >= A5_CYAN_G[ctx['t']])
    L, n = cc(_close(cand, 1) & region | body, structure=np.ones((3, 3), bool))
    keep = np.unique(L[body]); grown = np.isin(L, keep[keep > 0]) & (cand | body)
    return binary_fill_holes(_close(grown, 1)) & region & (np.isin(ids, own_d) | body)


POST_BODY[6000894] = _a5_cyan_body

# ---- HammerDude (Frank) -----------------------------------------------------------------------------------------------
# HammerDudeDJ (Frank002) attack / super waves: the trail's 'particle_stars_small_blink' flares (big soft glow discs, page
# ticks 281, 321, 371: round labels) and the super's yellow electric sparks ('shocked_yellow_delay') are twinkles /
# sparks, not the wave's light tiles (rule 3)
TRAIL_SKIP[6000477] = ('particle_stars_small_blink',)
TRAIL_SKIP[6000478] = ('particle_stars_small_blink', 'shocked_yellow')
# the round soft glow flares themselves use a generic disc shape that the bar clips list too (so they code as trail body):
# each flare is its own big draw, mostly under half opaque (t371: 3558 px, 1051 >= CUT), while a light tile's draw is
# opaque throughout. Body draws of which under half the pixels reach CUT are dropped (soft glows, rule 6).
A5_SOFT_DRAW = {6000477: 0.5, 6000478: 0.5}


def _a5_soft_draws(body, ctx):
    from scipy.ndimage import label as cc
    ids, cv = ctx['ids_h'], ctx['cv']; soft = np.zeros_like(body)
    for d in np.unique(ids[body]):
        if d <= 0: continue
        dm = ids == d
        if (cv[dm] >= CUT).mean() < A5_SOFT_DRAW[ctx['t']]: soft |= dm & body
    if not soft.any(): return body
    body = body & ~soft
    # a flare drawn over a light tile bites a round hole into it (the tile shows through the glow): each remaining tile
    # (connected part) takes back the flare pixels inside its convex hull
    L, n = cc(body)
    for k in range(1, n + 1):
        m = L == k
        if m.sum() >= 20: body |= hull_mask(m) & soft
    return body


for _t in A5_SOFT_DRAW: POST_BODY[_t] = _a5_soft_draws
# HammerDudeknight (Frank004): the ground cracks left by the ice spikes are the spike sprite tinted dark (old page t287,
# t289, t327, t331: brown cracks labelled red), as on Frank's default: DARK_DROP, consistent with 6000475/6000476
DARK_DROP.update({6000479, 6000480})
# HammerDudeSandsOfTimes (Frank008) attack / super: the ground cracks (frank_008_*ground_crack*, cyan then dark blue;
# page t287, t289, t325, t329, super t379) are drawn with an atlas shape its fire trail clips use too, so they were
# labelled. They are ground decals (rule 3). Their draws lie flat on the ground: eye depth minus the ground plane's
# (ground_depth at the wave's position) is a constant -0.077 +- 0.008 for every crack pixel (measured t281-t375), while
# the upright flames spread over -0.3 .. +0.5. Body draws with >= 60% of their pixels in that band are dropped.
A5_DECAL_BAND = {6000487: (-0.085, -0.068), 6000488: (-0.085, -0.068)}


def _a5_ground_decals(body, ctx):
    o, R, ids = ctx['o'], ctx['R'], ctx['ids_h']; wd = R.get('wd')
    if wd is None or o.get('anchor') is None or not body.any(): return body
    lo, hi = A5_DECAL_BAND[ctx['t']]
    q = wd - ground_depth(ctx['cam'], o['anchor'][0], o['anchor'][1], 1500, R['meta'].get('wref', 0.0))
    flat = (q >= lo) & (q <= hi)
    for d in np.unique(ids[body]):
        if d <= 0: continue
        m = body & (ids == d)
        if flat[m].mean() >= 0.6: body = body & ~m
    return body


for _t in A5_DECAL_BAND: POST_BODY[_t] = _a5_ground_decals
