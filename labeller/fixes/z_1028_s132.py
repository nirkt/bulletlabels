# 2026-10-28 (red1028 task): TEAM-AWARE version of z_1028_s132.py (replaces it, same name when installed). Part A now also works
# for the ENEMY wave (o['own'] False, red / pink / crimson streaks instead of blue): colour tests per team (enemy: r above b,
# b above g; brown debris told apart by g >= b + 8; dark crimson blades of the drawn body r >= b + 10, r >= 45; pink core
# r >= 200, r - g >= 90), a small closing of the final hull where it was cut at the brawler's cane. The own (blue) path is
# unchanged code: pixel-identical on sw2_132 (compare_own_check.txt). Part B (KnightPet) is untouched.
# 2026-10-28 round 3 for skin 132 (KnightKrampus, Knight): part A = the wave fix of the earlier fix2.py (accepted, unchanged);
# part B = a NEW KnightPet fix (replaces fix2's spawn_labels override, which re-ran the spawn matching with the bush draws
# hidden and thereby added small labels on UI; this one only REMOVES wrong draws from the pet's own label and adds nothing).
# Knight's attack wave, type 6001062 (Ash004Projectile1 = KnightProjectile1), sw2_132 attack 0 deg (object 2000002, ticks 269-277)
# and attack 90 deg (object 2000003, ticks 309-317). Findings: pairs 2135-2139, 2142-2146 (merge_split): the blue streaks were
# outlined as many small red fragments (most streaks are additive, the ID pass sees only the dark part of each). Rule 5/6: one
# convex hull of the whole attack, built from its own blue streaks only.
# Differences from the rejected fix.py: (1) the brown slash pieces (hands / rods, r clearly above b) are NOT part of the attack
# (debris of the skin's slash sprite, not the blue streaks): they are removed from the body before the hull; (2) the hull is
# built from the drawn body (blue part only) plus bright streak-coloured pixels (bright cyan-blue, not the soft glow) within a
# few px of the predicted label, so no glow / empty background is added; (3) the crate / wall cut is made on the hull's blue
# source pixels (streak pixels lying on the crate-coloured, nearer-than-ground geometry are never added) and the hull is only
# cut by depth where it overlaps real nearer geometry, then reduced to its largest piece and filled solid (no loops, no spurs).
# Not for the super (6001065) and not for ended waves (dropped before this hook, as before).
import numpy as _np
_K132_ATK = 6001062
_K132_NEAR = 6
_K132_CUT = 0.35
_K132_BLUE = 25      # b - r of a streak pixel (brown rod edges are below)
_K132_BR_G = 8      # enemy: brown debris has g >= b + 8, a red streak has g < b
_K132_RED_BG = 10   # enemy: b - g of a red streak pixel
_K132_DARK_RB = 10  # enemy: r - b of a dark crimson blade pixel of the drawn body
_K132_DARK_R = 45
_K132_BROWN = 2     # px kept clear around brown rod pixels


def _fix_k132_atk(body, ctx):
    from scipy.ndimage import label as _cc, binary_fill_holes as _fh, binary_opening as _bo
    img = ctx['img']; pred_any = ctx['pred_any']; region = ctx['region']; R = ctx['R']; cam = ctx['cam']; o = ctx['o']
    if img is None or not body.any(): return body
    im = img.astype(_np.int16); r, g, b = im[..., 0], im[..., 1], im[..., 2]
    if o.get('own', True):
        # streak pixels are saturated blue (b well above r); the brown rods and their dark edges / blue-purple flashes are not
        brown = (r >= b + 15) & (r >= 90)
        blue = (b >= r + _K132_BLUE) & (b >= 80) & ~_dilate(brown, _K132_BROWN)
        bb = body & blue
        core = (b >= 170) & (b - r >= 90) & (g - r >= 60) & _dilate(pred_any, _K132_NEAR) & region & ~_dilate(brown, _K132_BROWN)
    else:
        # enemy wave (red tint, o['own'] False): the streaks are crimson / pink (r well above b, b above g) instead of blue.
        # The brown debris (logs, canes, rocks: r > b and g clearly above b) is told apart by g: g >= b + 8 (a streak has g < b)
        brown = (r >= b + 15) & (r >= 90) & (g >= b + _K132_BR_G)
        red = (r >= b + _K132_BLUE) & (r >= 80) & (b >= g + _K132_RED_BG) & ~_dilate(brown, _K132_BROWN)
        # the dark crimson blades (drawn opaque enough for the ID pass, r only 45-80) count when they are in the drawn body
        dark = (r >= b + _K132_DARK_RB) & (r >= _K132_DARK_R) & (b >= g + 5) & ~_dilate(brown, _K132_BROWN)
        bb = body & (red | dark)
        core = (r >= 200) & (r - g >= 90) & (b >= g + _K132_RED_BG) & _dilate(pred_any, _K132_NEAR) & region & ~_dilate(brown, _K132_BROWN)
    m = bb | core
    if not m.any(): return body
    L, n = _cc(_dilate(m, 2), structure=_np.ones((3, 3)))
    has_body = _np.zeros(n + 1, bool); has_body[_np.unique(L[bb])] = True
    core_n = _np.bincount(L[core & m], minlength=n + 1)
    keep = has_body | (core_n >= 15); keep[0] = False
    m = m & keep[L]
    if not m.any(): return body
    hm = hull_mask(m)
    wd = R.get('wd'); wref = R['meta'].get('wref', 0.0)
    if wd is not None and o.get('logic') is not None:
        gd = ground_depth(cam, float(o['logic'][0]), float(o['logic'][1]), 300.0, wref)
        hm = hm & ((wd >= gd - _K132_CUT) | m)
    # the brown slash rods are debris: where they lie inside the hull they are cut out of it (their dark rim included)
    hm = hm & ~_dilate(_bo(brown, structure=_np.ones((3, 3))), 1)
    L2, n2 = _cc(hm, structure=_np.ones((3, 3)))
    if n2 > 1:
        sz = _np.bincount(L2.ravel())[1:]
        hm = L2 == 1 + int(sz.argmax())
    if not o.get('own', True):   # enemy: the cut at the brawler's cane (nearer than the ground) leaves a narrow notch / loop: closed
        from scipy.ndimage import binary_closing as _bc
        hm = _bc(_np.pad(hm, 3), structure=_np.ones((3, 3)), iterations=2)[3:-3, 3:-3]
    return _fh(hm)


POST_BODY[_K132_ATK] = _fix_k132_atk


# ---- Part B: KnightPet (spawnable, characters.csv did 16000197) label takes things that are not the pet.
# spawn_labels gives each runtime-texture quad (th 0, the sprite quads of characters / pets AND of the map's bush foliage) to the
# character nearest to it (within 80 px). For the pet that picks up (1) the translucent bush foliage quads at the top of the map
# (fs 4132660303, prog 112, no depth write, fixed screen position; pair 2154: bush-shaped label, tick 385-411) and (2) the
# Knight's own sprite quad (fs 1680474519 like the pet's, but 3x as large: bbox ~75 x 49 half-res px vs the pet's ~25 x 28)
# whenever a pet stands on the Knight and is nearer to the quad's centre than the Knight (sw2_132 ticks 399-407: 1700-1985 px
# label on the Knight). Fix (post-filter, for pets of that type only, adds nothing): of the draws that make up a pet's label,
# drop those with the bush foliage fragment shader or whose full on-screen extent is larger than a pet quad
# (> 34 px wide or > 36 px high at half resolution; pets measure up to 25 x 28); the label is rebuilt from the kept draws
# (small holes closed as before) and a pet label under 20 px (a sliver of a pet hidden behind the Knight) is dropped.
_K132B_PET = 16000197
_K132B_BUSH_FS = 4132660303
_K132B_MAXW, _K132B_MAXH = 34, 36
_K132B_MINPX = 20
if '_k132b_spawn_base' not in globals(): _k132b_spawn_base = spawn_labels   # the previous spawn_labels, also when exec'd twice


def spawn_labels(state, cam, R, taken):
    out = _k132b_spawn_base(state, cam, R, taken)
    if not out: return out
    ids_h = R['id'][::2, ::2]
    chars = {int(c[0]): c for c in state['chars']}
    for cid in list(out):
        c = chars.get(cid)
        if c is None or int(c[1]) != _K132B_PET: continue
        m = out[cid]; kept = _np.zeros_like(m)
        for d in _np.unique(ids_h[m]):
            dm = ids_h == d
            ys, xs = _np.nonzero(dm)
            if R['draws'].get(int(d), {}).get('fs') == _K132B_BUSH_FS: continue
            if xs.max() - xs.min() + 1 > _K132B_MAXW or ys.max() - ys.min() + 1 > _K132B_MAXH: continue
            kept |= dm & m
        if kept.any(): kept |= small_holes(kept, SPECK) & ~taken
        if kept.sum() < _K132B_MINPX: del out[cid]
        else: out[cid] = kept
    return out
