# Own-label fix (owner request 2026-11-08, "fix the 2 blue errors"). Prefix _o2_. Installed as py/fixes/z_1119_own2.py.
#
# (1) TWINS PETS (characters TwinsPet 16000230 and TwinsPetHyper 16000288; skins 088 / 181 / 193 / 208 all use the same character
#     and pet model). Enemy check, 181 TwinsVirus, spawn:TwinsPet: the OWN pet label sometimes takes in a bush rectangle (up to
#     ~3200 px; seen on a fresh capture of 181 at tick 407, and 3 to 6 ticks on each of 088 / 181 / 193 / 208).
#     CAUSE: spawn_labels() gives a pet every 3D-model draw (draw texture hash 0) whose centroid lies within CHAR_NEAR = 80 px of the
#     pet's projected line when the pet is the nearest character. For a few ticks (the bush row at the top is being shot at / the
#     heroes in it are hit) the bush foliage is drawn as model draws of ANOTHER shader (fs 4132660303, prog 134, blend 1) next to
#     the pet; their centroids lie 55-69 px from the pet's line, so they pass the 80 px test and the foliage rectangle became
#     part of the pet's label.
#     MEASURED on four captures (088 181 193 208, every tick that has a pet): the pet's own model draws (shader 1680474519) lie
#     0-14 px from its line; the foreign (foliage) draws 55-69 px; the floor draws 185+ px are already dropped by the 80 px test.
#     FIX: for a Twins pet a model draw whose centroid is farther than _O2_PET_NEAR = 30 px from the pet's line is not the pet's: it
#     is hidden from spawn_labels (its texture renamed in a shallow copy of R). Other spawnables (Voodoo, Knight, ... pets) keep
#     CHAR_NEAR; no other label is changed.
#
# (2) 361 BlackHoleStreetNinja_remake, Tara006UltiProjectile 6000422 (the flying eye / crystal of Tara's super; only skin 361 uses
#     the type). The OWN label was ragged: the left half of the centre spike and parts of the wings were missing.
#     CAUSE (measured on a fresh capture of 361, ticks 345-359): not the coverage cut (the crystal pixels have coverage 255). Half
#     of the crystal is drawn by three draws of the 4096 x 4096 atlas texture f9aa5e26 (effects_brawler.sc texture 0). The atlas
#     test of sprite_pixels (atlas_ok: pooled share of the texture's drawn pixels that show atlas alpha >= 64, bar 0.85) rates that
#     texture 0.73 pooled over the whole capture (it also carries many other sprites) and REFUSES it, so none of the crystal pixels
#     of those draws get a sprite code and they drop out of the label. The red clip is drawn from the same atlas shapes, but the
#     enemy run's pooled share lands on the other side of the bar, which is why the enemy label was complete and the own one not.
#     FIX: for type 6000422 only (its own, blue clip; sprite_pixels wrapper) the atlas test also accepts a texture when >= _O2_HIT =
#     50 % of its drawn pixels of THIS frame (at least 60) lie on the crystal clip's own sprite shapes (measured per frame: the right
#     texture 0.79-0.80, every other 4096 texture of the frame 0.00-0.001). The whole crystal then gets
#     its sprite codes like the enemy's. A per-type POST_BODY then closes slits and fills holes (one solid silhouette; the global
#     smoothing v2 runs after it). Enemy labels (red clip), other types, other skins' supers: untouched.
import numpy as np

# ---------------------------------------------------------------- (1) Twins pet
_O2_PET_NAMES = ('TwinsPet', 'TwinsPetHyper')
_O2_PET_NEAR = 30          # px (half resolution): the pet's own model draws lie within 14 px of its line (measured), foreign draws >= 55
_o2_spawn_prev = spawn_labels


def spawn_labels(state, cam, R, taken):
    import owners
    try:
        pets = []
        for c in state['chars']:
            ch = owners.character(int(c[1]))
            if ch['kind'] == 'spawnable' and int(c[2]) >= 0 and ch['name'] in _O2_PET_NAMES: pets.append(int(c[0]))
        if pets:
            ids_h = R['id'][::2, ::2]; chars = []
            for c in state['chars']:
                x, y, z = int(c[3]), int(c[4]), int(c[5])
                chars.append((int(c[0]), np.array(_proj(cam, x, y, z)), np.array(_proj(cam, x, y, z + CHAR_TOP))))
            draws = dict(R['draws']); changed = False
            for d, dr in R['draws'].items():
                if dr.get('th') != MODEL_TEX: continue
                m = ids_h == d
                if not m.any(): continue
                ys, xs = np.nonzero(m); p = np.array([xs.mean(), ys.mean()]); best = (CHAR_NEAR, None)   # as spawn_labels: nearest character
                for cid, a, b in chars:
                    ab = b - a; t = np.clip(np.dot(p - a, ab) / max(1e-6, np.dot(ab, ab)), 0, 1); dist = np.linalg.norm(a + t * ab - p)
                    if dist < best[0]: best = (dist, cid)
                if best[1] in pets and best[0] > _O2_PET_NEAR:
                    draws[d] = dict(dr, th='_o2_not_the_pet'); changed = True
            if changed:
                R2 = dict(R); R2['draws'] = draws
                return _o2_spawn_prev(state, cam, R2, taken)
    except Exception:
        pass
    return _o2_spawn_prev(state, cam, R, taken)


# ---------------------------------------------------------------- (2) Tara006 crystal
_O2_TARA = 6000422
_O2_HIT = 0.5              # share of an atlas candidate's drawn pixels (this frame) that must show the crystal clip's own sprite shapes
_O2_HIT_MIN = 60           # ... over at least this many pixels
_o2_in_tara = [False]
_o2_atlas_prev = atlas_ok
_o2_sprite_prev = sprite_pixels


def atlas_ok(R, th, draw_ids, file, tex):
    ok = _o2_atlas_prev(R, th, draw_ids, file, tex)
    if ok or not _o2_in_tara[0]: return ok
    # the pooled atlas share is poor evidence for this type: test THIS frame's pixels of the candidate draws directly against the
    # crystal's own shape map
    ids = R['id'][::2, ::2]; uv = R['uv']; cv = R['cv'][::2, ::2]
    m = np.isin(ids, draw_ids) & (uv[..., 0] >= 0) & (cv >= CUT)
    if m.sum() < _O2_HIT_MIN: return ok
    rm = region_map(_O2_TARA, True, file, tex); Hh, Ww = rm.shape
    ys, xs = np.nonzero(m); u = np.clip((uv[ys, xs, 0] * Ww).astype(int), 0, Ww - 1); v = np.clip((uv[ys, xs, 1] * Hh).astype(int), 0, Hh - 1)
    return bool((rm[v, u] > 0).mean() >= _O2_HIT)


def sprite_pixels(t, own, R):
    if t == _O2_TARA and own:
        _o2_in_tara[0] = True
        try: return _o2_sprite_prev(t, own, R)
        finally: _o2_in_tara[0] = False
    return _o2_sprite_prev(t, own, R)


_o2_post_prev = POST_BODY.get(_O2_TARA)


def _o2_tara_post(body, ctx):
    from scipy.ndimage import binary_fill_holes, binary_closing, label as cc
    if _o2_post_prev is not None: body = _o2_post_prev(body, ctx)
    if not body.any(): return body
    p = 3
    c = binary_closing(np.pad(body, p), structure=np.ones((3, 3), bool), iterations=2)[p:-p, p:-p] | body
    c = binary_fill_holes(c)
    L, n = cc(c)
    if n > 1:
        sz = np.bincount(L.ravel())[1:]; keep = [i + 1 for i in range(n) if sz[i] >= 0.2 * sz.max()]
        c = np.isin(L, keep)
    return c


POST_BODY[_O2_TARA] = _o2_tara_post
