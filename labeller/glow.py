"""Glow vs body: which layers of a projectile's display object are light around the projectile (glow) and which depict
the projectile itself (body), from the client's own data: per layer (one shape command drawn at a slot path) its blend
mode (the slot's blend byte, inherited down the clip tree), its colour transform (the concatenated multiplier alpha) and
the bitmap it samples. Rule and evidence: LABEL_SPEC.md 2, NOTES.md round 2.

A layer PAINTS when it can cover what lies under it with its own picture:
    blend normal or multiply   (additive and screen only add light: F = src + B, F = src + (1 - src) B, never hide B)
    and its bitmap has opaque texels (>= OPAQUE_SHARE of the texels it shows have alpha >= OPAQUE)
    and its colour transform draws it at least half opaque (multiplier alpha >= BODY_ALPHA / 255)
  body  = the painting layers;
  glow  = every other layer: additive / screen light, and normal-blend layers that are only a translucent wash
          (soft bitmaps with no opaque texel, or drawn below half opacity).
A clip that never paints (CLIP_LIGHT: its whole picture is light, e.g. an energy bolt) has no body by that rule; its
body is then its HARD light layers (an additive / screen layer whose bitmap has opaque texels, drawn at least half
opaque): the bolt's shape, not the soft blobs around it. Clips with neither are glow when the type has a 3D mesh (the
mesh is the body), and 'wave' otherwise: a melee wave or pulse whose drawing is exactly its hitbox, so every layer is
body and every pixel it changes is labelled (round 2, the user's decision). A layer's class is decided once per clip over
its animation (median multiplier alpha), so it does not change from frame to frame.
3D meshes: opaque and translucent materials are body, additive materials (blendMode 1, 6) glow.
  shape_stats(S, sid, cmd)            the bitmap of one shape command
  layer_class(t, own, q)              'body' | 'glow' | 'wave' of a placed polygon (projdraw.place dict) or Display.polys() tuple
  clip_layers(file, export)           the clip's kind ('paint' | 'light' | 'none') and every layer's class
  type_kind(t, own)                   'paint' | 'light' (hard light layers become body) | 'glow' (mesh = body) | 'wave'"""
import functools
import numpy as np
import sctex

OPAQUE = 0.9        # a texel is opaque from this alpha (premultiplied ASTC payload / 255)
SEEN = 0.02         # a texel is part of the bitmap from this alpha
OPAQUE_SHARE = 0.05 # a bitmap with opaque texels: at least this share of its seen texels (normal-blend bitmaps: 96 % of
                    # them have >= 80 % opaque texels, 3 % none; the cut anywhere in 0.01..0.2 changes 1.1 % of the area)
BODY_ALPHA = 128    # colour-transform alpha (0..255) of a painted layer: drawn at least half opaque


@functools.lru_cache(maxsize=None)
def _tex(path, k):
    import projdraw as PD, os
    return sctex.texture(PD.sc(os.path.relpath(path, PD.rc.ASSETS)), k)


def shape_stats(S, sid, cmd):
    """the bitmap of one shape command: texels inside its uv polygon -> n (texels with alpha >= SEEN), opaque (fraction
    of them with alpha >= OPAQUE), soft (fraction SEEN..OPAQUE), amax, lum (mean premultiplied rgb max / alpha of the
    seen texels: 1 = the colour fully carries the alpha), white (mean min(rgb) / alpha: near 1 = white light)"""
    from PIL import Image, ImageDraw
    tex_i, uv, xy = S.shapes[sid][cmd]
    T = sctex.texture(S, tex_i); h, w = T.shape[:2]
    p = np.asarray(uv, np.float64) / 65535.0 * [w, h]
    x0, y0 = np.floor(p.min(0)).astype(int); x1, y1 = np.ceil(p.max(0)).astype(int) + 1
    x0, y0 = max(0, x0), max(0, y0); x1, y1 = min(w, x1), min(h, y1)
    if x1 <= x0 or y1 <= y0: return dict(n=0, opaque=0.0, soft=0.0, amax=0.0, lum=0.0, white=0.0)
    m = Image.new('L', (x1 - x0, y1 - y0), 0); ImageDraw.Draw(m).polygon([tuple(v) for v in (p - [x0, y0])], fill=1)
    m = np.asarray(m, bool); px = T[y0:y1, x0:x1][m].astype(np.float32) / 255.0
    a = px[:, 3]; s = a >= SEEN
    if not s.any(): return dict(n=0, opaque=0.0, soft=0.0, amax=float(a.max()) if len(a) else 0.0, lum=0.0, white=0.0)
    return dict(n=int(s.sum()), opaque=float((a[s] >= OPAQUE).mean()), soft=float((a[s] < OPAQUE).mean()), amax=float(a.max()),
                lum=float((px[s, :3].max(1) / a[s]).mean()), white=float((px[s, :3].min(1) / a[s]).mean()))


def local_cam(cx, cy, s, W=None, H=None):
    """a camera that shows root-clip px (x right, y down, z 0) at s screen px per clip px, (cx, cy) at the centre of the
    2400 x 1080 frame (masks.project)"""
    import masks
    W = W or masks.W_FULL; H = H or masks.H_FULL
    return np.array([[2 * s / W, 0, 0, -2 * s * cx / W], [0, -2 * s / H, 0, 2 * s * cy / H], [0, 0, 1, 0], [0, 0, 0, 1]], np.float64)


def raster_local(S, polys, cam, bg=0.25):
    """composite of clip polygons (Display.polys() tuples, in draw order) over a grey background with the client's
    blend functions (masks.raster_poly) -> (H, W, 3) float image, full frame"""
    import masks
    img = np.full((masks.H_FULL, masks.W_FULL, 3), bg, np.float32)
    for (path, sid, cmd, tex, uv, xy, mul, add, bl) in polys:
        q = dict(world=np.c_[xy, np.zeros(len(xy))], uv=uv, tex=tex, mul=mul, add=add, blend=bl)
        r = masks.raster_poly(S, q, cam)
        if r is None: continue
        x0, y0, C, Tt = r; sl = img[y0:y0 + C.shape[0], x0:x0 + C.shape[1]]; sl *= Tt; sl += C
    return np.clip(img, 0, 1)


PAINT_BLENDS = {0, 3, 'normal', 'multiply'}
LIGHT_BLENDS = {8, 4, 'add', 'screen'}


@functools.lru_cache(maxsize=None)
def bitmap(file, sid, cmd):
    import projdraw as PD
    return shape_stats(PD.sc(file), sid, cmd)


def _fields(q):
    """(shape id, command, blend, multiplier alpha 0..255) of a placed polygon dict or a Display.polys() tuple"""
    if isinstance(q, dict): return q['shape'], q['cmd'], q['blend'], int(q['mul'][3])
    return q[1], q[2], q[8], int(q[6][3])


def paints(file, q):
    sid, cmd, bl, a = _fields(q)
    return bl in PAINT_BLENDS and a >= BODY_ALPHA and bitmap(file, sid, cmd)['opaque'] >= OPAQUE_SHARE


def hard_light(file, q):
    sid, cmd, bl, a = _fields(q)
    return bl in LIGHT_BLENDS and a >= BODY_ALPHA and bitmap(file, sid, cmd)['opaque'] >= OPAQUE_SHARE


def _layer_key(q):
    if isinstance(q, dict): return (tuple(q['path']), q['shape'], q['cmd'], q['blend'] if not isinstance(q['blend'], str) else _BLN[q['blend']])
    return (tuple(q[0]), q[1], q[2], q[8])


_BLN = {'normal': 0, 'multiply': 3, 'screen': 4, 'add': 8}


@functools.lru_cache(maxsize=None)
def clip_layers(file, export):
    """(kind, {layer key: class}) of a clip, decided once over its sampled animation (24 updates from up to 6 start
    frames spread over what rand() can give, flight 0..1000 permille, every special-visual state), so a layer keeps one class on every frame.
    A layer key is (slot path, shape, command, blend); its opacity is the MEDIAN multiplier alpha over the frames it is
    drawn. kind:
      'paint'  some layer paints (normal / multiply, opaque texels, median alpha >= BODY_ALPHA): those layers are body
      'light'  none paints, some light layer is hard (opaque texels, median alpha >= BODY_ALPHA): those are body
      'none'   neither (type_kind decides: 'glow' with a 3D mesh, else 'wave')"""
    import projdraw as PD, projtab
    T = projtab.table(); t = next((k for k, r in T.items() if r['file'] == file and (r['blue'] == export or r['red'] == export)), None)
    if t is None: return 'paint', {}
    own = T[t]['blue'] == export
    D0 = PD.Display(t, own, ground=False)
    if D0.sim is None: return 'paint', {}
    rng = PD.start_range(T[t], D0.S, D0.cid); m = max(rng) if isinstance(rng, tuple) else rng
    if T[t]['random_start'] not in (1, 2, 3, 4): m = max(1, D0.S.clips[D0.cid]['nframes'])
    A = {}   # every start frame up to 6, else 6 spread over the range (named children: each variant child is seen)
    for sf in sorted({(i * m) // min(m, 6) for i in range(min(m, 6))}):
        for ss in ((0, 1, 2) if T[t]['special_visual'] else (0,)):
            D = PD.Display(t, own, start_frame=sf, ground=False)
            for k in range(24):
                D.update(angle=0, travel=min(1000, 45 * k), special_state=ss, pos=(0, 0, 0))
                for q in D.polys(): A.setdefault(_layer_key(q), []).append(int(q[6][3]))
    def opaque_bmp(key): return bitmap(file, key[1], key[2])['opaque'] >= OPAQUE_SHARE
    med = {k: float(np.median(v)) for k, v in A.items()}
    paint = {k for k in A if k[3] in (0, 3) and med[k] >= BODY_ALPHA and opaque_bmp(k)}
    hard = {k for k in A if k[3] in (4, 8) and med[k] >= BODY_ALPHA and opaque_bmp(k)}
    if paint: kind, body = 'paint', paint
    elif hard: kind, body = 'light', hard
    else: kind, body = 'none', set()
    return kind, {k: 'body' if k in body else 'glow' for k in A}


def clip_kind(file, export):
    return clip_layers(file, export)[0]


@functools.lru_cache(maxsize=None)
def type_kind(t, own=True):
    """kind of what type t draws for its own / enemy team: the clip's 'paint' / 'light', or for a clip with neither
      'glow'  the type has a 3D mesh: the clip is all glow, the mesh is the body
      'wave'  no mesh (melee waves, gadget pulses): the drawing IS the hitbox (round 2, the user's decision): every
              layer is body, labelled by coverage (bulletlabels: every pixel the wave changes by >= 1/255)"""
    import projtab
    r = projtab.table()[t]; exp = r['blue'] if own else r['red']
    k = clip_kind(r['file'], exp) if (r['file'] and exp) else 'none'
    if k != 'none': return k
    return 'glow' if (r['blue_scw'] or r['red_scw']) else 'wave'


# types whose every normal / multiply layer is body (soft-textured rings painted over the ground are the bullet):
# only Lola's attack (every skin; Lolla003 is a wave), the user's decision 2026-10-18; others when asked
# {type: shape ids kept out (they stay glow)}: GiftsToy's trailing smear (shape 0) is not the button (user 2026-10-18)
NORMAL_IS_BODY = {6001087: set(), 6001091: set(), 6001095: set(), 6001097: set(), 6001099: set(), 6001101: {0}}
# layers labelled by their bitmap's shape, the colour transform's fade ignored (bulletlabels.unfaded): {type: shape ids}.
# Lola's default attack: its 3 trailing rings (shape 28, layer alpha 46 / 138 / 207) are all in the label (user 2026-10-18)
UNFADE = {6001087: {28}}


def layer_class(t, own, q):
    """'body', 'glow' or 'wave' (body, labelled by coverage) of one drawn clip layer of type t (own: the viewer's team):
    the layer's class decided over the clip's animation (clip_layers); a layer the sampling never saw falls back to the
    same rule on this frame"""
    import projtab
    r = projtab.table()[t]; file = r['file']; exp = r['blue'] if own else r['red']
    tk = type_kind(t, own)
    if tk == 'wave': return 'wave'
    if t in NORMAL_IS_BODY and _layer_key(q)[3] in PAINT_BLENDS and _fields(q)[0] not in NORMAL_IS_BODY[t]: return 'body'
    c = clip_layers(file, exp)[1].get(_layer_key(q))
    if c is not None: return c
    if paints(file, q): return 'body'
    if hard_light(file, q) and tk == 'light': return 'body'
    return 'glow'


def mesh_class(material):
    """'body' or 'glow' of a 3D mesh primitive by its material (glb.py: blendMode 1 / 6 additive)"""
    return 'glow' if (material or {}).get('blendMode') in (1, 6) else 'body'


def kinds_table(out_csv):
    """every type's clip kind (blue and red clips) -> CSV: id, name, file, blue, red, kind_blue, kind_red, mesh"""
    import csv, projtab
    T = projtab.table(); w = csv.writer(open(out_csv, 'w', newline=''))
    w.writerow(['id', 'name', 'file', 'blue', 'red', 'kind_blue', 'kind_red', 'mesh'])
    for t, r in sorted(T.items()):
        if not r['file']: continue
        kb = kr = ''
        try: kb = type_kind(t, True) if r['blue'] else ''
        except Exception as ex: kb = 'error'
        try: kr = type_kind(t, False) if r['red'] else ''
        except Exception as ex: kr = 'error'
        w.writerow([t, r['name'], r['file'], r['blue'], r['red'], kb, kr, int(bool(r['blue_scw'] or r['red_scw']))])


def split_table(out_csv):
    """per type (blue clip): share of the drawn picture in body and glow layers over 12 updates (flight 0..1000
    permille): area (strip triangles, clip px) x bitmap coverage (opaque + soft / 2) x multiplier alpha, per class"""
    import csv, projtab, projdraw as PD
    T = projtab.table(); w = csv.writer(open(out_csv, 'w', newline='')); w.writerow(['id', 'name', 'kind', 'mesh', 'body', 'glow']); done = {}
    for t, r in sorted(T.items()):
        if not r['file'] or not r['blue']: continue
        key = (r['file'], r['blue'], type_kind(t, True))
        if key not in done:
            acc = {'body': 0.0, 'glow': 0.0}
            try:
                D = PD.Display(t, True, ground=False)
                for k in range(12 if D.sim is not None else 0):
                    D.update(angle=0, travel=min(1000, 80 * k), pos=(0, 0, 0))
                    for q in D.polys():
                        xy = q[5]; area = sum(0.5 * abs((xy[i + 1][0] - xy[i][0]) * (xy[i + 2][1] - xy[i][1]) - (xy[i + 1][1] - xy[i][1]) * (xy[i + 2][0] - xy[i][0])) for i in range(len(xy) - 2))
                        b = bitmap(r['file'], q[1], q[2]); cov = b['opaque'] + 0.5 * b['soft']
                        c = layer_class(t, True, q); acc['body' if c == 'wave' else c] += area * cov * int(q[6][3]) / 255.0
            except Exception:
                pass
            tot = acc['body'] + acc['glow']; done[key] = (acc['body'] / tot, acc['glow'] / tot) if tot else (0.0, 0.0)
        bf, gf = done[key]
        w.writerow([t, r['name'], type_kind(t, True), int(bool(r['blue_scw'] or r['red_scw'])), round(bf, 3), round(gf, 3)])


if __name__ == '__main__':
    import sys
    if sys.argv[1:2] == ['table']: kinds_table(sys.argv[2])
    if sys.argv[1:2] == ['split']: split_table(sys.argv[2])
