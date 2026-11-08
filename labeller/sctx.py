"""Supercell textures `assets/sc3d/*.sctx` (3D model textures referenced as '<name>.pvr' by materials):
  u32 52, header flatbuffer ('SCTX'; fields 2 width, 3 height, 4 format byte, 6 level count, 7 decoded size),
  u32 decoded size, u32 n, n bytes of size-prefixed level records (flatbuffers: 0 width, 1 height, 2 offset into the
  decoded data), one zstd frame = the ASTC mip chain. The ASTC block size follows from level 0's size
  (= the next level's offset): ceil(w / bw) ceil(h / bh) 16 bytes [V: 6x6 for primo_005 1024^2, 8x8 for 8bit 512^2,
  4x4 for 128^2 textures; the decoded opacity map shows the expected shapes].
  rgba = texture('primo_005_atk_projectile_alpha')   (h, w, 4) uint8, level 0, rows top-down"""
import os, struct
import numpy as np
import rc

_C = {}
BLOCKS = [(4, 4), (5, 4), (5, 5), (6, 5), (6, 6), (8, 5), (8, 6), (8, 8), (10, 5), (10, 6), (10, 8), (10, 10), (12, 10), (12, 12)]


def _tbl(b, pos):
    vt = pos - struct.unpack_from('<i', b, pos)[0]; vs = struct.unpack_from('<H', b, vt)[0]
    return [(pos + o) if o else None for o in (struct.unpack_from('<H', b, vt + 4 + 2 * i)[0] for i in range((vs - 4) // 2))]


def path_of(name):
    """material texture name ('x.pvr#repeat', 'sc3d/x.sctx#repeat', ...) -> the .sctx file (None for an empty name)"""
    n = name.split('#')[0]
    if not n: return None
    for ext in ('.pvr', '.png', '.ktx', '.sctx'):
        if n.endswith(ext): n = n[:-len(ext)]
    if n.startswith('sc3d/'): n = n[5:]
    return os.path.join(rc.ASSETS, 'sc3d', n + '.sctx')


def texture(name):
    p = path_of(name)
    if p is None: return None
    if p in _C: return _C[p]
    import zstandard, texture2ddecoder
    d = open(p, 'rb').read()
    b = d[4:]; r = struct.unpack_from('<I', b, 0)[0]; f = _tbl(b, r)
    w = struct.unpack_from('<H', b, f[2])[0]; h = struct.unpack_from('<H', b, f[3])[0]
    tot, n = struct.unpack_from('<II', d, 52); q = 60; offs = []
    while q + 4 <= 60 + n:
        k = struct.unpack_from('<I', d, q)[0]; rb = d[q + 4:q + 4 + k]; rf = _tbl(rb, struct.unpack_from('<I', rb, 0)[0])
        offs.append(struct.unpack_from('<I', rb, rf[2])[0] if len(rf) > 2 and rf[2] is not None else 0); q += 4 + k
    z = zstandard.ZstdDecompressor().decompressobj().decompress(d[60 + n:])
    size0 = (sorted(set(offs) - {0}) or [len(z)])[0]
    bw, bh = next(((x, y) for x, y in BLOCKS if -(-w // x) * -(-h // y) * 16 == size0), (None, None))
    if bw is None: raise ValueError('no ASTC block size fits %s (%d x %d, %d bytes)' % (p, w, h, size0))
    img = np.frombuffer(texture2ddecoder.decode_astc(bytes(z[:size0]), w, h, bw, bh), np.uint8).reshape(h, w, 4)[:, :, [2, 1, 0, 3]].copy()
    _C[p] = img
    return img
