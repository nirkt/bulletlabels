"""Texture atlases of the game's sprite files (chunk 6 of SC v5/v6): KTX 1.1, ASTC 8x8 (glInternalFormat 0x93B7), each
texture stored twice (high and low resolution). The client uploads the HIGH-resolution copy unchanged (glCompressedTexImage2D)
[V: the content hashes of the textures the client samples (rendclone gen/dump.c hash64 of the level-0 upload, seed
w << 32 | h) equal those of the high-res KTX payloads: effects_brawler.sc 4096x4096 803c9bc31fb38fea, 2560x4064
55c149b3fb7e09fa, effects.sc 2782x3960 96e4e8f34ba6d8ba]. Texture index k of a shape command = k-th texture in chunk 6.
  rgba = texture(S, k)  -> (h, w, 4) uint8, row 0 = v 0 (top), PREMULTIPLIED as stored
  a = sample_alpha(rgba, uv_u16)"""
import re, struct
import numpy as np

_KTX = re.compile(rb'\xabKTX 11\xbb\r\n\x1a\n')
import collections
_CACHE = collections.OrderedDict()   # decoded atlases (64 MB each at 4096 x 4096): the CACHE_N most recently used
CACHE_N = 6
# ASTC glInternalFormat -> block size (RGBA 0x93B0.., sRGB 0x93D0..)
ASTC = {0x93B0 + i: b for i, b in enumerate([(4, 4), (5, 4), (5, 5), (6, 5), (6, 6), (8, 5), (8, 6), (8, 8), (10, 5), (10, 6),
                                            (10, 8), (10, 10), (12, 10), (12, 12)])}
ASTC.update({f + 0x20: b for f, b in list(ASTC.items())})


def ktx_list(S):
    """[(w, h, ifmt, payload bytes)] in file order"""
    c6 = S.ch[6] if len(S.ch) > 6 else b''; out = []
    for m in _KTX.finditer(c6):
        a = m.start(); h = struct.unpack_from('<13I', c6, a + 12); kv = h[12]; p = a + 64 + kv
        n = struct.unpack_from('<I', c6, p)[0]
        out.append((h[6], h[7], h[4], c6[p + 4:p + 4 + n]))
    return out


def textures_hi(S):
    """the uploaded (high-resolution) copy of each texture: texture k is the k-th of the root's texture vector; its two
    copies are adjacent in the chunk, the larger one is the high-res copy"""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'framegen'))
    import fbx
    c6 = S.ch[6]; r = struct.unpack_from('<I', c6, 0)[0]
    st, n = fbx.vec(c6, r + 4)
    K = ktx_list(S); pos = [m.start() for m in _KTX.finditer(c6)]
    out = []
    for k in range(n):
        t = fbx.deref(c6, st + 4 * k)
        # the copies of texture k lie after its table (flatbuffers write children first: the copies precede the next table)
        nxt = min([fbx.deref(c6, st + 4 * j) for j in range(n) if fbx.deref(c6, st + 4 * j) > t] + [len(c6)])
        cand = [i for i, p in enumerate(pos) if t < p < nxt] or [i for i, p in enumerate(pos) if p > t][:2]
        best = max(cand, key=lambda i: K[i][0] * K[i][1])
        out.append(K[best])
    return out


def texture(S, k):
    key = (S.path, k)
    if key not in _CACHE:
        import texture2ddecoder
        w, h, f, d = textures_hi(S)[k]
        assert f in ASTC, hex(f)
        bgra = texture2ddecoder.decode_astc(d, w, h, *ASTC[f])
        a = np.frombuffer(bgra, np.uint8).reshape(h, w, 4)
        _CACHE[key] = a[:, :, [2, 1, 0, 3]].copy()
        while len(_CACHE) > CACHE_N: _CACHE.popitem(last=False)
    _CACHE.move_to_end(key)
    return _CACHE[key]


def sample_alpha(tex, uv):
    """bilinear (GL_LINEAR) alpha at normalised u16 texture coordinates"""
    h, w = tex.shape[:2]
    u = np.asarray(uv, np.float64)[..., 0] / 65535.0 * w - 0.5; v = np.asarray(uv, np.float64)[..., 1] / 65535.0 * h - 0.5
    x0 = np.floor(u).astype(int); y0 = np.floor(v).astype(int); fx = u - x0; fy = v - y0
    A = tex[:, :, 3].astype(np.float64)
    g = lambda y, x: A[np.clip(y, 0, h - 1), np.clip(x, 0, w - 1)]
    return (g(y0, x0) * (1 - fx) * (1 - fy) + g(y0, x0 + 1) * fx * (1 - fy) + g(y0 + 1, x0) * (1 - fx) * fy + g(y0 + 1, x0 + 1) * fx * fy) / 255.0
