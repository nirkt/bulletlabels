"""Decoder for the game's sprite files assets/sc/*.sc, format version 6 (and 5), following the client's own loaders in
libg.so (module addresses, see UNDERSTANDING.md 3):
  file      = "SC" u16 version, u16 0, u32 header length, header flatbuffer (export name -> 16-byte id, counts),
              zstd frame z0 = size-prefixed flatbuffer chunks (0 data storage, 1 exports, 2 modifiers, 3 shapes,
              4 movie clips, 5 -, 6 textures), 1 byte, u32 n + flatbuffer of matrix-bank descriptors (0xceab74 ->
              0xcc1454), then one zstd frame per matrix bank (streamed, 0xcc0ffc).
  clip      = MovieClipOriginal flatbuffer table (0xcb30c0): f0 id, f1 name, f2 fps, f3 frame count, f4 flag,
              f5 child ids, f6 child names, f7 child blend bytes, f8 frames (u32 element count, u32 label),
              f9 / f13 inline frame elements, f10 matrix bank, f11 scaling grid, f12 streamed frame-element offset.
  bank      = [N_float x 24 B float 2x3][N_blocks x u32 block descriptors][N_u16 x u16 short matrices + delta
              streams][N_colour x 7 B colour transforms] .. frames region at its own offset (0xcc11d4).
  matrix i  = float (i < N_float), short (i < N_float + N_short: a, b, c, d / 1024, tx, ty * 0.05) or a delta
              compressed block of 16 (0xcc09dc -> 0xcbf4bc).
  frame     = element triples (child index, matrix index, colour index) delta-coded (0xcc0cc8), applied by 0xcb16dc.
  colour    = 7 bytes: r, g, b, a multipliers, r, g, b additions (element bytes 9..15).
Matrix convention (Flash): x' = a x + c y + tx, y' = b x + d y + ty.
  S = SC6(path); S.exports[name] -> id; S.clip(id) -> dict; S.frame(id, f) -> [(child id, matrix (6,), colour (7,), blend)]
  S.shape(id) -> [(texture index, uv (n,2) u16, xy (n,2) f32)]; S.draw(name, frames) -> polygons of a clip frame."""
import os, struct
import numpy as np


def _tbl(b, pos):
    vt = pos - struct.unpack_from('<i', b, pos)[0]
    vs = struct.unpack_from('<H', b, vt)[0]
    return [(pos + o) if o else None for o in (struct.unpack_from('<H', b, vt + 4 + 2 * i)[0] for i in range((vs - 4) // 2))]


def _f(b, pos, k):
    f = _tbl(b, pos)
    return f[k] if k < len(f) else None


def _u(b, pos, k, fmt='<I', default=None):
    p = _f(b, pos, k)
    return struct.unpack_from(fmt, b, p)[0] if p is not None else default


def _vec(b, pos, k):
    p = _f(b, pos, k)
    if p is None: return None
    t = p + struct.unpack_from('<I', b, p)[0]
    return t + 4, struct.unpack_from('<I', b, t)[0]


def _tables(b, pos, k):
    v = _vec(b, pos, k)
    if v is None: return []
    return [v[0] + 4 * i + struct.unpack_from('<I', b, v[0] + 4 * i)[0] for i in range(v[1])]


def _s32(x):
    x &= 0xffffffff
    return x - (1 << 32) if x & 0x80000000 else x


def _sx(x, left, right):
    """the client's '(int)(x << left) >> right' on a 32-bit value"""
    return _s32(x << left) >> right


BLEND = {0: 'normal', 3: 'multiply', 4: 'screen', 8: 'add', 12: 'erase', 15: 'erase2'}   # 0x1fe7bc table -> GL modes


class Bank:
    def __init__(self, raw, desc):
        self.raw = raw; self.d = desc
        nb, nu16, nshort, nfloat, ncol, foff = desc['blocks'], desc['u16'], desc['short'], desc['float'], desc['colours'], desc['frames']
        self.floats = np.frombuffer(raw, '<f4', nfloat * 6, 0).reshape(-1, 6)
        p = nfloat * 24
        self.blocks = np.frombuffer(raw, '<u4', nb, p); p += nb * 4
        self.u16_off = p; self.u16 = np.frombuffer(raw, '<u2', nu16, p); p += nu16 * 2
        self.colours = np.frombuffer(raw, np.uint8, ncol * 7, p).reshape(-1, 7)
        self.frames_off = foff
        self.nfloat, self.nshort = nfloat, nshort
        self._blk = {}

    def _short(self, i):
        """short matrix i (0-based in the u16 region, 6 x int16)"""
        return self.u16[6 * i:6 * i + 6].astype(np.int16).astype(np.int64)

    def _block(self, blk):
        """16 short matrices of compressed block blk (0xcbf4bc)"""
        if blk in self._blk: return self._blk[blk]
        if blk * 16 < self.nfloat + self.nshort:   # stored, not compressed
            i0 = blk * 16 - self.nfloat
            out = [self._short(i0 + k) for k in range(16)]
        else:
            u = self.u16; desc = int(self.blocks[blk]); base = desc & 0x1fff; p = desc >> 13
            v = [int(x) for x in self._short(base)]
            out = []
            for _ in range(16):
                c = int(u[p]); a, b, cc, d, tx, ty = v
                if c & 3 == 0:
                    tx += _sx(c, 23, 25); ty += _sx(c, 16, 25); p += 1
                else:
                    c1 = int(u[p + 1]); c2 = int(u[p + 2]) if p + 2 < len(u) else 0; k = c & 0xf
                    if k == 1:
                        tx += _sx(c << 14 | c1 << 30, 0, 18); ty += _sx(c1, 16, 18); p += 2
                    elif k == 2:
                        a += _sx(c, 21, 25); tx += _sx(c1, 23, 25); d += _sx(c << 14 | c1 << 30, 0, 25); ty += _sx(c1, 16, 25); p += 2
                    elif k == 3:
                        a += _sx(c, 17, 21); d += _sx(c << 6 | c1 << 22, 0, 21); tx += _sx(c1 << 11 | c2 << 27, 0, 21); ty += _sx(c2, 16, 21); p += 3
                    elif k == 5:
                        tx += _s32(c2 & 0xff) - (256 if c2 & 0x80 else 0) if False else ((c2 & 0xff) - 256 if c2 & 0x80 else (c2 & 0xff))
                        a += _sx(c, 21, 25); cc += _sx(c1, 23, 25); ty += _sx(c2, 16, 24); b += _sx(c << 14 | c1 << 30, 0, 25); d += _sx(c1, 16, 25); p += 3
                    elif k == 6:
                        c3 = int(u[p + 3])
                        a += _sx(c, 18, 22); b += _sx(c << 8 | c1 << 24, 0, 22); cc += _sx(c1 << 14 | c2 << 30, 0, 22); d += _sx(c2, 20, 22)
                        tx += _sx(c2 << 10 | c3 << 26, 0, 22); ty += _sx(c3, 16, 22); p += 4
                    elif k == 7:
                        c3 = int(u[p + 3]); c4 = int(u[p + 4])
                        a += _sx(c, 16, 20); b += _sx(c1, 20, 20); cc += _sx(c1 << 8 | c2 << 24, 0, 20); d += _sx(c2 << 12 | c3 << 28, 0, 20)
                        tx += _sx(c3 << 14 | c4 << 30, 0, 18); ty += _sx(c4, 16, 18); p += 5
                    elif k == 15:
                        a += c1; b += c2; cc += int(u[p + 3]); d += int(u[p + 4]); tx += int(u[p + 5]); ty += int(u[p + 6]); p += 7
                    else:
                        raise ValueError('matrix delta command %d at %d' % (k, p))
                v = [((x + 0x8000) & 0xffff) - 0x8000 for x in (a, b, cc, d, tx, ty)]   # stored as int16
                out.append(np.array(v, np.int64))
        self._blk[blk] = out
        return out

    def matrix(self, i):
        """(a, b, c, d, tx, ty) float32 like the client (0xcc09dc)"""
        if i < self.nfloat: return self.floats[i].astype(np.float64)
        if i < self.nfloat + self.nshort: s = self._short(i - self.nfloat)
        else: s = self._block(i >> 4)[i & 15]
        m = np.empty(6, np.float32); m[:4] = np.float32(s[:4]) * np.float32(0.0009765625); m[4:] = np.float32(s[4:]) * np.float32(0.05)
        return m.astype(np.float64)

    def frame_elements(self, off, f):
        """element triples of frame f of a clip whose streamed frame data starts at off (0xcb16dc + 0xcc0cc8)"""
        raw = self.raw; base = self.frames_off + off
        e = struct.unpack_from('<Q', raw, base + 8 + 8 * f)[0]
        start = base + (e & 0xffffffff); na = (e >> 32) & 0xffff; nb = e >> 48
        ref = np.frombuffer(raw, '<u2', na, start)          # reference triples
        cmd = np.frombuffer(raw, '<u2', nb - na, start + 2 * na)   # command stream
        if na == 0: return [tuple(int(x) for x in cmd[i:i + 3]) for i in range(0, len(cmd) - len(cmd) % 3, 3)]
        out = []; r = 0; c = 0; run = 0

        def t(k): return int(ref[r + k])
        while c < len(cmd):
            if run & 1:
                out.append((t(0), t(1), t(2))); r += 3; run >>= 1; continue
            x = int(cmd[c]); k = x & 7
            if x & 3 == 0:
                out.append((t(0), (t(1) + _sx(x, 23, 25)) & 0xffff, t(2)))
                out.append((t(3), (t(4) + (_s32(x << 16) >> 25)) & 0xffff, t(5)))
                r += 6; c += 1; run >>= 2
            elif k == 1:
                out.append((t(0), (t(1) + (_s32(x << 16) >> 19)) & 0xffff, t(2))); r += 3; c += 1; run >>= 1
            elif k == 2:
                out.append((t(0), (t(1) + _sx(x, 25, 28)) & 0xffff, (t(2) + (_s32(x << 16) >> 23)) & 0xffff)); r += 3; c += 1; run >>= 1
            elif k == 3:
                out.append((t(0), (t(1) + int(cmd[c + 1])) & 0xffff, (t(2) + (_s32(x << 16) >> 19)) & 0xffff)); r += 3; c += 2; run >>= 1
            elif k == 5:
                run = x >> 3; out.append((t(0), t(1), t(2))); r += 3; c += 1
            elif k == 6:
                r += (_s32(x << 16) >> 19) * 3; c += 1; run >>= 1
            elif k == 7:
                out.append(((_s32(x << 17) >> 20) & 0xffff, int(cmd[c + 1]), int(cmd[c + 2])))
                if x & 0x8000: r += 3
                c += 3; run >>= 1
        while run:   # trailing copy run
            if r + 3 > len(ref): break
            out.append((t(0), t(1), t(2))); r += 3; run >>= 1
        return out


class SC6:
    def __init__(self, path):
        import zstandard
        self.path = path
        d = open(path, 'rb').read(); self.d = d
        ver = struct.unpack_from('<H', d, 2)[0]
        if d[:2] != b'SC' or ver not in (5, 6): raise ValueError('not a v5/v6 sc file: %s' % path)
        hl = struct.unpack_from('<I', d, 6 if ver == 5 else 8)[0]
        s0 = (10 if ver == 5 else 12) + hl
        o = zstandard.ZstdDecompressor().decompressobj(); z = o.decompress(d[s0:]); rest = o.unused_data
        ch = []; p = 0
        while p + 4 <= len(z):
            n = struct.unpack_from('<I', z, p)[0]
            if p + 4 + n > len(z): break
            ch.append(z[p + 4:p + 4 + n]); p += 4 + n
        self.ch = ch
        root = lambda c: struct.unpack_from('<I', c, 0)[0]
        c0 = ch[0]; r0 = root(c0); self.c0, self.r0 = c0, r0
        self.strings = [c0[t + 4:t + 4 + struct.unpack_from('<I', c0, t)[0]].decode('utf8', 'replace') for t in _tables(c0, r0, 0)]
        pv = _vec(c0, r0, 5)
        self.points = np.frombuffer(c0, np.dtype([('x', '<f4'), ('y', '<f4'), ('u', '<u2'), ('v', '<u2')]), count=pv[1] // 12, offset=pv[0]) if pv else None
        v4 = _vec(c0, r0, 4); self.fe16 = np.frombuffer(c0, '<u2', v4[1], v4[0]) if v4 else None   # inline frame elements (u16)
        v7 = _vec(c0, r0, 7); self.fe32 = np.frombuffer(c0, '<u4', v7[1], v7[0]) if v7 else None
        c1 = ch[1]; r1 = root(c1); iv = _vec(c1, r1, 0); nv = _vec(c1, r1, 1)
        ids = np.frombuffer(c1, '<u2', iv[1], iv[0]) if iv else []; refs = np.frombuffer(c1, '<u4', nv[1], nv[0]) if nv else []
        self.exports = {self.strings[int(r)]: int(i) for i, r in zip(ids, refs)}
        self.shapes = {}
        c3 = ch[3]
        for t in _tables(c3, root(c3), 0):
            sid = _u(c3, t, 0, '<H', 0); cv = _vec(c3, t, 1); cmds = []
            if cv:
                for k in range(cv[1]):
                    _, tex, n, off = struct.unpack_from('<4I', c3, cv[0] + 16 * k); q = self.points[off:off + n]
                    cmds.append((int(tex), np.stack([q['u'], q['v']], 1).copy(), np.stack([q['x'], q['y']], 1).astype(np.float64)))
            self.shapes[sid] = cmds
        self.clips = {}
        c4 = ch[4]
        for t in _tables(c4, root(c4), 0):
            cid = _u(c4, t, 0, '<H', 0)
            kv = _vec(c4, t, 5); bv = _vec(c4, t, 7); fv = _vec(c4, t, 8); nmv = _vec(c4, t, 6)
            frames = [struct.unpack_from('<II', c4, fv[0] + 8 * i) for i in range(fv[1])] if fv else []
            self.clips[cid] = dict(
                id=cid, name=self._str(_u(c4, t, 1)), fps=(_u(c4, t, 2, '<B', 0) & 0x7f), nframes=_u(c4, t, 3, '<H', 0),
                flag=_u(c4, t, 4, '<B', 0), children=(np.frombuffer(c4, '<u2', kv[1], kv[0]).tolist() if kv else []),
                child_names=([self._str(x) for x in np.frombuffer(c4, '<u4', nmv[1], nmv[0])] if nmv else None),
                blend=(np.frombuffer(c4, np.uint8, bv[1], bv[0]).tolist() if bv else []),
                frames=[(int(np.int16(n & 0xffff)), self._str(lb)) for n, lb in frames],
                inline16=_u(c4, t, 9), bank=_u(c4, t, 10, '<I', 0) & 0x7f, grid=_u(c4, t, 11), stream=_u(c4, t, 12, '<I', None),
                inline32=_u(c4, t, 13))
        # matrix-bank descriptors after z0 (1 byte, u32 length, flatbuffer)
        self.banks_desc = []
        if len(rest) > 5:
            pos = len(d) - len(rest); pad = (-pos) % 4   # the descriptor length is 4-byte aligned in the file
            n = struct.unpack_from('<I', rest, pad)[0]; b = rest[pad + 4:pad + 4 + n]; self.stream0 = pos + pad + 4 + n
            names = ('index', 'offset', 'csize', 'dsize', 'blocks', 'u16', 'short', 'float', 'colours', 'frames', 'frames_size', 'flag', 'x12')
            for t in _tables(b, struct.unpack_from('<I', b, 0)[0], 0):
                f = _tbl(b, t)
                self.banks_desc.append({nm: (struct.unpack_from('<I', b, f[k])[0] if k < len(f) and f[k] is not None else 0) for k, nm in enumerate(names)})
        self._banks = {}

    def _str(self, i):
        return self.strings[i] if i else None

    def bank(self, k):
        if k not in self._banks:
            import zstandard
            ds = self.banks_desc[k]; a = self.stream0 + ds['offset']
            raw = zstandard.ZstdDecompressor().decompressobj().decompress(self.d[a:a + ds['csize']])
            assert len(raw) == ds['dsize'], (k, len(raw), ds['dsize'])
            self._banks[k] = Bank(raw, ds)
        return self._banks[k]

    def inline_elements(self, c, f):
        """frame elements stored in the data storage (v5 / small files): f9 -> u16 triples (storage field 4), f13 -> u32
        triples (storage field 7); frame k starts after the element counts of frames 0..k-1 (0xcb30c0)"""
        s = sum(n for n, _ in c['frames'][:f]); n = c['frames'][f][0]
        if c['inline32'] is not None and self.fe32 is not None: a = self.fe32[3 * (c['inline32'] + s):3 * (c['inline32'] + s + n)]
        else: a = self.fe16[(c['inline16'] or 0) + 3 * s:(c['inline16'] or 0) + 3 * (s + n)]
        return [tuple(int(x) for x in a[i:i + 3]) for i in range(0, len(a), 3)]

    def inline_bank(self):
        """matrices / colours of files without streamed banks: storage fields (see _storage_bank)"""
        if getattr(self, '_ib', None) is None: self._ib = _storage_bank(self)
        return self._ib

    def kind(self, i):
        return 'shape' if i in self.shapes else ('clip' if i in self.clips else 'other')

    def frame(self, cid, f):
        """[(child slot, child id, matrix (6,) or None, colour (7,) or None, blend byte)] of frame f of clip cid"""
        c = self.clips[cid]
        B = self.bank(c['bank']) if self.banks_desc else None; out = []
        if c['stream'] is not None: els = B.frame_elements(c['stream'], f)
        else: els = self.inline_elements(c, f)
        if B is None: B = self.inline_bank()
        for slot, mi, ci in els:
            out.append((slot, c['children'][slot], None if mi == 0xffff else B.matrix(mi), None if ci == 0xffff else B.colours[ci].copy(),
                        c['blend'][slot] if slot < len(c['blend']) else 0))
        return out

    def draw(self, cid, frame_of, m=None, col=None, path=(), out=None):
        """flatten clip cid: frame_of(path, clip dict) -> frame index of that (sub)clip instance; returns polygons
        [(path, shape id, command index, texture, uv (n,2), xy (n,2) in the root clip's space, colour (mul4, add3), blend)]"""
        if out is None: out = []
        if m is None: m = np.array([1, 0, 0, 1, 0, 0], np.float64)
        if col is None: col = (np.ones(4), np.zeros(3))
        c = self.clips[cid]
        if c['nframes'] == 0: return out
        f = frame_of(path, c)
        for slot, child, cm, cc, bl in self.frame(cid, f):
            if bl & 0x40: continue   # invisible child
            mm = m if cm is None else _mul(m, cm)
            cl = col if cc is None else _cmul(col, cc)
            if child in self.shapes:
                for k, (tex, uv, xy) in enumerate(self.shapes[child]):
                    out.append((path + (slot,), child, k, tex, uv, _apply(mm, xy), cl, bl & 0x3f))
            elif child in self.clips:
                self.draw(child, frame_of, mm, cl, path + (slot,), out)
        return out


def _mul(p, c):
    """parent * child (both a, b, c, d, tx, ty)"""
    a, b, cc, d, tx, ty = p; A, B, C, D, TX, TY = c
    return np.array([a * A + cc * B, b * A + d * B, a * C + cc * D, b * C + d * D, a * TX + cc * TY + tx, b * TX + d * TY + ty])


def _apply(m, xy):
    a, b, c, d, tx, ty = m
    return np.stack([a * xy[:, 0] + c * xy[:, 1] + tx, b * xy[:, 0] + d * xy[:, 1] + ty], 1)


def _cmul(col, c7):
    """concatenate colour transforms (parent (mul4, add3) with a child 7-byte transform), approximately as Flash does"""
    mul, add = col; cm = c7[:4] / 255.0; ca = c7[4:7].astype(np.float64)
    return (mul * cm, add + mul[:3] * ca)


class _InlineBank:
    """matrices and colour transforms stored in the data storage for files without streamed banks (storage field 6
    table: f0 float matrices 24 B each, f1 colour transforms 7 B each, f2 short matrices 12 B each: a, b, c, d / 1024,
    tx, ty * 0.05; a bank holds either floats (type 0) or shorts (type 1), 0xcb16dc)"""
    def __init__(self, floats, colours, shorts):
        self.floats = floats; self.colours = colours; self.shorts = shorts; self.nfloat = len(floats); self.nshort = len(shorts)

    def matrix(self, i):
        if len(self.floats): return self.floats[i].astype(np.float64)
        s = self.shorts[i]
        m = np.empty(6, np.float32); m[:4] = np.float32(s[:4]) * np.float32(0.0009765625); m[4:] = np.float32(s[4:]) * np.float32(0.05)
        return m.astype(np.float64)


def _storage_bank(S):
    c0, r0 = S.c0, S.r0
    banks = _tables(c0, r0, 6)
    floats = np.zeros((0, 6), np.float32); cols = np.zeros((0, 7), np.uint8); shorts = np.zeros((0, 6), np.int16)
    if banks:
        t = banks[0]
        v = _vec(c0, t, 0)
        if v: floats = np.frombuffer(c0, '<f4', v[1] * 6, v[0]).reshape(-1, 6)
        v = _vec(c0, t, 1)
        if v: cols = np.frombuffer(c0, np.uint8, v[1] * 7, v[0]).reshape(-1, 7)
        v = _vec(c0, t, 2)
        if v: shorts = np.frombuffer(c0, '<i2', v[1] * 6, v[0]).reshape(-1, 6)
    return _InlineBank(floats, cols, shorts)
