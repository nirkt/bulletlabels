"""Schema-less flatbuffer explorer: table fields with raw offsets, guesses of vectors / strings / subtables."""
import struct


def table(b, pos):
    """field offsets (absolute positions or None) of the table at pos"""
    vt = pos - struct.unpack_from('<i', b, pos)[0]
    vs, ts = struct.unpack_from('<HH', b, vt)
    return [(pos + struct.unpack_from('<H', b, vt + 4 + 2 * i)[0]) if struct.unpack_from('<H', b, vt + 4 + 2 * i)[0] else None
            for i in range((vs - 4) // 2)], ts


def deref(b, a):
    return a + struct.unpack_from('<I', b, a)[0]


def vec(b, a):
    """(start, count) of the vector referenced by the uoffset at a"""
    t = deref(b, a); return t + 4, struct.unpack_from('<I', b, t)[0]


def sizes(b, pos):
    """field byte widths guessed from the gaps between field positions inside the table"""
    f, ts = table(b, pos)
    ps = sorted((p, i) for i, p in enumerate(f) if p is not None) + [(pos + ts, -1)]
    return {i: ps[k + 1][0] - p for k, (p, i) in enumerate(ps[:-1])}


def describe(b, pos, depth=0, maxd=2, maxv=3):
    f, ts = table(b, pos); sz = sizes(b, pos); out = []
    for i, p in enumerate(f):
        if p is None: continue
        w = sz.get(i, 4); raw = b[p:p + w]
        s = '%s[%d] @%d w%d raw=%s' % ('  ' * depth, i, p - pos, w, raw.hex())
        if w == 4:
            v = struct.unpack_from('<I', b, p)[0]
            t = p + v
            if 0 < t < len(b) - 4:
                n = struct.unpack_from('<I', b, t)[0]
                if 0 < n < 10 ** 7 and t + 4 + n <= len(b):
                    s += ' -> vec/str n=%d head=%r' % (n, b[t + 4:t + 4 + min(n, 24)])
        out.append(s)
    return '\n'.join(out)
