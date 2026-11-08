"""The game's 3D models (assets/sc3d/*.glb): glTF 2.0 binary containers whose JSON chunk (type 'FLA2') is the glTF
document as a FLATBUFFER (fields in alphabetical glTF order) with a Supercell extension `SC_odin_format` stored as
FlexBuffers (loader 0xd605b0, which looks the keys up by name). Read here:
  root fields   0 accessors, 2 asset, 3 bufferViews, 4 buffers, 6 extensions (FlexBuffer), 7/8 extensionsRequired/Used,
                12 meshes, 13 nodes, 17 skins
  accessor      0 bufferView, 1 byteOffset, 2 componentType (u16), 3 count;  bufferView 0 buffer, 1 byteLength, 2 byteOffset
  primitive     1 extensions {SC_odin_format: {meshDataInfoIndex}}, 3 indices (accessor), 4 material
  node          1 children, 2 extensions {SC_odin_format: {parent}}, 4 matrix, 5 mesh, 6 name, 7 rotation (x y z w),
                8 scale, 9 skin, 10 translation
  skin          0 extensions {SC_odin_format: {bounds}}, 2 inverseBindMatrices (accessor, float32 4 x 4 column-major), 3 joints
  SC_odin_format (root): materials [{blendMode, constants, shader, variables{floats{cutout}, textures}}],
                meshDataInfos [{vertexDescriptors [{offset, stride, attributes [{name, format, offset}]}]}] (offsets
                into the BIN chunk), animation {frameRate, keyframeCount, packed{...}} (not decoded here)
Vertex formats: 30 = float32 x 3 (a_pos), 12 = int8 x 4 normalised (a_normal), 22 = int16 x 2 normalised (a_uv0) [V: the client
uploads the BIN chunk unchanged as the vertex buffer (rendclone dump blob e0c95b7d6758dbbf = hash64(BIN, 0x5eed) of
brock_renamed_rocket_blue.glb) and reads it with exactly these GL formats and offsets].
Packed node animation (SC_odin_format.animation.packed, loader 0xd68740..0xd691f0, sampler 0xd45ff8): anim_track().
Skinned vertices (a_boneindex format 3 = uint8 x 4, a_boneweights format 36 = one uint32): skin_positions().
  M = GLB(path); M.nodes, M.primitives(node) -> [(positions (n,3), triangles (m,3), material)], M.node_matrix(i)"""
import struct, json
import numpy as np

FMT = {30: ('<f4', 3), 12: ('<i1', 4), 22: ('<i2', 2),   # [V] the captured GL attribute formats of the same buffer
       3: ('<u1', 4), 36: ('<u4', 1),                     # a_boneindex (integer), a_boneweights (packed integer) [V captured]
       29: ('<f4', 2)}   # [R] the enum runs in groups of four (ubyte 0-3, byte 9-12, short 21-24, float 28-31): float32 x 2
                         # (a_uv0 of a few round-2 models, e.g. the wave roster's); not normalised


def _tbl(b, pos):
    vt = pos - struct.unpack_from('<i', b, pos)[0]
    vs = struct.unpack_from('<H', b, vt)[0]
    return [(pos + o) if o else None for o in (struct.unpack_from('<H', b, vt + 4 + 2 * i)[0] for i in range((vs - 4) // 2))]


def _vec(b, p):
    t = p + struct.unpack_from('<I', b, p)[0]
    return t + 4, struct.unpack_from('<I', b, t)[0]


def _tables(b, p):
    if p is None: return []
    s, n = _vec(b, p)
    return [s + 4 * i + struct.unpack_from('<I', b, s + 4 * i)[0] for i in range(n)]


def _f(f, k):
    return f[k] if k < len(f) else None


def _flex(b, p):
    if p is None: return None
    import flatbuffers.flexbuffers as fx
    s, n = _vec(b, p)
    return fx.GetRoot(bytearray(b[s:s + n])).Value


def _floats(b, p):
    if p is None: return None
    s, n = _vec(b, p)
    return np.frombuffer(b, '<f4', n, s).astype(np.float64)


class GLB:
    def __init__(self, path):
        d = open(path, 'rb').read(); self.path = path
        assert d[:4] == b'glTF', path
        n, t = struct.unpack_from('<II', d, 12); J = d[20:20 + n]; p = 20 + n
        nb, tb = struct.unpack_from('<II', d, p); self.BIN = d[p + 8:p + 8 + nb]; self.J = J
        r = struct.unpack_from('<I', J, 0)[0]; f = _tbl(J, r)
        self.ext = (_flex(J, _f(f, 6)) or {}).get('SC_odin_format') or {}
        self.views = []
        for v in _tables(J, _f(f, 3)):   # bufferViews (ASCII order puts them before 'buffers'): 0 buffer, 1 byteLength, 2 byteOffset
            vf = _tbl(J, v)
            g = lambda k: struct.unpack_from('<I', J, _f(vf, k))[0] if _f(vf, k) is not None else 0
            self.views.append(dict(buffer=g(0), byteLength=g(1), byteOffset=g(2)))
        self.accessors = []
        for a in _tables(J, _f(f, 0)):
            af = _tbl(J, a)
            g = lambda k, fmt='<I', dflt=0: struct.unpack_from(fmt, J, _f(af, k))[0] if _f(af, k) is not None else dflt
            bv = g(0); off = g(1) + (self.views[bv]['byteOffset'] if bv < len(self.views) else 0)
            self.accessors.append(dict(bufferView=bv, byteOffset=off, componentType=g(2, '<H', 5123), count=g(3)))
        self.meshes = []
        for m in _tables(J, _f(f, 12)):
            mf = _tbl(J, m); prims = []
            for pr in _tables(J, _f(mf, 3)):
                pf = _tbl(J, pr); e = (_flex(J, _f(pf, 1)) or {}).get('SC_odin_format') or {}
                g = lambda k: struct.unpack_from('<I', J, _f(pf, k))[0] if _f(pf, k) is not None else 0
                prims.append(dict(info=e.get('meshDataInfoIndex', 0), indices=g(3), material=g(4)))
            self.meshes.append(prims)
        self.nodes = []
        for nd in _tables(J, _f(f, 13)):
            nf = _tbl(J, nd); e = (_flex(J, _f(nf, 2)) or {}).get('SC_odin_format') or {}
            name = None
            if _f(nf, 6) is not None: s, k = _vec(J, nf[6]); name = J[s:s + k].decode('utf8', 'replace')
            ch = []
            if _f(nf, 1) is not None: s, k = _vec(J, nf[1]); ch = np.frombuffer(J, '<i4', k, s).tolist()
            self.nodes.append(dict(name=name, mesh=struct.unpack_from('<I', J, nf[5])[0] if _f(nf, 5) is not None else None,
                                   skin=struct.unpack_from('<I', J, nf[9])[0] if _f(nf, 9) is not None else None,
                                   parent=(e or {}).get('parent') if isinstance(e, dict) else None, children=ch,
                                   matrix=_floats(J, _f(nf, 4)), rotation=_floats(J, _f(nf, 7)), scale=_floats(J, _f(nf, 8)),
                                   translation=_floats(J, _f(nf, 10))))
        self.skins = []
        for sk in _tables(J, _f(f, 17)):
            sf = _tbl(J, sk); joints = []
            if _f(sf, 3) is not None: s_, k = _vec(J, sf[3]); joints = np.frombuffer(J, '<i4', k, s_).tolist()
            ibm = np.tile(np.eye(4), (len(joints), 1, 1))
            if _f(sf, 2) is not None:
                a = self.accessors[struct.unpack_from('<I', J, sf[2])[0]]
                ibm = np.frombuffer(self.BIN, '<f4', 16 * len(joints), a['byteOffset']).reshape(-1, 4, 4).transpose(0, 2, 1).astype(np.float64)
            self.skins.append(dict(joints=joints, ibm=ibm))
        for i, nd in enumerate(self.nodes):   # parents also from children lists
            for c in nd['children']:
                if self.nodes[c]['parent'] is None: self.nodes[c]['parent'] = i

    def local_matrix(self, i):
        nd = self.nodes[i]
        if nd['matrix'] is not None and len(nd['matrix']) == 16: return nd['matrix'].reshape(4, 4).T   # glTF column-major
        M = np.eye(4)
        if nd['scale'] is not None: M = np.diag(list(nd['scale']) + [1.0]) @ M
        if nd['rotation'] is not None:
            x, y, z, w = nd['rotation']; n = np.sqrt(x * x + y * y + z * z + w * w); x, y, z, w = x / n, y / n, z / n, w / n
            R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                          [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                          [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
            M[:3, :] = R @ M[:3, :]
        if nd['translation'] is not None: M[:3, 3] += nd['translation']
        return M

    def node_matrix(self, i):
        """model-space matrix of node i (parents applied), without animation"""
        M = self.local_matrix(i); p = self.nodes[i]['parent']; seen = {i}
        while p is not None and p not in seen:
            seen.add(p); M = self.local_matrix(p) @ M; p = self.nodes[p]['parent']
        return M

    def stream(self, info, name, raw_int=False):
        """one vertex attribute of a meshDataInfo as (n, k) float64, or in its stored integer type (raw_int)"""
        for vd in self.ext['meshDataInfos'][info]['vertexDescriptors']:
            for a in vd['attributes']:
                if a['name'] == name:
                    dt, k = FMT[a['format']]; isz = np.dtype(dt).itemsize
                    ends = [v['offset'] for mi in self.ext['meshDataInfos'] for v in mi['vertexDescriptors'] if v['offset'] > vd['offset']]
                    end = min(ends + [len(self.BIN)])
                    raw = np.frombuffer(self.BIN, np.uint8, end - vd['offset'], vd['offset'])
                    n = len(raw) // vd['stride']
                    rows = raw[:n * vd['stride']].reshape(n, vd['stride'])[:, a['offset']:a['offset'] + isz * k]
                    out = np.ascontiguousarray(rows).view(dt).reshape(n, k)
                    return out if raw_int else out.astype(np.float64)
        return None

    def attr_format(self, info, name):
        for vd in self.ext['meshDataInfos'][info]['vertexDescriptors']:
            for a in vd['attributes']:
                if a['name'] == name: return a['format']
        return None

    def skinned(self, info):
        return any(a['name'] == 'a_boneindex' for vd in self.ext['meshDataInfos'][info]['vertexDescriptors'] for a in vd['attributes'])

    def primitives(self, node):
        """[(positions (n,3) in the node's own space (bind pose when skinned), triangles (m,3) int, material dict,
        uv (n,2) or None, meshDataInfo index)] of a node's mesh"""
        nd = self.nodes[node]
        if nd['mesh'] is None: return []
        out = []
        for pr in self.meshes[nd['mesh']]:
            ac = self.accessors[pr['indices']]
            dt = {5121: '<u1', 5123: '<u2', 5125: '<u4'}[ac['componentType']]
            idx = np.frombuffer(self.BIN, dt, ac['count'], ac['byteOffset']).astype(np.int64)
            P = self.stream(pr['info'], 'a_pos')
            if P is None: continue
            UV = self.stream(pr['info'], 'a_uv0')
            if UV is not None and self.attr_format(pr['info'], 'a_uv0') == 22: UV = np.maximum(UV / 32767.0, -1.0)   # GL normalised int16
            mats = self.ext.get('materials') or []
            out.append((P, idx.reshape(-1, 3), mats[pr['material']] if pr['material'] < len(mats) else {}, UV, pr['info']))
        return out


def _ncomp(flags):
    """components per keyframe besides time (0xdaae84): rotation 4 (bit 1), translation 3 (bit 2), scale 1 / 3 (bit 3,
    bit 4), bit 0 one more"""
    n = (flags & 1) + 4 * ((flags >> 1) & 1) + (3 if flags & 4 else 0)
    if flags & 8: n += 3 if flags & 0x10 else 1
    return n


class Track:
    """one node's animation: rest pose (translation, rotation quaternion x y z w, scale), quanta, keyframe rows
    (time int16 over 0..32767 = keyframes 0..K-1, then the components)"""
    def __init__(self, T, Rq, S, tq, sq, flags, rows):
        self.T = np.asarray(T, np.float32); self.Rq = Rq; self.S = np.asarray(S, np.float32)
        self.tq = np.float32(tq); self.sq = np.float32(sq); self.flags = flags; self.rows = rows   # rows (K, 1 + ncomp) int16 or None

    def sample(self, t):
        """(translation, quaternion xyzw, scale) at time t in keyframes (0xd45ff8)"""
        f32 = np.float32; T = self.T.copy(); S = self.S.copy(); q = _qnorm(self.Rq.astype(np.float32))
        if self.rows is None or len(self.rows) < 2 or t <= 0: return T, q, S
        K = len(self.rows); end = f32(K - 1); fl = self.flags
        def comps(r):
            out = {}; k = 1
            if fl & 2: out['r'] = r[k:k + 4].astype(np.float32); k += 4
            if fl & 4: out['t'] = r[k:k + 3].astype(np.float32); k += 3
            if fl & 8:
                if fl & 0x10: out['s'] = r[k:k + 3].astype(np.float32); k += 3
                else: out['s'] = np.repeat(r[k:k + 1].astype(np.float32), 3); k += 1
            return out
        if t >= end:
            c = comps(self.rows[-1])
            if 'r' in c: q = _qnorm(c['r'])
            if 't' in c: T = T + c['t'] * self.tq
            if 's' in c: S = S + c['s'] * self.sq
            return T, q, S
        key = int(f32(f32(t) / end) * f32(32767.0)); times = self.rows[:, 0].astype(np.int64)
        i = int(np.searchsorted(times, key, 'right')); i = min(max(i, 1), K - 1)
        t0 = end * f32(3.051851e-05) * f32(times[i - 1]); t1 = end * f32(3.051851e-05) * f32(times[i])
        a = f32((f32(t) - t0) / (t1 - t0)) if t1 != t0 else f32(0.0)
        c0 = comps(self.rows[i - 1]); c1 = comps(self.rows[i])
        if 'r' in c0:
            r0, r1 = c0['r'], c1['r']
            q = _qnorm(r0 + (r1 - r0) * a if float(np.dot(r0, r1)) >= 0 else -r0 + (r1 + r0) * a)
        if 't' in c0: T = T + (c0['t'] + (c1['t'] - c0['t']) * a) * self.tq
        if 's' in c0: S = S + (c0['s'] + (c1['s'] - c0['s']) * a) * self.sq
        return T, q, S


def _qnorm(q):
    n = float(np.sqrt(np.float32(np.dot(q, q))))
    return (q / n).astype(np.float32) if n > 0 else np.array([0, 0, 0, 1], np.float32)


def _rle(data, p, size, ncomp, K):
    """0xdaaec0 -> 0xdaaed4..: count c > 0: c rows of ncomp values; c < 0: -c copies of the previous row"""
    rows = []; end = p + size; last = None
    while p < end:
        c = int(np.int16(data[p])); p += 1
        if c >= 1:
            for _ in range(c):
                last = data[p:p + ncomp].astype(np.int16); p += ncomp; rows.append(last)
        elif c < 0:
            for _ in range(-c): rows.append(last)
    f32 = np.float32; s0 = f32(32767.0) / f32(K - 1) if K > 1 else f32(0)
    out = np.zeros((len(rows), 1 + ncomp), np.int16)
    for k, r in enumerate(rows):
        out[k, 0] = np.int16(int(f32(f32(k) * s0))); out[k, 1:] = r
    return out


def anim_tracks(M):
    """{node index: Track} of a GLB's packed animation, plus (frameRate, keyframeCount, firstFrame, lastFrame)"""
    an = (M.ext or {}).get('animation')
    if not an or 'packed' not in an: return {}, None
    pk = an['packed']; K = int(an['keyframeCount'])
    def acc_bytes(i, dt):
        a = M.accessors[i]; n = a['count']
        isz = np.dtype(dt).itemsize
        return np.frombuffer(M.BIN, dt, n * (1 if dt != '<f4' else 1), a['byteOffset'])
    data = np.frombuffer(M.BIN, '<u2', M.accessors[pk['dataAccessor']]['count'], M.accessors[pk['dataAccessor']]['byteOffset'])
    na = M.accessors[pk['nodeAccessor']]
    rec = np.frombuffer(M.BIN, '<f4', na['count'], na['byteOffset'])
    ua = pk.get('uintAccessor')
    uq = None
    if ua is not None and M.accessors[ua]['count']:
        a = M.accessors[ua]; uq = np.frombuffer(M.BIN, '<i2', a['count'], a['byteOffset']).reshape(-1, 4)
    tracks = {}; p = 0; r = 0; un = 0
    for nd in pk['nodes']:
        if uq is None:
            x = rec[r:r + 12]; r += 12
            T, Rf, S, tq, sq = x[0:3], x[3:7], x[7:10], x[10], x[11]
            Rq = np.clip(np.round(Rf.astype(np.float32) * np.float32(32767.0)), -32767, 32767).astype(np.int16)
        else:
            x = rec[r:r + 8]; r += 8
            T, S, tq, sq = x[0:3], x[3:6], x[6], x[7]; Rq = uq[un]; un += 1
        fl = int(nd.get('flags', 0))
        if fl == 0:
            tracks[nd['nodeIndex']] = Track(T, Rq.astype(np.float32) * np.float32(3.051851e-05), S, tq, sq, 0, None); continue
        nc = _ncomp(fl); size = int(nd['dataSize'])
        rows = _rle(data, p, size, nc, K); p += size
        tracks[nd['nodeIndex']] = Track(T, Rq.astype(np.float32) * np.float32(3.051851e-05), S, tq, sq, fl | 1, rows)
    fv = lambda k, d: float(an.get(k)) if an.get(k) is not None else d   # (null frameRate: 30, what every other file has [?])
    return tracks, (fv('frameRate', 30.0), K, fv('firstFrame', 0.0), fv('lastFrame', K - 1.0))


def trs_matrix(T, q, S):
    x, y, z, w = [float(v) for v in q]
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    M = np.eye(4); M[:3, :3] = R * np.asarray(S, float)[None, :]; M[:3, 3] = T
    return M


_CACHE = {}


def load(path):
    """(GLB, tracks, anim info) cached per file"""
    if path not in _CACHE:
        M = GLB(path); tr, info = anim_tracks(M); _CACHE[path] = (M, tr, info)
    return _CACHE[path]


def _mat_quat(R):
    """unit quaternion (x, y, z, w) of a rotation matrix"""
    t = np.trace(R)
    if t > 0:
        r = np.sqrt(1 + t) * 2; return np.array([(R[2, 1] - R[1, 2]) / r, (R[0, 2] - R[2, 0]) / r, (R[1, 0] - R[0, 1]) / r, r / 4])
    i = int(np.argmax(np.diag(R))); j, k = (i + 1) % 3, (i + 2) % 3
    r = np.sqrt(1 + R[i, i] - R[j, j] - R[k, k]) * 2; q = np.zeros(4)
    q[i] = r / 4; q[j] = (R[j, i] + R[i, j]) / r; q[k] = (R[k, i] + R[i, k]) / r; q[3] = (R[k, j] - R[j, k]) / r
    return q


def local_trs(M, tracks, j, t):
    """(translation, quaternion x y z w, scale) of node j at time t: its track, else its rest TRS (a rest matrix is
    split into T, R, S)"""
    if j in tracks: return tuple(np.asarray(x, np.float64) for x in tracks[j].sample(t))
    nd = M.nodes[j]
    if nd['matrix'] is not None and len(nd['matrix']) == 16:
        X = nd['matrix'].reshape(4, 4).T; S = np.linalg.norm(X[:3, :3], axis=0)
        return X[:3, 3].copy(), _mat_quat(X[:3, :3] / S), S
    return (np.zeros(3) if nd['translation'] is None else nd['translation'].copy(),
            np.array([0, 0, 0, 1.0]) if nd['rotation'] is None else _qnorm(nd['rotation'].astype(np.float32)).astype(np.float64),
            np.ones(3) if nd['scale'] is None else nd['scale'].copy())


def _qmul(a, b):
    x1, y1, z1, w1 = a; x2, y2, z2, w2 = b
    return np.array([w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2, w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2])


def world_trs(M, tracks, i, t):
    """the client's node world transform (0xd46280, recursive over the parents), kept as T, q, S without shear:
    T = T_p + R(q_p) (S_p * T_l), q = q_p q_l, S = S_p * S_l"""
    chain = []; p = i; seen = set()
    while p is not None and p not in seen: seen.add(p); chain.append(p); p = M.nodes[p]['parent']
    T = np.zeros(3); q = np.array([0, 0, 0, 1.0]); S = np.ones(3)
    for j in reversed(chain):
        lt, lq, ls = local_trs(M, tracks, j, t)
        T = T + trs_matrix(np.zeros(3), q, np.ones(3))[:3, :3] @ (S * lt); q = _qmul(q, lq); S = S * ls
    return T, q, S


def node_world(M, tracks, i, t):
    """model matrix of node i at animation time t (keyframes): world_trs (0xd46280) as R(q) diag(S) + T. Equals the
    plain matrix product of local TRS matrices whenever no non-uniform scale meets a rotation below it"""
    return trs_matrix(*world_trs(M, tracks, i, t))


def anim_time(info, age_s):
    """animation time (keyframes) of a model at render age age_s: firstFrame + age x frameRate [V: Buzz005Ulti
    (firstFrame 1.25) captured at 2.75 / 4.25 after 0.05 / 0.1 s; the other captured files have firstFrame 0]. The
    sampler holds the last keyframe after K - 1; what lastFrame does when it is below K - 1 is not known [?]"""
    return (info[2] if info else 0.0) + age_s * (info[0] if info else 30.0)


def unpack_weights(v):
    """a_boneweights -> (n, 4) weights of a_boneindex x y z w, as the client's skinned vertex shader computes them
    (rendclone shaders/3888440857.vs): w_y = (v >> 21) & 2047, w_z = (v >> 10) & 2047, w_w = v & 1023, each x
    0.00024420025874860584735870361328125 (float32 1 / 4095), w_x = 1 - w_y - w_z - w_w; v = 0: bone x alone"""
    v = np.asarray(v, np.uint32).reshape(-1); f32 = np.float32; c = f32(0.00024420025874860584735870361328125)
    w1 = ((v >> 21) & 2047).astype(f32) * c; w2 = ((v >> 10) & 2047).astype(f32) * c; w3 = (v & 1023).astype(f32) * c
    W = np.stack([((f32(1) - w1) - w2) - w3, w1, w2, w3], 1)
    W[v == 0] = (1, 0, 0, 0)
    return W


def bone_matrices(M, tracks, skin, t):
    """(J, 4, 4) skin matrices at animation time t: joint model matrix x inverse bind matrix, in model space.
    Joint model matrix: a joint whose parent is a joint of the same skin = parent joint's matrix x its local TRS matrix
    (a plain product: a parent's non-uniform scale shears it); any other joint = node_world (0xd46280, no shear).
    [V: captured bones of 70 brawler skins (rendclone lib/poses, same shader): rigid joint chains with the rest
    translations; Willow's super (cap_b1, non-uniform animated scale on a non-joint parent): joint 0 by node_world
    and joints 1, 2 by the product match to 1e-5, the other compositions do not]"""
    sk = M.skins[skin]; js = set(sk['joints']); W = {}
    def jw(j):
        if j not in W:
            p = M.nodes[j]['parent']
            W[j] = jw(p) @ trs_matrix(*local_trs(M, tracks, j, t)) if p in js else node_world(M, tracks, j, t)
        return W[j]
    return np.stack([jw(j) @ ib for j, ib in zip(sk['joints'], sk['ibm'])]) if sk['joints'] else np.zeros((0, 4, 4))


def skin_positions(M, info, P, B):
    """positions of a skinned meshDataInfo under skin matrices B: sum_k w_k B[idx_k] p (the shader's blend of the 3 x 4
    rows, then the product with a_pos)"""
    bi = M.stream(info, 'a_boneindex', raw_int=True); bw = M.stream(info, 'a_boneweights', raw_int=True)
    n = len(P); bi = bi[:n].astype(np.int64); W = unpack_weights(bw[:n, 0]).astype(np.float64)
    bi = np.minimum(bi, len(B) - 1)
    Bv = (B[bi] * W[:, :, None, None]).sum(1)          # (n, 4, 4) blended matrix per vertex
    return np.einsum('nij,nj->ni', Bv[:, :3, :3], P) + Bv[:, :3, 3]


def posed_triangles(path, age_s):
    """triangles (m, 3, 3) in model space of every mesh node at render age age_s (s), with per-triangle material index
    and flags: animation time = age x frameRate, held at the last keyframe [V Brock004 / Colt010 captures].
    Rigid meshes: node_world x position. Skinned meshes: the blended skin matrices (bone_matrices of the node's skin)
    x position, and the mesh node's own transform is NOT applied [V: shelly_princess_geo draws whose mesh nodes differ
    by 7 units share one u_model]; the caller applies the object matrix in both cases.
    -> [(tris (m,3,3), material dict, skinned bool, uv (m,3,2) or None)]"""
    M, tracks, info = load(path)
    t = anim_time(info, age_s)
    out = []; bones = {}
    for i, nd in enumerate(M.nodes):
        if nd['mesh'] is None: continue
        X = None
        for P, tri, mat, UV, inf in M.primitives(i):
            if M.skinned(inf) and M.skins:
                sk = nd['skin'] if nd['skin'] is not None else 0
                if sk not in bones: bones[sk] = bone_matrices(M, tracks, sk, t)
                Pw = skin_positions(M, inf, P, bones[sk]); sm = True
            else:
                if X is None: X = node_world(M, tracks, i, t)
                Pw = P @ X[:3, :3].T + X[:3, 3]; sm = False
            out.append((Pw[tri], mat, sm, UV[tri] if UV is not None else None))
    return out
