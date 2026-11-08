"""Readers for replay --draws dumps (gen/dump.c) and the matching collector data (game state per gl_swap)."""
import json, os, sys, glob
import numpy as np

EB = os.path.dirname(os.path.abspath(__file__))   # vendor/ (state2.py)
if EB not in sys.path:
    sys.path.append(EB)

GL_T = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5124: np.int32, 5125: np.uint32, 5126: np.float32,
        5131: np.float16}


class Dump:
    def __init__(self, d):
        self.d = d
        self._blob = {}

    def frames(self):
        return sorted(int(os.path.basename(p)[1:]) for p in glob.glob(os.path.join(self.d, 'f[0-9]*')))

    def frame(self, f):
        """(draws, events): draws = draw dicts in order; events = all records (fbdesc / clear / draw) in order"""
        p = os.path.join(self.d, 'f%07d' % f, 'draws.jsonl')
        if os.path.exists(p + '.gz'):
            import gzip
            ev = [json.loads(l) for l in gzip.open(p + '.gz', 'rt')]
        else:
            ev = [json.loads(l) for l in open(p)]
        return [e for e in ev if 'i' in e], ev

    def blob(self, key):
        if not hasattr(self, '_idx'):
            self._idx = {}
            ip = os.path.join(self.d, 'blobs.idx')
            if os.path.exists(ip):
                for L in open(ip):
                    k, o, n = L.split(); self._idx[k] = (int(o), int(n))
                self._dat = np.memmap(os.path.join(self.d, 'blobs.dat'), np.uint8, 'r')
        if key in self._idx:
            o, n = self._idx[key]; return self._dat[o:o + n].tobytes()
        return open(os.path.join(self.d, 'blobs', key + '.bin'), 'rb').read()

    def attr(self, a):
        """attribute rows as an array (rows, size) of the GL type (not normalised)"""
        raw = np.frombuffer(self.blob(a['data']), np.uint8)[a.get('doff', 0):]
        dt = np.dtype(GL_T[a['type']]); n = a['size']; st = a['stride']; rows = a['rows']
        need = (rows - 1) * st + dt.itemsize * n
        raw = raw[:need]
        out = np.lib.stride_tricks.as_strided(raw, shape=(rows, dt.itemsize * n), strides=(st, 1)).copy()
        return out.view(dt).reshape(rows, n)

    def ubo(self, d, name, dtype=np.float32):
        b = d['ub'].get(name)
        if b is None or b['data'] == '0' * 16: return None
        return np.frombuffer(self.blob(b['data']), dtype)

    def indices(self, d):
        if not d.get('idx'): return None
        dt = np.uint16 if d['itype'] == 5123 else (np.uint8 if d['itype'] == 5121 else np.uint32)
        io = d.get('idoff', 0)
        return np.frombuffer(self.blob(d['idx'])[io:io + d['count'] * np.dtype(dt).itemsize], dt)


def tex_key(d, name):
    v = d['u'].get(name)
    return v['x'] if isinstance(v, dict) else None


def mat4(f16):
    """GL column-major 16 floats -> numpy 4x4 (row-major math)"""
    return np.asarray(f16, np.float64).reshape(4, 4).T


class Battle:
    """Collector output of one battle: meta + states per frame, indexed by gl_swap."""
    def __init__(self, bdir):
        import state2 as S2
        self.S2 = S2; self.dir = bdir
        self.meta = {}; self._st = {}
        for cj in sorted(glob.glob(os.path.join(bdir, 'chunk*.json'))):
            m = json.load(open(cj))['meta']
            z = np.load(cj.replace('.json', '.states.npz'))
            st, off = z['states'], z['offsets']
            for i, mm in enumerate(m):
                self.meta[mm['gl_swap']] = mm
                self._st[mm['gl_swap']] = (st, off, i)
        self.map = json.load(open(os.path.join(bdir, 'map.json'))) if os.path.exists(os.path.join(bdir, 'map.json')) else None
        self.tiles = np.load(os.path.join(bdir, 'map.npy')) if os.path.exists(os.path.join(bdir, 'map.npy')) else None

    def state(self, f):
        if f not in self._st: return None
        st, off, i = self._st[f]
        return self.S2.parse(st[off[i]:off[i + 1]].tobytes())
