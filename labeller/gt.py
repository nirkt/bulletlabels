"""Ground truth from the client's own draws: sprite quads of a dumped frame (rendclone's replay --draws dumps), their
world vertices and uv keys, matched to the shape commands of the game's sprite files (sc6.py).
  q = quads_of_frame(D, f) -> list of dict(key, P (n,3) world, uv (n,2) u16, col (n,4), col2 (n,4), blend, tex, draw, layer)
  index_clip(S, cid) -> {uv key: [(path..)]}"""
import numpy as np
import rc, dumps, quads as Q

_SC = {}


def sc(name):
    import sc6, os
    if name not in _SC: _SC[name] = sc6.SC6(os.path.join(rc.ASSETS, name))
    return _SC[name]


def raw_rows(D, d):
    a = d['a']['a_position']
    raw = np.frombuffer(D.blob(a['data']), np.uint8)[a.get('doff', 0):]
    st = a['stride']; rows = a['rows']; need = rows * st
    if len(raw) < need: raw = np.concatenate([raw, np.zeros(need - len(raw), np.uint8)])
    return raw[:need].reshape(rows, st)


def quads_of_frame(D, f, world_only=True):
    draws, _ = D.frame(f); out = []; nd = len(draws)
    for d in draws:
        if not Q.is_sprite(d): continue
        M = Q.camera_of(D, d)
        if world_only and not (M is not None and abs(M[3, 1]) > 1e-3): continue
        try:
            rows = raw_rows(D, d); qs = Q.quads(D, d)
        except Exception:
            continue
        st = d['a']['a_position']['stride']
        if st < 24: continue
        for q in qs:
            vr = rows[q['v']]
            out.append(dict(key=np.ascontiguousarray(vr[:, 12:16]).tobytes(), P=q['p'].astype(np.float64),
                            uv=vr[:, 12:16].copy().view('<u2').reshape(-1, 2), col=vr[:, 16:20].copy(), col2=vr[:, 20:24].copy(),
                            blend=tuple(d['blend']), draw=d['i'], layer=d['i'] / max(1, nd), cam=M, tex=Q.sprite_tex(d)))
    return out


def key_of_uv(uv):
    return np.ascontiguousarray(np.asarray(uv, '<u2')).tobytes()


def affine_fit(xy, P):
    """exact affine map local 2D -> world 3D: P = [x y 1] @ A (A 3x3); returns A, max residual"""
    X = np.c_[xy, np.ones(len(xy))]
    A, *_ = np.linalg.lstsq(X, P, rcond=None)
    return A, float(np.abs(X @ A - P).max())


class GTFile:
    """compact ground truth written by tools/gtcap.py (same quad dicts as quads_of_frame)"""
    def __init__(self, path):
        z = np.load(path, allow_pickle=False)
        self.z = {k: z[k] for k in z.files}
        fr = self.z['frame']; self._idx = {}
        order = np.argsort(fr, kind='stable')
        for f in np.unique(fr):
            self._idx[int(f)] = order[np.searchsorted(fr[order], f, 'left'):np.searchsorted(fr[order], f, 'right')]
        self.battle = str(self.z['battle']); self.rec = str(self.z['rec'])
        self.cams = {int(f): self.z['cams'][i].reshape(4, 4).T for i, f in enumerate(self.z['frames'])}
        self.blends = [tuple(b) for b in self.z['blends']]

    def frames(self):
        return sorted(self._idx)

    def quads(self, f):
        z = self.z; out = []
        for i in self._idx.get(f, []):
            s, n = int(z['start'][i]), int(z['npts'][i])
            uv = z['uv'][s:s + n]
            out.append(dict(key=np.ascontiguousarray(uv.astype('<u2')).tobytes(), P=z['pts'][s:s + n].astype(np.float64), uv=uv,
                            col=z['col'][s:s + n], col2=z['col2'][s:s + n], blend=self.blends[int(z['blend'][i])],
                            draw=int(z['draw'][i]), layer=int(z['draw'][i]) / max(1, int(z['ndraw'][i])), tex=int(z['tex'][i])))
        return out
