"""The client's own per-draw ID pass (rendclone glhost_id_rec/idpass.c, NBGL_ID_DIR): per dumped frame f%07d
  _id.z  NBID uint16 (h, w) id of the front-most non-additive draw (0 = nothing; draw ids are 1-based, see _draws.jsonl)
  _cv.z  NBCV uint8 (h, w) coverage (alpha of that fragment for blended draws, 255 opaque)
  _uv.z  NBUV float32 (h/2, w/2, 2) sprite atlas position (v_texCoord) of that fragment, -1 = none / not a sprite
  _wd.z  NBWD float32 (h/2, w/2) eye depth
  _draws.jsonl  per draw: id, prog, tex0, blend, bs, bd, th (texture content hash), tw, tht (texture size)
  _meta.json    frame (swap number), w, h, cam
  load(dir, frame) -> dict(id, cv, uv, wd, draws {id: row}, meta)"""
import json, os, zlib
import numpy as np


def _z(fn, dtype, ch=1):
    b = open(fn, 'rb').read(); w, h = np.frombuffer(b[4:12], np.int32)
    a = np.frombuffer(zlib.decompress(b[12:]), dtype)
    return a.reshape((h, w, ch) if ch > 1 else (h, w))


def frames(d):
    return sorted(int(f[1:8]) for f in os.listdir(d) if f.endswith('_meta.json'))


def load(d, f):
    p = os.path.join(d, 'f%07d' % f); out = dict(meta=json.load(open(p + '_meta.json')), dir=d)   # dir: the ID-pass folder (capture/id/<run>/)
    out['id'] = _z(p + '_id.z', np.uint16); out['cv'] = _z(p + '_cv.z', np.uint8)
    if os.path.exists(p + '_uv.z'): out['uv'] = _z(p + '_uv.z', np.float32, 2)
    if os.path.exists(p + '_wd.z'): out['wd'] = _z(p + '_wd.z', np.float32)
    out['draws'] = {}
    if os.path.exists(p + '_draws.jsonl'):
        for l in open(p + '_draws.jsonl'):
            if l.strip(): r = json.loads(l); out['draws'][r['id']] = r
    return out
