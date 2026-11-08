#!/usr/bin/env python3
"""Try a label fix before installing it: label a capture with the installed rules and again with a candidate fix file added, and
render every frame whose labels change: raw | before (red outlines) | after (yellow outlines), plus pixel counts.
  python3 debug/compare_fix.py CAPTURE my_fix.py [--out compare.png] [--max 24]
  python3 debug/compare_fix.py CAPTURE my_fix.py --skip z_1120_rules.py      (also leave an installed fix out: testing a replacement)
The candidate is exec'd after labeller/fixes/*.py in the labeller's namespace (IDLABELS_OVERRIDE), exactly as an installed fix would be.
A fix is ready when it changes the frames it should and NOTHING else: run it on several other skins' captures too."""
import argparse, os, pickle, subprocess, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))


def dump(cap, out, env):
    code = r'''
import sys, pickle, numpy as np
sys.path.insert(0, %r); import _common as C
res = {}
for f, s, objs, P, img in C.frames(%r):
    m = (P['ids'] >= 0) & (np.round(P['share'] * 255) >= 128)
    res[f] = (int(s['tick']), np.where(m, P['ids'], -1).astype(np.int16), [tuple(l) for l in P['lids']], img)
pickle.dump(res, open(%r, 'wb'))
''' % (HERE, cap, out)
    r = subprocess.run([sys.executable, '-c', code], env=env, capture_output=True, text=True)
    if r.returncode: sys.exit(r.stderr[-3000:])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('capture'); ap.add_argument('fix'); ap.add_argument('--skip', default='')
    ap.add_argument('--out', default='compare.png'); ap.add_argument('--max', type=int, default=24); a = ap.parse_args()
    import numpy as np
    from PIL import Image, ImageDraw
    sys.path.insert(0, HERE); import _common as C
    td = tempfile.mkdtemp()
    base = dict(os.environ); new = dict(os.environ, IDLABELS_OVERRIDE=os.path.abspath(a.fix))
    if a.skip: new['IDLABELS_SKIP'] = a.skip
    dump(a.capture, os.path.join(td, 'a.pkl'), base); dump(a.capture, os.path.join(td, 'b.pkl'), new)
    A = pickle.load(open(os.path.join(td, 'a.pkl'), 'rb')); B = pickle.load(open(os.path.join(td, 'b.pkl'), 'rb'))
    rows = []; tot_add = tot_rem = nchg = 0
    for f in sorted(A):
        tk, ia, la, img = A[f]; _, ib, lb, _ = B[f]
        ma, mb = ia >= 0, ib >= 0; add = int((mb & ~ma).sum()); rem = int((ma & ~mb).sum())
        if not add and not rem and np.array_equal(ia, ib): continue
        nchg += 1; tot_add += add; tot_rem += rem
        print('frame %d tick %d: +%d -%d px%s' % (f, tk, add, rem, '' if add or rem else ' (only the instance split changed)'))
        if len(rows) >= a.max: continue
        ys, xs = np.nonzero(ma | mb); y0, y1 = max(0, ys.min() - 30), min(img.shape[0], ys.max() + 31); x0, x1 = max(0, xs.min() - 30), min(img.shape[1], xs.max() + 31)
        def ov(ids, col):
            o = img.copy()
            for k in np.unique(ids[ids >= 0]): o[C.edge(ids == k)] = col
            return o[y0:y1, x0:x1]
        sep = np.full((y1 - y0, 4, 3), 255, np.uint8)
        r = np.concatenate([img[y0:y1, x0:x1], sep, ov(ia, (255, 0, 0)), sep, ov(ib, (255, 230, 0))], 1)
        I = Image.fromarray(r); ImageDraw.Draw(I).text((3, 2), 'tick %d +%d -%d' % (tk, add, rem), fill=(255, 255, 255)); rows.append(np.asarray(I))
    print('%d of %d frames change, +%d -%d px' % (nchg, len(A), tot_add, tot_rem))
    if rows:
        W = max(r.shape[1] for r in rows)
        Image.fromarray(np.concatenate([np.pad(r, ((0, 6), (0, W - r.shape[1]), (0, 0))) for r in rows], 0)).save(a.out); print('->', a.out)


if __name__ == '__main__':
    main()
