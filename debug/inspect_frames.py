#!/usr/bin/env python3
"""Show what the labeller does on some frames of a capture: raw | labels (one outline per label, in its class colour), plus a table
of every label (object id, part, class, type name, brawler, team, live or ended, pixels).
  python3 debug/inspect_frames.py CAPTURE --ticks 300-320 [--out sheet.png] [--crop] [--scale 2]
  python3 debug/inspect_frames.py CAPTURE --frames 2159,2163
Ticks / frames are in the label JSON of a dataset (labels/*.json: 'tick', 'frame') or in the review pages' row headers."""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C


def rng(s):
    out = set()
    for part in filter(None, s.split(',')):
        a, _, b = part.partition('-'); out |= set(range(int(a), int(b or a) + 1))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('capture'); ap.add_argument('--ticks', default=''); ap.add_argument('--frames', default='')
    ap.add_argument('--out', default='inspect.png'); ap.add_argument('--crop', action='store_true', help='crop to the labels (+40 px)')
    ap.add_argument('--scale', type=int, default=1); a = ap.parse_args()
    import numpy as np
    from PIL import Image, ImageDraw
    T, F = rng(a.ticks), rng(a.frames)
    if not T and not F: sys.exit('give --ticks or --frames')
    rows = []
    for f, s, objs, P, img in C.frames(a.capture, lambda f, t: (t in T) or (f in F)):
        ids = P['ids']; ok = (ids >= 0) & (np.round(P['share'] * 255) >= 128); M = P.get('meta') or {}
        live = {int(p[0]) for p in s['proj']} if len(s['proj']) else set()
        own = {o['id']: o['own'] for o in objs}
        o = img.copy(); print('frame %d tick %d' % (f, s['tick']))
        for k, lid in enumerate(P['lids']):
            m = ok & (ids == k)
            if not m.any(): continue
            me = M.get(lid[0]) or {}; code = C.KIND_CODE.get(me.get('kind'), 1)
            o[C.edge(m)] = C.KIND_COLOR[code]
            st = '' if lid[2] not in ('body', 'ground') else ('live' if int(lid[0]) in live else 'ENDED')
            print('  label %2d  obj %-9d %-6s %-10s %-28s %-12s %-6s %-5s %6d px' % (k, lid[0], lid[2], me.get('kind'), me.get('name'), me.get('brawler'),
                  {True: 'own', False: 'enemy'}.get(own.get(lid[0]), '-'), st, m.sum()))
        r = np.concatenate([img, np.full((img.shape[0], 4, 3), 255, np.uint8), o], 1)
        if a.crop and ok.any():
            ys, xs = np.nonzero(ok); y0, y1 = max(0, ys.min() - 40), min(img.shape[0], ys.max() + 41); x0, x1 = max(0, xs.min() - 40), min(img.shape[1], xs.max() + 41)
            r = np.concatenate([img[y0:y1, x0:x1], np.full((y1 - y0, 4, 3), 255, np.uint8), o[y0:y1, x0:x1]], 1)
        I = Image.fromarray(r); I = I.resize((I.width * a.scale, I.height * a.scale), Image.NEAREST)
        ImageDraw.Draw(I).text((4, 2), 'frame %d tick %d' % (f, s['tick']), fill=(255, 255, 0)); rows.append(np.asarray(I))
    if not rows: sys.exit('no labelled frame matched')
    W = max(r.shape[1] for r in rows)
    Image.fromarray(np.concatenate([np.pad(r, ((0, 6), (0, W - r.shape[1]), (0, 0))) for r in rows], 0)).save(a.out); print('->', a.out)


if __name__ == '__main__':
    main()
