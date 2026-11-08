"""Contact sheets of the body / glow split (LABEL_SPEC 2) -> out/glow/.
  real:  per projectile type seen in real-frame captures (tools/posecap.py --pix-every: frames.npz + states), the best
         example: crop of the real 1200 x 540 frame | body share (owned >= 0.5 red) | glow share | overlay (body contour
         red, glow >= 0.5 contour green). Labels come from the states alone (track + bulletlabels), not from the frame.
  model: per sampled type with no real frame, the module's own drawing at 8 updates: composite over grey | body share |
         glow share.
Tiles sorted by glow fraction (glow share / (glow + body share)): mostly-additive types first.
  python3 glowsheet.py real DIR [DIR ...] [--out ../out/glow]
  python3 glowsheet.py model N [--out ../out/glow]      (unrecorded types: every 'light' and 'wave' kind, 10 glow-only
                                                         clips with a mesh, the N/2 most additive 'paint' types, N/2 random)"""
import argparse, collections, csv, json, os, random
import numpy as np
from PIL import Image, ImageDraw
import rc, dumps, track, projtab, projdraw as PD, bulletlabels as BL, glow, masks

TILE = 96   # tile height px


def _edge(m):
    return m & ~(np.roll(m, 1, 0) & np.roll(m, -1, 0) & np.roll(m, 1, 1) & np.roll(m, -1, 1))


def _scale(a, s):
    return np.asarray(Image.fromarray(a).resize((a.shape[1] * s, a.shape[0] * s), Image.NEAREST))


def _panels(real, body, gl):
    """uint8 panels of one crop: real (h, w, 3) or None, body / glow shares (h, w)"""
    b = np.zeros(body.shape + (3,), np.uint8); b[..., :] = (np.clip(body, 0, 1) * 200)[..., None].astype(np.uint8)
    b[body >= 0.5] = [255, 60, 60]
    g = np.zeros(gl.shape + (3,), np.uint8); g[..., 1] = (np.clip(gl, 0, 1) * 255).astype(np.uint8)
    g[..., 0] = g[..., 2] = (np.clip(gl, 0, 1) * 90).astype(np.uint8)
    out = [b, g]
    if real is not None:
        ov = real.copy(); ov[_edge(body >= 0.5)] = [255, 0, 0]; ov[_edge(gl >= 0.5)] = [0, 255, 0]
        out = [real, b, g, ov]
    return out


def _tile(panels, title, sub):
    h, w = panels[0].shape[:2]; s = max(1, min(4, TILE // max(h, 1)))
    P = [_scale(p, s) for p in panels]
    sep = np.full((P[0].shape[0], 3, 3), 40, np.uint8)
    row = np.concatenate(sum([[p, sep] for p in P], [])[:-1], 1)
    head = np.full((26, row.shape[1], 3), 20, np.uint8)
    I = Image.fromarray(np.concatenate([head, row], 0)); d = ImageDraw.Draw(I)
    d.text((3, 1), title[:60], fill=(255, 255, 0)); d.text((3, 13), sub[:70], fill=(200, 200, 200))
    return np.asarray(I)


def _pages(tiles, out, prefix, per=8, width=1400):
    """pack tiles (sorted) into pages of `per` rows packed left to right"""
    os.makedirs(out, exist_ok=True); names = []
    for p in range(0, len(tiles), per):
        rows = []; cur = []; cw = 0
        for t in tiles[p:p + per]:
            if cur and cw + t.shape[1] > width: rows.append(cur); cur = []; cw = 0
            cur.append(t); cw += t.shape[1] + 6
        if cur: rows.append(cur)
        H = sum(max(t.shape[0] for t in r) + 6 for r in rows); W = max(sum(t.shape[1] + 6 for t in r) for r in rows)
        page = np.full((H, W, 3), 10, np.uint8); y = 0
        for r in rows:
            x = 0
            for t in r: page[y:y + t.shape[0], x:x + t.shape[1]] = t; x += t.shape[1] + 6
            y += max(t.shape[0] for t in r) + 6
        nm = os.path.join(out, '%s_%02d.png' % (prefix, p // per + 1)); Image.fromarray(page).save(nm); names.append(nm)
    return names


def kinds():
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'glow', 'clip_kinds.csv')
    return {int(r['id']): r for r in csv.DictReader(open(p))} if os.path.exists(p) else {}


def real(dirs, out):
    T = projtab.table(); K = kinds(); best = {}; stat = collections.defaultdict(lambda: [0, 0.0, 0.0, 0])
    for d in dirs:
        z = np.load(os.path.join(d, 'frames.npz')); want = {int(s): i for i, s in enumerate(z['swap'])}
        B = dumps.Battle(d); tr = track.Tracker(track.trajectories(B), rng=random.Random(0)); WF = track.frame_weights(B)
        sp = None; pt = None
        for f in sorted(B.meta):
            s = B.state(f)
            if s is None: sp = None; continue
            if pt is not None and s['tick'] - pt > 1: tr.fill(s['tick'] - pt - 1); sp = None
            v = int(B.meta[f].get('player', 0)); tr.step(f, s, sp, WF.get(f, -1), track.team_of(s, v), None, v); sp = s; pt = s['tick']
            if f not in want: continue
            cam = np.array(B.meta[f]['mat'], np.float64).reshape(4, 4).T
            objs = BL.frame_objects(tr); R = BL.labels(objs, cam); img = z['img'][want[f]]
            lids = R['lids']; gl = R['glow_lids']
            for o in objs:
                if not o['drawn']: continue
                bi = [k for k, l in enumerate(lids) if l[0] == o['id'] and l[2] == 'body']
                gi = [k for k, l in enumerate(gl) if l[0] == o['id']]
                bm = np.where(np.isin(R['ids'], bi), R['share'], 0.0) if bi else np.zeros(R['share'].shape)
                gm = np.where(np.isin(R['glow_ids'], gi), R['glow_share'], 0.0) if gi else np.zeros(R['share'].shape)
                ys, xs = np.nonzero((bm >= 0.1) | (gm >= 0.1))
                if not len(xs): continue
                st = stat[o['t']]; st[0] += 1; st[1] += float(bm.sum()); st[2] += float(gm.sum()); st[3] += int((bm >= 0.5).sum())
                x0, x1, y0, y1 = xs.min() - 6, xs.max() + 7, ys.min() - 6, ys.max() + 7
                lim = 300 if o.get('attack') is not None else 160   # a wave attack's region is large
                if x0 < 0 or y0 < 0 or x1 > img.shape[1] or y1 > img.shape[0] or x1 - x0 > lim or y1 - y0 > lim: continue
                others = ((R['ids'][y0:y1, x0:x1] >= 0) & ~np.isin(R['ids'][y0:y1, x0:x1], bi)).sum() + \
                         ((R['glow_ids'][y0:y1, x0:x1] >= 0) & ~np.isin(R['glow_ids'][y0:y1, x0:x1], gi)).sum()
                score = (bm >= 0.5).sum() + (gm >= 0.5).sum() + 0.3 * (gm >= 0.1).sum() - 3 * others
                if o.get('ended'): score *= 0.2   # a wave attack: show it while its projectiles still fly (drawing at its fullest)
                if o['t'] not in best or score > best[o['t']][0]:
                    best[o['t']] = (score, img[y0:y1, x0:x1].copy(), bm[y0:y1, x0:x1].astype(np.float32), gm[y0:y1, x0:x1].astype(np.float32),
                                    '%s f%d id%d' % (os.path.basename(d.rstrip('/')), f, o['id']))
        print(d, 'types so far', len(best), flush=True)
    rows = []; tiles = []
    for t, (sc, rc_, bm, gm, src) in best.items():
        n, bs, gs, bo = stat[t]; frac = gs / (gs + bs) if gs + bs else 0
        k = K.get(t, {}).get('kind_blue', '')
        rows.append(dict(type=t, name=T[t]['name'], kind=k, source='real ' + src, frames=n, body_px_per_frame=round(bo / n, 1),
                         body_share_per_frame=round(bs / n, 1), glow_share_per_frame=round(gs / n, 1), glow_fraction=round(frac, 3)))
        tiles.append((-frac, T[t]['name'], _tile(_panels(rc_, bm, gm), '%s %d  [%s]' % (T[t]['name'], t, k),
                                                     'glow %.0f%%  body %.0f px/frame  %s' % (100 * frac, bo / n, src))))
    tiles.sort(key=lambda x: (x[0], x[1]))
    pages = _pages([t for _, _, t in tiles], out, 'real')
    rows.sort(key=lambda r: (-r['glow_fraction'], r['name']))
    with open(os.path.join(out, 'real_index.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, list(rows[0])); w.writeheader(); w.writerows(rows)
    print('real: %d types, %d pages' % (len(rows), len(pages)))
    return rows


def model_tile(t, own=True, updates=40):
    """the module's drawing of type t, in its clip px: (composite, body share, glow share) at the frame (of the first
    `updates`) where its polygons cover the largest area: types that draw late (HideTime) or briefly get a tile too"""
    T = projtab.table(); r = T[t]; D = PD.Display(t, own, ground=False)
    if D.sim is None: return None
    best = (0.0, None)
    for k in range(updates):
        D.update(angle=0, travel=min(1000, 25 * k), pos=(0, 0, 0)); P = D.polys()
        if not P: continue
        xy = np.concatenate([p[5] for p in P]); a = float(np.prod(xy.max(0) - xy.min(0) + 1)) * len(P) ** 0.5
        if a > best[0]: best = (a, P)
    P = best[1]
    if not P: return None
    exp = r['blue'] if own else r['red']
    xy = np.concatenate([p[5] for p in P]); c = (xy.min(0) + xy.max(0)) / 2; ext = max(8.0, float((xy.max(0) - xy.min(0)).max()))
    s = min(3.0, 140.0 / ext); cam = glow.local_cam(c[0], c[1], s)
    comp = glow.raster_local(D.S, P, cam)
    L = masks.Layers(cam); LG = masks.Layers(cam)
    for p in P:
        q = dict(world=np.c_[p[5], np.zeros(len(p[5]))], uv=p[4], tex=p[3], mul=p[6], add=p[7], blend=p[8], shape=p[1], cmd=p[2], path=p[0])
        cls = glow.layer_class(t, own, q)
        if cls == 'wave': L.add_coverage('body', [q], D.S); LG.add_coverage('body', [q], D.S); continue
        if cls == 'body': L.add('body', [q], D.S)
        LG.add(cls, [q], D.S)
    H, W = masks.H_FULL // 2, masks.W_FULL // 2
    def full(Lx, lid):
        m = np.zeros((H, W), np.float32); sm = Lx.share_maps(True)
        if lid in sm:
            x0, y0, a = sm[lid]; ax0, ay0 = max(0, x0), max(0, y0); ax1, ay1 = min(W, x0 + a.shape[1]), min(H, y0 + a.shape[0])
            m[ay0:ay1, ax0:ax1] = a[ay0 - y0:ay1 - y0, ax0 - x0:ax1 - x0]
        return m
    bm, gm = full(L, 'body'), full(LG, 'glow')
    im = (comp.reshape(H, 2, W, 2, 3).mean((1, 3)) * 255).astype(np.uint8)
    cy, cx = H // 2, W // 2; hh = min(H // 2, int(ext * s / 2 / 2) + 8)
    sl = (slice(cy - hh, cy + hh), slice(cx - hh, cx + hh))
    return im[sl], bm[sl], gm[sl]


def model(n, out, seen=()):
    T = projtab.table(); K = kinds(); rng = random.Random(1)
    cand = [t for t, k in K.items() if k.get('kind_blue') and t not in seen]
    pri = [t for t in cand if K[t]['kind_blue'] in ('light', 'wave')]
    gm = [t for t in cand if K[t]['kind_blue'] == 'glow' and K[t]['mesh'] == '1']; rng.shuffle(gm); pri += gm[:10]
    try:
        A = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'glow', 'additive_share.json')))
        hi = sorted([t for t in cand if t not in pri and K[t]['kind_blue'] == 'paint'], key=lambda t: -A.get(str(t), {}).get('add', 0))[:n // 2]
    except Exception:
        hi = []
    rest = [t for t in cand if t not in pri and t not in hi]; rng.shuffle(rest)
    pick = pri + hi + rest[:max(0, n - len(hi))]
    tiles = []; rows = []
    for t in pick:
        try: r = model_tile(t)
        except Exception as ex: print('skip', t, ex); continue
        if r is None: continue
        im, bm, gm = r; bs, gs = float(bm.sum()), float(gm.sum()); frac = gs / (gs + bs) if gs + bs else 0
        k = K[t]['kind_blue']
        rows.append(dict(type=t, name=T[t]['name'], kind=k, mesh=K[t]['mesh'], glow_fraction=round(frac, 3)))
        P = [im] + _panels(None, bm, gm)
        tiles.append((-frac, T[t]['name'], _tile(P, '%s %d  [%s%s]' % (T[t]['name'], t, k, ', mesh = body' if K[t]['mesh'] == '1' else ''),
                                                 'module drawing | body | glow   glow %.0f%%%s' % (100 * frac, '  (wave: every pixel it changes = body)' if k == 'wave' else ''))))
    tiles.sort(key=lambda x: (x[0], x[1]))
    pages = _pages([t for _, _, t in tiles], out, 'model', per=12)
    rows.sort(key=lambda r: (-r['glow_fraction'], r['name']))
    with open(os.path.join(out, 'model_index.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, list(rows[0])); w.writeheader(); w.writerows(rows)
    print('model: %d types, %d pages' % (len(rows), len(pages)))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('mode'); ap.add_argument('args', nargs='+')
    ap.add_argument('--out', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'glow'))
    a = ap.parse_args()
    if a.mode == 'real': real(a.args, a.out)
    else:
        seen = set()
        p = os.path.join(a.out, 'real_index.csv')
        if os.path.exists(p): seen = {int(r['type']) for r in csv.DictReader(open(p))}
        model(int(a.args[0]), a.out, seen)
