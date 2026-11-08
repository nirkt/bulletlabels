"""Frame-by-frame sheets of solo attacks (posecap --solo --pix-every N captures, e.g. tools/gale_solo.sh): one row per
attack / super of player 0, its real frames in sequence, all cropped to the same window (the union of the labels over the
row). Per frame: the real frame with the labels drawn over it (body >= 0.5: red tint + red contour; glow >= 0.5: green
contour). Labels come from the states alone (track + bulletlabels), not from the frame.
  python3 seqsheet.py OUT_PREFIX DIR [DIR ...] [--per 8]   -> OUT_PREFIX_01.png ..."""
import argparse, json, os, random
import numpy as np
from PIL import Image, ImageDraw
import rc, dumps, track, projtab, bulletlabels as BL, glow
from glowsheet import _edge

MAXW = 260   # px per frame tile (crops wider than this are scaled down)
BOXES = []   # crop box of every row built by rows_of (exact_pages writes them next to the pages)


S01, S05 = 26, 128   # uint8 share thresholds (0.1, 0.5 of 255)


def rows_of(d, keep=None, idd=None):
    """[(title, [(frame, real, body, glow)], types)] one per attack / super of player 0"""
    T = projtab.table(); S = json.load(open(os.path.join(d, 'script.json')))['script']
    z = np.load(os.path.join(d, 'frames.npz')); want = {int(s): i for i, s in enumerate(z['swap'])}
    IMG = z['img']   # once: z['img'] reads the whole array on every access (and a stored view keeps that copy alive)
    if idd is None:   # the ID pass dumps next to the capture (posecap --id-dir <cap>/id): labels from the client's draws
        import glob
        g = sorted(glob.glob(os.path.join(d, 'id', '*', '')))
        idd = g[0] if g else None
    if idd:
        import idbuf, idlabels
        have = set(idbuf.frames(idd))
    B = dumps.Battle(d); fr = sorted(B.meta); tr = track.Tracker(track.trajectories(B), rng=random.Random(0)); WF = track.frame_weights(B)
    segs = []; sp = None; pt = None; seg = None
    for k, f in enumerate(fr):
        sh = next((p for p, v in sorted(S[k]['act'].items()) if v[1]), '0') if k < len(S) else '0'   # the firing player (10-28: enemy captures fire with player 1)
        btn = S[k]['act'][sh][1] if k < len(S) else 0
        if btn:
            seg = dict(phase=S[k]['phase'], ang=S[k]['act'][sh][2], frames=[], types=set()); segs.append(seg)
            seg['keep'] = keep is None or keep(seg['phase'], seg['ang'])
        s = B.state(f)
        if s is None: sp = None; continue
        if pt is not None and s['tick'] - pt > 1: tr.fill(s['tick'] - pt - 1); sp = None
        v = int(B.meta[f].get('player', 0)); tr.step(f, s, sp, WF.get(f, -1), track.team_of(s, v), None, v); sp = s; pt = s['tick']
        if seg is None or f not in want or not seg['keep']: continue
        cam = np.array(B.meta[f]['mat'], np.float64).reshape(4, 4).T
        objs = BL.frame_objects(tr)
        R = idlabels.labels(objs, cam, idbuf.load(idd, f), state=s, ended=list(tr.ended.values()), img=IMG[want[f]] if f in want else None) if idd and f in have else BL.labels(objs, cam)
        # shares kept as uint8 (share x 255; memory: a row holds every frame): 0.1 -> S01, 0.5 -> S05
        body = np.round(np.where(R['ids'] >= 0, R['share'], 0.0) * 255).astype(np.uint8)
        gl = np.round(np.where(R['glow_ids'] >= 0, R['glow_share'], 0.0) * 255).astype(np.uint8)
        kd = np.zeros(body.shape, np.int8)   # the label's kind (KIND_CODE; 0 = none or unknown: drawn as attack)
        M = R.get('meta', {})
        for k, lid in enumerate(R['lids']):
            kc = KIND_CODE.get((M.get(lid[0]) or {}).get('kind'), 1)
            kd[R['ids'] == k] = kc
            if lid[2] in ('spawn', 'area'): seg['types'].add(('%s:%s' % (lid[2], (M.get(lid[0]) or {}).get('name')), None))
        seg['types'] |= {(o['t'], glow.type_kind(o['t'], o['own'])) for o in objs if o.get('file') is not None or o.get('attack') is not None}
        seg['frames'].append((f, IMG[want[f]], body, gl, s['tick'], kd))
    out = []
    for g in segs:
        F = [x for x in g['frames']]
        m = np.zeros(F[0][2].shape, bool) if F else None
        for x in F: m |= (x[2] >= S01) | (x[3] >= S01)
        if not F or not m.any(): continue
        first = next(k for k, x in enumerate(F) if ((x[2] >= S01) | (x[3] >= S01)).any())
        last = max(k for k, x in enumerate(F) if ((x[2] >= S01) | (x[3] >= S01)).any())
        F = F[max(0, first - 1):last + 2]   # one frame before the first label and one after the last
        ys, xs = np.nonzero(m); H, W = m.shape
        x0, x1, y0, y1 = max(0, xs.min() - 20), min(W, xs.max() + 21), max(0, ys.min() - 20), min(H, ys.max() + 21)
        tnames = ', '.join(('%s [%s]' % (T[t]['name'], k)) if t in T else str(t) for t, k in sorted(g['types'], key=str))
        BOXES.append(dict(row='%s %s %.0f deg' % (os.path.basename(d.rstrip('/')), g['phase'], g['ang']), box=[int(x0), int(x1), int(y0), int(y1)],
                          ticks=[int(x[4]) for x in F], frames=[int(x[0]) for x in F]))
        out.append(('%s %s %.0f deg' % (os.path.basename(d.rstrip('/')), g['phase'], g['ang']), tnames,
                    [(tk, im[y0:y1, x0:x1], b[y0:y1, x0:x1], gg[y0:y1, x0:x1], kd[y0:y1, x0:x1]) for f, im, b, gg, tk, kd in F]))
    return out


def tile(f, im, b, g, k):
    ov = im.astype(np.float32); mb = b >= S05
    ov[mb] = ov[mb] * 0.6 + np.array([255, 0, 0]) * 0.4
    ov = ov.astype(np.uint8); ov[_edge(mb)] = [255, 0, 0]; ov[_edge(g >= S05)] = [0, 255, 0]
    I = Image.fromarray(ov)
    if I.width > MAXW: I = I.resize((MAXW, max(1, round(I.height * MAXW / I.width))), Image.BILINEAR)
    d = ImageDraw.Draw(I); d.text((2, 1), '%d' % k, fill=(255, 255, 0))
    return np.asarray(I)


def row_img(title, sub, F, per):
    T = [tile(f, im, b, g, k) for k, (f, im, b, g, *_) in enumerate(F)]
    h = max(t.shape[0] for t in T); w = T[0].shape[1]; lines = []
    for p in range(0, len(T), per):
        L = np.full((h, per * (w + 4), 3), 15, np.uint8)
        for j, t in enumerate(T[p:p + per]): L[:t.shape[0], j * (w + 4):j * (w + 4) + t.shape[1]] = t
        lines.append(L)
    body = np.concatenate(sum([[l, np.full((4, l.shape[1], 3), 15, np.uint8)] for l in lines], [])[:-1], 0)
    head = np.full((28, body.shape[1], 3), 30, np.uint8)
    I = Image.fromarray(np.concatenate([head, body], 0)); d = ImageDraw.Draw(I)
    d.text((3, 2), title + '   (frames 2 ticks apart; red = body label, green = glow)', fill=(255, 255, 0)); d.text((3, 15), sub[:180], fill=(210, 210, 210))
    return np.asarray(I)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('dirs', nargs='+'); ap.add_argument('--per', type=int, default=8)
    a = ap.parse_args(); rows = []
    for d in a.dirs:
        rows += [row_img(t, s, F, a.per) for t, s, F in rows_of(d)]; print(d, len(rows), 'rows', flush=True)
    # pages of at most ~2400 px height
    page = []; hgt = 0; n = 0
    def flush():
        nonlocal page, hgt, n
        if not page: return
        W = max(r.shape[1] for r in page); P = np.full((sum(r.shape[0] + 8 for r in page), W, 3), 0, np.uint8); y = 0
        for r in page: P[y:y + r.shape[0], :r.shape[1]] = r; y += r.shape[0] + 8
        n += 1; Image.fromarray(P).save('%s_%02d.png' % (a.out, n)); page = []; hgt = 0
    for r in rows:
        if page and hgt + r.shape[0] > 2400: flush()
        page.append(r); hgt += r.shape[0] + 8
    flush(); print(n, 'pages')


if __name__ == '__main__':
    main()


# ---- exact review sheets (manual check of every attack, 2026-10-18): raw crop | the same crop with a 1-px label outline,
# never scaled down, small crops scaled up; one PNG per skin page, each <= PAGE px so a viewer sees every pixel.
PAGE = 1900


def pick(F, n=8):
    """the frame before the first label, the first three labelled, two in between, the last labelled, one after"""
    lab = [k for k, x in enumerate(F) if ((x[2] >= S01) | (x[3] >= S01)).any()]
    if len(F) <= n or not lab: return list(range(len(F)))
    a, b = lab[0], lab[-1]; mid = list(np.linspace(a + 3, b - 1, 4)[1:3].round().astype(int)) if b - a > 5 else []
    ks = sorted({max(0, a - 1), a, min(a + 1, b), min(a + 2, b), *mid, b, min(len(F) - 1, b + 1)})
    return ks


# label kinds (the user's classes, 2026-10-19; idlabels.meta / owners.py) and their outline colours on the pages
KIND_CODE = dict(attack=1, super=2, spawnable=3, zone=4, gadget=5, starpower=5, spawnable_attack=6, target=7, throwable=8, pull=9)   # 6, 7: 10-24 (Penny); 8, 9: 10-28 (Tara super: eyeball = no-damage throwable, vortex = pull)
KIND_COLOR = {1: (255, 0, 0), 2: (255, 0, 255), 3: (0, 140, 255), 4: (255, 230, 0), 5: (0, 255, 255), 6: (255, 140, 0), 7: (255, 255, 255), 8: (255, 150, 200), 9: (150, 90, 255)}
LEGEND = 'red attack, magenta super, blue spawnable, orange spawnable attack, yellow zone, white landing marker, pink throwable (no damage), violet pull, cyan gadget/star power; green glow'


def exact_pair(tk, im, b, g, kd, s):
    up = lambda a: np.repeat(np.repeat(a, s, 0), s, 1)
    raw = up(im); mb = up(b >= S05); mg = up(g >= S05); K = up(kd)
    ov = raw.copy(); ov[_edge(mg)] = [0, 255, 0]
    for code, col in KIND_COLOR.items():   # one outline per kind (a kind's own edge, also where two kinds touch)
        m = mb & ((K == code) | ((K == 0) & (code == 1)))
        if m.any(): ov[_edge(m)] = col
    h, w = raw.shape[:2]
    if 2 * w + 3 <= PAGE: P = np.concatenate([raw, np.full((h, 3, 3), 255, np.uint8), ov], 1)
    else: P = np.concatenate([raw, np.full((3, w, 3), 255, np.uint8), ov], 0)
    head = np.full((14, P.shape[1], 3), 25, np.uint8); I = Image.fromarray(np.concatenate([head, P], 0))
    ImageDraw.Draw(I).text((2, 1), 'tick %d   raw | outline (x%d)' % (tk, s), fill=(255, 255, 0))
    return np.asarray(I)


def exact_rows(d, phases=('attack 0', 'attack 90', 'super')):
    out = []
    for title, sub, F in rows_of(d, keep=lambda ph, ang: ph == 'super' or (ph == 'attack' and ang in (0.0, 90.0))):
        if not any(title.split(' ', 1)[1].startswith(p + ' ') or title.split(' ', 1)[1] == p + ' deg' or title.split(' ', 1)[1].startswith(p) for p in phases): continue
        if title.split(' ', 1)[1].startswith('attack') and not any(title.split(' ', 1)[1] == p + ' deg' for p in phases): continue
        h, w = F[0][1].shape[:2]; s = int(max(1, min(3, 260 // max(h, w, 1))))
        ks = pick(F)
        for b in BOXES:
            if b['row'] == title: b['picked'] = [int(k) for k in ks]; b['scale'] = s
        out.append((title, sub, [exact_pair(*F[k], s) for k in ks]))
    return out


def exact_pages(prefix, d):
    BOXES.clear()
    """-> list of page files; each page: rows with a title bar, frame pairs packed left to right, wrapped at PAGE px"""
    blocks = []
    for title, sub, P in exact_rows(d):
        lines = []; cur = []; cw = 0
        for p in P:
            if cur and cw + p.shape[1] > PAGE: lines.append(cur); cur = []; cw = 0
            cur.append(p); cw += p.shape[1] + 6
        if cur: lines.append(cur)
        hdr = Image.new('RGB', (PAGE, 30), (45, 45, 70)); dd = ImageDraw.Draw(hdr)
        dd.text((3, 2), title + '      ' + LEGEND, fill=(255, 255, 0)); dd.text((3, 16), sub[:300], fill=(220, 220, 220))
        blocks.append(np.asarray(hdr))
        for L in lines:
            hh = max(p.shape[0] for p in L); row = np.full((hh + 6, PAGE, 3), 0, np.uint8); x = 0
            for p in L: row[:p.shape[0], x:x + p.shape[1]] = p; x += p.shape[1] + 6
            blocks.append(row)
    json.dump(BOXES, open(prefix + '_boxes.json', 'w')); BOXES.clear()
    names = []; cur = []; hgt = 0
    def flush():
        if not cur: return
        nm = '%s_%02d.png' % (prefix, len(names) + 1); Image.fromarray(np.concatenate(cur, 0)).save(nm, optimize=True); names.append(nm)
    for b in blocks:
        if cur and hgt + b.shape[0] > PAGE: flush(); cur = []; hgt = 0
        cur.append(b); hgt += b.shape[0]
    flush()
    return names
