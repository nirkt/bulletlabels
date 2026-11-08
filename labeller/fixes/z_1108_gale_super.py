# v2 (2026-11-04): same as gale_super_fix.py / installed z_1108_gale_super.py for OWN (blue) shots (pixel-identical, checked on 048 254 262 283
# 370 442), plus team-aware parameters for ENEMY (red) shots (o['own'] False): GS_ENEMY overrides per type. The red art has other colours:
# 254 pink slab with saturated dots (pc: close the dots out of the raw mask), 048 pink-white snow (no cyan rule; red >= blue; closing 14),
# 283 pink ghosts, 370 pink-white cloud + yellow turtles (no UI draws: the enemy view has a skull marker), 262 stars / 442 gears unchanged
# (installed pixel outline). tools/enemycmp.py with this file (IDLABELS_SKIP=z_1108_gale_super.py IDLABELS_OVERRIDE=<this file>): 254 048 283
# 370 262 442 all SAME (254 was DIFFERENT with the installed v1: recall 0.34).
# Gale (Blower) SUPER label, owner 2026-11-03: "make sure to improve gale's super label so that its around the white snow, not that weird pointy
# thing". ONE label per super shot in the white snow's OWN outline: the bright slab of the wave, filled, smooth, touching pieces merged; the
# light-blue puffs / bubbles / foam and the glow around it are out. Not a hull, not a rectangle. Replaces fix.py (the hull / rectangle version,
# kept for reference) and, for the super types only, the installed pixel labels of z_1027_s048 / z_1101_s048 / z_1107_dup_crow_poco_gale C /
# z_1028_s254 + z_1103_crow_barley_gale (254) / z_1027_s370 (+ the shared fixes). Types: 6000732 734 736 738 740 742 and 6001468.
# Exec'd in idlabels' namespace after the installed fixes (IDLABELS_OVERRIDE); names prefixed _gs_ / GS_ (no clash with _gc_ / _gcd_).
#
# Method (pixel-derived, a bullet-rule label):
#  1. search region from game data only: the wave's slab seen by the camera (ground rectangle over the live members, raised), grown by a margin,
#     so the brawler, walls, spawners and far puffs are never candidates;
#  2. snow pixels = frame pixels in the region whose WEAKEST channel is >= a per-skin threshold (white / pale snow, sand, ghost cyan; the blue puffs and
#     bubbles, glow, rim of other colours stay below it; 'gr': green at least gr above red for the cyan-white snow of the default skin);
#  3. opened (tiny bubbles stuck to the edge / spurs go), closed (gaps and the translucent patches over crates / bushes), holes filled, the largest
#     part kept with every part that is at least GS_KEEP of it (merged into the one shot label), edge smoothed (blur at 0.5) and grown GS_GROW px
#     for the anti-aliased rim;
#  4. skins whose super is not a snow slab (Nutcracker 262 stars, SteamPunk 442 gears: bright yellow / green bodies) keep their own installed
#     pixel label (the stars' / gears' outline), merged: closed, holes filled, one label per shot.
#  Lifetime: while a member of the shot still travels (finished == 0) -- as before, nothing after (the splash at the range is an end animation).
import numpy as _gn
from PIL import Image as _GI, ImageDraw as _GD
import bulletlabels as _GBL
from scipy.spatial import ConvexHull as _GHull
from scipy.ndimage import (binary_opening as _gs_open, binary_closing as _gs_close, binary_fill_holes as _gs_fill,
                           gaussian_filter as _gs_gauss, label as _gs_cc, binary_dilation as _gs_dil)

GS_TYPES = (6000732, 6000734, 6000736, 6000738, 6000740, 6000742, 6001468)
# per skin: thr = weakest-channel threshold of the snow; mode 'snow' = colour rule, 'keep' = installed pixel label merged
# region: slab model (ab behind / af ahead of the members' mean, e past the outermost members, z0..z1 logic height) + reg margin
GS_SKIN = {
    6000732: dict(mode='snow', thr=150, gr=10, ab=300, af=200, e=225, z0=250, z1=400),   # 048 default: white snow
    6001468: dict(mode='snow', thr=150, gr=10, ab=300, af=200, e=225, z0=250, z1=400),   # overcharged default (not in the sweep: 048 values)
    6000734: dict(mode='snow', thr=105, yb=40, cl=12, op=9, ab=350, af=200, e=250, z0=300, z1=600),   # 254 Trader: pale sand
    6000738: dict(mode='snow', thr=0, gmin=165, bmin=225, ab=200, af=300, e=250, z0=300, z1=500),   # 283 Busters: ghost cyan
    6000740: dict(mode='snow', thr=0, rmin=115, gr=15, turtle=1, cl=15, op=4, ab=500, af=200, e=400, z0=300, z1=800),   # 370 Deepsea: white cloud (blue foam out)
    6000736: dict(mode='keep', ab=300, af=200, e=200, z0=150, z1=450),            # 262 Nutcracker: stars
    6000742: dict(mode='keep', ab=300, af=250, e=250, z0=300, z1=500),            # 442 SteamPunk: gears
}
# ENEMY (red-tinted) shots (o['own'] False): the red art has other colours (254: pink slab with saturated pink dots that fall below the
# colour thresholds and cut the slab into thin necks), so the per-skin parameters can be overridden here; the own (blue) parameters above and
# their result are untouched. pc = closing radius applied to the raw colour mask before the clean-up (own: none).
GS_ENEMY = {
    6000734: dict(pc=4),
    6000732: dict(gr=None, rb=10, cl=14),      # 048: the red snow is pink-white (red >= blue, green below red), the cyan-white rule does not apply
    6001468: dict(gr=None, rb=10, cl=14),
    6000740: dict(noui=1, thr=185, rmin=None, gr=None, turtle=None, turtle2=1),   # 370: red art: the cloud is pink-white (weakest channel >= 185), the foam pink, the turtles yellow
    6000738: dict(gmin=140, bmin=175, rmin=235),     # 283: the ghosts are pink, not cyan
}
GS_UI_FS = (1735630204, 2825982420, 3318493823)     # fragment shaders of the UI draws
GS_OPEN = 5          # px: opening disk radius (bubbles / spurs)
GS_CLOSE = 4         # px: closing radius
GS_KEEP = 0.3       # parts at least this share of the largest part are kept
GS_SIGMA = 1.5
GS_GROW = 1


def _gs_disk(r):
    y, x = _gn.mgrid[-r:r + 1, -r:r + 1]
    return x * x + y * y <= r * r + 0.5


def _gs_proj(cam, X, Y, Z):
    X = _gn.asarray(X, float)
    Q = _gn.stack([X, -_gn.asarray(Y, float), 0.6 * _gn.asarray(Z, float) * _gn.ones_like(X), _gn.ones_like(X)], -1) @ _gn.asarray(cam, float).T
    return _gn.stack([(Q[..., 0] / Q[..., 3] * 0.5 + 0.5) * 1200, (1 - (Q[..., 1] / Q[..., 3] * 0.5 + 0.5)) * 540], -1)


def _gs_region(cam, mem, prm, shape):
    ob = [o for o in mem if o['finished'] in (0, 1)]
    G = _gn.array([[o['ground'][0], -o['ground'][1]] for o in ob], float); a = _gn.radians(ob[0]['angle'])
    ua = _gn.array([_gn.cos(a), _gn.sin(a)]); va = _gn.array([-_gn.sin(a), _gn.cos(a)])
    u = G @ ua; v = G @ va; uc = u.mean()
    pts = _gn.array([uu * ua + vv * va for uu in (uc - prm['ab'], uc + prm['af']) for vv in (v.min() - prm['e'], v.max() + prm['e'])])
    S = _gn.concatenate([_gs_proj(cam, pts[:, 0], pts[:, 1], prm['z0']), _gs_proj(cam, pts[:, 0], pts[:, 1], prm['z1'])])
    try: S = S[_GHull(S).vertices]
    except Exception: pass
    I = _GI.new('L', (shape[1], shape[0]), 0); _GD.Draw(I).polygon([tuple(p) for p in S], fill=1, outline=1)
    return _gn.asarray(I, bool)


def _gs_clean(m, close=GS_CLOSE, op=GS_OPEN, keep=GS_KEEP, sigma=GS_SIGMA, grow=GS_GROW):
    """opened, closed, holes filled, the largest part + the parts of at least GS_KEEP of it merged, smoothed, grown"""
    if not m.any(): return m
    p = close + op + 4
    mp = _gn.pad(m, p)
    if op: mp = _gs_open(mp, structure=_gs_disk(op))
    mp = _gs_fill(_gs_close(mp, structure=_gs_disk(close)))
    L, n = _gs_cc(mp)
    if n == 0: return m & False
    sz = _gn.bincount(L.ravel()); sz[0] = 0
    mp = _gn.isin(L, _gn.nonzero(sz >= max(1.0, keep * sz.max()))[0])
    if sigma: mp = _gs_gauss(mp.astype(_gn.float32), sigma) >= 0.5
    if grow: mp = _gs_dil(mp, structure=_gs_disk(grow))
    return _gs_fill(mp)[p:-p, p:-p]


_gs_prev_labels = labels


def labels(objs, cam, R, state=None, ended=(), img=None):
    P = _gs_prev_labels(objs, cam, R, state=state, ended=ended, img=img)
    mine = [o for o in objs if o['t'] in GS_SKIN]
    if not mine: return P
    ids, share, lids = P['ids'].copy(), P['share'].copy(), list(P['lids'])
    cv = R['cv'][::2, ::2]
    oid = {_GBL._lid(o) for o in mine} | {o['id'] for o in mine}
    old = [k for k, l in enumerate(lids) if l[0] in oid and l[2] == 'body']
    shots = {}
    for o in mine:
        if not o.get('ended'): shots.setdefault((o['t'], o['owner'], int(round(o['angle'])) % 360, o.get('first')), []).append(o)
    oldm = _gn.isin(ids, old) if old else _gn.zeros(ids.shape, bool)
    new = {}
    for key, mem in shots.items():
        t = key[0]; prm = dict(GS_SKIN[t])
        if not mem[0].get('own', True): prm.update(GS_ENEMY.get(t, {}))
        if not any(o['finished'] == 0 for o in mem): continue
        if prm['mode'] == 'snow':
            reg = _gs_region(cam, mem, prm, ids.shape)
            c = img.astype(_gn.int16); snow = c.min(-1) >= prm['thr']
            if prm.get('rmin') is not None: snow &= c[..., 0] >= prm['rmin']      # the cloud is pale (red well above 0); the deep-blue foam under it is not
            if 'gmin' in prm: snow &= (c[..., 1] >= prm['gmin']) & (c[..., 2] >= prm['bmin'])   # bright cyan (the ghost snowballs: dark centre, bright rim)
            if prm.get('rb') is not None: snow &= (c[..., 0] - c[..., 2]) >= prm['rb']      # red-tinted snow: red above blue (the lilac stone is not)
            if prm.get('gr') is not None: snow &= (c[..., 1] - c[..., 0]) >= prm['gr']      # cyan-white: green above red (the lilac stone of spawners / walls is not)
            if prm.get('turtle2'): snow |= (c[..., 0] >= 220) & (c[..., 1] >= 140) & (c[..., 2] <= 110)      # the yellow-orange turtles of the red skin
            if prm.get('turtle'): snow |= (c[..., 1] >= 170) & (c[..., 2] <= 215) & (c[..., 0] <= 160) & (c[..., 1] - c[..., 2] >= 20)   # the green turtles standing on the cloud
            if 'yb' in prm: snow &= (c[..., 0] - c[..., 2]) >= prm['yb']      # yellow sand: red well above blue (lilac stone is not)
            snow &= reg
            if prm.get('noui'):      # UI draws (the skull marker of the enemy view ...) are never the snow
                _ui = [i for i, dr in R['draws'].items() if dr.get('fs') in GS_UI_FS]
                if _ui: snow &= ~_gn.isin(R['id'][::2, ::2], _ui)
            if prm.get('pc'):      # enemy colours: close the dots out of the raw mask first
                pc = prm['pc']; snow = _gs_fill(_gs_close(_gn.pad(snow, pc + 2), structure=_gs_disk(pc)))[pc + 2:-pc - 2, pc + 2:-pc - 2]
            M = _gs_clean(snow, prm.get('cl', GS_CLOSE), prm.get('op', GS_OPEN))
        else:   # keep: the installed pixel label of this shot's objects, merged
            ks = [k for k, l in enumerate(lids) if l[0] in {_GBL._lid(o) for o in mem} | {o['id'] for o in mem} and l[2] == 'body']
            M = _gs_clean(_gn.isin(ids, ks), 2, 0, 0.0, 0, 0) if ks else None    # own outline: touching / near pieces merged, holes filled, no rounding
        if M is not None and M.any(): new[key] = (M, mem)
    if old: ids[oldm] = -1; share[oldm] = 0
    for key, (M, mem) in new.items():
        lid = (_GBL._lid(min(mem, key=lambda o: o['id'])), -1, 'body')
        if lid not in lids: lids.append(lid)
        k = lids.index(lid); m = M & (ids < 0)
        ids[m] = k; share[m] = _gn.maximum(cv[m] / 255.0, _GBL.CUT)
    P2 = dict(P); P2.update(ids=ids, share=share, lids=lids)
    return P2
