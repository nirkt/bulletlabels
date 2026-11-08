# round5 group "shelly5" (owner re-check 2026-11-06 evening, pages marked BAD; binding). Override file, exec'd in idlabels' namespace after all
# installed fixes (up to z_1114_cut5) and before the global smoothing v2. Global names carry the prefix _h5_.
#
# ---- 364 BanditGirlStar_remake: attack 6000004 (Shelly009Projectile), super 6000005 (Shelly009UltiProjectile); single-skin types (364 only) ----
# OWNER 11-06: "there are also some missing labels"; holes / ragged on attack 0 t265 / 267 / 269, attack 90 t305 / 307 / 309 / 313 / 317 / 319, super 45
# t347 / 349 / 353. Owner (coordinator relay 11-06 evening, same as 223): touching bullets MERGE into ONE label, no seams, no holes; bullets that do
# not touch stay separate. The big cyan hit star at the crate stays unlabelled.
# Cause: z_1113_shelly4 builds the star from the clip sprite's ring pixels (ID-pass code 1, coverage >= 110) only; where stars overlap, the front-most
#   draw of a pixel is the neighbour star's rim / glow, so the ring is torn, whole stars (their upper / left parts under the neighbours' draws) are
#   missing, and the closing leaves ragged bites and holes.
# Rule (data-derived, from the ID pass and the sprite atlas): a pixel belongs to a star when the front-most draw shows one of the star's own atlas
#   shapes (the star body shape, the thin rim shape where the sprite code is body / shared) at coverage >= _h5_CV. The union of those pixels, closed
#   (gaps between touching stars and seams), holes filled, thin slivers (the streak / glow of the trail) opened away, is one silhouette per connected
#   blob = ONE label per blob (object id of the nearest predicted star, k, 'body'). The violet hit bubble / hit star are other sprites (other
#   shapes): never in the mask.
#   Also: (a) ended pellets still drawn as saturated cyan stars (the clip shows only some of their facets as body pixels): the star-sprite draws'
#   bright cyan pixels of the blob that holds body pixels are added (owner: ended-but-visible pellets are labelled while drawn; the attack 90 t317 /
#   t319 'missing labels'); (b) a silhouette of at most _h5_ONESTAR px whose convex hull is at most _h5_HULLK x larger is one star with a flash / glow
#   blob over a part of it: the hull is taken (hidden part filled; t317 star). The big cyan hit star at the crate (other sprite) is not in the mask.
import numpy as _h5_np
from scipy.ndimage import (binary_fill_holes as _h5_fill, label as _h5_cc, binary_closing as _h5_closing, binary_opening as _h5_bopen,
                           binary_dilation as _h5_dil, distance_transform_edt as _h5_edt)
import bulletlabels as _h5_BL
import projtab as _h5_projtab

_h5_T364 = {6000004: dict(body=(56893,), rim=(0,), cv=60, cvw=25, close=2.0, col=(190, 205, 200, 4)),
            6000005: dict(body=(56873, 61), rim=(0,), cv=60, cvw=25, close=3.5, cyanrim=(130, 200, 220))}
_h5_MIN = 14
_h5_MINT = {6000005: 40}     # super: a star is >= 150 px; smaller shards are not stars
_h5_ONESTAR = {6000004: 210}   # px: larger silhouettes are several stars (a chain): no hull
_h5_HULLK = 1.9     # a silhouette whose convex hull is at most this much larger is one star: hidden bite filled
_h5_ENDSTAR = {6000004}
_h5_NEAR = {}      # attack: a silhouette must lie on a live pellet's predicted body (px reach) or where the installed label was
_h5_OPEN = 1.5


def _h5_disc(r):
    k = int(_h5_np.ceil(r)); y, x = _h5_np.ogrid[-k:k + 1, -k:k + 1]
    return (x * x + y * y) <= r * r + 1e-6


def _h5_t364f(P, objs, cam, R, img):
    out = P
    for t, cfg in sorted(_h5_T364.items()):
        sel = [o for o in objs if o['t'] == t]
        if not sel: continue
        lids = list(out['lids']); ids = out['ids'].copy(); share = out['share'].copy()
        keys = {_h5_BL._lid(o) for o in sel}
        old = [k for k, l in enumerate(lids) if l[0] in keys and l[2] == 'body']
        cv = R['cv'][::2, ::2]; C = sprite_pixels(t, sel[0]['own'], R)
        T = _h5_projtab.table()[t]
        SP = _a4_shape_pixels(R, T['file'], tuple(n for n in (T['blue'], T['red']) if n)) - 1
        B0 = _h5_np.isin(SP, cfg['body']) & _h5_np.isin(C, (1,)) & (cv >= cfg['cv'])
        # the thin rim shape (shape 0, sprite code shared) only where it clings to a star body (within 2 px): the same shape also draws the
        # violet hit bubble, which is not a bullet
        S0 = B0 | (_h5_np.isin(SP, cfg['rim']) & _h5_np.isin(C, (1, 5)) & (cv >= cfg['cv']) & _h5_dil(B0, iterations=2))
        if t in _h5_ENDSTAR and B0.any():
            # ended pellets still drawn as saturated cyan stars (the final phase of the pellet: the clip only shows some of its facets as body
            # pixels): whole star = the star-sprite draws' bright cyan pixels of the connected blob that holds body pixels (owner 11-06: ended
            # but visible pellets are labelled while drawn; t317 / t319 'missing labels')
            ids_h = R['id'][::2, ::2]; fsset = {R['draws'].get(int(d), {}).get('fs') for d in _h5_np.unique(ids_h[B0])}
            dm = [d for d, r in R['draws'].items() if r.get('fs') in fsset and r.get('blend') == 1 and not r.get('dmask')]
            im = img.astype(int)
            cy = (im[..., 2] > 190) & (im[..., 1] > 160) & (im[..., 0] < 150) & (cv >= 120) & _h5_np.isin(ids_h, dm)
            Lc, nc = _h5_cc(_h5_dil(cy, iterations=1) & (cy | _h5_np.isin(ids_h, dm)), structure=_h5_np.ones((3, 3)))
            for i in range(1, nc + 1):
                mi = Lc == i
                if (mi & B0).sum() >= 6: S0 = S0 | (mi & cy)
        if 'cvw' in cfg:
            # translucent parts of a star (a fading / spinning star shows its facets at coverage 25..60): kept only as thick blobs (the opening
            # with a disc of 3 px drops the thin halo ring and the necks the light trail makes between two stars)
            Wk = _h5_np.isin(SP, cfg['body']) & _h5_np.isin(C, (1,)) & (cv >= cfg['cvw'])
            Wk = _h5_fill(_h5_closing(_h5_np.pad(Wk, 5), structure=_h5_disc(2))[5:-5, 5:-5] | Wk)
            Wk = _h5_bopen(_h5_np.pad(Wk, 6), structure=_h5_disc(3))[6:-6, 6:-6]
            S0 = S0 | Wk
        if 'cyanrim' in cfg and S0.any():
            # round 8 (owner: super t345 / t349 'too small / cut off': stars of a carpet outside the label): the cyan rim of the super stars (thin bright
            # cyan outline) that touches a star pixel is star: the ring is added, closing + hole filling then give the whole star. Solid cyan blobs (the
            # hit star at the crate) are not rims (opening with a 3 px disc keeps only blobs thicker than a rim) and are left out.
            im3 = img.astype(int)
            rr, gg, bb = cfg['cyanrim']
            cyan = (im3[..., 0] <= rr) & (im3[..., 1] >= gg) & (im3[..., 2] >= bb)
            cyan = _h5_bopen(_h5_np.pad(cyan, 3), structure=_h5_np.ones((2, 2), bool))[3:-3, 3:-3]      # 1-px wires (the cyan lightning line of the hit) are no rim
            solid_c = _h5_bopen(_h5_np.pad(cyan, 5), structure=_h5_disc(3))[5:-5, 5:-5]
            cyan = cyan & ~_h5_dil(solid_c, iterations=2) & _h5_dil(S0, iterations=4)     # only the rim next to the star's own pixels (not cyan notes / glow farther out)
            Lc, nc = _h5_cc(_h5_dil(cyan, iterations=1) & (cyan | _h5_dil(S0, iterations=1)), structure=_h5_np.ones((3, 3)))
            seed = _h5_dil(S0, iterations=2)
            for i in _h5_np.unique(Lc[seed & cyan]):
                if i: S0 = S0 | ((Lc == i) & cyan)
        if 'col' in cfg and S0.any():
            # round 6 (owner: 'too small / cut off' t315, ragged t305): the star's bright cyan outer points / rim that the front-most draw of the
            # pixel hides (a neighbour's draw is in front): bright cyan raw pixels within _grow px of the star pixels are star too (the soft glow
            # and the light trails are dimmer; the green circle / music notes are not cyan)
            g_, b_, r_, grow = cfg['col']
            im2 = img.astype(int)
            S0 = S0 | ((im2[..., 1] >= g_) & (im2[..., 2] >= b_) & (im2[..., 0] <= r_) & _h5_dil(S0, iterations=grow))
        if not S0.any(): continue
        rc = cfg['close']; p = int(_h5_np.ceil(rc)) + 3
        S = _h5_closing(_h5_np.pad(S0, p), structure=_h5_disc(rc))[p:-p, p:-p] | S0
        S = _h5_fill(S)
        po = int(_h5_np.ceil(_h5_OPEN)) + 2
        so = _h5_bopen(_h5_np.pad(S, po), structure=_h5_disc(_h5_OPEN))[po:-po, po:-po]
        if so.sum() >= 0.8 * S.sum(): S = so | (S & _h5_dil(so, iterations=1) & (S0 | _h5_dil(S0, iterations=1)))
        L, n = _h5_cc(S, structure=_h5_np.ones((3, 3)))
        Pp = _h5_BL.labels(objs, cam, cut=0.01); pids, plids = Pp['ids'], Pp['lids']
        cen = {}
        for o in sel:
            lid = _h5_BL._lid(o)
            ks = [kk for kk, l in enumerate(plids) if l[0] == lid and l[2] == 'body' and (pids == kk).any()]
            if ks: cen[lid] = _h5_np.argwhere(_h5_np.isin(pids, ks)).mean(0)
        if not cen: cen = {_h5_BL._lid(sel[0]): _h5_np.zeros(2)}
        near = None
        if t in _h5_NEAR:
            live = {_h5_BL._lid(o) for o in sel if o.get('drawn', True)}
            kk = [q for q, l in enumerate(plids) if l[0] in live and l[2] == 'body' and (pids == q).any()]
            near = _h5_dil(_h5_np.isin(pids, kk), iterations=_h5_NEAR[t]) | _h5_dil(_h5_np.isin(ids, old), iterations=1)
        ids[_h5_np.isin(ids, old)] = -1
        cnt = {}; names = list(cen)
        for i in range(1, n + 1):
            m = L == i
            if m.sum() < _h5_MINT.get(t, _h5_MIN): continue
            hm = hull_mask(m)             # a single star with a glow / flash blob over a part of it: hidden part filled (hull only when nearly convex)
            if hm.sum() <= _h5_HULLK * m.sum() and m.sum() <= _h5_ONESTAR.get(t, 0):
                m = hm
            if near is not None and not (m & near).any(): continue
            c = _h5_np.argwhere(m).mean(0)
            lid = min(names, key=lambda q: float(_h5_np.hypot(*(cen[q] - c))))
            j = cnt.get(lid, 0); cnt[lid] = j + 1
            lp = (lid, j - 1, 'body')
            if lp not in lids: lids.append(lp)
            kp = lids.index(lp); ids[m & (ids < 0)] = kp
        nz = (ids != out['ids']) & (ids >= 0)
        share[nz] = _h5_np.maximum(cv[nz] / 255.0, _h5_BL.CUT); share[ids < 0] = 0
        out = dict(out); out.update(ids=ids, share=share.astype(_h5_np.float32), lids=lids)
    return out


_h5_prev_labels1 = labels


def labels(objs, cam, R, state=None, ended=(), img=None):
    P = _h5_prev_labels1(objs, cam, R, state=state, ended=ended, img=img)
    return _h5_t364f(P, objs, cam, R, img)


# ---- 460 BanditGirlCybernoodles: attack 6000018 (Shelly010Projectile); single-skin type (460 only) ----
# ROUND 6 (owner re-check): t309 'leaf / spark sprites above the bulbs labelled', t317 'too big: reaches into the green hit star', t313 'cut off':
#   neon yellow-green pixels (hit star, rings, leaf / lightning sprites, soft edge incl.) are now removed from every source (shape pixels, clip raster,
#   old label); the clip raster only fills next to the drawn bulb; the translucent cap on top of a bulb (shape 5) is included; _h5_CLOSE460 = 2.0 (only touching).
# OWNER 11-06 evening: attack 0 t273 and attack 90 t313 "too small / cut off"; coordinator relay (11-06 evening): touching bullets MERGE into ONE
# label, no seams (as 223); bullets that do not touch stay separate.
# Cause: z_1113_shelly4 (t460) takes each bulb from the clip data (the "raster"). A spinning bulb's clip pieces give only a thin strip of it (t313: the
#   raster is a narrow vertical slice, the drawn bulb is a wide round bulb), the gate then falls back to the older labels, which miss the bulb's
#   right half (t313 bulb 4) and, at t273, carry the yellow-green hit-effect rings (they use the same atlas shape as the bulb's ring).
# Rule (from the ID pass and the sprite atlas): the bulb = pixels whose front-most draw shows one of the bulb's own atlas shapes (disc, ring, cap, body:
#   _h5_SH460, sprite codes body / trail-over / shared) at coverage >= 128 (the translucent glow round it is below), minus the neon yellow-green hit
#   rings / lightning (colour), united with the clip raster (the parts a hit effect hides), closed 2 px, holes filled (the yellow symbol inside).
#   Connected blobs = ONE label each (touching bulbs one label); the blob is given to the object whose raster lies in it (else the nearest live object).
_h5_SH460 = (6, 7, 8, 15, 17, 71, 72)
_h5_MIN460 = 20
_h5_CLOSE460 = 2.0     # coordinator 11-06: only TOUCHING bulbs merge (3.6 bridged bulbs 6 px apart with necks)


def _h5_t460f(P, objs, cam, R, img):
    t = 6000018
    sel = [o for o in objs if o['t'] == t]
    if not sel: return P
    lids = list(P['lids']); ids = P['ids'].copy(); share = P['share'].copy()
    keys = {_h5_BL._lid(o) for o in sel}
    old = [k for k, l in enumerate(lids) if l[0] in keys and l[2] == 'body']
    cv = R['cv'][::2, ::2]; ids_h = R['id'][::2, ::2]
    T = _h5_projtab.table()[t]
    SP = _a4_shape_pixels(R, T['file'], tuple(n for n in (T['blue'], T['red']) if n)) - 1
    C = None
    for o in sel:
        c = sprite_pixels(o['t'], o['own'], R); C = c if C is None else _h5_np.where(C == 0, c, C)
    im = img.astype(int)
    neon = (im[..., 0] >= 130) & (im[..., 1] >= 170) & (im[..., 2] <= 130)
    neon = neon | (_h5_dil(neon, iterations=1) & (im[..., 1] >= 150) & (im[..., 2] <= 190))     # + the soft edge of the effect
    solid = [d for d, r in R['draws'].items() if r.get('dmask')]
    front = ~_h5_np.isin(ids_h, solid)
    # the floor (the solid draw covering the most pixels) lies behind the bulbs: bridges between two bulbs may cross it; walls / characters not
    cnts = {d: int((ids_h == d).sum()) for d in solid}
    floor = max(cnts, key=cnts.get) if cnts else None
    front2 = ~_h5_np.isin(ids_h, [d for d in solid if d != floor])
    Bm = _h5_np.isin(C, (1, 4, 5)) & _h5_np.isin(SP, _h5_SH460) & (cv >= 128) & ~neon & front
    # round 8 (owner t313: 'bottom edge clipped'): the bulb's dark navy outer rim (coverage 70..128 at its anti-aliased edge) next to the drawn bulb: navy
    # pixels (blue-dominant, not the violet halo) within 3 px of the bulb are bulb
    navy = (im[..., 2] >= 110) & (im[..., 0] <= 70) & (im[..., 1] <= 140) & (im[..., 2] - im[..., 0] >= 70)
    Bm = Bm | (_h5_np.isin(C, (1, 4, 5)) & _h5_np.isin(SP, _h5_SH460) & (cv >= 90) & navy & ~neon & front & _h5_dil(Bm, iterations=2))
    # the cap on top of the bulb (shape 5, translucent, coverage ~100): next to the drawn bulb it is part of it (cut-off checker: t313 / t315)
    Bm = Bm | (_h5_np.isin(C, (1, 4, 5)) & (SP == 5) & (cv >= 64) & ~neon & front & _h5_dil(Bm, iterations=3))
    ras = {}
    for o in sel:
        if not o.get('drawn', True): continue
        try: m = _sh4_raster(o, cam, lambda q: len(q['path']) <= 9) & front & ~neon     # round 6: the hit star / rings / leaf sprites are never in the label
        except Exception: m = None
        if m is not None and m.any(): ras[o['id']] = m
    S0 = Bm.copy()
    for m in ras.values(): S0 |= m & _h5_dil(Bm, iterations=4)     # round 6: the clip raster only fills what lies next to the drawn bulb (it is wider than a spinning bulb)
    # the installed label's thick pixels within 6 px of the above (the rod-like bulb seen edge-on has translucent edge pixels the shape rule misses): kept,
    # except the neon hit rings
    ok = _h5_np.isin(ids, old) & ~neon & front
    ok = _h5_bopen(_h5_np.pad(ok, 4), structure=_h5_disc(2))[4:-4, 4:-4]       # thin arcs of the hit effect do not survive the opening
    S0 = S0 | (ok & _h5_dil(S0, iterations=6))
    if not S0.any(): return P
    p = 7
    S = (_h5_closing(_h5_np.pad(S0, p), structure=_h5_disc(_h5_CLOSE460))[p:-p, p:-p] & ~neon & front2) | S0
    S = _h5_fill(S) & front2
    L, n = _h5_cc(S, structure=_h5_np.ones((3, 3)))
    Pp = _h5_BL.labels(objs, cam, cut=0.01); pids, plids = Pp['ids'], Pp['lids']
    cen = {}
    for o in sel:
        lid = _h5_BL._lid(o)
        ks = [kk for kk, l in enumerate(plids) if l[0] == lid and l[2] == 'body' and (pids == kk).any()]
        if ks: cen[lid] = _h5_np.argwhere(_h5_np.isin(pids, ks)).mean(0)
    if not cen: cen = {_h5_BL._lid(sel[0]): _h5_np.zeros(2)}
    # a blob must lie on a bulb: its clip raster, or where the installed label was (the same atlas shapes draw the lightning / trail bits)
    anchor = _h5_dil(_h5_np.isin(ids, old), iterations=1)
    for m in ras.values(): anchor = anchor | m
    ids[_h5_np.isin(ids, old)] = -1
    cnt = {}
    for i in range(1, n + 1):
        m = L == i
        if m.sum() < _h5_MIN460 or not (m & anchor).any(): continue
        ov = {oid: int((r & m).sum()) for oid, r in ras.items() if (r & m).any()}
        if ov:
            oid = max(ov, key=ov.get); lid = _h5_BL._lid(next(o for o in sel if o['id'] == oid))
        else:
            c = _h5_np.argwhere(m).mean(0); lid = min(cen, key=lambda q: float(_h5_np.hypot(*(cen[q] - c))))
        j = cnt.get(lid, 0); cnt[lid] = j + 1
        lp = (lid, j - 1, 'body')
        if lp not in lids: lids.append(lp)
        ids[m & (ids < 0)] = lids.index(lp)
    nz = (ids != P['ids']) & (ids >= 0)
    share[nz] = _h5_np.maximum(cv[nz] / 255.0, _h5_BL.CUT); share[ids < 0] = 0
    P2 = dict(P); P2.update(ids=ids, share=share.astype(_h5_np.float32), lids=lids)
    return P2


_h5_prev_labels2 = labels


def labels(objs, cam, R, state=None, ended=(), img=None):
    P = _h5_prev_labels2(objs, cam, R, state=state, ended=ended, img=img)
    return _h5_t460f(P, objs, cam, R, img)
