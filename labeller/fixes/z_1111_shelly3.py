# away_1105 round3 group "shelly3" (owner final review 2026-11-05, BAD pages, binding). Override file, exec'd in idlabels' namespace after
# all installed fixes. Prefix of every global: _sh3_
#
# Pages / types (each type id belongs to exactly one skin, owners.projectile):
#   240 ShotgunGirlWitch  attack 6000006 (Shelly003Projectile) + super 6000007 (Shelly003UltiProjectile)   OWNER: "a little too ragged, some are missing labels"
#   341 ShotgunGirlBrawloween attack 6000012 + super 6000013                                               OWNER: "labels are ragged and some are missing labels"
#   349 ShotgunGirlDancer attack 6000014 + super 6000015                                                    OWNER: "ragged, and some are missing labels"
#   364 BanditGirlStar_remake attack 6000004 + super 6000005                                               OWNER: "some are ragged, some have holes"
#
# Cause (measured on the fresh captures hull_shelly3_NNN): the pellets (one logic object each, a flame ghost / leaf / gem sprite drawn from the
# clip's body shapes) overlap in clusters. The installed chain (z_1027 stack merge, z_1103 per-pellet closing, z_1104 per-draw split, z_1105
# per-anchor split, z_1107 silhouette, fix2 discs for 341 / 349) hands every pellet only a part of its drawn sprite: the pixels of one sprite
# are split between neighbours by the claim, bridged by closings, or replaced by a disc around the pellet's eye. So (a) big parts of the
# ghost / leaf sprites stay unlabelled ("missing") and (b) the outline of what is labelled follows the claim cuts and closing bridges
# ("ragged", "holes").
# Rule (owner: one label per bullet, in the bullet's own outline, filled solid, smooth; separate bullets never merged): for the drawn pellets of
# these types (finished-but-still-drawn pellets included: same ruling as 015 / 390) the label is rebuilt from the drawn sprite itself:
#   1. sprite pixels = front-most draw shows a clip body / over-body shape (code 1 / 4, coverage >= CUT); 341 pods also code 5 near the prediction
#      (as idlabels' SHARED_NEAR), 341 super also the opaque fins (code 2, coverage >= CUT);
#   2. seeds = each pellet's predicted body (BL.labels partition); every sprite pixel goes to the pellet with the nearest seed pixel (Voronoi, no
#      seam), pixels farther than SH3_RMAX(D) from every seed are not a pellet (hit streaks, splash lumps) and are dropped;
#   3. label = sprite pixels (opened 1.5 px against thin slivers) + the predicted body where nothing opaque of something else is on top (so a
#      translucent / additive ghost keeps its outline, and hidden parts keep their shape); supers also keep the installed label (rear leaves);
#   4. per pellet: closing, holes filled, spurs / frays opened away (disc opening, radius per type), Gaussian smoothing at 0.5, small detached
#      parts dropped; 1-4 px channels between touching pellets are closed (nearest label), no seam line.
# Installed fixes are kept (this file runs after them and only replaces the body labels of these 8 types).
import numpy as _sh3_np
from scipy.ndimage import (binary_fill_holes as _sh3_fill, label as _sh3_cc, distance_transform_edt as _sh3_edt,
                           binary_closing as _sh3_bclose, binary_opening as _sh3_bopen, gaussian_filter as _sh3_gauss)
import bulletlabels as _sh3_BL

SH3_TYPES = {6000004, 6000005, 6000006, 6000007, 6000012, 6000013, 6000014, 6000015}
SH3_RMAXD = {6000012: 3.0, 6000005: 30.0, 6000007: 30.0, 6000013: 30.0, 6000015: 30.0}   # supers: several leaves / gems per object
SH3_GLOWOP = {6000013}   # 341 super: the fish's fins are the clip's 'glow' layer but drawn opaque (coverage >= CUT): part of the fish
SH3_PRED_ON_SPRITE = {6000004, 6000005}   # 364 stars: the predicted body only where a drawn star sprite is (its span between two stars is no star)
SH3_OLD = {6000007, 6000013, 6000015}   # supers: the installed label is kept as part of the pellet (rear translucent leaves)
SH3_RMAX = 6.0        # px (label res): a sprite pixel farther than this from every seed of the type is not a pellet of this shot
SH3_CLOSE = 1         # px: disc closing between the parts of one pellet
SH3_SIGMA = 1.1       # px: Gaussian smoothing of the pellet outline (threshold 0.5)
SH3_GAPC = 2          # px: channels between the labels of one shot up to 2 x this wide are closed
SH3_MIN = 12          # px: a pellet label smaller than this is not a label (a speck)
SH3_PART = 0.2        # detached parts of one pellet smaller than this share of its largest part are dropped


def _sh3_disc(r):
    k = int(_sh3_np.ceil(r)); y, x = _sh3_np.ogrid[-k:k + 1, -k:k + 1]
    return (x * x + y * y) <= r * r + 1e-6


def _sh3_closed(m, r):
    p = int(_sh3_np.ceil(r)) + 2
    return _sh3_bclose(_sh3_np.pad(m, p), structure=_sh3_disc(r))[p:-p, p:-p]


def _sh3_opened(m, r):
    p = int(_sh3_np.ceil(r)) + 2
    return _sh3_bopen(_sh3_np.pad(m, p), structure=_sh3_disc(r))[p:-p, p:-p]


SH3_OPEN = {6000012: 4.2}          # 341 pods: round bodies; the splash tails of the trail are thin / small lumps, opened away
SH3_NOKEEP = {6000012}


def _sh3_clean(m, free, keep=None, ropen=2.0):
    """one solid, smooth pellet silhouette from its pixels m; may only grow into the pixels in `free`"""
    if not m.any(): return m
    g = _sh3_closed(m, SH3_CLOSE) & (m | free)
    g = _sh3_fill(g | m) & (g | m | free)
    L, n = _sh3_cc(g, structure=_sh3_np.ones((3, 3)))
    if n > 1:
        sz = _sh3_np.bincount(L.ravel())[1:]
        g = _sh3_np.isin(L, 1 + _sh3_np.nonzero(sz >= SH3_PART * sz.max())[0])
    for r_ in sorted({ropen, max(2.0, ropen - 1.0), 2.0}, reverse=True):     # spurs / frays / thin slivers of other sprites
        op = _sh3_opened(g, r_)
        if op.sum() >= 0.8 * g.sum():
            g = _dilate(op, 1) & g
            if keep is not None: g = g | (m & keep)             # the predicted ghost outline itself keeps its tips
            break
    sm = _sh3_gauss(g.astype(_sh3_np.float32), SH3_SIGMA) >= 0.5
    sm &= (g | free)
    if sm.sum() >= 0.8 * g.sum(): g = sm
    g = _sh3_fill(g)
    L, n = _sh3_cc(g, structure=_sh3_np.ones((3, 3)))
    if n > 1:
        sz = _sh3_np.bincount(L.ravel())[1:]
        g = _sh3_np.isin(L, 1 + _sh3_np.nonzero(sz >= SH3_PART * sz.max())[0])
    return g


_sh3_prev_labels = labels


def labels(objs, cam, R, state=None, ended=(), img=None):
    P = _sh3_prev_labels(objs, cam, R, state=state, ended=ended, img=img)
    sel = [o for o in objs if o['t'] in SH3_TYPES]
    if not sel: return P
    ids = P['ids'].copy(); share = P['share'].copy(); lids = list(P['lids'])
    H, W = ids.shape
    cv = R['cv'][::2, ::2]; ids_h = R['id'][::2, ::2]
    # the prediction (partition of the cluster by pellet), as idlabels computes it
    Q = _sh3_BL.labels(objs, cam, cut=0.01)
    qids, qlids, qsh = Q['ids'], Q['lids'], Q['share']
    changed = False
    for t in sorted({o['t'] for o in sel}):
        ot = [o for o in sel if o['t'] == t]
        live = [o for o in ot if o.get('drawn', True)]
        # remove the installed body labels of this type's objects (rebuilt below); ground labels stay
        oldk = {}
        for o in ot:
            lb = (_sh3_BL._lid(o), -1, 'body')
            if lb in lids: oldk[o['id']] = lids.index(lb)
        if not live:
            continue
        C = sprite_pixels(t, live[0]['own'], R)
        spr = ((C == 1) | (C == 4)) & (cv >= CUT)
        seed = _sh3_np.full((H, W), -1, _sh3_np.int32)
        # sprites shared by the clip and its trail (code 5: the 341 pods): the pod's own near the predicted body, as idlabels does (SHARED_NEAR)
        c5 = (C == 5)
        for j, o in enumerate(live):
            # predicted body of this pellet
            ks = [k for k, l in enumerate(qlids) if l[0] == _sh3_BL._lid(o) and l[2] == 'body']
            if ks:
                pm = _sh3_np.isin(qids, ks) & (qsh >= _sh3_BL.CUT)
                seed[pm & (seed < 0)] = j
        if not (seed >= 0).any(): continue
        d, (iy, ix) = _sh3_edt(seed < 0, return_indices=True)
        own = seed[iy, ix]
        if c5.any():
            spr = spr | (c5 & (cv >= CUT) & (d <= min(SHARED_NEAR, SH3_RMAXD.get(t, SHARED_NEAR))))
        if t in SH3_GLOWOP: spr = spr | ((C == 2) & (cv >= CUT))
        U = spr & (d <= SH3_RMAXD.get(t, SH3_RMAX))
        # remove old labels of this type's objects
        oldmask = {}
        for k in oldk.values():
            m = ids == k; oldmask[k] = m.copy(); ids[m] = -1; share[m] = 0
        newk = []
        masks = []; keeps = []
        wd = R.get('wd'); wref = R['meta'].get('wref', 0.0)
        for j, o in enumerate(live):
            m = U & (own == j)
            if m.any():                                           # thin slivers of other sprites (a hit streak beside the pellet) off
                mo = _dilate(_sh3_opened(m, 1.5), 1) & m
                if mo.sum() >= 0.6 * m.sum(): m = mo
            # the pellet's predicted body (where the sprite is drawn translucent / additive the ID pass shows no opaque body: the
            # prediction is the ghost's own outline there), where nothing nearer the camera is drawn in front
            ks = [k for k, l in enumerate(qlids) if l[0] == _sh3_BL._lid(o) and l[2] == 'body']
            if ks:
                pm = _sh3_np.isin(qids, ks) & (qsh >= _sh3_BL.CUT)
                if wd is not None and o.get('anchor') is not None:
                    dz = _eye_depth(cam, o['anchor'], wref)[0]
                    pm &= wd >= dz - DEPTH_TOL
                if t in SH3_PRED_ON_SPRITE: pm &= _dilate(spr, 2)
                pm &= ~((cv >= CUT) & (C == 0))                   # an opaque draw of something else on top (explosion flash, a character): not the pellet
                m = m | pm
                keeps.append(pm)
            else: keeps.append(None)
            if t in SH3_OLD:
                k = oldk.get(o['id'])
                if k is not None: m = m | oldmask[k]
            masks.append(m)
        taken = _sh3_np.zeros((H, W), bool)
        # larger pellet sprites first would favour them; keep it deterministic: by object order
        for j, o in enumerate(live):
            m = masks[j]
            if not m.any(): continue
            free = (ids < 0) & ~taken & ((own == j) | (d > 0))
            g = _sh3_clean(m, free & (own == j), None if t in SH3_NOKEEP else keeps[j], SH3_OPEN.get(t, 2.0))
            g &= ~taken & (ids < 0)
            if g.sum() < SH3_MIN: continue
            lb = (_sh3_BL._lid(o), -1, 'body')
            if lb not in lids: lids.append(lb)
            kb = lids.index(lb)
            ids[g] = kb; share[g] = _sh3_np.maximum(cv[g] / 255.0, _sh3_BL.CUT); taken |= g
        # gaps between the pellets' labels (cleaning each pellet alone leaves 1-3 px channels along the shared edges): given to the nearest label
        kk = [k for k, l in enumerate(lids) if l[2] == 'body' and l[0] in {_sh3_BL._lid(o) for o in live}]
        if kk:
            Ul = _sh3_np.isin(ids, kk)
            cnt = _sh3_np.zeros((H, W), _sh3_np.int8)
            for k in kk:
                mk = ids == k
                if mk.any(): cnt += _dilate(mk, SH3_GAPC).astype(_sh3_np.int8)
            gap = (cnt >= 2) & ~Ul & (ids < 0) & ((C > 0) | (seed >= 0))     # between two touching pellets only; sprite / predicted pixels, never bare ground
            if gap.any():
                _, (iy, ix) = _sh3_edt(~Ul, return_indices=True)
                ids[gap] = ids[iy[gap], ix[gap]]; share[gap] = _sh3_np.maximum(cv[gap] / 255.0, _sh3_BL.CUT)
        changed = True
    if not changed: return P
    P2 = dict(P); P2.update(ids=ids, share=share.astype(_sh3_np.float32), lids=lids)
    return P2
