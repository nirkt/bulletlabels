# installed 11-02 from out/sweep2/haikutasks/siteai3/shelly_poco_gale/fix.py (regression round: 240/280 per-pellet + seams, 256 Poco loop/spur out, 283/370 Gale a90 one oval per snowball; single-skin types)
# Fix file, task shelly_poco_gale (regression-sweep fix round 2026-11-02): skins 240, 256, 280, 283, 370.
# Exec'd in idlabels' namespace after the installed fixes (IDLABELS_OVERRIDE). Keyed to explicit type ids only.
# Each part names the reviewer finding it answers and why it is a real defect (checked on sw2_NNN captures).
#
# A. 240 Shelly003 attack (type 6000006, ShotgunGirlWitch), attack 0 (t265) and attack 90 (t305-t313).
#    Finding: 'attack-0 pellet cloud is one hull; Shelly's pellets should each have their own outline'.
#    Real: one shot = five pellet objects (2000015-19 / 2000020-24), all five with the same predicted region, so the claim
#    gave the whole cluster to the first object (one label, one outline). The installed z_1027_s240 stack rule merges the
#    touching pellets into ONE label on purpose (10-27 hole fix); the newer owner rule for Shelly is one outline per pellet
#    (no hull). Fix: the cluster's pixels (the union of the pellets' labels, whatever the stack rule did) are split per
#    pellet by the nearest pellet anchor (projected, the pellet's own position), and a 1-px seam is left between two
#    pellets along their shared edge (the nearer pellet keeps the edge pixel), so each pellet shows its own outline. Each
#    piece is kept whole (holes filled, specks under 8 px dropped). No hull. Only the pixels already labelled are split;
#    nothing new is added. Ended / finished pellets stay unlabelled (they are not members).
#
# B. 280 Shelly 'Princess' attack (6000010, attack 90 t305) and super (6000011, super t345-t351).
#    Finding: 'Shelly pellets are merged under one outline in attack90 t305 and super t347'.
#    Real on the page: the labels are already one per pellet (fix2 disc / per-pellet closing), but touching pellets have
#    no seam, so the outline drawn around the union reads as one merged shape. Fix: a 1-px seam between touching pellets
#    of these two types (the farther pellet by depth gives up the shared edge pixel), as 364 and 240 super (z_1104_shelly).
#
# C. 370 BlowerDeepsea and 283 BlowerBusters Gale attack 90 (types 6000739 / 6000737, stage angle 270; 442 = 6000741 is
#    NOT touched here). Findings: 370 'alternate bulbs have no outline' (t309, t313) and 283 'attack90 wave split into
#    several ragged outline pieces' (t305, t307); reviewer also asked for a hull.
#    Checked: 370 t305 / t309 the live snowballs 2000026 (and 2000024 at t305) hold 1-9 label px while they are drawn and
#    not ended: a real missing label (the claim gave their shared pixels to the neighbour that came first). 283 pieces are
#    ragged, with spurs and a thin sliver. Owner rule: separate snowballs keep their own outlines, so NO hull over the row.
#    Fix (like z_1104's attack-0 column fix, for the attack-90 row): each live snowball owns the body pixels (drawn body
#    codes 1 and 5, coverage >= 128) inside an ellipse around its own projected anchor, ownership by the nearest normalised
#    anchor; its label is the convex hull of those pixels, holes filled. Ellipse 370: 7.5 x 13 px; 283: 9 x 16 px (bigger
#    ghost sprites). Only while at least 3 of the row's snowballs are labelled by the base (the row is alive).
#
# D. 256 PocoStarr attack 90 (6000343, Poco005Projectile, zone), t315: a small detached loop at the top right of the zone
#    joined by a thin neck (the 'zigzag interior line / notch' of the finding). Handled in part D below.

import numpy as _spg_np
from scipy.ndimage import binary_fill_holes as _spg_fill, label as _spg_cc, binary_dilation as _spg_dil
from scipy.ndimage import binary_opening as _spg_open
from scipy.spatial import ConvexHull as _spg_hull
from PIL import Image as _spg_Image, ImageDraw as _spg_Draw
import masks as _spg_masks


def _spg_anchor(o, cam):
    a = _spg_masks.project(cam, o['anchor'][None])[0][0]
    return _spg_np.array([a[0] / 2.0, a[1] / 2.0])      # projection is in the 2400 x 1080 frame; labels are 1200 x 540


def _spg_hullmask(pts, shape):
    I = _spg_Image.new('L', (shape[1], shape[0]), 0)
    if len(pts) < 3:
        for x, y in pts: I.putpixel((int(x), int(y)), 1)
        return _spg_np.array(I).astype(bool)
    try:
        h = _spg_hull(_spg_np.asarray(pts, float))
    except Exception:
        return None
    _spg_Draw.Draw(I).polygon([tuple(map(float, pts[k])) for k in h.vertices], fill=1, outline=1)
    return _spg_np.array(I).astype(bool)


def _spg_seam(owner, dist, H, W):
    """1-px seam along every shared edge between two different owners: the pixel farther from its own anchor goes"""
    rm = _spg_np.zeros((H, W), bool)
    for dy, dx in ((0, 1), (1, 0)):
        Aa = (slice(0, H - dy), slice(0, W - dx)); Bb = (slice(dy, H), slice(dx, W))
        pair = (owner[Aa] >= 0) & (owner[Bb] >= 0) & (owner[Aa] != owner[Bb])
        if not pair.any(): continue
        da = dist[Aa]; db = dist[Bb]
        rm[Aa] |= pair & (da > db)
        rm[Bb] |= pair & (da <= db)
    return rm


_spg_prev_labels = labels


def labels(objs, cam, R, state=None, ended=(), img=None):
    import bulletlabels as BL
    P = _spg_prev_labels(objs, cam, R, state=state, ended=ended, img=img)
    H, W = P['ids'].shape
    cv = R['cv'][::2, ::2]
    ids, lids, share = P['ids'].copy(), list(P['lids']), P['share'].copy()
    changed = False

    # ---- A: 240 Shelly003 attack (6000006): split the pellets' union per pellet, with seams -----------------------------
    mem = []
    for o in objs:
        if o['t'] != 6000006 or o.get('finished', 0) or o.get('ended') or o.get('anchor') is None: continue
        lid = (BL._lid(o), -1, 'body')
        if lid not in lids: lids.append(lid)     # a pellet with no label yet (the claim gave it nothing) still owns its share
        mem.append((o, lids.index(lid)))
    if len(mem) >= 2:
        ks = [k for _, k in mem]
        U = _spg_np.isin(ids, ks)
        if U.any():
            ys, xs = _spg_np.nonzero(U)
            A = _spg_np.array([_spg_anchor(o, cam) for o, _ in mem])
            d = _spg_np.sqrt((xs[:, None] - A[None, :, 0]) ** 2 + (ys[:, None] - A[None, :, 1]) ** 2)
            owner = _spg_np.full((H, W), -1, _spg_np.int32); dist = _spg_np.full((H, W), _spg_np.inf, _spg_np.float32)
            owner[ys, xs] = d.argmin(1); dist[ys, xs] = d.min(1)
            rm = _spg_seam(owner, dist, H, W)
            keep = U & ~rm
            ids[U] = -1; share[U] = 0
            for j, (o, k) in enumerate(mem):
                m = keep & (owner == j)
                if not m.any(): continue
                L, n = _spg_cc(m)
                if n > 1:
                    sz = _spg_np.bincount(L.ravel())[1:]
                    m = _spg_np.isin(L, 1 + _spg_np.nonzero(sz >= 8)[0])
                m = _spg_fill(m) & (((owner == j)) | ((owner < 0) & (ids < 0))) & ~rm
                if not m.any(): continue
                ids[m] = k; share[m] = _spg_np.maximum(cv[m] / 255.0, BL.CUT)
            changed = True

    # ---- B: 280 Shelly (6000010 attack, 6000011 super): one-pixel seam between touching pellets ----------------------------
    wd = R.get('wd')
    if wd is not None:
        types_b = {6000010, 6000011}
        kb = set()
        for o in objs:
            if o['t'] not in types_b or o.get('finished', 0) or o.get('ended'): continue
            if o['t'] == 6000010 and not (60 <= float(o.get('angle', 0)) % 360 <= 120): continue   # 280 attack: the attack-90 row only (shot angle 79-101)
            lid = (BL._lid(o), -1, 'body')
            if lid in lids: kb.add(lids.index(lid))
        if len(kb) >= 2:
            own_b = _spg_np.isin(ids, list(kb))
            rm = _spg_np.zeros((H, W), bool)
            for dy, dx in ((0, 1), (1, 0)):
                Aa = (slice(0, H - dy), slice(0, W - dx)); Bb = (slice(dy, H), slice(dx, W))
                pair = own_b[Aa] & own_b[Bb] & (ids[Aa] != ids[Bb])
                if not pair.any(): continue
                da = wd[Aa]; db = wd[Bb]
                rm[Aa] |= pair & (da > db)
                rm[Bb] |= pair & (da <= db)
            if rm.any():
                ids[rm] = -1; share[rm] = 0; changed = True

    # ---- C: Gale attack 90 snowball rows (6000737 = 283, 6000739 = 370; stage angle 270) ---------------------------------
    GA90 = {6000737: (9.0, 16.0, (1, 2, 5), 48),    # 283 BlowerBusters: big translucent ghost snowballs (their body is
            #                                             mostly the clip glow code 2, coverage 50-200), about 12.6 px apart
            6000739: (7.5, 13.0, (1, 5), 128)}      # 370 BlowerDeepsea: opaque round bulbs, about 12.6 px apart
    on = [o for o in objs if o['t'] in GA90 and not o.get('ended') and not o.get('finished', 0)
          and o.get('anchor') is not None and int(o.get('stage_angle', -1)) % 360 == 270]
    if on:
        base_n = 0
        for o in on:
            lid = (BL._lid(o), -1, 'body')
            if lid in lids and (ids == lids.index(lid)).sum() >= 15: base_n += 1
        if base_n >= 3:
            yy, xx = _spg_np.mgrid[0:H, 0:W]
            ks, A, RX, RY, Cs, CODES, CUTS = [], [], [], [], [], [], []
            for o in on:
                lid = (BL._lid(o), -1, 'body')
                if lid not in lids: lids.append(lid)
                ks.append(lids.index(lid)); A.append(_spg_anchor(o, cam))
                rx, ry, cd, cc_ = GA90[o['t']]; RX.append(rx); RY.append(ry); CODES.append(cd); CUTS.append(cc_)
                Cs.append(sprite_pixels(o['t'], o['own'], R))
            ks_arr = _spg_np.array(ks)
            gone = _spg_np.isin(ids, ks_arr); ids[gone] = -1; share[gone] = 0
            dmin = _spg_np.full((H, W), _spg_np.inf, _spg_np.float32); own = _spg_np.full((H, W), -1, _spg_np.int32)
            for j in range(len(on)):
                dj = _spg_np.sqrt(((xx - A[j][0]) / RX[j]) ** 2 + ((yy - A[j][1]) / RY[j]) ** 2)
                m = dj < dmin; dmin[m] = dj[m]; own[m] = j
            for j in range(len(on)):
                sel = _spg_np.isin(Cs[j], list(CODES[j])) & (cv >= CUTS[j]) & (own == j) & (dmin <= 1.0)
                if sel.sum() < 3: continue
                pts = _spg_np.c_[_spg_np.nonzero(sel)[1], _spg_np.nonzero(sel)[0]]
                hm = _spg_hullmask(pts, (H, W))
                if hm is None: continue
                hm = _spg_fill(hm & (own == j) & (dmin <= 1.0)) & (own == j)
                if not hm.any(): continue
                hm = hm & (ids < 0)
                ids[hm] = ks[j]; share[hm] = _spg_np.maximum(cv[hm] / 255.0, BL.CUT)
            changed = True

    # ---- D: 256 PocoStarr attack 90 (6000343, zone): a detached loop joined by a thin neck is cut off ------------------
    # (the finding's 'zigzag interior line and notch' at t315; the zone itself stays the cloud's one solid piece)
    zk = []
    for o in objs:
        if o['t'] != 6000343 or o.get('ended'): continue
        lid = (BL._lid(o), -1, 'body')
        if lid in lids: zk.append(lids.index(lid))
    for k in sorted(set(zk)):
        Z = ids == k
        if Z.sum() < 200: continue
        # small enclosed holes of the zone (a dark spot inside the cloud): filled
        H_ = _spg_fill(Z) & ~Z & (ids < 0)
        L, n = _spg_cc(H_)
        if n:
            sz = _spg_np.bincount(L.ravel())[1:]
            small = _spg_np.isin(L, 1 + _spg_np.nonzero(sz <= 60)[0])
            if small.any():
                ids[small] = k; share[small] = _spg_np.maximum(cv[small] / 255.0, BL.CUT); changed = True
                Z = ids == k
        # a thin arm / loop joined by a neck (5 px opening): cut off, the main part kept with 2 px of its edge
        op = _spg_open(Z, structure=_spg_np.ones((5, 5), bool))
        L, n = _spg_cc(op)
        if n < 1: continue
        sz = _spg_np.bincount(L.ravel())[1:]
        main = _spg_np.isin(L, 1 + _spg_np.nonzero(sz >= 0.15 * sz.max())[0])
        cut = Z & ~_spg_dil(main, iterations=2)            # what the opening cut off, beyond 2 px of the main part
        Lc, nc = _spg_cc(cut)
        if nc:
            szc = _spg_np.bincount(Lc.ravel())[1:]
            drop = _spg_np.isin(Lc, 1 + _spg_np.nonzero(szc <= 150)[0])   # thin spurs / loops only: blobs (puffs) stay
            if drop.any():
                ids[drop] = -1; share[drop] = 0; changed = True
    if not changed: return P
    P2 = dict(P); P2['ids'] = ids; P2['share'] = share; P2['lids'] = lids
    return P2
