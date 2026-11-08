# hosted review 2026-11-01 fix (Haiku fix, reviewed by the coordinator, installed 11-01 LOCAL ONLY; source out/sweep2/haikutasks/site1101/s201/fix.py; runs after z_1101_*)
# Skin 201 (TntGuyProgressionSkinLevel9), hosted review 2026-11-01, task s201. THE USER (page note, binding): 'each attack
# should get a hull'. Flagged: super 45 t345, t347, t349, t355, t361, t367 (no reason given); the same page's attack 0 and
# attack 90 rows show the same defect (the labels are split into several red / magenta pieces per projectile).
#
# Types (projtab; the same two types are used by the other TntGuyProgressionSkinLevel3..11 skins, see the report):
#   6000289 Mike012Projectile       (attack: dynamike_012_atk_projectile, a yellow bone-like piece with cyan swirl heads;
#                                    a shot is 2 projectiles, ids 2000006/2000007 at attack 0, 2000008/2000009 at attack 90)
#   6000290 Mike012UltiProjectile   (super: dynamike_012_ulti_projectile, one object per shot: a box with a teal bar, a yellow
#                                    chick on top and two cyan swirls)
# The overcharged variants (6000291..6000293) are not touched.
#
# What was wrong: the label of each projectile was its drawn pieces (the yellow arms, the swirl discs, the box faces) as
# separate outlines with notches and holes; the super had 3-6 pieces per tick. Cyan spark dots of the effects touching the
# box were labelled too, and the super's label reached them.
#
# Fix (keyed to the two types only; a labels() wrapper like z_1101 / s073):
#  1. Own body per projectile: the largest part of its label, plus the other parts that have clip-body pixels of their own
#     (sprite code 1 of the type's clip, the arm tips and the rim of the disc) or are at least 12 px. Parts without clip-body
#     pixels and under 12 px are the sparks / glow dots of the effects: dropped.
#  2. Groups: projectiles of one shot whose convex hulls touch fuse into one visible attack (attack 0 t265-277, attack 90
#     t305-309: one hull over the group). Separate projectiles (attack 0 t281, attack 90 t313-321, the super) keep one hull each.
#     The group's label id is its lowest object id; the other member ids stay empty on that tick (the ShieldTank / Emz
#     convention). A projectile that has a label keeps one: no group loses its label.
#  3. ONE convex hull per group over the group's own body (rule 'hull' of ONE_LABEL). Unlabelled pixels inside the hull are
#     taken (the translucent swirl centres and the shading between the arms, which the ID pass does not attribute to the
#     clip): the label is one solid piece in the hull's outline, no holes, no notches.
#  4. Props: an opaque world draw (blend 0, depth write) that is nearer the camera than every visible part of the group is
#     a prop in front and is cut from the hull. This is inactive on all frames of this skin (checked), it is a safety for
#     walls / crates standing in front of an attack.
#  5. Glow, effects and the brawler are never part of the hull's body: the body is the projectile's label as it was before
#     the fix (its clip / trail pixels chosen by the existing rules), minus the specks above. The hull grows only into
#     unlabelled pixels, never into another object's label; no other type's label is changed.
#
# Other types and skins are not changed. The wrapper leaves every label that is not one of the two types as it was.

import numpy as _np

S201_TYPES = {6000289, 6000290}
S201_SPECK = 12          # px: a part smaller than this with no clip-body pixel of its own is an effect speck
_s201_prev_labels = labels


def labels(objs, cam, R, state=None, ended=(), img=None):
    import bulletlabels as BL
    from scipy.ndimage import label as _cc
    P = _s201_prev_labels(objs, cam, R, state=state, ended=ended, img=img)
    if not any(o['t'] in S201_TYPES for o in objs): return P
    ids, lids, share = P['ids'], P['lids'], P['share']
    wd = R.get('wd'); cv = R['cv'][::2, ::2]
    ids_h = R['id'][::2, ::2]
    wopaque = [int(d) for d, r in R['draws'].items() if r.get('blend') == 0 and r.get('dmask') == 1]
    wworld = _np.isin(ids_h, wopaque)
    mem = []; codes = {}
    for o in objs:   # pass 1: each projectile's own body
        if o['t'] not in S201_TYPES or o.get('ended'): continue
        lid = (BL._lid(o), -1, 'body')
        if lid not in lids: continue
        k = lids.index(lid); m = ids == k
        if not m.any(): continue
        L, n = _cc(m, structure=_np.ones((3, 3), bool))
        sz = _np.bincount(L.ravel()); sz[0] = 0; big = int(sz.argmax())
        keep = L == big
        ck = (o['t'], o['own'])
        if ck not in codes: codes[ck] = sprite_pixels(o['t'], o['own'], R)   # the type's own sprite codes (1 = clip body)
        Cc = codes[ck]
        for j in range(1, n + 1):
            if j == big: continue
            cj = L == j
            if sz[j] >= S201_SPECK or (Cc[cj] == 1).any(): keep |= cj
        ids[m & ~keep] = -1                                  # the specks are dropped
        mem.append(dict(o=o, k=k, keep=keep, hull=hull_mask(keep)))
    # pass 2: projectiles whose hulls touch form one visible attack: one hull over the group's bodies
    par = list(range(len(mem)))
    def find(i):
        while par[i] != i: par[i] = par[par[i]]; i = par[i]
        return i
    for i in range(len(mem)):
        for j in range(i + 1, len(mem)):
            if (mem[i]['hull'] & mem[j]['hull']).any(): par[find(j)] = find(i)
    groups = {}
    for i in range(len(mem)): groups.setdefault(find(i), []).append(i)
    for g in groups.values():
        G = sorted(g, key=lambda i: mem[i]['o']['id']); car = mem[G[0]]['k']
        K = _np.zeros(ids.shape, bool)
        for i in G: K |= mem[i]['keep']
        hm = hull_mask(K)
        if wd is not None and K.any():   # props in front: opaque world draws nearer than every visible part of the group
            hm = hm & ~(wworld & (wd < float(wd[K].min()) - DEPTH_TOL) & ~K)
        new = hm & ~K & (ids < 0)        # unlabelled pixels only: other objects' labels stay as they are
        for i in G: ids[mem[i]['keep']] = car
        ids[new] = car
        share[new | K] = _np.maximum(cv[new | K] / 255.0, BL.CUT)
    return P
