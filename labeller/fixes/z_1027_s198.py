# sweep 10-27 fix for skin 198 (Haiku fix, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s198/)
# s198 (AssaultShotgunKaiju, sweep 2026-10-27; pairs 3473 tick 267 attack 0 deg, 3481 tick 307 attack 90 deg).
# Type 6001027 = Griff005Projectile (logic AssaultShotgunProjectile, clip griff_005_atk_projectile, trail griff_005_atk_trail):
# the shotgun's pellets. Its label carried extra parts that are not the pellet: the X-shaped yellow-white sparks drawn
# between the pellet groups (tick 267: label pieces at x 670-677, shapes 47 and 55, 10 px from the pellet; tick 309: sparks
# above the cluster), and small specks / dashes around the chest flame at spawn (tick 307 and later). Rule 3: sparks and
# particles are never labelled; the attack body only.
# Fix, on the final label of each such projectile (after the claims between neighbouring objects and the clean-up): its
# 8-connected parts that do not touch the main part and are smaller than S198_SPECK px are dropped (no gap bridging: a
# chain of small pieces across a gap must not join the body). The main part (the largest) is always kept, so a projectile never loses its label. The pellet body (50 px and
# up in every frame checked) is unchanged. Keyed to the type id only; other skins that use this type get the same rule.
S198_SPECK = 40          # px: a detached part smaller than this is dropped (final label pixels, 1200 x 540 frame)
S198_TYPES = {6001027}   # Griff005Projectile

_s198_prev_labels = labels


def _s198_drop_detached(P, objs):
    import bulletlabels as BL
    from scipy.ndimage import label as _cc, generate_binary_structure as _gbs
    S8 = _gbs(2, 2)
    ids, lids, share = P['ids'], P['lids'], P['share']
    ltype = {}
    for o in objs: ltype.setdefault(BL._lid(o), o['t'])
    for k in np.unique(ids[ids >= 0]):
        if lids[k][2] != 'body' or ltype.get(lids[k][0]) not in S198_TYPES: continue
        m = ids == k
        ys, xs = np.nonzero(m); y0, y1, x0, x1 = max(0, ys.min() - 2), ys.max() + 3, max(0, xs.min() - 2), xs.max() + 3
        sm = m[y0:y1, x0:x1]
        L, n = _cc(sm, S8)     # 8-connected parts of the label as it is
        if n <= 1: continue
        sz = np.bincount(L[sm], minlength=n + 1); sz[0] = 0
        main = int(np.argmax(sz))
        drop = [j for j in range(1, n + 1) if j != main and sz[j] < S198_SPECK]
        if not drop: continue
        d = np.isin(L, drop) & sm
        sid = ids[y0:y1, x0:x1]; ssh = share[y0:y1, x0:x1]
        sid[d] = -1; ssh[d] = 0
        if os.environ.get('S198_DEBUG'): print('S198 label', int(k), 'parts', sz[1:].tolist(), 'dropped', [int(sz[j]) for j in drop])


def labels(objs, cam, R, state=None, ended=(), img=None):
    P = _s198_prev_labels(objs, cam, R, state=state, ended=ended, img=img)
    _s198_drop_detached(P, objs)
    return P
