# AI-review batch 2026-11-01 fix (Haiku fix, reviewed by the coordinator, installed 11-01 LOCAL ONLY; source out/sweep2/haikutasks/siteai/misc_c/fix.py)
# misc_c fix file (AI-review findings 2026-11-01; skins 398, 444, 508; 426, 452, 548, 461 are reported, see report.md).
# Exec'd in idlabels' namespace after the installed fixes (IDLABELS_OVERRIDE). Every change is keyed to explicit type ids.
# Owner rules applied: holes and specks out; nothing cut off; thin lines / splintered multi-piece attacks of ONE object =
# one convex hull per object; separate projectiles keep separate labels.
#
# 398 ConductorSupercellMake, attack 6001374 (Conductor003Projectile, rows attack 0 and attack 90). Reviewer (tick 321):
#   "a small interior hole sits inside the cloud label". Cause: the orbs of one cloud are separate projectiles with
#   separate labels (kept, as the owner decided); where two orb labels meet, a 26 px diamond between them is covered by
#   neither label, so the union of the labels has an enclosed gap. Also 1 px seams between touching orbs (ticks 275-315)
#   and a 3 px detached speck at the base of a cyan dart (tick 277). Fix (labels() wrapper, this type's labels only):
#   specks of 6 px or less detached from the label's main part are dropped; enclosed gaps of the union of this attack's
#   labels up to 60 px are given to the nearest label of the attack. Super (6001375) and other ticks are unchanged.

_T398_ATT = 6001374
_T398_GAP = 60      # px: enclosed gaps of the attack's labels up to this size are filled
_T398_SPECK = 6     # px: detached pieces of this attack's labels up to this size are dropped
_t398_prev_labels = labels


def labels(objs, cam, R, state=None, ended=(), img=None):
    from scipy.ndimage import binary_fill_holes, label as cc, distance_transform_edt
    import bulletlabels as BL
    P = _t398_prev_labels(objs, cam, R, state=state, ended=ended, img=img)
    tl = {BL._lid(o) for o in objs if o['t'] == _T398_ATT}
    ids, lids = P['ids'], P['lids']
    ks = [k for k, l in enumerate(lids) if l[0] in tl and l[2] == 'body']
    if not ks: return P
    for k in ks:   # detached specks of this attack's labels
        m = ids == k
        if not m.any(): continue
        L, n = cc(m)
        if n < 2: continue
        sz = np.bincount(L.ravel())[1:]
        for c in range(n):
            if sz[c] <= _T398_SPECK and sz[c] < 0.1 * sz.max(): ids[L == c + 1] = -1
    u = np.isin(ids, ks)
    if u.any():   # enclosed gaps between the attack's labels (seams, the diamond): given to the nearest label of the attack
        gap = binary_fill_holes(u) & ~u & (ids < 0)
        if gap.any():
            GL, gn = cc(gap); small = np.zeros(gap.shape, bool)
            for j in range(1, gn + 1):
                gj = GL == j
                if gj.sum() <= _T398_GAP: small |= gj
            if small.any():
                _, (iy, ix) = distance_transform_edt(~u, return_indices=True)
                ids[small] = ids[iy[small], ix[small]]
                cv = R['cv'][::2, ::2]
                P['share'][small] = np.maximum(cv[small] / 255.0, BL.CUT)
    return P


# 444 InsectManBrawlentine, attack 90 deg: Angelo004Projectile (6001436, the heart arrow). Reviewer (tick 307): "the arrow
#   label leaves the yellow heart lobe beside the stem outside the outline". Checked: the upper-right lobe of the lower
#   heart, beside the stem (frame ~600-603, 362-365), is outside the label; its yellow pixels (shape 19 of this clip)
#   are drawn but not body. Fix: the yellow pixels of shape 19 that touch the body (1 px) join the body. Only a few px
#   per tick (ticks 307-313); the stem itself is solid (no hole).
_T444_ARROW = 6001436
_T444_FILE = 'sc/effects_brawler_angelo.sc'
_T444_CLIPS = ('angelo_004_atk_projectile', 'angelo_004_atk_projectile_red')
_T444_LOBE = 19 + 1   # shape id + 1, as _a4_shape_pixels gives it


def _t444_arrow_body(body, ctx):
    if not body.any(): return body
    SP = _a4_shape_pixels(ctx['R'], _T444_FILE, _T444_CLIPS)
    near = _dilate(body, 1) & ctx['region'] & (SP == _T444_LOBE)
    return body | near


POST_BODY[_T444_ARROW] = _t444_arrow_body


# 508 (6000907, leaf blade) is NOT changed here: a per-object hull overlaps (two objects share the same blades at tick 265)
#   and grows over background (ticks 305-307); see report.md (needs_user).
