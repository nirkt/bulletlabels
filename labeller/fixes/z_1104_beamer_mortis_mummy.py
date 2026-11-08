# installed 11-01 from out/sweep2/haikutasks/siteai2/beamer_mortis_mummy/fix2.py (2nd AI sweep: 405 beam tail shape 133 grown, 482 Mortis003 glow >=64 grown back, 578 tiny swoosh fragment out, 582 Emz006 detached spur out; single-skin types)
# AI-review second sweep 2026-11-01, task beamer_mortis_mummy: complete fix (fix2.py = fix.py + the second pass).
# Exec'd after all installed fixes (IDLABELS_OVERRIDE). Skins 405, 482, 578, 582 changed; 433 and 605 not changed here.
#
# (1) 578 UndertakerSB, Mortis 007 super (6000303), tick 345 (reviewer: "tiny label on the yellow swoosh"): the label was a
#     29 px fragment, not the vortex. A body under 100 px is dropped before the one-hull step. 343 none, 345 none (was 29 px),
#     347 / 349 / 353 unchanged. Type named by no other skin.
#
# (2) 405 BeamerCarnival, Beamer005Projectile (6001306), attack 0 deg, ticks 265-269 (reviewer: "beam label covers only the
#     yellow core; the teal tail and the bright front stay outside"). Cause (shapediag on sw2_405): the teal tail is the
#     beam's own clip body shape 133 (807 px, unlabelled). The tail lies beyond the NEAR window of the prediction, so the
#     per-type rule never reaches it. Fix: connected growth from the label through shape 133 only (the clip's own body shape:
#     not the glow layer 0, not the core shapes, not the trail), coverage >= 32, at most 14 px from the label. Only for a
#     HORIZONTAL beam (principal axis |cos| >= 0.8, attack 0 deg): on the vertical attack 90 deg rows the same growth runs over
#     the brawler's dress where the beam overlays it (not grown). The bright orange tip beyond the core is the glow layer
#     (halo, shape 0) and stays out. Checked on 265 / 267 / 269 against the raw frame: the outline follows the teal tail up to
#     the muzzle and the yellow core; no brawler pixels taken. Attack 90 rows (305-313) unchanged.
#
# (3) 482 UndertakerBazaar, Mortis 003 super (6000300), tick 345-353 (reviewer: "super tornado label is a wedge that leaves
#     the lower-left of the spinning tornado outside"). Cause (traced on sw2_482): the installed step a3_mortis003 drops 1510
#     translucent glow pixels (coverage median 63) across the whole vortex, and _mort_clean drops 894 more glow pixels at the
#     lower-left. Fix: after the installed chain, the glow pixels of coverage >= 64 that touch the body (connected growth,
#     at most 6 px, from the BASE body, so the dropped glow is available) are added back before the one hull. Halo (< 64)
#     stays out. Result 345: 5327 -> 6236 px, 349: 6098 -> 6338 px. The remaining lower-left outside at 345 is the Undertaker
#     brawler's coat in front of the vortex (never labelled).
#
# (4) 582 MummyDemon, Emz006Projectile (6000688), attack 90 deg tick 305 (reviewer: "notch running down into the blue flames
#     and a detached speck at lower left"). Cause (labels of sw2_582, connected parts of the stack label): at 305 the stack
#     label has two parts, the cloud (1194 px) and a detached 144 px piece below it (x 598-611, y 298-317: the spur into the
#     flames). The piece is 12 % of the largest part, so a share rule alone (3 %) would keep it; it touches no predicted body
#     (predicted-body overlap 0), which is the discriminator. Fix: after the Emz stack merge, a part smaller than 15 % of the
#     largest part of the label that does not touch any member's predicted body is dropped. At 307 / 309 / 311 the label is
#     one part and the counts are unchanged (2098 / 3335 / 3369). A radius-3 notch closing was tried: it takes 0 pixels, so it
#     is not used.
#
# Not changed here: 605 Emz007 (6000689), tick 305 "two bright lower lobes outside" (not_fixed: growth through the cloud's
# clip shapes 3 / 37 / 4 adds part of the lobes but leaves loops and a separate closed piece at 307; not clean).
# 433 super (6001311) tick 357 "label across the brawler body": not changed; needs_user (see report).

# ---- (1) 578 ----
_s578_prev_post = POST_BODY[6000303]


def _s578_min_body(body, ctx):
    body = _s578_prev_post(body, ctx)
    if body.any() and int(body.sum()) < 100:
        return body & False      # a speck of the swoosh at the super's first tick, not the vortex
    return body


POST_BODY[6000303] = _s578_min_body

# ---- (2) 405 ----
from scipy.ndimage import binary_dilation as _f2_bd405, label as _f2_cc405
F2_405_TYPE = 6001306
F2_405_SHAPES = {133}                    # the beam's own body shape (teal tail)
F2_405_CV = 32                           # coverage from which a shape-133 pixel is the beam
F2_405_REACH = 14                        # px: the tail may reach this far from the label
F2_405_HORIZ = 0.8                       # |cos| of the principal axis: only a horizontal beam (attack 0 deg)
_f2_prev_405 = POST_BODY.get(F2_405_TYPE)


def _f2_beam405(body, ctx):
    if _f2_prev_405 is not None: body = _f2_prev_405(body, ctx)
    if not body.any(): return body
    ys, xs = np.nonzero(body)
    if len(xs) < 3: return body
    C = np.cov(np.stack([xs, ys]).astype(float))
    w, V = np.linalg.eigh(C); vx = abs(V[0, np.argmax(w)])
    if vx < F2_405_HORIZ: return body
    T = projtab.table()[F2_405_TYPE]
    names = tuple(n for n in (T['blue'], T['red']) if n)
    SP = _a4_shape_pixels(ctx['R'], T['file'], names)
    cand = np.isin(SP - 1, list(F2_405_SHAPES)) & (ctx['cv'] >= F2_405_CV) & _f2_bd405(body, iterations=F2_405_REACH)
    cand = cand | body
    L, n = _f2_cc405(cand, structure=np.ones((3, 3), bool))
    keep = np.unique(L[body]); keep = keep[keep > 0]
    return body | (np.isin(L, keep) & cand)


POST_BODY[F2_405_TYPE] = _f2_beam405

# ---- (3) 482 ----
F2_482_TYPE = 6000300
F2_482_GLOW_CV = 64                      # translucent body of the vortex's lower-left edge; the halo is below 64
F2_482_GROW = 6                          # px: the glow edge is taken at most this far out, connected to the body
_f2_mort_prev = POST_BODY[F2_482_TYPE]   # the installed chain (a3_mortis003, then _mort_clean)


def _f2_mort482(body, ctx):
    out = _f2_mort_prev(body, ctx)
    if not out.any(): return out
    gl = body & (ctx['C'] == 2) & (ctx['cv'] >= F2_482_GLOW_CV) & ~out
    add = np.zeros_like(out); cur = out.copy()
    from scipy.ndimage import binary_dilation as _f2_bd482
    for _ in range(F2_482_GROW):
        g = _f2_bd482(cur, structure=np.ones((3, 3), bool)) & gl & ~add
        if not g.any(): break
        add |= g; cur = cur | g
    return out | add


POST_BODY[F2_482_TYPE] = _f2_mort482

# ---- (4) 582 ----
F2_582_TYPES = {6000688}
F2_582_SHARE = 0.15                      # a part of the stack label smaller than this share of its largest part ...
_f2_prev_labels = labels


def labels(objs, cam, R, state=None, ended=(), img=None):
    import bulletlabels as BL
    P = _f2_prev_labels(objs, cam, R, state=state, ended=ended, img=img)
    mine = [o for o in objs if o['t'] in F2_582_TYPES]
    if not mine: return P
    lids = P['lids']; ids = P['ids']
    mem = {(BL._lid(o), -1, 'body') for o in mine}
    ks = [k for k, l in enumerate(lids) if l in mem]
    if not ks: return P
    U = np.isin(ids, ks)
    if not U.any(): return P
    from scipy.ndimage import label as _f2_cc
    L, n = _f2_cc(U, structure=np.ones((3, 3), bool))
    if n < 2: return P
    sz = np.bincount(L.ravel())[1:]; big = sz.max()
    Pr = BL.labels(mine, cam, cut=0.01)  # the predicted bodies (logic position), not the drawn parts
    pids = Pr['ids']; plids = Pr['lids']
    pk = [k for k, l in enumerate(plids) if l in mem]
    pu = np.isin(pids, pk)
    if pu.shape != U.shape:
        pu = np.repeat(np.repeat(pu, U.shape[0] // pu.shape[0], 0), U.shape[1] // pu.shape[1], 1)
    pu_d = _f2_bd405(pu, iterations=3) if pu.any() else pu
    for j in range(n):
        c = L == (j + 1)
        if sz[j] == big: continue
        if sz[j] < F2_582_SHARE * big and not (c & pu_d).any():
            ids[c] = -1                 # a detached speck or spur of the stack label, away from every predicted cloud
    return P
