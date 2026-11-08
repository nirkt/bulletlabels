# sweep 10-27 fix for skin 336 (Sonnet revision, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s336/)
# Beamer 003 attack (type 6001302, Beamer003Projectile; skin 336 Beamerharvest). Sweep 2026-10-27, task s336: pairs 6288 and
# 6289 (attack 0 deg, ticks 265 and 267: the beam tip is cut off, the red outline ends 15-20 px short of the white front) and
# pairs 6293 and 6294 (attack 90 deg, ticks 305 and 307: the lower tip is cut off, the outline closes above the white end).
# Cause: the beam's white core (shape 150 of sc/effects_brawler.sc) is drawn as glow-coded, translucent sprite pixels (coverage
# 26-40 at the front). idlabels takes glow pixels into the body only inside the aligned prediction, so the front of the core
# beyond the logic beam's end was left out. The same shape also forms a faint fringe along the sides and a blue translucent
# tail below the white core, which must stay out (rules: no soft halo, no glow).
# Measured on sw2_336 (base and fixed label masks per tick, dump336.py / white336.py): the tip pixels lie 0-19 px beyond the
# far end of the label along its axis; the side fringe lies beside the label; the tail is blue (min RGB 90-140) and the white
# front is min RGB >= 150 (tick 305: 94 white pixels beyond the end, 307: 94, 309: 113, 311: 100, 313: 0, 315: 25).
# Fix (POST_BODY, after the per-type rules): add the core's pixels (shape 150, coverage >= BEAM3_CV, inside the label region)
# that lie beyond the body's own far end along its principal axis, are white (min RGB >= BEAM3_WHITE) and touch the body
# (8-connected). The candidates are opened with a 3x3 element first (1-2 px slivers along an edge are not the front), and
# the notches where the added front meets the old label are closed with the core's own pixels only. Nothing else is added:
# no pixel of the old label is removed, and nothing changes unless the body is elongated (the axis is defined).
# Checked on ticks 265-269 (attack 0 deg) and 305-315 (attack 90 deg): removed 0 px on every tick; added 89-122 px per tick
# on 265-311 (the tip), 0 on 313 and 315 (after the opening). Other types are not touched.
# (the final run's masks are fix0c.npz and fix90d.npz beside this file)
BEAM3_TYPE = 6001302
BEAM3_FILE = 'sc/effects_brawler.sc'
BEAM3_CORE = 150         # atlas shape of the beam's white core (its front and shaft); shape id, +1 in _a4_shape_pixels
BEAM3_CV = 16            # coverage (0..255) from which a core pixel is the beam (the glow-coded front is at 26-40)
BEAM3_WHITE = 225        # min(R, G, B) of the frame from which a core pixel is the PURE white front (cyan block, pale fade and blue tail are below it)


def _beam3_core(body, ctx):
    from scipy.ndimage import label as cc
    if not body.any() or ctx.get('img') is None: return body
    t = ctx['t']; T = projtab.table()[t]
    names = tuple(sorted({n for f, n in _trail_clips(t) if f == BEAM3_FILE} | {n for n in (T['blue'], T['red']) if n}))
    SP = _a4_shape_pixels(ctx['R'], BEAM3_FILE, names)   # shape id + 1 per pixel, 0: none
    core = (SP == BEAM3_CORE + 1) & ctx['region'] & (ctx['cv'] >= BEAM3_CV)
    if not core.any(): return body
    ys, xs = np.nonzero(body); P = np.stack([ys, xs], 1).astype(float); mu = P.mean(0)
    w, V = np.linalg.eigh(np.cov((P - mu).T)); v = V[:, np.argmax(w)]
    if w.max() < 4 * max(w.min(), 1e-6): return body      # not elongated: no axis, nothing beyond an end
    proj = (P - mu) @ v; lo, hi = proj.min(), proj.max()
    cy, cx = np.nonzero(core); pr = (np.stack([cy, cx], 1).astype(float) - mu) @ v
    beyond = np.zeros(core.shape, bool); beyond[cy, cx] = (pr > hi + 0.5) | (pr < lo - 0.5)
    white = ctx['img'].min(-1) >= BEAM3_WHITE
    cand = core & beyond & white
    from scipy.ndimage import binary_opening
    cand = binary_opening(cand, structure=np.ones((3, 3), bool))   # 1-2 px slivers along an edge (tick 315) are not the front
    if not cand.any(): return body
    L, n = cc(cand | body, structure=np.ones((3, 3), bool))
    keep = np.unique(L[body]); keep = keep[keep > 0]
    out = body | (np.isin(L, keep) & cand)
    # the joint between the old label and the added front leaves small notches (ticks 307, 309 of 6001302): closed, core only
    from scipy.ndimage import binary_closing
    cl = binary_closing(out, structure=np.ones((3, 3), bool), iterations=2)
    return out | (cl & core)


POST_BODY[BEAM3_TYPE] = _beam3_core
