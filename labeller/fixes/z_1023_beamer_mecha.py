# Beamer Mecha 433 super: no hexagon at the gun (user Q3, 2026-10-23; written by a Haiku agent, reviewed; out/sweep2/bugtrial2/)
# Haiku fix for Beamer Mecha 433 super (Beamer007UltiProjectile, type 6001311, clip mandy_007_ulti_projectile).
# The user's decision (2026-10-23): REMOVE the big cyan hexagon frame at the gun (drawn while the beam starts; in the
# label at ticks 361-363) from the super's label; keep the beam shaft and the arrow head exactly as they are.
# Cause: the hexagon's rim and the arrow head are both atlas shape 40 of the clip (a solid vector fill: every command of
# shape 40 samples the same texel), so the shape cannot tell them apart. Separation uses the projectile's own predicted
# body (ctx['pred_body']: its logic position and radius, game data, not screen coordinates): a shape-40 part is the
# arrow head when at least HEX433_IN of its pixels lie in that prediction (measured: arrow parts 1.00, hexagon parts
# 0.00-0.06 on ticks 361-363). Guard: nothing is changed unless some part is the arrow head (the prediction covers it).
# Leftovers of a removed label part with no pixel in the prediction (the hexagon's own edge) are dropped; the beam's
# shapes (4, 36, 18, 37, 0, 79) are never removed; no-shape pixels within 2 px of the removed rim are dropped unless a
# kept beam shape lies within 2 px.
HEX433_TYPE = 6001311
HEX433_FILE = 'sc/effects_brawler_mandy.sc'
HEX433_RIM = 40          # atlas shape of the hexagon's rim (the arrow head uses the same shape)
HEX433_IN = 0.5          # share of a shape-40 part's pixels inside the projectile's predicted body from which it is the arrow


def _beamer433_hex(body, ctx):
    if not body.any(): return body
    from scipy.ndimage import label as cc
    pb = ctx['pred_body']                                # the projectile's own predicted body (bool, as body)
    names = ('mandy_007_ulti_projectile', 'mandy_007_ulti_projectile_red') + tuple(
        n for f, n in _trail_clips(HEX433_TYPE) if f == HEX433_FILE)
    SP = _a4_shape_pixels(ctx['R'], HEX433_FILE, names)  # shape id + 1 per pixel, 0: none
    rim = body & (SP == HEX433_RIM + 1)
    if not rim.any(): return body
    L, n = cc(rim); hexm = np.zeros_like(body); arrow = False
    for j in range(1, n + 1):
        part = L == j
        if (part & pb).sum() >= HEX433_IN * part.sum(): arrow = True
        else: hexm |= part
    if not (arrow and hexm.any()): return body           # guard: the prediction must cover the arrow head
    beam = body & (SP != 0) & (SP != HEX433_RIM + 1)     # kept beam shapes of the clip and its trail
    out = body & ~hexm
    out &= ~(_dilate(hexm, 2) & ~_dilate(beam, 2) & (SP == 0))   # no-shape edge pixels of the removed rim
    orig, _ = cc(body)                                   # label parts (4-connected) that held hexagon rim
    for k in np.unique(orig[hexm]):
        if k == 0: continue
        rest = out & (orig == k)
        if rest.any() and not (rest & pb).any(): out &= ~rest   # leftovers with no pixel in the arrow's prediction
    return out


POST_BODY[HEX433_TYPE] = _beamer433_hex


# the beam's label frays (thin strips, jagged edges, mostly at its start): one convex hull of the beam + arrow head, after
# the hexagon is removed (the user, 2026-10-23: labels are fed to a model, they must be clean)
ONE_LABEL[HEX433_TYPE] = 'hull'
