# Task shotgun_twins (owner complaints, round siteai5, 2026-11-03): skins 173 AssaultShotgunArcade (super cards),
# 193 TwinsAngel (attack 90 arrow). Exec'd in idlabels' namespace after the installed fixes (IDLABELS_OVERRIDE).
#
# (1) 173 super, type 6001021 Griff003UltiProjectile (the rainbow cards; PLAIN BULLET rule: one label per card, no hull over
#     the shot). Owner: "solid, no interior lines". A card partly under the white flash (t355) lost its hidden part: its label
#     had a notch and a separate strip. Fix: the hidden part is filled from the convex hull of the card's OWN final label (only
#     unclaimed pixels are added, one card at a time, so a hull never reaches a neighbouring card); thin spurs under 3 px are
#     opened away. Measured on sw2_173: cards at t353-359 gain or keep their pixels; pellets (6001019) untouched.
# (2) 176 super, type 6001025 (the bottles): NO code here. The global fill-and-smooth step (zz_1108_fill_smooth.py) already
#     fills the holes and smooths the ragged outlines of each bottle (checked on sw2_176 t343-359: one label per bottle).
# (3) 193 attack 90, type 6001408 Twins003ThrowerProjectile (the arrow). Owner: "the whole arrow (head AND shaft) inside the
#     label". At t305-307 the shaft below the head was outside the label. The shaft is the cyan rod of the arrow's own draw
#     (fragment shader 1680474519, 1024 px texture, no sprite code), in the lowest rows of the body. Fix: only rod pixels that
#     touch the body (within 8 px) are added, up to 25 px from it. The big cyan orb above the head and the twinkles stay out.
#     Runs after z_1102_s193 (wrapped). Measured: t305 +68 px, t307 +34 px, all other ticks unchanged.

_st5_prev_labels = labels


def _st5_disc(r):
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r


def labels(objs, cam, R, state=None, ended=(), img=None):
    import bulletlabels as BL
    P = _st5_prev_labels(objs, cam, R, state=state, ended=ended, img=img)
    cards = [o for o in objs if o['t'] == 6001021]
    if not cards: return P
    ids = P['ids'].copy(); share = P['share'].copy(); lids = P['lids']
    cv = R['cv'][::2, ::2]
    for o in cards:
        lid = (BL._lid(o), -1, 'body')
        if lid not in lids: continue
        k = lids.index(lid); m = ids == k
        if not m.any(): continue
        add = hull_mask(m) & ~m & (ids < 0)
        if add.any():
            ids[add] = k; share[add] = np.maximum(cv[add] / 255.0, BL.CUT)
        # thin spurs (a sliver reaching into the flash beside a card) are released: opening with a 3 px disc keeps the card
        from scipy.ndimage import binary_opening
        m2 = ids == k
        op = binary_opening(m2, structure=_st5_disc(3)) & m2
        if op.any() and op.sum() < m2.sum():
            drop = m2 & ~op
            ids[drop] = -1; share[drop] = 0
    P['ids'] = ids; P['share'] = share
    return P

ST5_FEATHER_CLIPS = ('melodie_002_feather_06', 'melodie_002_feather_06_flip')
ST5_ARROW = 6001408
ST5_SHAFT_MAX = 25     # px: how far from the arrow's own body the attached shaft may reach
_st5_prev_post_193 = POST_BODY.get(ST5_ARROW)


ST5_ROD_FS = 1680474519   # fragment shader of the draw that carries the arrow's rod at attack 90 (t305-307): no sprite code
ST5_ROD_TW = 1024
ST5_TOUCH = 8           # px: the rod counts when it comes within this distance of the body (a gap of the wing's pixels)
ST5_ROD_BAND = 6         # px: the rod counts only in the lowest rows of the body (its bottom edge and below)


def _st5_rod_draws(R):
    return [i for i, dd in R['draws'].items() if dd.get('tw') == ST5_ROD_TW and dd.get('fs') == ST5_ROD_FS and not dd.get('dmask')]


def _st5_arrow_shaft(body, ctx):
    from scipy.ndimage import label as cc, binary_fill_holes, distance_transform_edt
    if not body.any(): return body
    cv, img = ctx['cv'], ctx['img']
    rd = _st5_rod_draws(ctx['R'])
    if not rd: return body
    # the rod hangs below the head: cyan pixels of its draw in the lowest rows of the body only (the same draw also holds
    # the big cyan orb above the head, which is not the arrow); only the part that touches the body is added
    im = img.astype(np.int16)
    cyan = (im[..., 1] > 60) & (im[..., 2] >= 0.8 * im[..., 1]) & (im[..., 2] <= 1.3 * im[..., 1]) & (im[..., 0] <= 0.75 * im[..., 1])   # cyan: G ~ B, R low (not the armour's light blue)
    yb = np.nonzero(body.any(1))[0].max()
    low = np.zeros(body.shape, bool); low[max(0, yb - ST5_ROD_BAND):, :] = True
    feat = np.isin(ctx['ids_h'], rd) & cyan & low & (cv >= 32)
    Lf, nf = cc(feat)
    if nf == 0: return body
    touch = _dilate(body, ST5_TOUCH)
    ks = np.unique(Lf[touch & feat]); ks = ks[ks > 0]
    if ks.size == 0: return body
    add = np.isin(Lf, ks) & feat
    add &= distance_transform_edt(~body) <= ST5_SHAFT_MAX
    body = body | add
    return binary_fill_holes(_close(body, 1) | body)


def _st5_post_193(body, ctx):
    if _st5_prev_post_193 is not None: body = _st5_prev_post_193(body, ctx)
    return _st5_arrow_shaft(body, ctx)


POST_BODY[ST5_ARROW] = _st5_post_193
