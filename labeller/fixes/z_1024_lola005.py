# Darkcircus Lola 005 (Lolla006Projectile 6001099): head + band by clip shapes, trail pieces out (D-shape, spurs, loops) (Haiku task 31, reviewed 10-24)
# Task 31 (skin 005 DuplicatorDarkcircus, Lola): the attack Lolla006Projectile (type 6001099; clip lolla_006_atk_projectile /
# _red, trail lolla_006_atk_trail; its end effects lolla_006_atk_hit / _reached are not labelled, end leftovers are not labelled).
# The user's decision (2026-10-24): each projectile = ONE clean label: its bright head plus the band of segments trailing
# behind it that the projectile draws while it flies (as labelled now), no spurs, specks, holes or frayed edges.
# Found (sw2_005, ticks 269-275): (a) spurs on the band are trail-clip pieces (shapes 624-628) inside the predicted body;
# (b) the D-shaped label right of the head at t269 is trail shape 624: the leftover of obj 2000018, which ended at t269 at the
# wall (ended list), so it goes; (c) the right-most projectile reaching the crate (t273-275) carried trail pieces and
# no-shape edge bits into the crate.
# Fix: the band and head are the clip shapes 13414 (ribbon), 13419 and 13412 (head). Trail-clip pixels are removed; shape-0
# and no-shape pixels are kept only next to the head / band (1-2 px edges); gaps of 2 px between the band's own pieces are
# closed (only pixels of its own draws); pieces with none of the head / band's own shapes (edge bits) are dropped; holes filled.
LOLA005_ATK = 6001099
LOLA005_FILE = 'sc/effects_brawler.sc'
LOLA005_CLIP = ('lolla_006_atk_projectile', 'lolla_006_atk_projectile_red')
LOLA005_HEAD = (13414, 13419, 13412)   # clip shapes of the band (13414) and the head (13419, 13412)
LOLA005_EDGE = 2                       # px: shape-0 / no-shape pixels kept within this of the head or band
LOLA005_GAP = 2                        # px: gaps between the band's own pieces closed (drawn pixels only)


def _lola005_atk(body, ctx):
    if not body.any(): return body
    from scipy.ndimage import binary_fill_holes, label as cc
    R = ctx['R']
    SP = _a4_shape_pixels(R, LOLA005_FILE, LOLA005_CLIP)          # clip shape id + 1 per pixel, 0: none of the clip's shapes
    trail = tuple(n for f, n in _trail_clips(LOLA005_ATK) if f == LOLA005_FILE)
    TR = _a4_shape_pixels(R, LOLA005_FILE, trail) > 0             # trail-clip pieces: spurs, the D, particles
    core = body & np.isin(SP, [s + 1 for s in LOLA005_HEAD])
    if not core.any(): return body & ~TR
    rest = body & ~TR & ~core
    m = core | (rest & _dilate(core, LOLA005_EDGE))
    own_d = np.unique(ctx['ids_h'][core]); own_d = own_d[own_d > 0]   # the draws that show its own head / band
    m = _close(m, LOLA005_GAP) & ctx['region'] & ~TR & (body | (np.isin(ctx['ids_h'], own_d) & _dilate(core, LOLA005_GAP)))
    L, n = cc(m, structure=np.ones((3, 3)))     # 8-connected: diagonal touches are one piece
    if n > 1:   # pieces with none of the head / band's own shapes (edge bits) are dropped unless they are the largest part
        own = np.unique(L[core & m]); own = own[own > 0]
        sz = np.bincount(L.ravel())[1:]; big = 1 + int(np.argmax(sz))
        keep = np.isin(np.arange(n + 1), np.r_[own, big]) & (np.arange(n + 1) > 0)
        m = keep[L]
    return binary_fill_holes(m)


POST_BODY[LOLA005_ATK] = _lola005_atk
