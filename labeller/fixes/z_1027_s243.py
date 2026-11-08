# sweep 10-27 fix for skin 243 (Haiku fix, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s243/)
# sweep 10-27 fix for skin 243 GunslingerPirate (Haiku task s243, sweep 2026-10-27; findings: pair 4398 = attack 90 deg, tick 315,
# too_big; pair 4399 = attack 90 deg, tick 317, ragged). Type 6000024 is Colt's bullet Colt003Projectile, used only by skin 243 (the
# GunslingerPirate conf is the only one listing it): the skin's blue skull orb with a flame trail behind it. At the pool edge the lowest
# orb of the shot is in front of a purple impact splash (spiky, drawn on the water). The splash is drawn with the bullet's own clip
# shapes (shared atlas polygons), so the shape test labels it: its pixels, and the 1-2 px edge grown from them, joined the orb's label
# as loops and spurs running down to the bottom of the frame (ticks 315-323). The verifier's note: an impact effect, not the attack.
# Fix (keyed to 6000024 only): the orb's disc sits about 1.7 projected logic radii (pr) from its projected centre on the side it flies
# to; its visual centre is about 0.55 pr lower than the projected one. Body pixels IN FRONT of the orb (on the side it flies to, from
# o['angle']) and farther than S243_R pr from its projected centre are dropped: no part of the bullet lies there, the splash does. The
# disc and the flame behind it (along <= 0) are unchanged; nothing is added. Measured on sw2_243 ticks 303-325: the lowest orb keeps
# all its disc pixels (273, 267, 276, 283, 273 on ticks 315-323, before and after); the removed pixels are the splash only; the attack 0
# deg and super rows of the skin are unchanged.
# Runs as a POST_BODY hook, before the hull / claim stages, chaining any earlier POST_BODY for the same type.

S243_TYPE = 6000024
S243_R = 1.9          # projected logic radii (pr) from the projected centre: the disc reaches about 1.7 pr; the splash lies beyond
S243_WIN = 150        # px: window around the orb in which the front cut is applied


def _s243_front(body, ctx):
    o = ctx['o']; cam = ctx['cam']
    if not body.any() or cam is None or o.get('anchor') is None or o.get('angle') is None or o.get('radius') is None:
        return body
    x, yr, zr = [float(v) for v in o['anchor']]        # the anchor is in render coords (x, -y, 0.6 z)
    lx, ly, lz = x, -yr, zr / 0.6                       # logic position of the orb
    cx, cy = _proj(cam, lx, ly, lz)
    a = np.deg2rad(float(o['angle']))
    p2 = _proj(cam, lx + 100 * np.cos(a), ly + 100 * np.sin(a), lz)   # on screen: the direction it flies to
    ux, uy = p2[0] - cx, p2[1] - cy
    nu = float(np.hypot(ux, uy))
    if nu < 1e-6: return body
    ux, uy = ux / nu, uy / nu
    pr = float(np.hypot(*(np.array(_proj(cam, lx + float(o['radius']), ly, lz)) - np.array([cx, cy]))))
    if pr < 1: return body
    H, W = body.shape
    y0, y1 = max(0, int(cy) - S243_WIN), min(H, int(cy) + S243_WIN + 1)
    x0, x1 = max(0, int(cx) - S243_WIN), min(W, int(cx) + S243_WIN + 1)
    if y1 <= y0 or x1 <= x0: return body
    yy, xx = np.mgrid[y0:y1, x0:x1]
    dx, dy = xx - cx, yy - cy
    drop = ((dx * ux + dy * uy) > 0) & ((dx * dx + dy * dy) > (S243_R * pr) ** 2)
    out = body.copy()
    out[y0:y1, x0:x1] &= ~drop
    return out


_s243_prev = POST_BODY.get(S243_TYPE)


def _s243_body(body, ctx):
    if _s243_prev is not None: body = _s243_prev(body, ctx)
    return _s243_front(body, ctx)


POST_BODY[S243_TYPE] = _s243_body
