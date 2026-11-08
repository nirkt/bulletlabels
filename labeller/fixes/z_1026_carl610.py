# Carl Carreta 610 (Carl010Projectile 6000621/6000622): crescent incl. teal caps and blue rim, smooth, green swoosh out (user 10-26 Q3; Haiku task 33, reviewed; replaces z_1023_carl610.py)
# Carl 610 (WhirlwindCarreta, skin 010) crescent attack: the teal end caps are IN, the green glow swoosh stays OUT; one clean
# solid crescent (Haiku task 33, 2026-10-26; replaces py/fixes/z_1023_carl610.py, task 7). The user's decision (binding,
# 2026-10-26): the teal end caps at the crescent's tips are part of the crescent (the notches at the tips are wrong); the
# bright green glow swoosh stays out (earlier decision, Q12). The label must be clean and smooth: one solid crescent in its
# own outline, no notches, no chops, no cut-offs, no holes, stable from frame to frame.
# Types: 6000621 Carl010Projectile (present in the sw2_610 capture, ticks 265-271) and 6000622 Carl010Projectile2 (same clip,
# not present in the capture). Clip: sc/effects_brawler_carl.sc carl_010_atk_projectile (+ trail carl_010_atk_trail). The
# attack-90 row (ticks 303-313) has no projectile in the capture; the page has no super row.
# Cause (measured on sw2_610): the crescent's own shapes are 6, 29 (cyan rim), 33 (inner stroke), 56 (blue body), 57
# (dark band), 58, 59 (teal; the two end caps are shape 59 at the tips). Their pixels are translucent: ID-pass coverage
# 20-127, and the pipeline's body keeps only coverage >= 128 (CUT), so the caps' and rims' translucent parts, the inner stroke
# and the dark band were missing: notches at the tips, a V-cut under the upper cap (t267), chops. The green swoosh is shape 0
# (outer arc), shape 53 (tail) and the green-dominant pixels of shape 52 (its inner edge); all three are dropped.
# Rule (keyed to the two type ids):
#  1. the crescent's own shapes CARL610_CORE at coverage >= CARL610_CV (every such pixel measured in the region is >= 20),
#     in the draws the body already uses (other draws in the scene, e.g. the launch puff, are not the attack) and in the region;
#  2. shape 52 pixels that are blue-dominant (the bright outer rim) within 2 px of the core, same draws and coverage;
#     its green-dominant pixels (the swoosh's inner edge) are out;
#  3. pixels with no shape of the clip within 1 px of the core that are in the body (sprite edges);
#  4. only parts connected to the body; closed (gaps up to about 4 px: the notches and the gaps between the strokes), holes
#     filled, specks below 2 % of the largest part dropped. No hull: the crescent is concave and stays in its own outline.
CARL610_FILE = 'sc/effects_brawler_carl.sc'
CARL610_CLIPS = ('carl_010_atk_projectile', 'carl_010_atk_trail')
CARL610_CORE = (6, 29, 33, 56, 57, 58, 59)   # the crescent's own shapes (59: the teal end caps)
CARL610_RIM = 52                              # shared: blue-dominant = bright outer rim (kept), green-dominant = swoosh (out)
CARL610_CV = 20                               # coverage from which a cap (shape 59) pixel is the attack (measured lower bound)
# the user (10-26, after the first version): the label was too big, it took in the glow: only the crescent shape. The
# crescent's other shapes count from CARL610_CV_BODY (their translucent outer fringe is the glow), the caps from CARL610_CV
CARL610_CV_BODY = 20   # (a higher cut fragments the see-through crescent; the glow is removed by the rim step below)
CARL610_TYPES = (6000621, 6000622)


def _t33_carl610(body, ctx):
    from scipy.ndimage import binary_fill_holes, binary_opening, label as cc
    if not body.any(): return body
    SP = _a4_shape_pixels(ctx['R'], CARL610_FILE, CARL610_CLIPS)   # shape id + 1, 0: no shape of this clip
    ids = ctx['ids_h']; cv = ctx['cv']; region = ctx['region']; img = ctx['img']
    own_draw = np.isin(ids, np.unique(ids[body & (ids > 0)]))      # the draws the attack's body already uses
    # the translucent edge of the crescent within 3 px of its body (further out it is the soft haze of the sprite: measured
    # on tick 271, a spur and a 7 px bulge into the blue glow)
    ok = region & (cv >= CARL610_CV) & own_draw & _dilate(body, 3)
    # shape 33 (the inner stroke) lies between the swoosh's shapes 52 and 53 at the upper-left tip (tick 271): its translucent
    # pixels are not added, only the body's own (measured: a spur up into the swoosh otherwise)
    core = np.isin(SP, [s + 1 for s in CARL610_CORE if s not in (33, 59)]) & ok & (cv >= CARL610_CV_BODY)
    core |= (SP == 60) & ok                                     # the teal end caps (shape 59) from CARL610_CV
    core |= np.isin(SP, [34]) & body & region                     # shape 33's body pixels as before (inner stroke)
    if not core.any(): return body
    near = _dilate(core, 2)
    rim = np.zeros_like(core)
    if img is not None:
        blue = img[..., 2] >= img[..., 1]
        rim = (SP == CARL610_RIM + 1) & blue & near & ok & (cv >= CARL610_CV_BODY)
    edge = (SP == 0) & body & _dilate(core | rim, 1)              # sprite edges (no shape) next to the crescent
    m = (core | rim | edge) & region
    L, n = cc(m)
    if not n: return body & False
    keep = np.unique(L[body & m]); keep = keep[keep > 0]          # parts connected to the attack's body only
    m = np.isin(L, keep)
    if not m.any(): return body & False
    m = binary_opening(m, iterations=2)                           # thin spurs (1-4 px) cut away: a spur up into the swoosh (tick 271)
    if not m.any(): return body & False
    m = _close(m, 2) & region                                     # gaps and notches up to about 4 px
    m = binary_fill_holes(m) & region
    L, n = cc(m)
    if n > 1:
        sz = np.bincount(L.ravel())[1:]
        m = np.isin(L, 1 + np.nonzero(sz >= max(12, 0.02 * sz.max()))[0])
    return m


for _t in CARL610_TYPES:
    _prev_t33 = POST_BODY.get(_t)   # the agent-3 launch-puff rule (6000621) first, then the crescent rule
    POST_BODY[_t] = (lambda prev: (lambda body, ctx: _t33_carl610(prev(body, ctx) if prev else body, ctx)))(_prev_t33)


# Coordinator, 10-26 (the user: 'the label is too big, it also includes the glow. it should only include the crescent shape'):
# the crescent = the region enclosed by its cyan rim line (shape 29) and the teal caps (59), filled; its inner edge from the
# label above (so the fill never bridges into the hollow of the C); the translucent glow outside the rim is out.
import os as _os
CRF_SET = [int(v) for v in _os.environ.get("CRF_SET", "30,60").split(",")]
CRF_CLOSE = int(_os.environ.get('CRF_CLOSE', '3'))
def _crf(body, ctx):
    from scipy.ndimage import binary_fill_holes, label as cc
    if not body.any(): return body
    SP = _a4_shape_pixels(ctx['R'], CARL610_FILE, CARL610_CLIPS)
    cv = ctx['cv']; region = ctx['region']
    m = np.isin(SP, CRF_SET) & (cv >= 20) & region & _dilate(body, 4)
    if not m.any(): return body
    m = binary_fill_holes(_close(m, CRF_CLOSE)) & region & _dilate(body, 1)   # inner edge: the previous label's
    L, n = cc(m)
    if n > 1:
        sz = np.bincount(L.ravel())[1:]; m = L == (1 + sz.argmax())
    return m
for _t in CARL610_TYPES:
    _pp = POST_BODY.get(_t)
    POST_BODY[_t] = (lambda prev: (lambda body, ctx: _crf(prev(body, ctx) if prev else body, ctx)))(_pp)
