# cut5 fix file (round 4b, 2026-11-06) -> W/py/fixes/z_1114_cut5.py. Every global name here has the prefix _c5_.
# Owner 11-06 on page 148: "part of the label just gets cut off ... check every label with a script" (cutcheck4 finds the real ones).
#
# FIX 1  HIDDEN-PARTS FILL for the object types below (generalises z_1113_sticky4, which only fixed Grom 148). Cutcheck (cutcheck4, 'covered'
# tier) found parts of an object's own body that no label covers because an effect draws over them:
#   * Grom (CrossBomber) SUPER cube while it spawns at the thrower (t343-355: the cyan swoosh / HP bar hide half of it): 061 6000958 (and its
#     overcharged copy 6000960), 126 6000964 (egg), 130 6000968, 135 6000972 (cube + sword + disc), 142 6000976;
#   * 142 Grom attack stars / blades 6000974 / 6000975 and the landed super 6000977 (t371-375), 148 landed super 6000981 (top fringe, t375),
#     457 IceDude cone + ball 6000859 (orange drill cone cut, t271-273 and the whole attack-90 row), 405 Beamer flame 6001306 (edge),
#     018 RocketGirl super rocket 6000054 (the rocket tip hidden by smoke, t385).
# Rule (hidden parts rule, owner 10-28): the object's PREDICTED body (game state), shifted onto the drawn body (<= 16 px, or <= 4 px for
# types whose neighbours touch), is trusted only when it covers >= 85 % of the drawn clip body; the part of it that is hidden by an effect
# draw (no depth write, no UI draw, not the scene background) or by nothing the ID pass sees is added to the label (opened by 1 px so
# anti-aliasing rims do not grow the label; 'thin' types: only within 3 px of the label). World geometry / characters / HP bar (opaque, depth
# write) never count as hiders. Applied AFTER the type's own installed POST_BODY.
import numpy as _c5_np


from scipy.ndimage import binary_opening as _c5_open
_c5_UI_FS = (1735630204, 2825982420, 3318493823)


def _c5_shift(a, dy, dx):
    o = np.zeros_like(a); h, w = a.shape
    o[max(0, dy):h + min(0, dy), max(0, dx):w + min(0, dx)] = a[max(0, -dy):h - max(0, dy), max(0, -dx):w - max(0, dx)]
    return o


def _c5_pred_fill(body, ctx, trust=0.85, own=False, t_thin=False):
    """hidden-parts rule: the predicted body (game state), shifted onto the drawn body, fills what a translucent overlay (swoosh, HP bar
    is opaque and stays out) hides. Only where the shifted prediction fits the drawn body (>= 85 %)."""
    pb = ctx['pred_body']
    drawn = (ctx['C'] == 1) & (ctx['cv'] >= CUT) & ctx['region']
    if own and pb.any(): drawn = drawn & _dilate(pb, 4)   # touching bullets: judge the fit on this bullet's own neighbourhood
    if not body.any() or not pb.any() or drawn.sum() < 30: return body
    ys, xs = np.nonzero(pb | drawn); H, Wd = pb.shape
    y0, y1, x0, x1 = max(0, ys.min() - 20), min(H, ys.max() + 21), max(0, xs.min() - 20), min(Wd, xs.max() + 21)
    mc = drawn[y0:y1, x0:x1]; pc = pb[y0:y1, x0:x1]; ms = max(1, int(mc.sum()))
    fit = lambda q: (q & mc).sum() / ms
    q = pc; R_ = 4 if own else 16
    if fit(q) < trust:
        best = (fit(q), 0, 0)
        for dy in range(-R_, R_ + 1, 2):
            for dx in range(-R_, R_ + 1, 2):
                f = fit(_c5_shift(pc, dy, dx))
                if f > best[0] + 0.02: best = (f, dy, dx)
        q = _c5_shift(pc, best[1], best[2])
    if fit(q) < trust: return body
    Q = np.zeros_like(pb); Q[y0:y1, x0:x1] = q
    draws = ctx['R']['draws']
    # what may hide the body without being a legitimate occluder: effect sprites (no depth write; the cyan swoosh, flashes) or nothing
    # the ID pass sees (additive only). Never world geometry / characters (depth write), UI draws, the scene background texture.
    eff = [d for d, dr in draws.items() if not dr.get('dmask') and dr.get('fs') not in _c5_UI_FS
           and not str(dr.get('th', '')).startswith('1d304abe')]
    hid = Q & ~body & (np.isin(ctx['ids_h'], eff) | (ctx['ids_h'] == 0))
    add = (hid & _dilate(body, 3)) if t_thin else _c5_open(hid, structure=np.ones((3, 3), bool), iterations=1)
    return body | add


def _c5_wrap_hull(t):
    prev = POST_BODY.get(t)

    def f(body, ctx):
        if prev is not None: body = prev(body, ctx)
        return _c5_pred_fill(body, ctx, own=(t in (6000967, 6000859, 6000974, 6000975)), t_thin=(t in (6000859, 6000981, 6001306)))
    POST_BODY[t] = f


for _c5_t in (6000974, 6000975, 6000977, 6000054, 6001306, 6000981, 6000859, 6000967, 6000958, 6000960, 6000964, 6000968, 6000972, 6000976):
    _c5_wrap_hull(_c5_t)


# ---------------------------------------------------------------------------------------------------------------------------------
# FIX 2  Grom Franken (skin 130) attack, type 6000967 = Grom003Projectile2 (only skin 130 uses it), and Grom Supercell-Make (skin 142) attack
# type 6000975 = Grom005Projectile2 (same burst, same defect): the burst of 3-4 blobs of one shot at the
# crate (ticks ~285, ~325) is crossed by the white hit star: the labels of the touching blobs have an inner seam / notch where the star
# covers them ("130 attack t287 blob notch", cutcheck; owner 11-06 on 223: touching bullets of one shot = ONE label, no seams / holes).
# Labels of this type that touch (<= 2 px) become one label (union, closed with radius 4 inside effect draws / empty pixels only, holes filled).
import bulletlabels as _c5_BL
from scipy.ndimage import binary_fill_holes as _c5_fill
_c5_prev_labels = labels
_c5_MERGE = (6000967, 6000975)


def labels(objs, cam, R, state=None, ended=(), img=None):
    P = _c5_prev_labels(objs, cam, R, state=state, ended=ended, img=img)
    ids = P['ids']; ks = []
    for o in objs:
        if o['t'] not in _c5_MERGE: continue
        lid = (_c5_BL._lid(o), -1, 'body')
        if lid in P['lids']:
            k = P['lids'].index(lid)
            if (ids == k).any() and k not in ks: ks.append(k)
    if len(ks) < 2: return P
    par = {k: k for k in ks}

    def find(a):
        while par[a] != a: a = par[a]
        return a
    D = {k: _dilate(ids == k, 2) for k in ks}
    for a in ks:
        for b in ks:
            if a < b and (D[a] & (ids == b)).any(): par[find(b)] = find(a)
    groups = {}
    for k in ks: groups.setdefault(find(k), []).append(k)
    ih = R['id'][::2, ::2]; draws = R['draws']
    eff = [d for d, dr in draws.items() if not dr.get('dmask') and dr.get('fs') not in _c5_UI_FS and not str(dr.get('th', '')).startswith('1d304abe')]
    okpx = np.isin(ih, eff) | (ih == 0)
    for k0, g in groups.items():
        if len(g) < 2: continue
        u = np.isin(ids, g)
        u2 = _c5_fill(_close(u, 7) & (okpx | u)) & ((ids < 0) | u)
        P['ids'][u2] = k0
        P['share'][u2 & ~u] = _c5_BL.CUT
    return P


# ---------------------------------------------------------------------------------------------------------------------------------
# FIX 3  RocketGirlDefault (Brock 018) super rockets, type 6000054 (RocketGirlUltiProjectile; only skin 018 and its enemy copies): the star
# spikes of the flame head are cut ("cutcheck: drawn body pixels outside the label over many ticks 347-387", about 30 % of the star).
# Cause (measured): a3_agent3's smoke filter (_a3_brock_mesh, hooked by z_1100_s018) drops every whole draw that shows (almost) no clip-only
# sprite; the star's rim shares those draws with the trail sprites, so its clip-body pixels (code 1, coverage >= CUT) go with them. Fix: after
# the installed hooks, the clip-body pixels of THIS object (within 2 px of its predicted label region) are put back and the gaps of 1 px closed;
# smoke (code 4 / 5 only draws) stays out because only code 1 pixels are added.
_c5_prev18 = POST_BODY.get(6000054)


def _c5_rocket18(body, ctx):
    if _c5_prev18 is not None: body = _c5_prev18(body, ctx)
    if not body.any(): return body
    add = (ctx['C'] == 1) & (ctx['cv'] >= CUT) & _dilate(ctx['pred_any'], 2) & ctx['mine'] if 'mine' in ctx else None
    return body if add is None else (body | add)


POST_BODY[6000054] = _c5_rocket18
