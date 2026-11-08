# misc group fix (owner's BAD marks, final review 2026-11-04/05), capture group misc5. Exec'd in idlabels' namespace.
# Every global name has the prefix _ab5m_ / AB5M_. Keyed to explicit type ids; the owner's notes are binding.
#
# 397 GunSlingerWoody (Colt010Projectile 6000038 attack, Colt010UltiProjectile 6000039 super; both single-skin types):
#   owner: "holes, and there are tiny super labels getting spawned everywhere, even though its just a normal attack".
#   The shot is a 3D model (bulb + fins, draws with fragment shader AB5M_397_FS) under a sprite cone of stacked tiers. The
#   label covered only the cone and bits of the bulb (holes, ragged), and where smoke / splash hid the shot it shrank to
#   specks. Fix: the whole shot = cone + the model's own pixels (connected, near its prediction), closed and filled; when
#   effects hide most of it, the (aligned) predicted body stands in for the hidden part (hidden-parts rule) but never over
#   solid map objects.
import numpy as _ab5m_np
from scipy.ndimage import (label as _ab5m_cc, binary_fill_holes as _ab5m_fh, binary_closing as _ab5m_bc,
                           binary_dilation as _ab5m_bd, binary_opening as _ab5m_bo)

AB5M_397 = (6000038, 6000039)
AB5M_397_FS = 1098844569                    # fragment shader hash of the shot's 3D model draws (textures 128x128 / 32x32, attack and super)
AB5M_397_NEAR = 6                           # px (half res) around the prediction in which the model may be taken


def _ab5m_disk(r):
    y, x = _ab5m_np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r + r // 2


def _ab5m_close(m, r):
    p = r + 2
    return _ab5m_bc(_ab5m_np.pad(m, p), structure=_ab5m_disk(r))[p:-p, p:-p]


def _ab5m_largest(m, frac=0.2):
    L, n = _ab5m_cc(m, structure=_ab5m_np.ones((3, 3), bool))
    if n <= 1: return m
    sz = _ab5m_np.bincount(L.ravel())[1:]
    return _ab5m_np.isin(L, 1 + _ab5m_np.nonzero(sz >= frac * sz.max())[0]) & m


def _ab5m_hidden(out, ctx):
    """hidden parts (smoke, splash, dust drawn over the shot): the predicted body stands in where an effect sprite (no depth write,
    not the shot's own sprite) is in front of it and nothing solid is; never over walls / bushes / characters (they write depth)"""
    R = ctx['R']; ids_h = ctx['ids_h']; wd = R.get('wd'); o = ctx['o']
    if wd is not None and o.get('anchor') is not None:
        dz = _eye_depth(ctx['cam'], o['anchor'], R['meta'].get('wref', 0.0))[0]; vis = wd >= dz - DEPTH_TOL
    else: vis = _ab5m_np.ones(out.shape, bool)
    solid = [d for d, r in R['draws'].items() if r.get('dmask')]
    eff = ~_ab5m_np.isin(ids_h, solid) & (ids_h > 0)
    return ctx['pred_body'] & vis & eff & (ctx['C'] == 0) & ~out


AB5M_TIER_B = 110        # ... and blue channel (the green aura of the owner's brawler is not)
AB5M_TIER_G2 = 240
AB5M_TIER_G = 190        # frame green channel from which a pixel of the shot's cone sprite is the solid tiers (beam stays below)


def _ab5m_397_hook(prev):
    def fn(body, ctx):
        body = prev(body, ctx) if prev else body
        R = ctx['R']; ids_h = ctx['ids_h']; cv = ctx['cv']; img = ctx['img']; C = ctx['C']
        mesh = [d for d, r in R['draws'].items() if r.get('fs') == AB5M_397_FS and r.get('dmask')]
        M = _ab5m_np.isin(ids_h, mesh) & (cv >= CUT) if mesh else _ab5m_np.zeros(body.shape, bool)
        pa = ctx['pred_any']
        near = _ab5m_bd(pa, structure=_ab5m_disk(AB5M_397_NEAR))
        cand = M & near
        # the solid tiers: pixels of its own cone sprite that are bright in the frame
        far = _ab5m_bd(pa, structure=_ab5m_disk(26))
        tiers = (C == 2) & far & (img[..., 1] >= AB5M_TIER_G) & (img[..., 2] >= AB5M_TIER_B)
        strong = tiers & (img[..., 1] >= AB5M_TIER_G2)          # the tiers are saturated; the aura-lit beam is not
        tiers = tiers & _ab5m_bd(strong, structure=_ab5m_disk(2))
        cand = cand | tiers
        body = body & ~((C == 2) & ~tiers & ~M)          # the beam (dim part of the cone sprite) that the prediction let in
        base = body | (M & near)
        L, n = _ab5m_cc(cand | body, structure=_ab5m_np.ones((3, 3), bool))
        keep = _ab5m_np.unique(L[base]); keep = keep[keep > 0]
        add = _ab5m_np.isin(L, keep) & cand & ~base
        pad = 4
        op = _ab5m_bo(_ab5m_np.pad(add | base, pad), structure=_ab5m_disk(2))[pad:-pad, pad:-pad]   # frayed spurs of the tiers off
        out = base | (add & op)
        # hidden parts (smoke, splash, dust drawn over the shot): the predicted body stands in where an effect sprite (no depth
        # write) is in front of it and nothing solid is; never over walls / bushes / characters (they write depth)
        wd = R.get('wd'); o = ctx['o']
        if wd is not None and o.get('anchor') is not None:
            dz = _eye_depth(ctx['cam'], o['anchor'], R['meta'].get('wref', 0.0))[0]; vis = wd >= dz - DEPTH_TOL
        else: vis = _ab5m_np.ones(body.shape, bool)
        solid = [d for d, r in R['draws'].items() if r.get('dmask')]
        eff = ~_ab5m_np.isin(ids_h, solid) & (ids_h > 0)
        hid = ctx['pred_body'] & vis & eff & (C == 0) & ~out   # not its own cone / glow sprite (the beam is in the prediction)
        out = out | hid
        out = _ab5m_close(out, 1) & (out | near | far & _ab5m_bd(out, structure=_ab5m_disk(3)))
        out = _ab5m_fh(out)
        out = _ab5m_largest(out, 0.2)
        return out
    return fn


for _t_ab5m in AB5M_397:
    POST_BODY[_t_ab5m] = _ab5m_397_hook(POST_BODY.get(_t_ab5m))


# 306 GunslingerPaintball super (Colt005UltiProjectile 6000029; single-skin type): owner "super is broken, it has big holes in it".
#   The comet's labelled pixels were fragments of its head and tail streaks (the translucent outer layer, shape 427, sat under the
#   coverage cut). The attack's installed rule (_rg_306: outer layer in at coverage >= 32, holes filled, splash out) is applied to
#   the super too, then the pieces are closed (the head and its streaks are one comet) and filled.
AB5M_306_SUPER = (6000029,)
AB5M_306_CLOSE = 3


def _ab5m_306_hook(prev):
    def fn(body, ctx):
        body = prev(body, ctx) if prev else body
        b = _rg_306(body, ctx)
        if not b.any(): return body
        b = b | _ab5m_hidden(b, ctx)
        b = _ab5m_close(b, AB5M_306_CLOSE) & (b | ctx['region'])
        b = _ab5m_fh(b)
        return _ab5m_largest(b, 0.2)
    return fn


for _t_ab5m in AB5M_306_SUPER:
    POST_BODY[_t_ab5m] = _ab5m_306_hook(POST_BODY.get(_t_ab5m))


# 328 GunslingerMonkey (Colt007Projectile 6000032 attack, Colt007UltiProjectile 6000033 super; single-skin types):
#   owner: "super labels are a little ragged and attack labels are broken, they include some background and are ragged".
#   The labels took pixels of two soft glow shapes of the clips: shape 145 (the blue glow tube / disc around the banana) and
#   shape 0 (soft discs / the ground shadow), taken in since FADE_CUT lowered the cut for the banana's translucent head layer.
#   Fix: those two shapes are never body; the banana's own shapes (the peel / flesh frame sprites, translucent parts at
#   coverage >= 32) connected to what is left are body; closed, holes filled, one piece. Shapes are the clips' own ids.
AB5M_328 = {6000032: ('colt_007_atk_projectile', 'colt_007_atk_projectile_red'),
            6000033: ('colt_007_ulti_projectile', 'colt_007_ulti_projectile_red')}
AB5M_328_GLOW = (0 + 1, 145 + 1)     # shape ids + 1 of the glow tube / soft discs
AB5M_328_CV = {6000032: 32, 6000033: 80}   # attack: the translucent head layer is banana; super: opaque parts only (its flesh rim is soft)
AB5M_328_CLOSE = {6000032: 2, 6000033: 3}


def _ab5m_328_hook(t, prev):
    def fn(body, ctx):
        body = prev(body, ctx) if prev else body
        names = AB5M_328[t]
        SP = _a4_shape_pixels(ctx['R'], 'sc/effects_brawler.sc', names if ctx['own'] else (names[1],))
        core = body & ~_ab5m_np.isin(SP, AB5M_328_GLOW)
        if not core.any(): return body
        cand = ctx['region'] & (SP > 0) & ~_ab5m_np.isin(SP, AB5M_328_GLOW) & (ctx['cv'] >= AB5M_328_CV[t])
        L, n = _ab5m_cc(cand | core, structure=_ab5m_np.ones((3, 3), bool))
        keep = _ab5m_np.unique(L[core]); keep = keep[keep > 0]
        out = core | (_ab5m_np.isin(L, keep) & cand)
        out = _ab5m_close(out, AB5M_328_CLOSE[t]) & (out | ctx['region'])
        out = _ab5m_fh(out)
        return _ab5m_largest(out, 0.2)
    return fn


for _t_ab5m in AB5M_328:
    POST_BODY[_t_ab5m] = _ab5m_328_hook(_t_ab5m, POST_BODY.get(_t_ab5m))


# 207 AssaultShotgunDragon (Griff006Projectile 6001031 attack pellets, and 6001032 its Sp1Buddy copy; only skin 207 uses them):
#   owner (page 207_..._01, "attack 90" t315 and t319): "holes / ragged". After the installed z_1109_mortis_dragon re-cut the pellet
#   (gem + crystal + gold clasp) is one piece but its outline is still not clean: the gold clasp has a U-shaped notch open to
#   the outside (the global hole fill cannot close it), and thin spurs (the gold thread at the left end) stick out. Final step on
#   these types' labels only: thin spurs opened away (radius 2), then closed (radius 6: the clasp's notch, the gap between gem
#   and crystal), holes filled, outline smoothed; only unclaimed pixels are added, growth at most 7 px, never loses its label.
AB5M_207 = (6001031, 6001032)
AB5M_207_OPEN = 3
AB5M_207_CLOSE = 9
AB5M_207_SIGMA = 1.2
AB5M_207_GROW = 10
_ab5m_prev_labels = labels


def _ab5m_207_clean(m):
    from scipy.ndimage import gaussian_filter
    p = 12
    mp = _ab5m_np.pad(m, p)
    op = _ab5m_bo(mp, structure=_ab5m_disk(AB5M_207_OPEN))
    if op.sum() < 0.6 * mp.sum(): op = mp                      # a small bullet: no opening
    cl = _ab5m_bc(op | False, structure=_ab5m_disk(AB5M_207_CLOSE))
    cl = _ab5m_fh(cl)
    g = gaussian_filter(cl.astype(_ab5m_np.float32), AB5M_207_SIGMA) >= 0.5
    g = _ab5m_fh(g)
    return g[p:-p, p:-p]


def labels(objs, cam, R, state=None, ended=(), img=None):
    import bulletlabels as BL
    P = _ab5m_prev_labels(objs, cam, R, state=state, ended=ended, img=img)
    mine = [o for o in objs if o['t'] in AB5M_207]
    if not mine: return P
    ids = P['ids'].copy(); share = P['share'].copy(); lids = P['lids']; cv = R['cv'][::2, ::2]
    for o in mine:
        lid = (BL._lid(o), -1, 'body')
        if lid not in lids: continue
        k = lids.index(lid); m = ids == k
        if not m.any(): continue
        s = _ab5m_207_clean(m) & _ab5m_bd(m, structure=_ab5m_disk(AB5M_207_GROW))
        if (s & m).sum() < 0.5 * m.sum(): continue
        add = s & ~m & (ids < 0)
        ids[add] = k; share[add] = _ab5m_np.maximum(cv[add] / 255.0, BL.CUT)
        drop = m & ~s; ids[drop] = -1; share[drop] = 0
    P2 = dict(P); P2.update(ids=ids, share=share)
    return P2


# 293 BarkeepDrunken (Barley_007_atk / Barley_007_ulti zones; skin 293 only), row "super 45", t411. OWNER NOTE: "exclude zone for the
#   tiny walls is kinda broken". The map's low fence row at the bottom (tile code 13, the grey arches; the same map as skins 102 /
#   199) is a 590-unit box in z_1028_zone184 (marked "unverified" there); its far top edge projects above the fence's drawn top and
#   cuts a strip of visible zone ground off above the arches (z_1109_zones measured the real fence top at ~310 units, drawn top
#   edge, and set it for the Chronomancer rows). The same fence box (310, inset 20) is used while the chain labels these Barkeep
#   zones, so the cut follows the drawn top edge of the tiny walls.
AB5M_293_ZONES = ('Barley_007_atk', 'Barley_007_ulti')
AB5M_293_FENCE = (310.0, 20.0)
AB5M_293_CODE = 13
_ab5m_prev_zone_labels = zone_labels


def zone_labels(state, cam, R, taken, own_team=None):
    import owners
    hit = False
    for it in state.get('items', []):
        Z = owners.zone(int(it[1]))
        if Z and Z.get('Name') in AB5M_293_ZONES: hit = True; break
    if not hit: return _ab5m_prev_zone_labels(state, cam, R, taken, own_team)
    old = Z184_BOX.get(AB5M_293_CODE)
    Z184_BOX[AB5M_293_CODE] = AB5M_293_FENCE
    try:
        return _ab5m_prev_zone_labels(state, cam, R, taken, own_team)
    finally:
        if old is not None: Z184_BOX[AB5M_293_CODE] = old
