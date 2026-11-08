# installed 11-02 from out/sweep2/haikutasks/siteai5/dup_crow_poco_arcade_gale/fix.py (owner complaints: 005 tail tip, 027 dagger hulls, 028 attack holes + super t353 kept while drawn, 048 Gale wall incl. edge puffs; 041 unchanged)
# Owner round 2026-11-02 (sweep 4 complaints), task dup_crow_poco_arcade_gale (skins 005, 027, 028 (default, pages _01 and _02), 041, 048).
# ONE fix file for the group; exec'd after all installed fixes (IDLABELS_OVERRIDE). Keyed to type ids only.
#
# A. skin 027 CrowDefault, type 6000308 (CrowProjectile, the Crow default attack): "holes / ragged" on attack 0 t265 and attack 90
#    t305 / t307. Each dagger = the convex hull of its own blade (clip body, code 1, inside its predicted body dilated 3 px): the green
#    skull swirls lying on the blade are inside the hull and are filled in; swirls on the trail beside the blade stay out. One outline
#    per dagger (the blades that touch at their backs are split by the predicted body, so a hull never spans two daggers).
#    Replaces the z_1100_s027 result for this type. Other Crow skins that share the type id: see the report (re-check).
# B. skin 005 DuplicatorDarkcircus, type 6001099 (Lolla006Projectile, the Duplicator attack): owner note "the tail of the attack should
#    also be labeled" (FIXED 10-29, re-check); AI check at attack 90 t323: the tail's far tip is outside. The tip is drawn with the clip
#    glow code (2) although its shape is one of the projectile's own body shapes (13412 / 13414 / 13419) at coverage >= 64. These
#    code-2 pixels of the own shapes are added to the body before the z_1101_s005a rule (gaps, holes, one piece joined to the head).
#    The trail haze (particle_gas_seq_1) and glow shape 0 stay out.
# C. skin 048 BlowerDefault, super 45, type 6000732 (BlowerUltiProjectile): site review "as: Gale super wall partly labelled"; AI check:
#    the upper-right puffs outside the hull at t345-t353. Owner rule: the wall is ONE hull covering the whole wall incl. its puffs. The band
#    takes the wall's puffs from coverage 32 (the translucent edge puffs), joined to the bright core; one convex hull; the prop / olive-glow
#    cut and the one-owner rule of z_1101_s048 are kept. Replaces z_1101_s048's band for this type.
# D. skin 028 DeadMariachiDefault_02, super 45, type 6000336 (DeadMariachiUltiProjectile): owner complaint "super 45 t353: missing label"
#    (binding). The three members have finished = 1 at t353 but are still drawn (they disappear at t355). A member that has finished but is
#    still drawn keeps its region for the label (finished treated as 0), so the still-visible cloud is labelled at t353. Only this type.
# E. skin 028 DeadMariachiDefault_01, attack, type 6000335 (DeadMariachiProjectile): owner complaint "holes / ragged" at attack 0 t271 / t275,
#    attack 90 t309 / t315 / t317. Holes enclosed by the label (up to 1500 px, not another label's pixels) are filled, so the cloud's
#    outline is one closed outer edge with no inner loop. The z_1100_s028 cell rule is kept. The t275 seam to the ended member's cloud
#    is by design (owner decision 10-28, z_1100_s028) and is not changed here.
#
# Not changed: skin 041 (Arcade): see the report (one hull per tracked projectile object is already the case; open question).
import numpy as np
from scipy.ndimage import binary_dilation as _g_dil, label as _g_cc

# ---- A. 027 Crow dagger hull --------------------------------------------------------------------------------------
CROW27_TYPE = 6000308
CROW27_PRED_DIL = 3
_crow27_prev = POST_BODY.get(CROW27_TYPE)


def _crow27_dagger(body, ctx):
    if _crow27_prev is not None:
        body = _crow27_prev(body, ctx)
    if not body.any(): return body
    C = ctx['C']; pb = ctx['pred_body']
    b1 = body & (C == 1)
    if b1.sum() < 3 or not pb.any(): return body
    own = b1 & _g_dil(pb, iterations=CROW27_PRED_DIL)
    if own.sum() < 3: return body
    return hull_mask(own) & ctx['region']


POST_BODY[CROW27_TYPE] = _crow27_dagger

# ---- B. 005 Duplicator tail tip -----------------------------------------------------------------------------------
LOLA006_ATK_F = 6001099
LOLA006_FILE_F = 'sc/effects_brawler.sc'
LOLA006_CLIP_F = ('lolla_006_atk_projectile', 'lolla_006_atk_projectile_red')
LOLA006_TRAIL_F = ('particle_gas_seq_1',)
LOLA006_BODY_F = (13412, 13414, 13419)
LOLA006_CUT_F = 64
_lola005_prev_post = POST_BODY.get(LOLA006_ATK_F)


def _lola006_tip(body, ctx):
    R = ctx['R']; C = ctx['C']; cv = ctx['cv']; region = ctx['region']
    SP = _a4_shape_pixels(R, LOLA006_FILE_F, LOLA006_CLIP_F)
    TR = _a4_shape_pixels(R, LOLA006_FILE_F, LOLA006_TRAIL_F) > 0
    own_shape = np.isin(SP, [s + 1 for s in LOLA006_BODY_F])
    extra = own_shape & (C == 2) & (cv >= LOLA006_CUT_F) & region & ~TR
    if extra.any():
        body = body | extra
    if _lola005_prev_post is not None:
        body = _lola005_prev_post(body, ctx)
    return body


POST_BODY[LOLA006_ATK_F] = _lola006_tip

# ---- C. 048 Gale super wall: one hull over the whole wall incl. its puffs -----------------------------------------
G48_TYPE = 6000732
G48_SWF = 'sc/effects_brawler.sc'
G48_CLIPS = ('gale_def_ulti_projectile', 'gale_def_ulti_projectile_red')
G48_SHAPE = 64
G48_CUT = 32
G48_CORE = 170
G48_JOIN = 6
G48_GREEN = 60
G48_PROP_PX = 15000
_g48_state = {'R': None, 'owner': None, 'mask': None}


def _g48_band(ctx):
    R = ctx['R']
    if _g48_state['R'] is R:
        return _g48_state['mask']
    cv = R['cv'][::2, ::2]; img = ctx['img']
    SP = _a4_shape_pixels(R, G48_SWF, G48_CLIPS)
    opaque = (SP == G48_SHAPE) & (cv >= G48_CUT)
    core = opaque & (img.min(-1) >= G48_CORE)
    mask = np.zeros(opaque.shape, bool)
    if core.any():
        L, n = _g_cc(_g_dil(opaque, iterations=2), structure=np.ones((3, 3)))
        cnt = np.bincount(L[core], minlength=n + 1); cnt[0] = 0
        A = opaque & np.isin(L, np.nonzero(cnt)[0][1:])
        L2, n2 = _g_cc(_g_dil(A, iterations=G48_JOIN), structure=np.ones((3, 3)))
        cnt2 = np.bincount(L2[core], minlength=n2 + 1); cnt2[0] = 0
        band = A & (L2 == int(np.argmax(cnt2)))
        ys, xs = np.nonzero(band)
        if len(ys) >= 3:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(np.stack([xs, ys], 1).astype(np.float64))
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            gy, gx = np.mgrid[y0:y1, x0:x1]
            inside = np.ones(gx.shape, bool)
            for a, b, c in hull.equations:
                inside &= (a * gx + b * gy + c) <= 1e-9
            mask[y0:y1, x0:x1] = inside
    ids = R['id'][::2, ::2]; cntd = np.bincount(ids.ravel())
    occ = [d for d, dr in R['draws'].items() if dr.get('dmask') and not dr.get('blend') and d < len(cntd) and 0 < cntd[d] <= G48_PROP_PX]
    if occ and mask.any():
        mask &= ~np.isin(ids, occ)
    im = img.astype(np.int16)
    mask &= ~((im[..., 1] - im[..., 2] >= G48_GREEN) & (im[..., 1] >= im[..., 0] - 10))
    if mask.any():
        L, n = _g_cc(mask, structure=np.ones((3, 3)))
        cnt = np.bincount(L.ravel()); cnt[0] = 0
        mask = L == int(np.argmax(cnt))
    _g48_state.update(R=R, owner=None, mask=mask)
    return mask


def _fix_blower_super_wall4(body, ctx):
    import bulletlabels as BL
    if ctx.get('img') is None:
        return body
    if _g48_state['owner'] is None or _g48_state['R'] is not ctx['R']:
        _g48_band(ctx)
    if _g48_state['owner'] is None:
        _g48_state['owner'] = BL._lid(ctx['o'])
    if BL._lid(ctx['o']) != _g48_state['owner']:
        return np.zeros_like(body)
    return _g48_state['mask'].copy()


POST_BODY[G48_TYPE] = _fix_blower_super_wall4

# ---- D. 028 Poco super: still-drawn finished member keeps its label ----------------------------------------------
_p028s_prev_labels = labels


def labels(objs, cam, R, state=None, ended=(), img=None):
    objs2 = []
    for o in objs:
        if o['t'] == 6000336 and o.get('finished', 0) and o.get('drawn', True) and not o.get('ended'):
            o = dict(o, finished=0)
        objs2.append(o)
    return _p028s_prev_labels(objs2, cam, R, state=state, ended=ended, img=img)


# ---- E. 028 Poco attack: enclosed holes filled -------------------------------------------------------------------
F028_TYPE = 6000335
F028_MAX_HOLE = 1500
_f028_prev_labels = labels


def labels(objs, cam, R, state=None, ended=(), img=None):
    from scipy.ndimage import binary_fill_holes as _f028_fill
    P = _f028_prev_labels(objs, cam, R, state=state, ended=ended, img=img)
    ids, lids = P['ids'], P['lids']; share = P['share']
    ks = [k for k, l in enumerate(lids) if l[2] == 'body' and any(o['id'] == l[0] and o['t'] == F028_TYPE for o in objs)]
    for k in ks:
        m = ids == k
        if not m.any(): continue
        h = _f028_fill(m) & ~m & (ids < 0)
        if not h.any(): continue
        L, n = _g_cc(h)
        sz = np.bincount(L.ravel())
        small = np.zeros(n + 1, bool); small[1:] = sz[1:] <= F028_MAX_HOLE
        fill = small[L] & h
        ids[fill] = k; share[fill] = np.maximum(share[fill], 0.5)
    P['ids'] = ids; P['share'] = share
    return P
