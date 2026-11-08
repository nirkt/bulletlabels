# cutoff3 fix file, exec'd after all installed fixes (IDLABELS_OVERRIDE). Unique prefix _c3_.
#
# (1) 405 BeamerCarnival, Beamer005Projectile (6001306), attack 90 deg row, ticks 305-311 (owner: "too small / cut off"):
#     the vertical beam's label covered only the yellow core; the visible teal tail above it (the clip's own body shape 133)
#     stayed outside. z_1104_beamer_mortis_mummy (2) already grows through shape 133 for a HORIZONTAL beam (attack 0 deg) and
#     leaves the vertical beam alone (there the growth ran over the brawler's dress / took the faint dark glow column).
#     Here, for the VERTICAL beam only, the label grows (connected, at most _c3_REACH405 px from the label) through the
#     pixels of shape 133 that are the bright teal of the tail: coverage >= 100, bright (max RGB >= 160), teal not yellow-green
#     (R <= 105: the little green leaf sprite beside the beam also uses shape 133) and not water blue (G >= B - 20: on the
#     lake the faint tail is blue). The faint dark teal fade of the tail (and the soft halo) stays out.
from scipy.ndimage import binary_dilation as _c3_bd, label as _c3_cc
_c3_T405 = 6001306
_c3_prev405 = POST_BODY.get(_c3_T405)
_c3_REACH405 = 15       # coordinator 11-05: 30 -> 15 (t311 grew far into the dark teal column)
_c3_CV405 = 100
_c3_BRIGHT405 = 160
_c3_HORIZ405 = 0.8       # same split as z_1104: |cos| of the principal axis >= this = horizontal beam (not changed here)


def _c3_beam405(body, ctx):
    if _c3_prev405 is not None: body = _c3_prev405(body, ctx)
    if not body.any(): return body
    ys, xs = np.nonzero(body)
    if len(xs) < 3: return body
    w, V = np.linalg.eigh(np.cov(np.stack([xs, ys]).astype(float)))
    if abs(V[0, np.argmax(w)]) >= _c3_HORIZ405: return body      # horizontal beam: left as z_1104 has it
    T = projtab.table()[_c3_T405]
    names = tuple(n for n in (T['blue'], T['red']) if n)
    SP = _a4_shape_pixels(ctx['R'], T['file'], names)
    im = ctx['img'].astype(np.int16); r, g, b = im[..., 0], im[..., 1], im[..., 2]
    teal = (im.max(2) >= _c3_BRIGHT405) & (r <= 105) & (g >= b - 20)
    cand = np.isin(SP - 1, [133]) & (ctx['cv'] >= _c3_CV405) & teal & _c3_bd(body, iterations=_c3_REACH405)
    cand = cand | body
    L, n = _c3_cc(cand, structure=np.ones((3, 3), bool))
    keep = np.unique(L[body]); keep = keep[keep > 0]
    return body | (np.isin(L, keep) & cand)


POST_BODY[_c3_T405] = _c3_beam405

# (2) 419 ArcadeAngel, Arcade006Projectile (6000568), both attack rows (owner: "some are cut off"): the crystal's white / yellow
#     lower faces, the lower-left rim and the pointed tail were left outside the label (installed rule = the front-most
#     draws only, ID-pass coverage >= 128). The projectile is drawn from the clip (arcade_006_atk_projectile): a big soft aura
#     (shape 1, alpha 123, blend normal), six opaque body pieces (shape 18, the blue faces / tail) and four opaque face pieces
#     (shape 0, the white / yellow lower faces), plus additive glows (shape 289, shape 3, shape 1 add: not body). Same structure
#     as the Arcade 002 tile fix (z_1103_arcade_electro (1)): the body is rasterised from the game's own geometry: every
#     opaque (mul alpha 255) normal-blend piece of shape 18 and shape 0, coverage >= 1/2, then the ID-pass pixels of this
#     projectile next to it (2 px), closed and filled: ONE solid label in the crystal's own outline. The aura and the glows stay out.
from scipy.ndimage import binary_closing as _c3_bc, binary_fill_holes as _c3_fh
_c3_T419 = 6000568
_c3_SHAPES419 = (0, 18)
_c3_prev419 = POST_BODY.get(_c3_T419)


def _c3_pieces419(o, cam):
    import masks
    S = PD.sc(o['file']); m = np.zeros((masks.H_FULL, masks.W_FULL), bool)
    for q in o['parts'].get('clip', []):
        if q['shape'] not in _c3_SHAPES419 or q.get('blend') != 'normal' or int(q['mul'][3]) < 250: continue
        r = masks.raster_poly(S, q, cam)
        if r is None: continue
        x0, y0, C, T = r
        mm = (1.0 - T[..., 0]) >= 0.5
        m[y0:y0 + mm.shape[0], x0:x0 + mm.shape[1]] |= mm
    return m[::2, ::2]


def _c3_arcade006(body, ctx):
    if _c3_prev419 is not None: body = _c3_prev419(body, ctx)
    o = ctx['o']
    if not body.any(): return body
    m = _c3_pieces419(o, ctx['cam'])
    if not m.any(): return body
    m = m | (ctx['mine'] & _dilate(m, 2) & (ctx['C'] <= 2))
    m = _c3_bc(np.pad(m, 3), iterations=2)[3:-3, 3:-3] | m
    return _c3_fh(m)


POST_BODY[_c3_T419] = _c3_arcade006

# (3) 457 IceDudeStrikers, Lou007Projectile (6000859), attack 90 deg row, ticks 315 and 319 (owner: "too small / cut off"; old
#     AI note: "ragged, cross over the cone and blobs"): each projectile is an orange cone on a cyan ball. The installed solid
#     rule (z_1103_icedude_ropedude (3)) only closes gaps <= 3 px, so the label has bites in the ball, the cone's stripes /
#     outline cut into it, tail spurs and unlabelled ball pieces. Fix: after that rule, the projectile's own sprite pixels
#     (ID-pass, same object) at coverage >= 90 within 3 px of the label are added, then the silhouette is closed with a larger
#     radius (4), holes filled and spurs opened away: ONE solid label for cone + ball (a smooth silhouette). Neighbouring
#     projectiles are never merged because only this object's own pixels are used. At tick 319 the label also took a detached
#     piece of the splash's bubble ring (a hit effect, not the attack): parts that do not lie on the object's own predicted
#     body are dropped (never the last one).
from scipy.ndimage import binary_opening as _c3_bo
_c3_T457 = 6000859
_c3_prev457 = POST_BODY.get(_c3_T457)


def _c3_lou007(body, ctx):
    if _c3_prev457 is not None: body = _c3_prev457(body, ctx)
    if not body.any(): return body
    near = _dilate(ctx['pred_any'], 5)
    L0, n0 = _c3_cc(body, structure=np.ones((3, 3), bool))          # detached bits away from our own predicted body (the splash's
    if n0 > 1:                                                      # bubble rings, a neighbour's cone) must not be bridged by the closing
        on = np.array([(near[L0 == k]).mean() for k in range(1, n0 + 1)])
        sz0 = np.bincount(L0.ravel())[1:]
        keep0 = (on >= 0.25) | (sz0 == sz0.max())
        body = np.isin(L0, 1 + np.nonzero(keep0)[0])
    # the region of the neighbouring projectiles overlaps ours: our own pixels only near our own prediction
    own = ctx['mine'] & (ctx['C'] <= 2) & (ctx['cv'] >= 90) & near
    b = body | (own & _dilate(body, 3))
    b = _c3_bc(np.pad(b, 8), iterations=4)[8:-8, 8:-8] | b
    b = _c3_fh(b)
    o2 = _c3_bo(np.pad(b, 8), iterations=4)[8:-8, 8:-8]
    if o2.any(): b = o2 | (b & _dilate(o2, 3))     # thin bridges to the next projectile / tail spurs opened away, corners restored
    b = b & (near | body)
    L, n = _c3_cc(b)
    if n > 1:
        sz = np.bincount(L.ravel())[1:]
        # a part is the projectile (cone + ball) only if it is not small against the largest part and lies on the object's own
        # predicted body (>= 25 % of its pixels within 5 px of it): a detached part away from the prediction (the bubble ring of
        # the splash after a hit) is a hit effect and is dropped; never all parts
        fr = np.array([(near[L == k]).mean() for k in range(1, n + 1)])
        ok = (sz >= 0.3 * sz.max()) & (fr >= 0.25)
        if not ok.any(): ok = sz == sz.max()
        b = np.isin(L, 1 + np.nonzero(ok)[0])
    return b


POST_BODY[_c3_T457] = _c3_lou007

# (4) 460 BanditGirlCybernoodles, Shelly010Projectile (6000018), both attack rows (owner: "some labels are missing or cut off"):
#     each shot is a bulb (cap, teal ring, dark-blue disc with the yellow sign). The installed rule labels only the clip's
#     'body' sprites near the predicted (narrow) body: an inner oval of the bulb (315: three bulbs cut to a narrow oval; 317: a bulb
#     that shows only its dark disc had no label). Measured on sw hull capture 460: the disc, the ring and the cap (shapes 6, 7, 8,
#     15, 17, 71, 72 at coverage >= 128) are drawn with the sprites the clip's trail shares, so the sprite code is 5 (shared; 4 'trail over
#     the body': never body) around the oval. The translucent streak / halo sprites (shapes 5 and 7 at alpha 128) are below the
#     coverage cut. Fix (labels wrapper, after all rules): opaque (coverage >= 128) code-4 pixels of those shapes that no label
#     has, within BULB_REACH px of a label of this type, go to the NEAREST label of the type (so neighbouring bulbs stay
#     separate; a disc that touches no label is a finished object's fading bulb: left unlabelled), then every label of the type is closed and its holes filled: ONE solid label for the whole bulb.
#     Type 6000018 is named by this skin only (Shelly010 = the Cybernoodles skin).
from scipy.ndimage import distance_transform_edt as _c3_edt
_c3_T460 = 6000018
_c3_SHAPES460 = (6, 7, 8, 15, 17, 71, 72)
_c3_REACH460 = 14
_c3_prev_labels460 = labels


def labels(objs, cam, R, state=None, ended=(), img=None):
    import bulletlabels as BL
    P = _c3_prev_labels460(objs, cam, R, state=state, ended=ended, img=img)
    mine = [o for o in objs if o['t'] == _c3_T460]
    if not mine: return P
    ids, lids, share = P['ids'], P['lids'], P['share']
    lab = {BL._lid(o): o for o in mine}
    ks = [k for k, l in enumerate(lids) if l[0] in lab and l[2] == 'body']
    if not ks: return P
    own = np.isin(ids, ks)
    if not own.any(): return P
    T = projtab.table()[_c3_T460]
    names = tuple(n for n in (T['blue'], T['red']) if n)
    SP = _a4_shape_pixels(R, T['file'], names)
    C = None
    for o in mine:
        c = sprite_pixels(o['t'], o['own'], R); C = c if C is None else np.where(C == 0, c, C)
    cv = R['cv'][::2, ::2]
    cand = ((C == 4) | (C == 5)) & (cv >= CUT) & np.isin(SP - 1, list(_c3_SHAPES460)) & (ids < 0)
    near = _dilate(own, _c3_REACH460) & cand
    L, n = _c3_cc(near | own, structure=np.ones((3, 3), bool))
    d, (iy, ix) = _c3_edt(~own, return_indices=True)
    new = ids.copy(); sh = share.copy()
    for j in range(1, n + 1):
        c = (L == j) & cand
        if not c.any(): continue
        if (L == j)[own].any():                        # touches a bulb: its pixels go to the nearest label (neighbours stay apart)
            tk = c & (d <= _c3_REACH460)
            new[tk] = ids[iy[tk], ix[tk]]; sh[tk] = np.maximum(sh[tk], CUT)
    for k in ks:                                       # each bulb solid: gaps closed, holes filled (only unlabelled pixels)
        m = new == k
        if not m.any(): continue
        f = _c3_bc(np.pad(m, 3), iterations=2)[3:-3, 3:-3] | m
        f = _c3_fh(f) & (new < 0)
        new[f] = k; sh[f] = np.maximum(sh[f], CUT)
    P2 = dict(P); P2.update(ids=new, share=sh)
    return P2
