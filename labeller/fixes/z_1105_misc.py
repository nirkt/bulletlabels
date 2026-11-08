# installed 11-02 from out/sweep2/haikutasks/siteai3/misc/fix.py (regression round: 412 zone 17000777 not cut by bushes, 373 Poco008 notch closed)
# Fix file for the regression-sweep findings of 2026-11-02, task misc (skins 359, 373, 412, 529, 548). Exec'd in idlabels'
# namespace (IDLABELS_OVERRIDE) after all installed fixes (py/fixes/z_1101..z_1104). Keyed to explicit type / did ids only.
#
# ---- 412 SamuraiJapanOvercharge, super zone 17000777 (Samurai003CrossSlashArea), blade hull at t389 (page _02 super 45):
# the hull of the blade (z_1103_samurai_whirl) is cut by the depth test against everything drawn in front of the ground, so
# the green bushes at the lower right cut its edge into a ragged, notched outline (RULES 7: zones are never cut by bushes,
# never effects). Fix: for this did only, the bush pixels (green, in front) are not counted as occluders in the depth cut;
# walls and crates still cut. Only did 17000777 changes (the other Samurai zones 583 / 630 keep their behaviour).

F412_DID = 17000777
F412_GREEN = 30            # green excess (G - max(R, B)) of a bush pixel; floor, walls, crates and the blade are far below
_f412_img = [None]
_f412_prev_labels = labels


def labels(objs, cam, R, state=None, ended=(), img=None):
    _f412_img[0] = img
    return _f412_prev_labels(objs, cam, R, state=state, ended=ended, img=img)


_f412_prev_zone = zone_labels


def zone_labels(state, cam, R, taken, own_team=None):
    out = _f412_prev_zone(state, cam, R, taken, own_team)
    img = _f412_img[0]
    if not state or img is None or R.get('wd') is None: return out
    items = [it for it in state.get('items', []) if int(it[1]) == F412_DID]
    if not items: return out
    wd = R['wd']; H, W = wd.shape
    im = img[::2, ::2] if img.shape[:2] != (H, W) else img
    green = (im[..., 1].astype(int) - np.maximum(im[..., 0], im[..., 2]).astype(int)) > F412_GREEN
    wd2 = wd.copy(); wd2[green] = 1e9              # green pixels are not in front: not an occluder for this zone
    R2 = dict(R); R2['wd'] = wd2
    out2 = _f412_prev_zone(state, cam, R2, taken, own_team)
    for it in items:
        iid = int(it[0])
        if iid in out2: out[iid] = out2[iid]
        else: out.pop(iid, None)
    return out


# ---- 373 DeadMariachiOnceupon, Poco008 attack (type 6000349), attack 90 t311: the label of the top-left wisp has a notch
# between two of its leaves (the background shows through a gap open to the outside: RULES, holes / ragged: one clean piece
# in the object's own outline). Fix: after the Poco rule and the barrel cut, a small closing (radius F373_CLOSE, frame px)
# of this type's own body, limited to the region; the cloud of this type is one solid outline (Poco rule, RULES 7).
# Only type 6000349 changes (skin 373 has no other user of this type).

F373_CLOSE = 4
from scipy.ndimage import binary_closing as _f373_bc, binary_fill_holes as _f373_fh


def _f373_disc(r):
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    return (xx * xx + yy * yy) <= r * r


_f373_prev_post = POST_BODY.get(6000349)


def _f373_close(body, ctx):
    if _f373_prev_post is not None:
        body = _f373_prev_post(body, ctx)
    if not body.any(): return body
    P = F373_CLOSE + 1
    cl = _f373_bc(np.pad(body, P), structure=_f373_disc(F373_CLOSE))[P:-P, P:-P] | body
    add = _f373_fh(cl) & ctx['region']          # only pixels are added (the previous body is kept as it is)
    return body | add


POST_BODY[6000349] = _f373_close
