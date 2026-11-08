# AI-review batch 2026-11-01 fix (Haiku fix, reviewed by the coordinator, installed 11-01 LOCAL ONLY; source out/sweep2/haikutasks/siteai/beamer/fix.py)
# AI-review fixes, task beamer (skins 336, 405, 433), 2026-11-01. Override file: exec'd after all installed fixes.
#
# (1) 336 Beamerharvest, attack 0 deg, tick 269 (reviewer: "too small / cut off"): the beam's bright front is cut off, the
#     outline ends about 12-19 px before the white front. Same mechanism as z_1027_s336 (type 6001302, shape 150 core beyond
#     the body's far end). At tick 269 the front is slightly dimmer than at 265-311: its pixels have min(R,G,B) 203-219
#     (the old BEAM3_WHITE = 225 excludes them); the blue tail is 55-170 and the side fringe 150-190 beside it. Unlabelled
#     shape-150 pixels beyond x=760 at 269: 240 px before, 172 px after. BEAM3_WHITE is read only by the 6001302 hook of
#     z_1027_s336, so this changes that type alone. Also at attack 90 deg (ticks 307, 309) the lower white tip gains a few
#     px (19 and 78); no other tick of 6001302 changes (265, 267: +13 to +17 px at the front).
BEAM3_WHITE = 200

# (2) 336 Beamerharvest, super 6001303 (Beamer003UltiProjectile, the crystal cube shield), tick 373 (reviewer: "the two
#     small cyan cube debris pieces are labelled and should not be"): the shield has shattered (end animation). The tracker
#     still lists the super until tick 375, so the 'ended' flag does not help; what remains are three small shards of the
#     cube sprite (BODY_ONLY keeps the largest of them). Label body size (image-res px): ticks 357-369 1571-1842 (intact
#     shield), 371 1052 (still the shield), 373 114 (three shards: 60, 30, 24 px). A body under SUPER_SHIELD_MIN is the
#     broken-up debris and is dropped; an intact shield is never that small. Owner rule: end animations are not labelled.
SUPER_SHIELD_TYPE = 6001303
SUPER_SHIELD_MIN = 500       # image-res px (540x1200) of the label body


def _super_shield_shards(body, ctx):
    if not body.any(): return body
    if int(body.sum()) < SUPER_SHIELD_MIN:
        return body & False   # the shattered shield: its leftover shards are an end animation, not the attack
    return body


POST_BODY[SUPER_SHIELD_TYPE] = _super_shield_shards

# (3) 405 BeamerCarnival, super 6001307 (one object per shot: the blue-cyan beam and its yellow / green arrow head),
#     ticks 361 and 369 (reviewer: "covered by several adjacent outlines with dividing lines across the streak instead of
#     one hull"). Confirmed on sw2_405: the label of the one object is two parts (body + ground kinds) split by a boundary
#     across the beam (361: 2377 + 467 px; 369: 3050 + 1025 px). The ground kind is assigned after the per-type hooks, so a
#     POST_BODY cannot join it. Owner rule: thin lines / streaks = ONE convex hull over the whole attack. Fix: one hull over
#     the union of the object's body and ground pixels on every tick of the super (a labels() wrapper, as z_1027_s014).
#     The hull fills unlabelled background inside it, but not bright-yellow pixels (the hit star at the bend: ticks 357,
#     361, 369; the star lies on the hull edge at 361) nor orange pixels (the crate at tick 365): checked, 0 such pixels are
#     added on any tick. No labelled pixel of the attack is removed (lost px 0 on 357-373).
S405_TYPE = 6001307
_s405_prev_labels = labels


def labels(objs, cam, R, state=None, ended=(), img=None):
    import numpy as _np5
    import bulletlabels as BL
    P = _s405_prev_labels(objs, cam, R, state=state, ended=ended, img=img)
    lab = {BL._lid(o) for o in objs if o['t'] == S405_TYPE}
    if not lab: return P
    ids, lids, share = P['ids'], P['lids'], P['share']
    kb = [k for k, l in enumerate(lids) if l[0] in lab and l[2] == 'body']
    kk = [k for k, l in enumerate(lids) if l[0] in lab and l[2] in ('body', 'ground')]
    if not kb or not kk: return P
    ours = _np5.isin(ids, kk)
    if not ours.any(): return P
    h = hull_mask(ours)                       # one convex hull over the whole attack (body and ground parts)
    take = h & ((ids < 0) | ours)             # never take another object's label
    if img is not None:                       # the hull fills no bright-yellow hit star or orange crate that is not labelled
        im = img.astype(_np5.int16); r, g, b = im[..., 0], im[..., 1], im[..., 2]
        flash = ((r > 200) & (g > 200) & (b < 120)) | ((r > 180) & (g > 70) & (g < 170) & (b < 110))
        take &= ours | ~flash
    new = ids.copy(); new[ours] = -1; new[take] = kb[0]
    sh = share.copy(); sh[take] = _np5.maximum(sh[take], BL.CUT)
    P2 = dict(P); P2.update(ids=new, share=sh)
    return P2


# (4) 433 BeamerMecha, super 6001311 (Beamer007UltiProjectile, the spear: a bright blue-white shaft and its head), tick 373
#     (reviewer: "the spear outline covers only a thin sliver along its lower edge, leaving the bright beam body outside").
#     At 373 the label is 1396 px against 5102 px at 369; shape 36 (the spear's main clip body) has 866 px outside the label,
#     and those pixels are translucent (their coverage is under the 128 cut). FADE_CUT lowers the cut for this type's
#     translucent body (as for Ghost / Frank types in idlabels): the shaft is labelled at 373 (2411 px, shape 36 unlabelled
#     60 px). Side effect to check by eye: the shaft's far end at 369 grows by about 1300 px (the label reaches the far
#     translucent end of the shaft) and at 357 / 359 the outline reaches the brawler's arm slightly as before.
FADE_CUT[6001311] = 32
