# installed 11-02 from out/sweep2/haikutasks/siteai4/samurai/fix.py (owner rulings: 412 creature teal in + one solid label, 412/583/630 parts under the name plate filled from the predicted body)
# Owner rulings fix round 2026-11-02 (RULINGS_1102.md), task samurai: samurai_teal and nameplate. Exec'd in idlabels'
# namespace (IDLABELS_OVERRIDE) after all installed fixes. Changes keyed to explicit type ids only.
#
# ---- 1. samurai_teal (ruling, binding): skin 412 SamuraiJapanOvercharge, super creature Samurai003UltiProjectile
#         (type 6001513, ticks ~345-359). The lower teal part of the creature IS part of the creature: the teal body
#         sprites of its trail clips (effects_brawler_kenji.sc, shapes 0 / 1 / 89) were left out of the label, so the
#         label was a notched wedge (green spikes + part of the belly). Fix: those teal shapes are added when they touch
#         the label body (connected to it) and lie in the object's region; only opaque pixels (coverage >= CUT), so the
#         soft halo stays out; holes inside the creature are filled. Then (coordinator, 11-02) the notches the creature's
#         outline still has (a dark crease between its upper body and lower-left lobe at t353, sparkles over its top edge
#         at t355 / t357) are closed by SAM_CREATURE_CLOSE px, inside the creature's region: one solid label.
# ---- 2. nameplate (ruling, binding; coordinator 11-02: a general hidden-parts rule, so skin 412 too): the UI name plate
#         (draw fs SAM_PLATE_FS, the "...ERY ROSA" name plate and the "8000" bar) hides part of a super's round / block
#         outline and the label was cut along the plate edge: 630 super t345-t347 (orb cut at the lower left), 583 super
#         t345-t349, and the creature of 412 at t345-t349. Fix, per frame: where the plate covers at least SAM_PLATE_MIN
#         pixels of the object's predicted body (ctx['pred_body'], the game's own prediction of its outline), the label
#         is filled from that predicted body (the plate's letters too: they are drawn by another draw). Frames the plate
#         does not cover (e.g. 630 t349 and later) are unchanged.
# Only types 6001512 (583), 6001513 (412) and 6001514 (630) change; each keeps every label pixel it had.

from scipy.ndimage import label as _ccS, binary_fill_holes as _bfhS

SAM_CREATURE = 6001513
SAM_TEAL_FILE = 'sc/effects_brawler_kenji.sc'   # the file of the creature's clips (its blue / red glow and its trail clips)
SAM_TEAL_SHAPES = (0, 1, 89)                    # the creature's teal body shapes of that file (shape ids)
SAM_TEAL_REACH = 70                             # px: reach of the creature's own teal draws beyond its region (see below)
SAM_CREATURE_CLOSE = 3                          # px: notches of the creature's outline closed (crease, sparkles over its edge)


def _sam_creature_teal(body, ctx):
    """the creature's label body plus the connected teal creature shapes of its region (opaque only), holes filled"""
    if not body.any(): return body
    t, R = ctx['t'], ctx['R']
    T = projtab.table()[t]
    names = tuple(sorted({n for f, n in _trail_clips(t) if f == SAM_TEAL_FILE} | {T.get('blue'), T.get('red')} - {None, ''}))
    if not names: return body
    SP = _a4_shape_pixels(R, SAM_TEAL_FILE, names)
    teal = np.isin(SP, [s + 1 for s in SAM_TEAL_SHAPES])
    cv = ctx['cv']; reg = ctx['region']
    # the creature's own teal draws (the draws that show its teal shapes inside its region), searched out to SAM_TEAL_REACH px
    # of its predicted body: the upper lobe the region edge cut off. Only those draws, so other effects do not join.
    c0 = teal & reg & (cv >= CUT)
    dr = np.unique(ctx['ids_h'][c0]); dr = dr[dr > 0]
    cand = teal & _dilate(ctx['pred_any'], SAM_TEAL_REACH) & (cv >= CUT) & np.isin(ctx['ids_h'], dr)
    L, n = _ccS(cand | body, structure=np.ones((3, 3), bool))
    keep = np.unique(L[body]); keep = keep[keep > 0]
    out = body | (np.isin(L, keep) & cand)
    out = out | (_bfhS(out) & reg & ((cv >= CUT) | out))     # enclosed holes of the creature (dark belly spots)
    out = out | (_close(out, SAM_CREATURE_CLOSE) & reg)      # notches of its outline (crease, sparkles over the edge)
    return out


SAM_PLATE_FS = 3318493823      # the UI name plate's draw shader hash (stable across captures, see the brief)
SAM_PLATE_MIN = 20             # predicted-body pixels under the plate needed before the fill applies


def _sam_plate_fill(body, ctx):
    pb = ctx['pred_body']
    if not pb.any(): return body
    R = ctx['R']
    ui = np.isin(ctx['ids_h'], [d for d, dr in R['draws'].items() if dr.get('fs') == SAM_PLATE_FS])
    hidden = pb & ui & ~body
    if hidden.sum() < SAM_PLATE_MIN: return body
    return body | (pb & ~body)


def _sam_creature_full(body, ctx):
    """412 creature: teal shapes, holes and notches (part 1), then the name plate fill (part 2); the plate fill can
    enclose a dark spot of the creature (its eye at t345), so the holes are filled again inside the creature's region"""
    out = _sam_plate_fill(_sam_creature_teal(body, ctx), ctx)
    if not out.any(): return out
    return out | (_bfhS(out) & ctx['region'])


POST_BODY[SAM_CREATURE] = _sam_creature_full
POST_BODY[6001512] = _sam_plate_fill           # 583 SamuraiStarrForce super (Samurai002)
POST_BODY[6001514] = _sam_plate_fill           # 630 SamuraiWindstock super (Samurai004)
