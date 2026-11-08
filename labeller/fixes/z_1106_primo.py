# installed 11-02 from out/sweep2/haikutasks/siteai4/primo/fix.py (owner ruling primo_fist: Primo first-update object unlabelled, all types 6000398-6000410; primo_spark 609 partly)
# Owner rulings 2026-11-02 (answer sheet RULINGS_1102.md), task primo: primo_fist + primo_spark.
# Exec'd in idlabels' namespace after all installed fixes (IDLABELS_OVERRIDE), keyed to the Primo type ids 6000398-6000410.
#
# 1. primo_fist (binding): El Primo's punch object is drawn at the brawler's fist on its FIRST update, with the blast texture,
#    then the same object becomes the blast. The owner rules that first-update fist sprite is a MUZZLE FLASH: unlabelled.
#    Data rule: an object of these types whose first update is the current tick (len(o['path']) == 1, one entry per update)
#    has an empty body. From its second update on (the blast) the label is the installed one, unchanged.
#    Checked on 511 (6000403, objects 2000006 at t269 and 2000010 at t309), 475 (6000399, 2000006 at t269, 2000010 at t309) and
#    609 (6000409, 2000006 at t269 and 2000010 at t309: already empty before this fix).
#
# 2. primo_spark (binding, 609): the fist label is the WHOLE fist silhouette, including the part under the yellow spark bar.
#    The bar is drawn over the fist from the effects sheet 10ab7fd9 (blend 1, its own draws, ID-pass code 4 "trail over"); the
#    fist's pixels under it are not labelled (a hole, with an outline loop round the bar, t305-t315 on skin 609).
#    Data rule (only where the bar is drawn over the fist; nothing else changes):
#      - the fist's predicted body (the projectile's silhouette from the game data, aligned as labels() does) is the outline;
#        the label must agree with it (>= PRIMO_SPARK_AGREE of the label inside it), else nothing is changed;
#      - added pixels: inside the predicted body and the label's convex hull, not labelled, drawn from the sheet and bright
#        yellow-green (G >= 235, B <= 170: the bar; cyan beams and glow fail the test);
#      - only bar pieces that touch the label, lie mostly near its core (PRIMO_SPARK_NEAR; sparks beside the fist do not),
#        and survive a 5 px opening (no thin edge strips) are added; gaps are closed (radius PRIMO_SPARK_CLOSE, frame px)
#        and holes filled, inside the predicted body only, so the bar's tip reaching out of the silhouette stays out.
#    Verified: 511 and 475 (blast types) unchanged apart from the first-tick objects; 609 nothing lost, pixels added only.
#    Known imperfections are listed in the report.
import numpy as np

PRIMO_TYPES = set(range(6000398, 6000411))   # Primo projectile types (skins 025 467 468 475 500 510 511 533 553 558 577 609 625)
PRIMO_SPARK_SHEET = '10ab7fd9'               # texture hash of the effects sheet the spark bar is drawn from (blend 1, 434 x 746)
PRIMO_SPARK_RGB = (235, 170)                 # the bar is bright yellow-green: G >= 235 and B <= 170 (cyan beams and glow fail)
PRIMO_SPARK_AGREE = 0.7                      # share of the label inside the predicted body, required for the silhouette fill
PRIMO_SPARK_NEAR = 0.4                       # share of a bar piece that must lie near the fist's core (spark pieces beside the fist: 0-0.38)
PRIMO_SPARK_CLOSE = 2                        # px: gap closing radius around the bar's edges (inside the silhouette only)


def _primo_disk(k):
    yy, xx = np.mgrid[-k:k + 1, -k:k + 1]
    return (xx * xx + yy * yy) <= k * k


def _primo_spark(body, ctx):
    """primo_spark: the fist's whole silhouette (predicted body), the spark bar's hidden part included (see header 2)"""
    from scipy.ndimage import binary_fill_holes, binary_closing
    if not body.any(): return body
    C = ctx['C']; reg = ctx['region']; ids = ctx['ids_h']; img = ctx['img']
    pb = align(ctx['pred_body'], ctx['mine'], C, reg) & reg
    if not pb.any() or (body & pb).sum() < PRIMO_SPARK_AGREE * body.sum(): return body
    sheet = [d for d, v in ctx['R']['draws'].items() if str(v.get('th', '')).startswith(PRIMO_SPARK_SHEET) and v.get('blend') == 1]
    if not sheet: return body
    im = img.astype(np.int16)
    bar = np.isin(ids, sheet) & (im[..., 1] >= PRIMO_SPARK_RGB[0]) & (im[..., 2] <= PRIMO_SPARK_RGB[1])
    # the bar lies over the fist: its pixels count only inside the label's own convex hull (the fist's silhouette, the bar's
    # tip that reaches out stays out) and only where they touch the label (sparks of other objects beside it are not taken)
    from scipy.ndimage import label as cc, binary_dilation, binary_opening
    cand = pb & ~body & bar & hull_mask(body)
    if not cand.any(): return body
    L, n = cc(cand); touch = binary_dilation(body, iterations=1)
    # a spark piece beside the fist (not over it) is not part of the fist: most of a kept piece lies near the fist's core
    from scipy.ndimage import binary_fill_holes as _fh
    core = binary_opening(_fh(body), structure=_primo_disk(4)); near = binary_dilation(core, iterations=3) if core.any() else body
    keep = []
    for kk in np.unique(L[touch & cand]):
        if kk <= 0: continue
        comp = L == kk
        if (comp & near).sum() >= PRIMO_SPARK_NEAR * comp.sum(): keep.append(kk)
    keep = np.array(keep, dtype=int)
    add = np.isin(L, keep) if keep.size else np.zeros_like(cand)
    if not add.any(): return body
    # thin strips of the bar's edge (under 5 px wide) are not the fist: only added pixels that survive an opening are kept
    from scipy.ndimage import binary_opening
    add = add & binary_opening(body | add, structure=_primo_disk(2))
    if not add.any(): return body
    out = body | add
    k = PRIMO_SPARK_CLOSE; p = k + 2
    cl = binary_closing(np.pad(out, p), structure=_primo_disk(k))[p:-p, p:-p]
    out = binary_fill_holes(cl | out) & ((pb & hull_mask(body | add)) | body)
    return out


def _primo_rule(body, ctx):
    """one body rule for every Primo type: first update = muzzle flash (unlabelled), later updates = fist / blast"""
    if len(ctx['o']['path']) <= 1:            # primo_fist: the first update's sprite at the fist (muzzle flash)
        return body & False
    return _primo_spark(body, ctx)


def _primo_wrap(t):
    prev = POST_BODY.get(t)
    def fn(body, ctx):
        if prev is not None: body = prev(body, ctx)
        return _primo_rule(body, ctx)
    return fn


for _t in sorted(PRIMO_TYPES):
    POST_BODY[_t] = _primo_wrap(_t)
