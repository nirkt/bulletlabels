# installed 11-01 from out/sweep2/haikutasks/siteai2/bo_ruffs_shaman/fix.py (2nd AI sweep: 264 flame hull, 318 blob smooth, 334 column; 295 arrow hull DISABLED)
# Second AI-review sweep (2026-11-01), task bo_ruffs_shaman. Override file, exec'd in idlabels' namespace after all installed fixes
# (IDLABELS_OVERRIDE). Skins 264, 295, 318, 334 (322 and 393: no change, see the report).
# Owner rules used: one label per projectile in the object's own outline, solid, no notches / spurs / holes; thin or splintered
# attack of ONE object = one convex hull (brawler bands out, see _bo_hull_clear in z_1103_bo_ruffs_shaman.py); the flame, the arrow
# and the splash are one object each; glow and other effects stay out.
from scipy.ndimage import binary_fill_holes as _bo2_fill, binary_opening as _bo2_open, label as _bo2_cc


def _bo2_keep_joined(m, body, region):
    """the parts of m joined to the object's own body (8-connected), inside region"""
    L, n = _bo2_cc(m, structure=np.ones((3, 3), bool))
    keep = np.unique(L[body]); keep = keep[keep > 0]
    return np.isin(L, keep) & region


# (1) 295 BowDudeSanta, attack 90 (and attack 0) arrow, type 6000365 (Bo005Projectile: one arrow per object: a bright head with a
#     fire tail and a thin blue shaft, drawn from several sprites). Reviewer tick 305: "labelled only on its glowing head while the
#     shaft above it is outside". In the new capture the shaft comes out as ragged strips and gaps (ticks 309 to 315, three arrows).
#     Owner rule: thin / multi-piece attack of one object = one convex hull per shot; separate arrows keep separate labels.
# NOT INSTALLED (coordinator 11-01): the hull spanned several arrows from the brawler down (compare_295.png t307-t317)
# _bo_hull_post(6000365)


# (2) 264 RuffsRonin, flames, type 6000890 (Ruffs002Projectile: two flames per shot). Reviewer ticks 265 / 267 (attack 0) "ragged with
#     interior lines", 305 / 311 (attack 90) "the blue flame body above the bulb is outside". The installed z_1103 hook keeps the bulb
#     and the cyan tongues only (shape 85); the owner's rule: the label covers the whole flame body, bulb and coloured flame above it,
#     never the soft glow. Fix: one solid convex hull per flame (its own body, brawler bands out): the ragged interior lines and the
#     notches go, the flame's blue body between its tongues and its bulb is inside. The blue tail above the top curl is NOT in the
#     ID pass (no sprite code, additive over the ground) and is not reached by this fix: see the report.
_bo_hull_post(6000890)


# (3) 318 ShamanGummybear, splash blob, type 6000232 (Shaman006Projectile, radius 250; grows and shrinks over ~10 ticks; attack 0
#     and attack 90). Reviewer ticks 273 / 309: "holes / ragged: lumpy outline with notches". Owner rule: one solid piece, smooth edges,
#     no notches. Fix: close the bays (3 px), fill holes, open 1-px spurs, keep the main part (ragged bay pixels only). Crates and
#     walls in front are not part of the blob and stay cut (closing is limited to the object's own region).
def _bo2_smooth_blob(body, ctx):
    if not body.any(): return body
    reg = ctx['region']
    m = (_close(body, 3) & reg) | body
    m = _bo2_fill(m) & reg | body
    m = _bo2_open(m, np.ones((3, 3), bool))
    m = _bo2_keep_joined(m, body, reg) | (body & _bo2_open(body, np.ones((3, 3), bool)))
    return _bo2_fill(m) & reg


POST_BODY[6000232] = _bo2_smooth_blob


# (4) 334 PuppeteerWasteland, attack 90, type 6001320 (Puppeteer002Projectile: a bright cyan column). Reviewer tick 305: "the label
#     covers only the middle of the bright column; its upper part above the label is outside". The column's bright cyan part is
#     additive (no non-additive draw of its own), so the ID pass only has its middle. Fix: the cyan additive pixels (G, B high,
#     G clearly above R, so not the green zone ring, not the purple robe, not the white flash) joined to the object's own body,
#     inside a 60 px band around its prediction, then one convex hull (a thin streak).
def _bo2_column(body, ctx):
    img = ctx['img']
    if img is None or not body.any(): return body
    im = img.astype(np.int16)
    cy = (im[..., 1] >= 190) & (im[..., 2] >= 220) & (im[..., 1] - im[..., 0] >= 80)
    band = _dilate(ctx['pred_any'], 60)
    m = (body | (cy & band)) & band
    m = _bo2_keep_joined(m, body, band) | body
    return _bo2_fill(m) & band | body


POST_BODY[6001320] = _bo2_column
_bo_hull_post(6001320)
