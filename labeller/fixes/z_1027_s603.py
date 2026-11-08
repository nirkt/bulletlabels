# sweep 10-27 fix for skin 603 (Haiku fix, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s603/)
# Skin 603 BullDudeDemon (sweep 2026-10-27, pairs 10325 t271 and 10330 t307, plus the same flash on t273 and t313):
# Bull011Projectile (6000051, bull_011_atk_projectile) is drawn in the same batched draws (texture 766x696) as a white
# four-point impact star at the crate / wall. The star's pixels take the clip's shapes (none, 172, 171, 181, 20) and stayed in
# the label as red spurs and hatched fragments (rule 4: flashes are never labelled). The bullets are blue / cyan with frame
# min(R, G, B) <= 132 (p99 of their labelled pixels 94-127); the flash core is >= 130 (median 132-182). So:
#  1. the flash core (min(R, G, B) >= 130) is cut from the body;
#  2. within FLASH_R px of the core (the star's arms, thin cyan rays of 1-3 px) the body is opened (rays cut off, the bullet's
#     blobs, 10+ px across, survive the opening);
#  3. holes the cut leaves inside the cloud are filled (only those touching the cut).
# The cut runs only where the strong core (>= 150, >= 40 px) is in the body: t271 and t307 (the impact stars); the weak
# cyan star of t273 and t313 and all other frames stay as they are.
BULL011_FLASH_MIN = 130
BULL011_FLASH_R = 14
BULL011_OPEN = 3
BULL011_HOLE = 40   # px: holes the cut leaves inside the cloud up to this size are filled (bigger gaps stay as they are)
BULL011_CORE_MIN = 150   # min(R, G, B) of the strong core that gates the cut (the flash of t271 / t307; t273's weak cyan
BULL011_CORE_N = 40      # star reaches 150 in under 10 px, and cutting it there only made new loops: left as it was)


def _bull011_flash(body, ctx):
    from scipy.ndimage import binary_fill_holes, binary_opening
    img = ctx['img']
    if img is None or not body.any(): return body
    flash = img.min(-1) >= BULL011_FLASH_MIN
    core = body & flash
    if not core.any() or (body & (img.min(-1) >= BULL011_CORE_MIN)).sum() < BULL011_CORE_N: return body
    b = body & ~flash
    zone = _dilate(core, BULL011_FLASH_R)
    b = np.where(zone, binary_opening(b, iterations=BULL011_OPEN), b) & b   # rays out of the zone only
    # enclosed gaps only, drawn by its own sprites (C > 0, as idlabels fills its holes): ground / shadow inside the cloud stays out
    add = binary_fill_holes(b) & ~b & ~flash & ctx['region'] & (ctx['C'] > 0)
    if add.any():
        from scipy.ndimage import label as cc
        L, n = cc(add); sz = np.bincount(L.ravel()); touch = np.unique(L[_dilate(core, 1) & add])
        touch = touch[(touch > 0) & (sz[touch] <= BULL011_HOLE)]
        if touch.size: b = b | np.isin(L, touch)
    return b


POST_BODY[6000051] = _bull011_flash
