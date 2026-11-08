# s118 round 3 (10-28, skin 118 JesterWoodspirit): spark rim rings / spark-sprite bridges labelled on the ghosts.
# Type 6001270 (Chester002Projectile, Jester's default attack: a cyan ghost with a green rim, sparks trailing it).
# Findings 1844-1846 (attack 0 deg, ticks 265-269) and 1850-1854 (attack 90 deg, ticks 307-315). Cause (ID pass, by sprite
# shape): the label takes in (a) the TRAIL spark sprites (chester_002_atk_spark_01..03 / particle_gas_seq, shapes 64065-64068,
# 47) and (b) small instances of the clip's thin rim-ring shape (63, with 407) that are drawn around the free-floating green
# sparks: a thin loop or a spur from the ghost to a spark. The ghost's OWN pixels (its other shapes) are never touched, so no
# flat cuts into the ghost / its green rim (the rejected fix2 used a morphological opening around the ghost core).
# Rule: within one ghost label, the "spark-type" pixels (shapes of the trail spark sprites + shapes 63, 407, 0 of the clip)
# are split in 8-connected pieces; a piece lying mostly (>= OUT_FRAC) outside the ghost core (all the label's other pixels,
# closed, grown by NEAR px) is a spark piece and is removed. The ghost's own rim ring (shape 63 hugging the body) stays.
# Runs in the labels() wrapper before the global clean-up (z_1099), which then drops the leftover crumbs / fills holes.
# Affects every skin that fires type 6001270 (shared by the Jester skins).
SP_NEAR = 2        # px (label px): the ghost core grown by this counts as 'inside'
SP_OUT_FRAC = 0.5  # a spark-type piece with at least this share of its pixels outside the grown core is removed
SP_CLIP_SPARK = (63, 407, 0)
SP_STUB = 2        # a spark-type pixel with at most this many label neighbours (8-conn) is a protruding end
SP_SPECK = 25      # px
SP_TRIM = 3        # trim rounds
SP_TYPE_FRAC = 0.6  # a draw is spark-type when this share of its label pixels are spark-type shapes


def _s118_shapes(R, t):
    """(shape id + 1 per pixel of the type's clip, same for its trail clips) at label resolution"""
    T = projtab.table()[t]; clip = {}; trail = {}
    for k in ('blue', 'red'):
        if T.get(k): clip.setdefault(T['file'], []).append(T[k])
    for f, n in _trail_clips(t): trail.setdefault(f, []).append(n)
    def run(d):
        out = None
        for f, names in d.items():
            sp = _a4_shape_pixels(R, f, tuple(sorted(set(names))))
            out = sp if out is None else np.where(out > 0, out, sp)
        return out
    return run(clip), run(trail)


def _s118_clean_label(m, clipSP, trailSP, did):
    """did: draw id per pixel (label resolution). Spark-type draws: >= SP_TYPE_FRAC of their label pixels are spark-type shapes"""
    from scipy import ndimage as nd
    cand = np.zeros_like(m)
    if trailSP is not None: cand |= (trailSP > 0)
    if clipSP is not None:
        for s in SP_CLIP_SPARK: cand |= (clipSP == s + 1)
    cand &= m
    if not cand.any(): return m
    spark_draw = np.zeros_like(m)
    for d in np.unique(did[m]):
        dm = m & (did == d)
        if (dm & cand).sum() >= SP_TYPE_FRAC * dm.sum(): spark_draw |= dm
    if not spark_draw.any(): return m
    core = m & ~spark_draw
    if core.sum() < 120: return m      # thin / hidden ghost: nothing certain, never lose a projectile
    st = np.ones((3, 3), bool)
    core = nd.binary_closing(np.pad(core, 4), structure=st, iterations=2)[4:-4, 4:-4]
    core = nd.binary_fill_holes(core)
    near = nd.binary_dilation(core, structure=st, iterations=SP_NEAR)
    L, n = nd.label(spark_draw, structure=st)
    out = m.copy()
    for i in range(1, n + 1):
        c = L == i
        if (c & ~near).sum() >= SP_OUT_FRAC * c.sum(): out &= ~c
    # stubs: what is left of a spark-type draw as a protruding end (<= SP_STUB label neighbours) is trimmed, a few rounds
    sd = spark_draw & out
    for _ in range(SP_TRIM):
        nb = nd.convolve(out.astype(np.int8), np.ones((3, 3), np.int8), mode='constant') - 1
        e = sd & out & (nb <= SP_STUB)
        if not e.any(): break
        out &= ~e
    # separate specks of the label (< SP_SPECK px, next to a large piece): dropped here, the global clean-up bridges 6 px
    L, n = nd.label(out, structure=st)
    if n > 1:
        sz = np.bincount(L.ravel()); sz[0] = 0
        if sz.max() >= 10 * SP_SPECK: out &= (sz[L] >= SP_SPECK)
    return out


_s118_labels_prev = labels


def labels(objs, cam, R, state=None, ended=(), img=None):
    import bulletlabels as BL
    P = _s118_labels_prev(objs, cam, R, state=state, ended=ended, img=img)
    mine = {BL._lid(o) for o in objs if o['t'] == 6001270}
    if not mine: return P
    ids, lids, share = P['ids'], P['lids'], P['share']
    clipSP = trailSP = None; got = False
    for k, lid in enumerate(lids):
        if lid[2] != 'body' or lid[0] not in mine: continue
        m = ids == k
        if not m.any(): continue
        if not got:
            clipSP, trailSP = _s118_shapes(R, 6001270); got = True
        d = m & ~_s118_clean_label(m, clipSP, trailSP, R['id'][::2, ::2])
        if d.any():
            ids[d] = -1; share[d] = 0
    return P
