# sweep 10-27 fix for skin 284 (Haiku fix, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s284/)
# sweep 10-27 fix for skin 284 HookDudeSwamp (task s284: out/sweep2/haikutasks/sweep1027/s284/task.md). Haiku fix agent.
# The swamp blob attacks of HookDude: attack 6000505 (Gene004Projectile, cyan blob, attack class) and super 6000507
# (Gene004UltiProjectile, the 3D blob with leaves, super class). Findings (sweep 2026-10-27, capture sw2_284):
#  - pair 5262 (attack 0 deg, tick 265) and pair 5267 (attack 90 deg, tick 305): the label took in detached cyan dots and a
#    droplet beside the blob (tick 265: a 52 px piece of 646 px; tick 305: a 62 px teardrop of 707 px, joined by a thin neck).
#  - pair 5270 (attack 90 deg, tick 311): a 1 px speck of the green trail was labelled (the shared clean-up only drops
#    specks not joined to the blob within its bridging distance, and this one is joined).
#  - pair 5277 (super 45 deg, tick 353): thin magenta spikes at the lower-left and lower-right edges, where the splash
#    droplets sit, and small separate specks (5-9 px at half resolution) near the droplets.
# Fix: (1) an opening with a disk of GENE_OPEN px (half resolution) removes the thin spikes and necks; the blob's own
# outline is unchanged by it. The opening is skipped when it would remove more than half of a small body (early,
# small frames keep their body). (2) Only the pieces of the opened body that are at least GENE_KEEP_FRAC of its largest
# piece are kept: the main blob and any large part of it; detached dots, droplets and specks are dropped.
# Only the two types above are changed. Their other frames are unaffected unless they have such parts.
GENE_TYPES = (6000505, 6000507)
GENE_OPEN = {6000505: 2, 6000507: 1}   # px (half resolution): the opening's disk radius per type (attack: necks and specks
#   thinner than about 5 px; super: radius 1 only, so the leaf tips of the 3D model, 3-5 px wide, stay labelled)
GENE_KEEP_FRAC = 0.2   # pieces smaller than this share of the largest piece are dropped (detached dots and specks)


def _gene_disk(r):
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r


def _gene_body(body, ctx):
    from scipy.ndimage import binary_opening, label as cc
    if not body.any(): return body
    op = binary_opening(body, structure=_gene_disk(GENE_OPEN[ctx['t']]))
    if op.sum() < 0.5 * body.sum(): op = body   # small body: no opening (its thin parts are most of it)
    L, n = cc(op, structure=np.ones((3, 3), bool))
    if n == 0: return body
    sz = np.bincount(L.ravel())[1:]
    keep = [k + 1 for k in range(n) if sz[k] >= GENE_KEEP_FRAC * sz.max()]
    return np.isin(L, keep)


for _gt in GENE_TYPES:
    POST_BODY[_gt] = _gene_body
