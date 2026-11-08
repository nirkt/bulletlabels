# AI-review batch 2026-11-01 fix (Haiku fix, reviewed by the coordinator, installed 11-01 LOCAL ONLY; source out/sweep2/haikutasks/siteai/speedy_mecha_maisie/fix.py)
# Task speedy_mecha_maisie (AI-reviewer findings 2026-11-01), fix for skins 618 and 622. Exec'd in idlabels' namespace
# (IDLABELS_OVERRIDE). Findings handled here:
#
# 618 SpeedyEsports, attack 0 t275 ("holes / ragged": the lightning outline is a zigzag mesh with interior lines) and the
# same type on attack 90 (t305-t319, separate bolts). Type 6000720 = Max006Projectile, the lightning bolt. Owner rule:
# thin lines / streaks = one convex hull per shot, separate projectiles keep separate outlines. ONE_LABEL 'hull' on the
# whole type was tested and rejected: on attack 90 it drew one hull over 3-4 separate bolts. This hook takes the hull per
# connected part of the label, so each bolt of the volley keeps its own hull and the zigzag mesh is replaced by a clean
# hull. Known limit (needs_user): on attack 0 at t275 the bolt splits into 2-3 parts that still get separate hulls;
# grouping parts by 2-4 px merges the attack 90 bolts too, so no distance rule separates the two.
#
# 622 MaisieDragons, attack 90 t313 ("too small / cut off": the dark blue flame plume above the swirl is outside the
# outline). Type 6001334 = Maisie005Projectile. Its dark blue flame plume is translucent (coverage below the 128 cut at
# t313: 0 px at >= 128), so it dropped out while it stays labelled at t305-t309. FADE_CUT (coverage cut for translucent
# bodies, as Beamer's and Willow's attacks) lets its flame body in from coverage 32. The glow around it stays out.

from scipy.ndimage import label as _cc6000720


def _fix_6000720(body, ctx):
    if not body.any(): return body
    L, n = _cc6000720(body, structure=np.ones((3, 3), bool)); out = np.zeros_like(body)
    for k in range(1, n + 1):
        m = L == k
        out |= hull_mask(m) if m.sum() >= 3 else m     # one hull per connected part (one per bolt)
    return out


POST_BODY[6000720] = _fix_6000720
FADE_CUT[6001334] = 32
