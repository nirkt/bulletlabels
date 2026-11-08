# sweep 10-27 fix for skin 396 (Haiku fix, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s396/)
# sweep 2026-10-27 task s396 (skin 396 MechanicJessie), pairs 7463 / 7464 / 7465 (attack 0 deg, ticks 267, 269, 271)
# and the same type on the attack 90 deg row of this skin (ticks 305-319, same defect, not in the findings).
# Type 6000194 Jessie009Projectile1 (clip jessie_009_atk_projectile, red jessie_009_atk_projectile_red; trail
# jessie_009_atk_trail). The wave is a bright cyan-white streak. Its clip is drawn as several nested chevron arcs (the
# streak's outer edges and bright inner arcs), and the ID pass sees only those arcs, so the label was several nested
# contours with unfilled gaps between them (rule 6: an attack drawn as several lines / streaks gets ONE label, the convex
# hull of the whole attack).
# Approaches considered:
#   1. ONE_LABEL 'hull' on the labelled body (chosen): uses only the attack's own clip pixels (the arcs), so it is the hull
#      of the streak; the cyan rings beside it are trail draws drawn over the body (code 4), never body, and stay out.
#   2. close + fill holes of the arcs (the object's own outline filled solid): rejected, the arcs are open chevrons
#      with wide gaps at several ticks, closing them would need a radius large enough to take in the glow.
#   3. a colour / brightness threshold on the frame: rejected, the streak's brightness changes with the scene.
# Verified (count_check.py): one part per tick after the fix (before: 4-13 parts with specks), on ticks 265-271 and
# 305-317; the hull only adds pixels, so every kept pixel stays. Ticks 263, 273 and 319 have no label before or after.
# Nothing else is changed: only this type id, and the hull is taken only over pixels the label already has.
ONE_LABEL[6000194] = 'hull'
