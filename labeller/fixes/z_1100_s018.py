# human review 2026-10-28 fix for skin 018 (Haiku fix, reviewed by the coordinator, installed 10-29; source out/sweep2/haikutasks/human1028/s018/fix.py; tested via IDLABELS_OVERRIDE after all fixes incl. z_1099_cleanup, hence z_1100_)
# human1028 / s018 (2026-10-28): RocketGirlDefault super, type 6000054 (RocketGirlUltiProjectile, glb sc3d/brock_renamed_rocket_blue.glb).
# User finding on page 02 (super 45 deg, ticks 347 / 363 / 377 / 393): "some rocket parts arent labeled".
# Cause: the rocket's white missile body is a 3D mesh (program 116, no atlas uv), drawn in two shared draws (44, 45) for all
# the super's rockets. idlabels' own_draw_pixels gives a draw to one projectile only (one owner per draw), so the other
# rockets kept only the blue flame of their clip. Same mechanism as the Brock 003 Hotrod / Brock 010 mesh rockets (a3_agent3).
# Fix: the same POST_BODY hook (_a3_brock_mesh): each rocket takes the mesh parts connected to its own predicted body; the
# smoke puffs of its trail (clip sprites shared with the trail, codes 4 / 5) are dropped there.
POST_BODY[6000054] = _a3_brock_mesh
