# enemy-label fix 2026-11-04 (task away_1104/enemy, group brock2 follow-up: skin 159 RocketGirlRanger, skin 172 RocketGirlDelivery)
# Same cause as z_1109_enemy_brock (152 / 154): z_1101_s152brock adds the super rocket's 3D missile body by the mesh draw's texture hash
# (G29_MESH), the red (enemy) rockets use the same mesh with another texture hash, so the enemy label kept only the flare.
#   6000062 Brock005UltiProjectile (159): red hash 6c9d940cbd51dbf8   (blue 9122958aa6b4a41c)
#   6000064 Brock006UltiProjectile (172): red hash 33797f8ad1c60c97   (blue c5a635924fc6b3f3)
# The red hashes are appended to G29_MESH (read at call time); blue hashes and own shots are never touched.
_EN4C_RED = {6000062: ('6c9d940cbd51dbf8',),
             6000064: ('33797f8ad1c60c97',)}
for _t, _hs in _EN4C_RED.items():
    G29_MESH[_t] = tuple(G29_MESH[_t]) + tuple(h for h in _hs if h not in G29_MESH[_t])
