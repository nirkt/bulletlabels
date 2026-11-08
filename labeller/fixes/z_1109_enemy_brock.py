# enemy-label fix 2026-11-04 (task away_1104/enemy, group brock2: skins 152 RocketGirlBeach, 154 RocketGirlCny, 155 RocketGirlHotrod)
# Cause (super rockets 6000060 = 152 Brock004UltiProjectile, 6000056 = 154 Brock002UltiProjectile): z_1101_s152brock adds the rocket's
# 3D missile body (the mesh draw: blend 0, dmask 1, one instanced draw for all rockets) by the draw's TEXTURE HASH (G29_MESH). The red
# (enemy) rockets use the same mesh with another texture, so another hash (152: aa37fdd9aa533817, blue 8371787794963ea9;
# 154: bc8f96e3b0a784ca, blue 25da29a95031ec59): no mesh pixels were found and the enemy rocket kept only its flare (recall 0.44 / 0.64).
# Fix: the red hashes are added to G29_MESH (read at call time by _g29_rocket), so the same rule runs on the red mesh. Blue hashes are
# never touched and a red hash is never drawn in an own shot, so own labels are pixel-identical.
# Skin 155 (zones only, no rocket change) and the zone rows of 152 / 154: scene differences (false positives), nothing changed there.
_EN4B_RED = {6000060: ('aa37fdd9aa533817',),      # 152 RocketGirlBeach Brock004UltiProjectile
             6000056: ('bc8f96e3b0a784ca',)}      # 154 RocketGirlCny Brock002UltiProjectile
for _t, _hs in _EN4B_RED.items():
    G29_MESH[_t] = tuple(G29_MESH[_t]) + tuple(h for h in _hs if h not in G29_MESH[_t])
