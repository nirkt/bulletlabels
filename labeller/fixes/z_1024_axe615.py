# Axe Juggler Gym 615 attack: the mesh silhouette, filled, glow / spark out (Haiku task 20, reviewed)
# task 20: skin 615 AxeJugglerGym (29001362, brawler AxeJuggler). Its attack is AxeJuggler004Projectile (6001426): the
# Gym skin's 'Projectiles' column in csv_logic/skin_confs.csv is the only one that names it (the default skin uses
# AxeJugglerProjectile), so this type is this skin's alone.
#
# The projectile is a 3D mesh (sc3d melodie_004 geo: face, ears, handle and dark blade). Its opaque mesh draws are
# the ID pass's draws with fragment shader 697340477, blend 0, depth write (the ground draws use other shaders:
# 623557017 / 3366365837; the glow and sparks are additive sprites: blend 1). The current label is the mesh
# plus pieces of the additive glow / spark sprites that are 'trail over' sprites of this type (code 4, taken where
# they fall in the predicted body), and it is holed where the ground shows through the mesh's window and between
# its ear and head (enclosed holes, 4-33 px).
#
# The user's rules: a projectile's label is its outline, fully labelled inside (one piece, no holes); sparks, halos
# and glows are not labelled. So: the body = the mesh draws' pixels in the projectile's region, the connected parts
# touching its predicted body, holes filled. The spark at the top right (tick 305, 309) and the cyan glint are
# additive sprites and fall out. Nothing else changes: the trailing ground decal and the lime end effect at tick 313
# (after the logic has finished and the body has hidden) are not this mesh and stay unlabelled.
AXE004_MESH_FS = 697340477


def _axe004_mesh(body, ctx):
    from scipy.ndimage import label as cc, binary_fill_holes
    R = ctx['R']; ids_h = ctx['ids_h']
    mesh = [d for d, dr in R['draws'].items() if dr.get('fs') == AXE004_MESH_FS and dr.get('blend') == 0]
    if not mesh: return body
    m = ctx['region'] & np.isin(ids_h, mesh) & (ctx['cv'] >= CUT)
    if not m.any(): return body
    L, n = cc(m)
    keep = np.unique(L[m & _dilate(ctx['pred_body'], 2)]); keep = keep[keep > 0]
    if not keep.size: return body
    return binary_fill_holes(np.isin(L, keep))


POST_BODY[6001426] = _axe004_mesh
