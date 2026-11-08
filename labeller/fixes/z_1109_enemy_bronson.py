# 2026-11-04 enemy-label fix, group bronson (skins 136 / 141 / 147: WeaponThrower Darkcircus, SupercellMake, SteamPunk).
# Executed in idlabels' namespace; everything below only acts when ctx['own'] is False (the enemy / red art): the own (blue)
# labels go through the installed hooks untouched.
import numpy as _en4b_np

# ---- Bronson004Projectile (6001228): the installed hook _a4_fix_6001228 takes the bright pixels of the BLUE clip
# bronson_004_atk_trail_afterimage and keeps blue >= red. The enemy art is drawn by bronson_004_atk_trail_afterimage_red
# (pink; blue-clip lookup finds nothing, the label was only the ID-pass rim). Same rule on the red clip, orange particles
# (G above B, the mirror of the blue test) kept out.
def _en4b_bright_clip_red(ctx, swf, clips, vmin=200):
    from scipy.ndimage import label as cc, binary_fill_holes
    img = ctx['img']
    if img is None: return _en4b_np.zeros_like(ctx['region'])
    SP = _a4_shape_pixels(ctx['R'], swf, tuple(clips)); im = img.astype(_en4b_np.int16)
    m = (SP > 0) & ctx['region'] & (im.max(-1) >= vmin) & (im[..., 1] <= im[..., 2] + 10)
    m = _close(m, 2); L, n = cc(m)
    if not n: return m
    sz = _en4b_np.bincount(L.ravel())[1:]; m = _en4b_np.isin(L, 1 + _en4b_np.nonzero(sz >= 0.15 * sz.max())[0])
    m = binary_fill_holes(m)
    m &= im[..., 1] <= im[..., 2] + 30
    return m

_en4b_prev_6001228 = POST_BODY.get(6001228)
def _en4b_fix_6001228(body, ctx):
    if ctx.get('own'): return _en4b_prev_6001228(body, ctx) if _en4b_prev_6001228 else body
    return _en4b_bright_clip_red(ctx, 'sc/effects_brawler.sc', ('bronson_004_atk_trail_afterimage_red',))
POST_BODY[6001228] = _en4b_fix_6001228


# ---- thrown super meshes (6001230 Darkcircus, 6001234 SupercellMake, 6001238 SteamPunk): the enemy run draws a big pink
# puff / muzzle burst (translucent sprites of the red glow clip and other effects) over the mesh for the first ticks: the ID
# pass then keeps the puff itself as 'body' (C == 1 sprites away from the mesh) and loses the covered part of the mesh. The
# blue run shows the mesh (its glow is small). Enemy rule: body = ID-pass body within 2 px of the predicted mesh silhouette,
# plus the predicted silhouette pixels where a translucent draw (coverage < 255) of the glow / an effect (class 0 or 1) is
# the front-most thing drawn (the mesh under a translucent overlay); opaque things in front (walls, brawlers) still cut it.
def _en4b_cover(body, ctx):
    if ctx.get('own'): return body
    pb = ctx['pred_body']
    if pb is None or not pb.any(): return body
    cov = pb & ~body & (ctx['ids_h'] > 0) & (ctx['cv'] < 255) & ((ctx['C'] == 0) | (ctx['C'] == 1))
    return (body & _dilate(pb, 2)) | cov

def _en4b_mk(t):
    prev = POST_BODY.get(t)
    def f(body, ctx):
        if prev is not None: body = prev(body, ctx)
        return _en4b_cover(body, ctx)
    POST_BODY[t] = f
for _t in (6001230, 6001234, 6001238): _en4b_mk(_t)
