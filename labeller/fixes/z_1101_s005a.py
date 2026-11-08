# hosted review 2026-10-29 fix (Haiku fix, reviewed by the coordinator, installed 10-29 LOCAL ONLY; source out/sweep2/haikutasks/site1029/s005/fix.py; tested via IDLABELS_OVERRIDE after all fixes incl. z_1100_*, hence z_1101_)
# Skin 005 DuplicatorDarkcircus, the Duplicator's clone (spawnable DuplicatorPet, character type 16000199; the super's spawn,
# owned by player 0). Hosted review 2026-10-29, row "super 45", the user's finding "too small / cut off" at tick 371: the
# clone's translucent upper body and head (drawn faded while it spawns) are outside its label, which had only its opaque
# lower body (the 3D model draw, coverage >= CUT). Coordinator clarification: the whole visible clone body belongs in the
# label; not the green glow / spawn veil under it, not the cyan ring under it, not the floating green skulls.
#
# Cause (sw2_005, ticks 359-411): while the clone spawns, a translucent green veil (a blended glow sprite, draws with texture
# th 7694b067) lies over it, so the ID pass shows the veil, not the clone, over its faded head and body. Only the opaque 3D
# model draw is labelled then (tick 365: 41 px, 369: 361 px, 371: 960 px, 375: 1422 px). The clone is pink; the veil is
# green, the background dark blue-grey, the skulls teal, the ring light blue.
# Fix (labels() wrapper, runs after every installed fix): the clone's label is its opaque part plus
#   (a) its pink pixels (min(R, B) - G >= CLONE_PINK) and its pale lilac pixels (the fin, faded to grey) inside the hull of
#       its pink / opaque parts, inside a box around its ground point (CLONE_BOX), not taken by another label;
#   (b) once the opaque part is a large pink body (>= CLONE_HULL_MIN px, half pink): the faded cup under the veil (hull of
#       the body, not dark, not green, not the bright spawn flash), gaps and notches up to 2 x CLONE_SMOOTH px closed,
#       the enclosed interior filled (not green, not the flash);
#   then one piece (the part touching the opaque body). Nothing outside the clone's box and nothing of another label is taken.
# Not solved: at ticks 361-367 the clone is under the veil at full strength; its body is green-grey there and cannot be told
# from the veil by colour, so the label covers only its pink parts (the cup rim and horn at 365, the lower body at 367).
# Other Duplicator skins use the same pet (character 16000199): the same rule applies to them (same brawler, same spawn).
import numpy as np
from scipy.ndimage import binary_fill_holes as _cl_fill, binary_closing as _cl_close, label as _cl_cc

CLONE_CHAR = 16000199   # DuplicatorPet (characters.csv Minion_Duplicate)
CLONE_PINK = 30         # min(R, B) - G from which a frame pixel is the clone's pink (veil green, background blue-grey: below)
CLONE_BOX = (26, 64, 14)   # px: half width, up, down from the clone's ground point (its drawn size on this skin: 41 x 47 px)
CLONE_SMOOTH = 6       # px: gaps / notches of the clone's outline up to 2 x this closed (smooth outline)
CLONE_PALE = 110       # max channel from which a pale (lilac-grey) frame pixel can be the clone's pale fin
CLONE_PALE_MAX = 200   # ... below this (the spawn flash is white: 200 and more)
CLONE_DARK = 100       # max channel below which a frame pixel is the dark background
CLONE_WHITE = 150      # min channel from which a frame pixel is the bright veil (white-cyan), not the clone
CLONE_HOLE_MAX = 200   # min channel from which an enclosed interior pixel is the spawn flash, not the clone's glass
CLONE_GREEN = 60       # G - (R + B) / 2 above which a pixel is green (teal skulls, the veil): never added by (b)
CLONE_HULL_MIN = 300   # opaque clone pixels from which (b) applies (the faded cup)
CLONE_NOTGREEN = 6     # G - (R + B) / 2 at most this for the pale fin (about -10; the veil: +20 and more)
_clone_prev = labels


def labels(objs, cam, R, state=None, ended=(), img=None):
    P = _clone_prev(objs, cam, R, state=state, ended=ended, img=img)
    if state is None or img is None: return P
    import bulletlabels as BL
    ids, lids, share = P['ids'], P['lids'], P['share']
    cv = R['cv'][::2, ::2]
    im = img.astype(np.int16)
    pink = (np.minimum(im[..., 0], im[..., 2]) - im[..., 1]) >= CLONE_PINK
    H, W = ids.shape; yy, xx = np.mgrid[0:H, 0:W]
    ids = ids.copy(); share = share.copy(); changed = False
    for c in state['chars']:
        cid, did, own = int(c[0]), int(c[1]), int(c[2])
        if did != CLONE_CHAR or own < 0: continue
        ax, ay = _proj(cam, int(c[3]), int(c[4]), 0)
        box = (np.abs(xx - ax) <= CLONE_BOX[0]) & (yy >= ay - CLONE_BOX[1]) & (yy <= ay + CLONE_BOX[2])
        lid = (cid, -1, 'spawn')
        if lid not in lids: continue          # no opaque body drawn yet (nothing to extend)
        k = list(lids).index(lid)
        mine = ids == k
        if not mine.any(): continue
        free = (ids < 0) | mine
        seed = mine | (pink & box & free)
        # (a): the pink and pale parts inside the hull of the clone's pink / opaque parts (never the veil or the background)
        hl = hull_mask(seed) & box & free
        pale = (im.max(-1) >= CLONE_PALE) & (im.max(-1) < CLONE_PALE_MAX) & ((im[..., 1] - (im[..., 0] + im[..., 2]) / 2.0) <= CLONE_NOTGREEN)
        m = seed | (hl & (pink | pale))
        # (b) below never takes the bright veil (min channel >= CLONE_WHITE) or green pixels (the veil, the skulls)
        greenness = im[..., 1] - (im[..., 0] + im[..., 2]) / 2.0          # the veil and the teal skulls are green, the clone is not
        dark_ok = (im.min(-1) < CLONE_WHITE) & (greenness <= CLONE_GREEN) & box & free
        # (b): only once the opaque part is a large pink body (not the bright spawn flash, not a small horn tip)
        if mine.sum() >= CLONE_HULL_MIN and (pink & mine).sum() >= 0.5 * mine.sum():
            m = m | (hull_mask(m) & ~m & dark_ok & (im.max(-1) >= CLONE_DARK))
            mc = _cl_close(np.pad(m, CLONE_SMOOTH), iterations=CLONE_SMOOTH)[CLONE_SMOOTH:-CLONE_SMOOTH, CLONE_SMOOTH:-CLONE_SMOOTH]
            m = m | (mc & dark_ok)
            m = m | (_cl_fill(m) & free & box & (im.min(-1) < CLONE_HOLE_MAX) & (greenness <= CLONE_GREEN))   # enclosed interior (highlights too), not the flash
        m = m & free
        L, n = _cl_cc(m, structure=np.ones((3, 3)))
        if n > 1:   # one piece: the parts touching the opaque body
            keep = np.unique(L[mine & m]); keep = keep[keep > 0]
            m = np.isin(L, keep)
        new = m & ~mine
        if not new.any(): continue
        ids[new] = k; share[new] = np.maximum(cv[new] / 255.0, BL.CUT); changed = True
    if not changed: return P
    P2 = dict(P); P2.update(ids=ids, share=share)
    return P2
