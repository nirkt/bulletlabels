# hosted review 2026-11-01 fix (Haiku fix, reviewed by the coordinator, installed 11-01 LOCAL ONLY; source out/sweep2/haikutasks/site1101/s182/fix2.py; runs after z_1101_*)
# Skin 182 RopeDudeGodzilla (Buzz's super, the boat), hosted review 2026-11-01, task s182. THE USER (binding): 'this attack
# needs a hull'; flagged super 45 at t345 and t347: 'holes / ragged'.
# The super is Buzz007UltiProjectile (type 6000993, logic RopeDudeUltiProjectile): one boat per shot (object 2000010), labelled
# on ticks 345 and 347 only. Its label was the boat's drawn body with a slot cut along the dark tow rope that lies across the
# boat (the rope is its own draw, 139, not part of the boat, not labelled), and the pale fin on the left of the rope was left
# out, so the outline ran ragged along the rope.
# Fix (one convex hull per projectile per tick, rules 5/6 and the coordinator's clarification):
#  1. the boat's body: the labelled body, minus its detached small parts (the boat's bubbles; the same < 40 px rule as
#     z_1027_s182.py, applied here because that rule runs after the hull);
#  2. plus the boat's own pale fins: the pixels of its clip shape 63 (effects_brawler.sc) that touch that body (4-connected),
#     drawn opaque enough (coverage >= CUT), in the projectile's region. Detached bubbles, which also show shape 63, stay out;
#  3. ONE convex hull of 1 + 2: the rope lying over the boat is filled in (hidden parts rule: the label keeps the full shape),
#     the rope outside the boat is not hull material and stays out;
#  4. props in front: solid map geometry (walls, crates: z_1023 solid tiles) that is nearer the camera than the boat is cut out
#     of the hull again; the largest piece is kept, holes filled (no loops, no spurs).
# Only type 6000993 is touched. Other skins of this brawler with the same super type get the same rule (same clip, same type).

import numpy as _np

S182_SUPER = 6000993            # Buzz007UltiProjectile
S182_DETACHED_PX = 40           # half-res px: body parts smaller than this, detached from the boat, are not the boat
S182_SHAPE = 63 + 1             # the boat's own atlas shape of the clip (shape id + 1, the _a4_shape_pixels convention)


def _fix_s182_super(body, ctx):
    from scipy.ndimage import label as _cc, binary_fill_holes as _fh
    if not body.any(): return body
    R = ctx['R']; o = ctx['o']; cam = ctx['cam']; cv = ctx['cv']; region = ctx['region']
    # 1. the boat's body without its detached small parts (bubbles beside it)
    L, n = _cc(body)
    sz = _np.bincount(L.ravel())[1:]
    keep = _np.zeros(n + 1, bool); keep[1:] = sz >= S182_DETACHED_PX; keep[1 + int(_np.argmax(sz))] = True
    main = keep[L] & body
    # 2. the boat's own fins: its clip shape 63 touching the body
    T = projtab.table()[S182_SUPER]
    names = tuple(nm for nm in (T.get('blue'), T.get('red')) if nm)
    SP = _a4_shape_pixels(R, T['file'], names)
    cand = ((SP == S182_SHAPE) & (cv >= CUT) & region) | main
    L2, n2 = _cc(cand)
    touch = _np.unique(L2[main]); touch = touch[touch > 0]
    U = main | (_np.isin(L2, touch) & cand)
    # 3. one convex hull
    hm = hull_mask(U)
    # 4. props in front of the boat (solid map geometry nearer the camera than the boat) are cut out of the hull
    try: world = _zs_solid(cam, R)
    except Exception: world = None
    wd = R.get('wd'); wref = R['meta'].get('wref', 0.0)
    if world is not None and world.any() and wd is not None and o.get('anchor') is not None:
        dz = _eye_depth(cam, o['anchor'], wref)[0]
        cut = hm & world & (wd < dz - DEPTH_TOL) & ~U
        if cut.any():
            hm = hm & ~cut
            L3, n3 = _cc(hm); s3 = _np.bincount(L3.ravel())[1:]
            hm = L3 == 1 + int(_np.argmax(s3))
    return _fh(hm | U)


POST_BODY[S182_SUPER] = _fix_s182_super


# ---- Attack row of the same skin: Buzz007Projectile (type 6000994, logic RopeDudeProjectile), attack 0 deg (objects 2000000-
# 2000004) and attack 90 deg (2000005-2000009). Coordinator decision (2026-11-01, after the user's note 'this attack needs a
# hull'): the attack label is two separate small pieces on one bright streak (the core's shape 133 pieces), which the user rejects.
# Fix, per object per tick:
#  (a) a FINISHED projectile (the game's own flag 'finished' set: the shot has hit and shatters on) is not labelled. On its first
#      finished tick the ID pass shows the impact burst (cyan trail sprites, shapes 884 / 45690 at attack 0 t269, object 2000001;
#      the big elongated label at attack 90 t309, object 2000006): hit / impact animation, rule 2 (dropped, as in s064);
#  (b) a flying projectile: ONE convex hull over its streak: its own core pieces plus the white / light streak pixels
#      (min(R, G, B) >= S182_WHITE) that touch them, within the projectile's region. The soft blue glow (min channel well below
#      the streak's) and the cyan spikes on the flame side stay out of the streak pixels.
# Only type 6000994 is touched.
S182_ATK = 6000994
S182_WHITE = 170     # min(R, G, B) of the bright streak; the soft blue glow lies below ~120


def _fix_s182_atk(body, ctx):
    from scipy.ndimage import label as _cc, binary_fill_holes as _fh
    if ctx['o'].get('finished'): return _np.zeros_like(body)      # the impact burst of a finished shot: no label
    if not body.any(): return body
    img = ctx['img']; region = ctx['region']
    white = (img.min(-1) >= S182_WHITE) & region
    cand = white | body
    L, n = _cc(cand)
    touch = _np.unique(L[body]); touch = touch[touch > 0]
    U = body | (_np.isin(L, touch) & white)                     # the core pieces + the streak pixels touching them
    return _fh(hull_mask(U))                                    # ONE convex hull over the streak


POST_BODY[S182_ATK] = _fix_s182_atk
