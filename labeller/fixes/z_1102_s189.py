# hosted review 2026-11-01 fix (Haiku fix, reviewed by the coordinator, installed 11-01 LOCAL ONLY; source out/sweep2/haikutasks/site1101/s189/fix2.py; runs after z_1101_*)
# Skin 189 ClusterBombDudeGodzilla, fix 2 (coordinator's review of fix.py, 2026-11-01): THE USER: "attack should get a hull".
# Tick's basic attack, Tick006Projectile (type 6000536, clip tick_006_atk1_projectile, sc/effects_brawler.sc): one object per
# shot (2000003 at attack 0 deg, ticks 265-279; 2000004 at attack 90 deg, ticks 305-319): three glowing orbs.
# Result: ONE convex hull per shot per tick over the orbs' own body (one label per shot).
#   1. Body components that lie on the projectile's predicted body (ctx pred_body: the orbs' footprints from the game's sprite
#      placement) are kept; a component is kept when at least half its pixels lie in the prediction. This drops the blue bubble
#      puff above the head at attack 90 (ticks 305-309: same atlas shape 63 as the orbs, 2-4 px away from them) and the streak specks.
#   2. Fix 2: the kept orbs are grown by one pixel into the orbs' own sprite pixels (clip body code 1, shape 0 or 63; plus the
#      bright pixels of shape 0, G >= 190 and B >= 240, the light ring, also where the glow code marks them; the soft halo and
#      the streak shapes are not grown into). The body threshold (coverage >= 128) and the thin-part opening
#      (THIN_OFF[6000536] = 2 of a0_agent0) cut the anti-aliased rim of each orb by a pixel here and there; the hull's edge then
#      ran across the rim (coordinator: the right orb of tick 319 was cut). The union is taken over all of these orb pixels, then
#      the hull, repeated until no orb pixel lies outside it.
#   3. ONE_LABEL 'hull': the convex hull of that union (one label per shot per tick).
# Not applied: the game's 'finished' flag (1 at ticks 279 and 319, the orbs are still drawn intact there): their hull is kept.
# The thin white speed streaks beside the orbs stay out (except where they touch an orb's own pixels), as before.
# Only type 6000536. The super (Tick006UltiProjectile, ClusterBombPet spawnable) is not touched. Skins using the same
# projectile: ClusterBombDudeGodzillaDark and ClusterBombDudeGodzillaGold (csv skin_confs Projectiles = Tick006Projectile...).
from scipy.ndimage import label as _s189_cc, binary_dilation as _s189_dil

_S189_ATK = 6000536
_S189_MIN_IN = 0.5      # a body component is the projectile's when at least this share of its pixels lie in the prediction
_S189_FILE = 'sc/effects_brawler.sc'


def _fix_s189_atk2(body, ctx):
    if not body.any(): return body
    pb = ctx['pred_body']
    if not pb.any(): return body
    L, n = _s189_cc(body)
    keep = [k for k in range(1, n + 1) if (pb & (L == k)).sum() >= _S189_MIN_IN * (L == k).sum()]
    if not keep: return body
    K = np.isin(L, keep)
    C = ctx['C']; reg = ctx['region']
    # the orb's own sprite shapes (shape 0 = the disc, shape 63 = its ring; value = shape id + 1): the streak's shapes and
    # the puffs are not grown into
    t = ctx['t']; T = projtab.table()[t]
    names = tuple(sorted({n for f, n in _trail_clips(t) if f == _S189_FILE} | {n for n in (T['blue'], T['red']) if n}))
    SP = _a4_shape_pixels(ctx['R'], _S189_FILE, names)
    orb = (C == 1) & reg & ((SP == 1) | (SP == 64))
    # the bright rim of the disc (shape 0; G >= 190, B >= 240: the light ring, not the soft halo) counts as the orb too, also
    # where the glow code marks it
    img = ctx['img']; bright = (img[..., 1] >= 190) & (img[..., 2] >= 240)
    U = K | (_s189_dil(K, structure=np.ones((3, 3), bool)) & (orb | ((SP == 1) & reg & bright)))   # one pixel around the body only
    # the hull's chords cut a pixel or two of the orb's own sprite pixels at its edge: they join the union, the hull is
    # taken again, until no orb pixel of the shot lies outside it (at most 3 rounds)
    for _ in range(3):
        H = hull_mask(U)
        extra = orb & _s189_dil(U, structure=np.ones((3, 3), bool)) & ~H
        if not extra.any(): break
        U = U | extra
    return U


POST_BODY[_S189_ATK] = _fix_s189_atk2
ONE_LABEL.update({_S189_ATK: 'hull'})
