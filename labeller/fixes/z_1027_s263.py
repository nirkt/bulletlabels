# sweep 10-27 fix for skin 263 (Haiku fix, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s263/)
# sweep 10-27 fix for skin 263 IceDudeCrooner (Haiku fix, task out/sweep2/haikutasks/sweep1027/s263/), finding pair 4850:
# super 45 deg, tick 411, spur_or_speck: the yellow zone outline has a thin spike at the top centre that follows the stem
# of a cyan music note.
#
# Cause: the super is the lingering zone IceDudeValentineUltiArea (area effect did 17000394, Effect lou_003_ulti_area,
# Radius 1100, area row 397 of area_effects_skin; the zone of projectile type 6000850, Lou003UltiProjectile). zone_labels
# takes every pixel of the zone's sprites, including the small particles the effect throws around the cloud: music notes
# (particle_note_circle_blue / _red), snow (particle_blizzard_snow_*), hearts (particle_blizzard_heart_*) and twinkles
# (gen_twinkle_particles_L). A note at the rim pulls its stem out into the zone label as a spike.
#
# Fix, this zone only (did 17000394; every other zone and skin is untouched):
#  - the particle pixels of the zone's label are kept only where they lie at least S263_RIM px inside the cloud's own
#    outline (its body clips, filled over its inner holes). Particles at the rim or outside the cloud are not the zone
#    (rules 3 and 7), so the stems and note heads that stick out go; a snowflake well inside the cloud stays as it was;
#  - holes inside the cloud that touch the dropped particle pixels, up to S263_HOLE px and not touching another label, are
#    filled, and detached specks of the zone label under S263_SPECK px are dropped;
#  - the result is always a subset of the label as it was: no pixel is added (checked: 0 added on all 35 super ticks),
#    the body pixels are never removed, and the blue orb in front of the cloud (tick 363) stays unlabelled as before.
# The attack rows of skin 263 (attack 0 / attack 90) are identical before and after (checked on ticks 263, 275, 303, 321).

S263_ZONE_DID = 17000394
S263_ZONE_NAME = 'IceDudeValentineUltiArea'
S263_PARTICLE = ('note', 'heart', 'snow', 'twinkle', 'gen_')   # lower-case clip name parts of the drifting particles
S263_RIM = 6       # px: a particle pixel counts as the zone only at least this far inside the cloud's outline
S263_SPECK = 10    # px: detached pieces of the zone label smaller than this are dropped
S263_HOLE = 60     # px: holes inside the cloud touching the dropped particles are filled up to this size

_s263_prev_sprites = sprites
_s263_prev_zone_labels = zone_labels


def _s263_clips(did, particles):
    import owners
    Z = owners.zone(did)
    if not Z or Z.get('Name') != S263_ZONE_NAME: return []
    clips = [(Z.get('FileName'), Z.get(c)) for c in ZONE_EXPORTS if Z.get(c) and Z.get('FileName')]
    for eff in (Z.get('Effect'), Z.get('LoopingEffect')): clips += effect_clips(eff)
    return [c for c in clips if c[1] and any(k in c[1].lower() for k in S263_PARTICLE) == particles]


def sprites(t, own):
    """('s263', particles, did): the zone's body clips (particles False) or its particle clips (True)"""
    if isinstance(t, tuple) and len(t) == 3 and t[0] == 's263' and t[2] == S263_ZONE_DID:
        return _clips_sprites(_s263_clips(t[2], t[1]))
    return _s263_prev_sprites(t, own)


def zone_labels(state, cam, R, taken, own_team=None):
    import owners
    from scipy.ndimage import binary_fill_holes, binary_closing, distance_transform_edt, label as _s263_cc
    out = _s263_prev_zone_labels(state, cam, R, taken, own_team)
    for it in state.get('items', []):
        iid, did = int(it[0]), int(it[1])
        if did != S263_ZONE_DID or iid not in out: continue
        Z = owners.zone(did)
        if not Z or Z.get('Name') != S263_ZONE_NAME: continue
        m, odid, own = out[iid]
        if not m.any(): continue
        body = sprite_pixels(('s263', False, did), own, R) > 0
        part = sprite_pixels(('s263', True, did), own, R) > 0
        bm = body & ~taken
        if not bm.any(): continue
        cl = binary_closing(np.pad(bm, 2), iterations=2)[2:-2, 2:-2] | bm
        filled = binary_fill_holes(cl)                               # the cloud's own outline (its inner holes filled)
        depth = distance_transform_edt(filled)                       # px inside the outline (0 outside)
        drop = m & part & ~body & (depth < S263_RIM)                 # particles at the rim or outside the cloud
        new = m & ~drop
        holes = binary_fill_holes(new) & ~new                        # holes the dropped particles left inside the cloud
        if holes.any():
            HL, hn = _s263_cc(holes)
            for j in range(1, hn + 1):
                hj = HL == j
                if hj.sum() <= S263_HOLE and (hj & drop).any() and not (hj & taken).any(): new |= hj
        L, n = _s263_cc(new)
        if n > 1:
            sz = np.bincount(L.ravel()); small = np.nonzero(sz < S263_SPECK)[0]; small = small[small > 0]
            if small.size: new &= ~np.isin(L, small)
        if new.sum() == m.sum(): continue
        if new.any(): out[iid] = (new, odid, own)
        else: out.pop(iid, None)
    return out
