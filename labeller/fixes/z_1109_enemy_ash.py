# enemy-label fix 2026-11-04 (task away_1104/enemy, group ash: skins 129 / 131 / 138 / 150; types 6001043, 6001046, 6001049, 6001055 + 6001067)
# Cause: an Ash wave attack is drawn by its TRAIL effect (ash_0xx_atk_trail); the enemy version of it is the effect
# 'ash_0xx_atk_trail_red' whose emitters ('..._red') show red twin clips ('ash_005_atk_lv1_candy_01_red', 'ash_006_atk_1_sword_red',
# 'ash_008_akt_particle_03_rotate_01_r_red', the flame 'spike_006_lobby_fire_01' of Ash 003 ...). The pipeline only knows the BLUE trail
# clips (wave._emitters has no '_red' emitters), so (1) sprites(t, own=False) has no atlas polygon: the red sprites are never found
# in the ID pass and the label of an enemy wave is missing or only the prediction; (2) the installed fixes (z_1024_ash138,
# z_1027_s129, z_1101_s129knight) key on the BLUE shape ids of the clips (27811 / 27842.. / 7743 ..) which the red clips do not have.
# Fix: (1) sprites() of these types for own=False is built from the red twin clips (same classification by clip name);
# (2) _a4_shape_pixels() asks for the red twin clips too (listed FIRST, so a blue polygon wins on an atlas overlap) and reports every
# red-only shape id as its blue counterpart (twin clips have the same shape list, position by position), so every rule written
# for the blue art applies unchanged to the red art; (3) the cyan cube of the super (ash_006_ulti_projectile_red: 27744 -> 27745) too.
# Own (blue) shots: no red polygon / red shape is ever drawn, shape ids of blue clips are never touched.
import numpy as _en4a_np
import json as _en4a_json
_EN4A_TYPES = (6001043, 6001046, 6001049, 6001055, 6001067)
_EN4A_PAIRS = None      # [(file, blue clip, red clip)]
_EN4A_LUT = None        # red shape id + 1 -> blue shape id + 1
_EN4A_RED_MODE = [False]
# ash_006_atk_1_sword_red -> ash_006_atk_1_sword shapes (same per-shape draw counts: 142 80 224 231~234 219 231~234 126; 27836 = a second 219-draw flame)
_EN4A_SWORD = {27829: 27838, 27830: 27839, 27831: 27840, 27832: 27841, 27833: 27842, 27834: 27843, 27835: 27844, 27836: 27842}


def _en4a_pairs():
    """[(swf, blue clip name, red twin clip name)] of the trails (emitter by emitter) and the projectile clips of the types"""
    global _EN4A_PAIRS
    if _EN4A_PAIRS is not None: return _EN4A_PAIRS
    EC = emitter_clips(); FX = wave._effects(); out = []
    for t in _EN4A_TYPES:
        T = projtab.table()[t]
        if T['blue'] and T['red'] and T['blue'] != T['red']: out.append((T['file'], T['blue'], T['red']))
        for eb in ('trail', 'special_trail'):
            eff = T[eb]
            if not eff: continue
            effr = eff + '_red' if (eff + '_red') in FX else eff
            for emb, emr in zip(FX.get(eff, []), FX.get(effr, [])):
                cb = EC.get(emb) or []; cr = EC.get(emr) or []
                for a in cb:   # the twin is the clip named '<blue>_red' where there is one, else the clip in the same place of the emitter
                    if a['name'] + '_red' in PD.sc(a['swf']).exports and any(b['name'] == a['name'] + '_red' for b in cr):
                        out.append((a['swf'], a['name'], a['name'] + '_red'))
                if len(cb) == len(cr):
                    for a, b in zip(cb, cr):
                        if a['name'] != b['name'] and not any(a['name'] + '_red' == x['name'] for x in cr): out.append((a['swf'], a['name'], b['name']))
    _EN4A_PAIRS = list(dict.fromkeys(out))
    return _EN4A_PAIRS


def _en4a_lut():
    global _EN4A_LUT
    if _EN4A_LUT is not None: return _EN4A_LUT
    lut = _en4a_np.arange(65536, dtype=_en4a_np.int32); blue_ids = set(); conflict = set()
    for f, b, r in _en4a_pairs():
        S = PD.sc(f); cb = S.exports.get(b); cr = S.exports.get(r)
        if cb is None or cr is None: continue
        sb = sorted(_clip_shapes(S, cb)); sr = sorted(_clip_shapes(S, cr)); blue_ids |= {s for s, k in sb}
        if b == 'ash_006_atk_1_sword':   # the red sword has one more flame shape (27836, a copy of 27833's 219 draws): table below
            continue
        if len(sb) != len(sr): continue
        for (s1, k1), (s2, k2) in zip(sb, sr):
            if s1 != s2:
                if lut[s2 + 1] not in (s2 + 1, s1 + 1): conflict.add(s2)
                lut[s2 + 1] = s1 + 1
    for s2, s1 in _EN4A_SWORD.items(): lut[s2 + 1] = s1 + 1
    for s in blue_ids | conflict: lut[s + 1] = s + 1     # a shape id of a blue clip is never renamed
    _EN4A_LUT = lut
    return lut


def _en4a_red_trail(t):
    T = projtab.table()[t]; FX = wave._effects(); EC = emitter_clips(); out = []
    for eb in ('trail', 'special_trail', 'special_trail2'):
        eff = T.get(eb) or ''
        if not eff: continue
        effr = eff + '_red' if (eff + '_red') in FX else eff
        for em in FX.get(effr, []):
            for mc in EC.get(em) or []: out.append((mc['swf'], mc['name']))
    for swf, n in _en4a_prev_tc(t):   # a red twin the red emitter list does not name (ash_008 shoes: the blue list has 4 clips, the red one 3)
        if n + '_red' in PD.sc(swf).exports and (swf, n + '_red') not in out: out.append((swf, n + '_red'))
    return out


_en4a_prev_tc = _trail_clips


def _trail_clips(t):
    if _EN4A_RED_MODE[0] and t in _EN4A_TYPES: return _en4a_red_trail(t)
    return _en4a_prev_tc(t)


_en4a_prev_sprites = sprites


def sprites(t, own):
    if not own and not isinstance(t, tuple) and t in _EN4A_TYPES:
        _EN4A_RED_MODE[0] = True
        try: return _en4a_prev_sprites(t, own)
        finally: _EN4A_RED_MODE[0] = False
    return _en4a_prev_sprites(t, own)


_en4a_prev_sp = _a4_shape_pixels


def _a4_shape_pixels(R, swf, names):
    names = tuple(names); bl = set(names); extra = []
    for f, b, r in _en4a_pairs():
        if f == swf and b in bl and r not in bl and r not in extra: extra.append(r)
    if not extra: return _en4a_prev_sp(R, swf, names)
    out = _en4a_prev_sp(R, swf, tuple(extra) + names)     # red twins first: on an atlas overlap the blue polygon (drawn later) wins
    return _en4a_lut()[out] if out.size else out
