# sweep 10-27 fix for skin 068 (Haiku fix, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s068/)
# Skin 068 JetpackGirlDefault, sweep 2026-10-27, finding pair 1237 (super 45 deg, tick 395, spur_or_speck).
# Janet's super (JetpackGirlUltiProjectile, types below) has NO label at all: the user's decision of 2026-10-23 (Q18),
# carried out for the body by z_1024_janet_super_off.py (POST_BODY zeroes it). That fix does not touch the glow map,
# where the super's predicted translucent glow layer still gave a lone green outline: a 13-14 px speck at glow share
# 0.51 in the white flame column, ticks 389-397 (the rest of the glow stays at about 0.33, below the outline threshold).
# This fix applies the same decision to the glow: every glow part of these types is left out of the glow map. The body
# labels (ids, share) of these types are already empty and are not touched. Other skins with these types (JetpackGirl
# Valkyrie, Popstar, DarkFairy, Moon, Christmas, SupercellMake, Surfer, MHA: their super rows) lose their green glow too,
# if any: the same super, the same decision.
JANET_SUPER_GLOW_OFF = (6001156, 6001157, 6001159, 6001161, 6001163, 6001165, 6001167, 6001169, 6001171, 6001173)
_labels_s068_prev = labels


def labels(objs, cam, R, state=None, ended=(), img=None):
    P = _labels_s068_prev(objs, cam, R, state=state, ended=ended, img=img)
    gi, gs, gl = P.get('glow_ids'), P.get('glow_share'), P.get('glow_lids')
    if gi is None or gs is None or not gl or not (gi >= 0).any(): return P
    tmap = {o['id']: o['t'] for o in objs}
    kk = [k for k, lid in enumerate(gl) if lid[2] == 'glow' and tmap.get(lid[0]) in JANET_SUPER_GLOW_OFF]
    if not kk: return P
    m = np.isin(gi, kk)
    if not m.any(): return P
    gi = gi.copy(); gs = gs.copy(); gi[m] = -1; gs[m] = 0
    P = dict(P); P['glow_ids'] = gi; P['glow_share'] = gs
    return P
