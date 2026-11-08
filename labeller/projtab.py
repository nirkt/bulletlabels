"""Per projectile type (data id 6000000 + row of csv_logic/projectiles_skin.csv) the values the client's display code
reads (LogicProjectileData ctor 0xaf65f8, column indices 0xaf9a50, getters 0xafa698..0xafab38, display object ctor
0x539f28 and update 0x53a938). Skin columns come from the type's own skin row except PreExplosionScale,
AnimateByDistanceTravelled and VisualizeEndPoint, which come from ParentProjectileForSkin when that is set (0xaf65f8);
logic columns come from csv_logic/projectiles_logic.csv by LogicProjectile name.
  T = table() -> {type id: dict}"""
import csv, os
import rc

RENDERING = {'': 0, 'Use360Frames': 1, 'UseZFrames': 2, 'DoNotRotateClip': 3, 'UseVerticalMirroring': 4}   # 0xaf65f8 +0xc4
_T = None


def _rows(path):
    r = list(csv.reader(open(path, encoding='utf8')))
    return r[0], r[2:]


def _int(v, default=0):
    try: return int(v)
    except Exception: return default


def _bool(v):
    return str(v).strip().lower() == 'true'


def table():
    global _T
    if _T is not None: return _T
    hs, S = _rows(os.path.join(rc.ASSETS, 'csv_logic', 'projectiles_skin.csv'))
    hl, L = _rows(os.path.join(rc.ASSETS, 'csv_logic', 'projectiles_logic.csv'))
    logic = {r[0]: dict(zip(hl, r)) for r in L}
    skin = [dict(zip(hs, r)) for r in S]
    byname = {s['Name']: s for s in skin}
    out = {}
    for i, s in enumerate(skin):
        par = byname.get(s['ParentProjectileForSkin']) if s['ParentProjectileForSkin'] else None
        lg = logic.get(s['LogicProjectile'], {})
        pv = par if par is not None else s
        out[6000000 + i] = dict(
            id=6000000 + i, name=s['Name'], logic=s['LogicProjectile'], parent=s['ParentProjectileForSkin'] or None,
            file=s['FileName'], blue_scw=s['BlueSCW'], red_scw=s['RedSCW'], blue=s['BlueExportName'], red=s['RedExportName'],
            shadow=s['ShadowExportName'], blue_glow=s['BlueGroundGlowExportName'], red_glow=s['RedGroundGlowExportName'],
            pre_file=s['PreExplosionExportFileName'], pre_blue=s['PreExplosionBlueExportName'], pre_red=s['PreExplosionRedExportName'],
            pre_scale=_int(pv['PreExplosionScale']), hide_time=_int(s['HideTime']), scale=_int(s['Scale']),
            random_start=_int(s['RandomStartFrame']), rendering=RENDERING.get(s['Rendering'], 0),   # strcmp (0xe9779c -> 0x11a5200): case-sensitive, unknown -> 0
            rendering_name=s['Rendering'], anim_by_dist=_bool(pv['AnimateByDistanceTravelled']), special_visual=_bool(s['SpecialVisualState']),
            hide_faster=_bool(s['HideFaster']), color_mod=_bool(s['UseColorMod']),
            color=tuple(_int(s[k]) for k in ('RedMul', 'GreenMul', 'BlueMul', 'RedAdd', 'GreenAdd', 'BlueAdd')),
            ground_basis=_bool(s['GroundBasis']), trail=s['TrailEffect'], special_trail=s['SpecialTrailEffect'],
            special_trail2=s['SpecialTrailEffect2'], trail_behavior=_int(s['TrailBehaviorType']),
            hit_env=s['HitEffectEnv'], hit_char=s['HitEffectChar'], range_effect=s['MaxRangeReachedEffect'], cancel_effect=s['CancelEffect'],
            # logic (LogicProjectileData)
            speed=_int(lg.get('Speed')), gravity=_int(lg.get('Gravity')), radius=_int(lg.get('Radius')), early_ticks=_int(lg.get('EarlyTicks')),
            indirect=_bool(lg.get('Indirect')), is_beam=_bool(lg.get('IsBeam')), shot_by_hero=_bool(lg.get('ShotByHero')),
            scale_start=_int(lg.get('ScaleStart')) or 100, scale_end=_int(lg.get('ScaleEnd')) or 100,
            zoffset=_int(lg.get('Zoffset')), pre_explosion_ms=_int(lg.get('PreExplosionTimeMs')), travel_type=lg.get('TravelType', ''),
            constant_fly_time=_bool(lg.get('ConstantFlyTime')), disabled=_bool(s['Disabled']) or _bool(lg.get('Disabled')),
            spawn_character=lg.get('SpawnCharacter', ''), bouncing=_bool(lg.get('IsBouncing')))
    _T = out
    return out


if __name__ == '__main__':
    import collections
    T = table()
    print(len(T), 'types')
    for k in ('rendering', 'random_start', 'anim_by_dist', 'special_visual', 'hide_time', 'is_beam', 'color_mod', 'ground_basis'):
        print(k, collections.Counter(v[k] for v in T.values()).most_common(8))
    print(T[6000000])
