"""What every labelled object is (the user's classes, 2026-10-19): normal attacks are 'attack', supers 'super', characters
a brawler spawns 'spawnable', lingering damage zones 'zone' (gadgets 'gadget', star powers 'starpower'), each with the
brawler (and skin) it belongs to, so a label reads e.g. Shelly attack / ShotgunGirl skin Bandit.
  projectile(t) -> dict(kind, brawler, skin)       t: projectile type id (6000000 + projectiles_skin row)
  character(did) -> dict(kind 'hero' | 'spawnable' | None, name, type)   did: 16000000 + characters row
  zone(did) -> dict(name, logic, file, ...)          did: 17000000 + area_effects_skin row
From the game tables (csv_logic): characters (WeaponSkill, UltimateSkill, OverchargedUltimateSkill, AutoAttackProjectile),
skills (Projectiles, OverchargedProjectiles, SummonedCharacters), accessories (gadgets: Skill), cards (star powers:
Skill, Projectiles), skin_confs (skin projectiles), projectiles_skin (ParentProjectileForSkin), projectiles_logic
(ChainBullets, SpawnCharacter), area_effects_logic (BulletExplosionBullet). A projectile no table names (spawned by
another one) inherits from its parent; the rest fall back on the name ('Ulti' -> super)."""
import csv, functools, os
import rc

KINDS = ('attack', 'super', 'spawnable', 'zone', 'gadget', 'starpower')


def _rows(name):
    r = list(csv.reader(open(os.path.join(rc.ASSETS, 'csv_logic', name + '.csv'), encoding='utf8')))
    h = r[0]; return h, [dict(zip(h, x)) for x in r[2:]]


def _split(v): return [x for x in (v or '').split(';') if x]


@functools.lru_cache(maxsize=None)
def _tables():
    _, chars = _rows('characters'); _, skills = _rows('skills'); _, acc = _rows('accessories'); _, cards = _rows('cards')
    _, confs = _rows('skin_confs'); _, pskin = _rows('projectiles_skin'); _, plog = _rows('projectiles_logic')
    _, alog = _rows('area_effects_logic'); _, askin = _rows('area_effects_skin')
    return dict(chars=chars, skills={s['Name']: s for s in skills if s['Name']}, acc=acc, cards=cards, confs=confs,
                pskin=pskin, plog={p['Name']: p for p in plog if p['Name']}, alog={a['Name']: a for a in alog if a['Name']},
                askin=askin)


def _skill_projectiles(S, name):
    s = S['skills'].get(name) or {}
    return _split(s.get('Projectiles')) + _split(s.get('OverchargedProjectiles')) + _split(s.get('OverchargedChainProjectile'))


@functools.lru_cache(maxsize=None)
def _base_map():
    """{projectile skin name of a default projectile: (kind, brawler name)}"""
    S = _tables(); out = {}
    heroes = {c['Name']: c for c in S['chars'] if c.get('Type') in ('Hero', 'Her0')}
    for c in S['chars']:
        n = c['Name']; minion = (c.get('Type') or '').startswith('Minion')
        if not n: continue
        for col, kind in (('WeaponSkill', 'attack'), ('UltimateSkill', 'super'), ('OverchargedUltimateSkill', 'super')):
            for p in _skill_projectiles(S, c.get(col)):
                out.setdefault(p, ('spawnable' if minion else kind, n))
        for p in _split(c.get('AutoAttackProjectile')) + _split(c.get('OverchargedAutoAttackProjectile')):
            out.setdefault(p, ('spawnable' if minion else 'attack', n))
    owner_of_minion = {}
    for c in heroes.values():
        for col in ('WeaponSkill', 'UltimateSkill', 'OverchargedUltimateSkill'):
            s = S['skills'].get(c.get(col)) or {}
            for m in _split(s.get('SummonedCharacters')) + _split(s.get('OverchargedSummonedCharacters')): owner_of_minion.setdefault(m, c['Name'])
        for m in _split(c.get('Pet')) + _split(c.get('AltPet')): owner_of_minion.setdefault(m, c['Name'])
    for p, (k, n) in list(out.items()):   # a spawnable's own projectiles belong to the brawler that spawns it
        if k == 'spawnable' and n in owner_of_minion: out[p] = (k, owner_of_minion[n])
    for a in S['acc']:
        for p in _skill_projectiles(S, a.get('Skill')): out.setdefault(p, ('gadget', _hero_prefix(a['Name'], heroes)))
    for c in S['cards']:
        for p in _skill_projectiles(S, c.get('Skill')) + _split(c.get('Projectiles')):
            out.setdefault(p, ('starpower', _hero_prefix(c['Name'], heroes)))
    # projectiles spawned by projectiles (chain bullets) or by zones (explosion bullets) inherit the parent's
    for _ in range(3):
        for p, (k, n) in list(out.items()):
            lg = S['plog'].get(_logic_of(p)) or {}
            for q in _split(lg.get('ChainBullets')): out.setdefault(q, (k, n))
        for a in S['alog'].values():
            for q in _split(a.get('BulletExplosionBullet')): pass   # zone -> bullet: the zone's owner (state) decides
    return out


def _hero_prefix(name, heroes):
    best = ''
    for h in heroes:
        if name.startswith(h) and len(h) > len(best): best = h
    return best or None


@functools.lru_cache(maxsize=None)
def _logic_map():
    return {p['Name']: p['LogicProjectile'] for p in _tables()['pskin']}


def _logic_of(p): return _logic_map().get(p, p)


@functools.lru_cache(maxsize=None)
def _skin_of_projectile():
    """{projectile skin name: skin conf name} from skin_confs.Projectiles"""
    out = {}
    for c in _tables()['confs']:
        for p in _split(c.get('Projectiles')): out.setdefault(p, c['Name'])
    return out


@functools.lru_cache(maxsize=None)
def projectile(t):
    """dict(kind, brawler, skin, name) of projectile type t"""
    S = _tables(); i = t - 6000000
    if not 0 <= i < len(S['pskin']): return dict(kind=None, brawler=None, skin=None, name=None)
    row = S['pskin'][i]; name = row['Name']; base = name; seen = set()
    while (S['pskin'][_index()[base]].get('ParentProjectileForSkin') if base in _index() else '') and base not in seen:
        seen.add(base); base = S['pskin'][_index()[base]]['ParentProjectileForSkin']
    M = _base_map()
    k = M.get(base) or M.get(name) or M.get(_logic_of(name)) or M.get(_logic_of(base))
    if k is None:   # not named by any skill: the skin conf that lists it, else the brawler of the longest name prefix
        guess = 'super' if 'Ulti' in name else 'attack'
        conf = _conf_by_name().get(_skin_of_projectile().get(name))
        if conf:
            ch = _split(conf.get('Character'))[:1]; c = _char_by_name().get(ch[0]) if ch else None
            if c and (c.get('Type') or '').startswith('Minion'): k = ('spawnable', _minion_owner().get(c['Name']))
            elif c: k = (guess, c['Name'])
        if k is None:
            pre = max(((len(_common(n, q)), b) for n in (base, name) for q, (kk, b) in M.items() if b and len(_common(n, q)) >= 5),
                      default=(0, None))[1]
            if pre is None:   # a brawler's name it starts with (CrowUltiProjectile)
                hs = [c['Name'] for c in _tables()['chars'] if c.get('Type') in ('Hero', 'Her0') and c['Name'] and name.startswith(c['Name'])]
                pre = max(hs, key=len) if hs else None
            k = (guess, pre)
    skin = _skin_of_projectile().get(name)
    if skin is None and base != name:   # a skin's chain / extra projectile no conf lists: the conf of its siblings
        best = max(((len(_common(name, q)), c) for q, c in _skin_of_projectile().items()), default=(0, None))
        if best[0] >= len(name) - 2: skin = best[1]
    return dict(kind=k[0], brawler=k[1], skin=skin, name=name)


def _common(a, b):
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]: n += 1
    return a[:n]


@functools.lru_cache(maxsize=None)
def _conf_by_name(): return {c['Name']: c for c in _tables()['confs']}


@functools.lru_cache(maxsize=None)
def _char_by_name(): return {c['Name']: c for c in _tables()['chars'] if c['Name']}


@functools.lru_cache(maxsize=None)
def _minion_owner():
    S = _tables(); out = {}
    for c in S['chars']:
        if c.get('Type') not in ('Hero', 'Her0'): continue
        for col in ('WeaponSkill', 'UltimateSkill', 'OverchargedUltimateSkill'):
            sk = S['skills'].get(c.get(col)) or {}
            for m in _split(sk.get('SummonedCharacters')) + _split(sk.get('OverchargedSummonedCharacters')): out.setdefault(m, c['Name'])
        for m in _split(c.get('Pet')) + _split(c.get('AltPet')): out.setdefault(m, c['Name'])
    return out


def runtime(t, src_did, owner_hero_did=None):
    """dict(kind, brawler, skin, name) of a projectile in a state: its source character (proj 'src' -> chars dataId)
    decides spawnable vs a brawler's own; the brawler comes from the source (or the owner player's hero)"""
    r = dict(projectile(t)); c = character(src_did) if src_did else dict(kind=None)
    if c.get('kind') == 'spawnable':
        r['kind'] = 'spawnable'; r['brawler'] = spawnable_owner(c['name']) or (character(owner_hero_did)['name'] if owner_hero_did else r['brawler'])
    elif c.get('kind') == 'hero': r['brawler'] = c['name']
    return r


def spawnable_owner(name):
    """the brawler that spawns character name (a Minion), None if unknown"""
    return _minion_owner().get(name)


@functools.lru_cache(maxsize=None)
def _index():
    return {p['Name']: i for i, p in enumerate(_tables()['pskin'])}


@functools.lru_cache(maxsize=None)
def character(did):
    S = _tables(); i = did - 16000000
    if not 0 <= i < len(S['chars']): return dict(kind=None, name=None, type=None)
    c = S['chars'][i]; ty = c.get('Type') or ''
    kind = 'hero' if ty in ('Hero', 'Her0') else 'spawnable' if ty.startswith('Minion') else None
    return dict(kind=kind, name=c['Name'], type=ty)


@functools.lru_cache(maxsize=None)
def zone(did):
    S = _tables(); i = did - 17000000
    if not 0 <= i < len(S['askin']): return None
    r = dict(S['askin'][i]); r['logic_row'] = S['alog'].get(r.get('Logic')) or {}
    return r


ZONE_MIN_MS = 1000   # an area effect lasting at least this long is a lingering damage zone; shorter ones are explosions
AREA_COLS_P = ('SpawnAreaEffectObject', 'SpawnAreaEffectObject2', 'SpawnAreaEffectTrail', 'CustomObject', 'CustomObject2',
               'OnCreateAreaEffect', 'SpawnAreaEffectObjectsAdditional')
AREA_COLS_S = ('AreaEffectObject', 'AreaEffectObject2', 'OverchargedAreaEffectObject', 'OverchargedAreaEffectObject2')


@functools.lru_cache(maxsize=None)
def _area_parents():
    """{area effect name: (kind, brawler)} of the projectile / skill that spawns it"""
    S = _tables(); out = {}
    for i, r in enumerate(S['pskin']):
        for c in AREA_COLS_P:
            for a in _split(r.get(c)):
                if a not in out:
                    p = projectile(6000000 + i); out[a] = (p['kind'], p['brawler'])
    heroes = [c for c in S['chars'] if c.get('Type') in ('Hero', 'Her0')]
    for c in heroes:
        for col, kind in (('WeaponSkill', 'attack'), ('UltimateSkill', 'super'), ('OverchargedUltimateSkill', 'super')):
            sk = S['skills'].get(c.get(col)) or {}
            for cc in AREA_COLS_S:
                for a in _split(sk.get(cc)): out.setdefault(a, (kind, c['Name']))
    for a in S['acc']:
        sk = S['skills'].get(a.get('Skill')) or {}
        for cc in AREA_COLS_S:
            for x in _split(sk.get(cc)) + _split(a.get('AreaEffect')): out.setdefault(x, ('gadget', _hero_prefix(a['Name'], {h['Name']: h for h in heroes})))
    return out


@functools.lru_cache(maxsize=None)
def area(did):
    """dict(kind 'zone' | the spawner's kind (an explosion), brawler, name, ms) of area effect did (17000000 + skin row)"""
    Z = zone(did)
    if not Z: return dict(kind=None, brawler=None, name=None, ms=0)
    lg = Z.get('logic_row') or {}; ms = int(lg.get('TimeMs') or 0)
    par = _area_parents().get(Z['Name']) or _area_parents().get(Z.get('Logic')) or (None, None)
    kind = 'zone' if ms >= ZONE_MIN_MS else (par[0] or 'attack')
    return dict(kind=kind, brawler=par[1], name=Z['Name'], ms=ms)
