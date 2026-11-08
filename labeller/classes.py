"""Class schemes for a dataset written by labeller/export.py: every instance carries kind, team, name (its type), brawler / skin and
owner_brawler / owner_skin, so any scheme can be made afterwards without capturing again.

  python3 bl.py classes DATASET --mode MODE [--kinds attack,super] [--masks] [--class-list FILE]

MODE (team = own / enemy: own = the camera's brawler's, enemy = everyone else's):
  all                  one class for everything labelled
  team                 own, enemy                                  (e.g. one class for every enemy projectile)
  kind                 attack, super, spawnable, zone, ...
  kind_team            own/attack, enemy/attack, own/super, ...
  brawler              the brawler who fired / spawned it (owner_brawler)
  brawler_team         own/Colt, enemy/Colt, ...
  brawler_kind_team    own/Colt/attack, enemy/Colt/super, ...
  skin, skin_team, skin_kind_team      the same with the skin (owner_skin)
  type, type_team      the projectile / character / area type (name: e.g. Ash005Projectile1)
Writes DATASET/classes/MODE/classes.json ([{id, name, count}], id 0 = background), DATASET/classes/MODE/instances.json
({frame stem: {instance id: class id}}) and with --masks DATASET/classes/MODE/masks/<stem>.png (16-bit class map).
--class-list FILE: reuse the class ids of an earlier classes.json (new classes get new ids after them) so several datasets share ids."""
import glob, json, os

MODES = ('all', 'team', 'kind', 'kind_team', 'brawler', 'brawler_team', 'brawler_kind_team', 'skin', 'skin_team', 'skin_kind_team',
         'type', 'type_team')


def class_of(inst, mode):
    """the class name of one instance (a dict of labels/*.json) under MODE; None = not labelled in this scheme"""
    team = inst.get('team') or 'unknown'; kind = inst.get('kind') or 'attack'
    br = inst.get('owner_brawler') or inst.get('brawler') or 'unknown'
    sk = inst.get('owner_skin') or inst.get('skin') or 'unknown'
    ty = inst.get('name') or 'unknown'
    return {'all': 'labelled', 'team': team, 'kind': kind, 'kind_team': team + '/' + kind, 'brawler': br, 'brawler_team': team + '/' + br,
            'brawler_kind_team': '%s/%s/%s' % (team, br, kind), 'skin': sk, 'skin_team': team + '/' + sk,
            'skin_kind_team': '%s/%s/%s' % (team, sk, kind), 'type': ty, 'type_team': team + '/' + ty}[mode]


def build(dataset, mode, kinds=None, masks=False, class_list=None, log=print):
    import numpy as np
    from PIL import Image
    if mode not in MODES: raise SystemExit('mode must be one of: ' + ', '.join(MODES))
    files = sorted(glob.glob(os.path.join(dataset, 'labels', '*.json')))
    if not files: raise SystemExit('%s: no labels/*.json (python3 bl.py label ... first)' % dataset)
    names = []
    if class_list: names = [c['name'] for c in sorted(json.load(open(class_list)), key=lambda c: c['id']) if c['id'] > 0]
    count = {}; per = {}
    rows = [(f, json.load(open(f))) for f in files]
    for f, d in rows:
        for i in d['instances']:
            if kinds and i.get('kind') not in kinds: continue
            c = class_of(i, mode); count[c] = count.get(c, 0) + 1
    for c in sorted(count):
        if c not in names: names.append(c)
    cid = {c: k + 1 for k, c in enumerate(names)}
    out = os.path.join(dataset, 'classes', mode); os.makedirs(out, exist_ok=True)
    if masks: os.makedirs(os.path.join(out, 'masks'), exist_ok=True)
    for f, d in rows:
        stem = os.path.basename(f)[:-5]; m = {}
        for i in d['instances']:
            if kinds and i.get('kind') not in kinds: continue
            m[str(i['id'])] = cid[class_of(i, mode)]
        per[stem] = m
        if masks:
            inst = np.array(Image.open(os.path.join(dataset, d['mask'])))
            lut = np.zeros(int(inst.max()) + 1, np.uint16)
            for k, v in m.items():
                if int(k) < len(lut): lut[int(k)] = v
            Image.fromarray(lut[inst]).save(os.path.join(out, 'masks', stem + '.png'))
    json.dump([dict(id=0, name='background', count=0)] + [dict(id=cid[c], name=c, count=count.get(c, 0)) for c in names],
              open(os.path.join(out, 'classes.json'), 'w'), indent=1)
    json.dump(per, open(os.path.join(out, 'instances.json'), 'w'))
    log('%s: %d classes, %d frames -> %s' % (mode, len(names), len(rows), out))
    return out
