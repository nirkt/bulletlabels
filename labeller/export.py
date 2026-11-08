"""Training-data export: a capture (framegen output with its ID pass) -> images + instance masks + per-frame JSON.

  python3 bl.py label CAPTURE OUT_DIR [--motion] [--overlay]

Per labelled frame (the frames the capture saved with an ID pass, 1200 x 540):
  OUT/images/<cap>_f<frame>.png    the game frame (RGB)
  OUT/masks/<cap>_f<frame>.png     16-bit instance map: 0 = background, i = instance i of the JSON (one owner per pixel)
  OUT/labels/<cap>_f<frame>.json   {image, mask, width, height, frame, tick, camera, instances: [...]}
  OUT/overlays/<cap>_f<frame>.png  (--overlay) the frame with every instance outlined in its class colour
An instance = one game object (a projectile, a spawned character, a zone); the label rules are those of labeller/idlabels.py +
labeller/fixes (docs/LABEL_RULES.md). Instance fields:
  id, object_id, kind (attack | super | spawnable | spawnable_attack | zone | gadget | starpower | target | throwable | pull),
  kind_code (classes.json), brawler, skin, name (the projectile / character / area type), team (own = player 0's, the
  camera's brawler; enemy = anyone else), owner_player / owner_brawler / owner_skin / owner_skin_id (who fired / spawned it),
  parts {body / ground / spawn / area: pixels}, bbox [x0, y0, x1, y1] (inclusive),
  area (pixels), and with --motion: motion {dir, angle_deg, dir_screen, end_world, end_screen, end_reason, end_tick, ticks_left,
  ...} (labeller/fixes/zz_1109_motion.py) for projectiles.
A pixel belongs to a label when its owner covers at least half of it (the same threshold as the review pages).
Other class schemes (one class per brawler / skin / type, with or without ally / enemy): python3 bl.py classes DATASET --mode ..."""
import glob, json, os, random, sys

KIND_CODE = dict(attack=1, super=2, spawnable=3, zone=4, gadget=5, starpower=5, spawnable_attack=6, target=7, throwable=8, pull=9)
KIND_COLOR = {1: (255, 0, 0), 2: (255, 0, 255), 3: (0, 140, 255), 4: (255, 230, 0), 5: (0, 255, 255), 6: (255, 140, 0),
              7: (255, 255, 255), 8: (255, 150, 200), 9: (150, 90, 255)}
SHARE_MIN = 128   # share x 255 >= 128 (0.5): as the review pages (seqsheet.S05)


def classes():
    names = {1: 'attack', 2: 'super', 3: 'spawnable', 4: 'zone', 5: 'gadget/starpower', 6: 'spawnable_attack', 7: 'target (landing marker)',
             8: 'throwable (no damage)', 9: 'pull'}
    return [dict(code=c, name=names[c], color=list(KIND_COLOR[c])) for c in sorted(names)]


def _edge(m):
    import numpy as np
    e = np.zeros_like(m)
    e[1:] |= m[1:] & ~m[:-1]; e[:-1] |= m[:-1] & ~m[1:]; e[:, 1:] |= m[:, 1:] & ~m[:, :-1]; e[:, :-1] |= m[:, :-1] & ~m[:, 1:]
    return e


def export_capture(cap, out, motion=False, overlay=False, every=1, log=print):
    """-> number of frames written. motion=True needs the IDLABELS_MOTION_* switches set BEFORE idlabels is imported (bl.py does)."""
    import numpy as np
    from PIL import Image
    import rc, dumps, track, bulletlabels as BL, idbuf, idlabels
    sys.setrecursionlimit(5000)
    cap = os.path.abspath(cap); name = os.path.basename(cap.rstrip('/'))
    ids_dirs = sorted(glob.glob(os.path.join(cap, 'id', '*', '')))
    if not ids_dirs: raise SystemExit('%s: no ID pass (capture with --id-dir, bl.py capture does)' % cap)
    idd = ids_dirs[0]; have = set(idbuf.frames(idd))
    z = np.load(os.path.join(cap, 'frames.npz')); want = {int(s): i for i, s in enumerate(z['swap'])}; IMG = z['img']
    for sub in ('images', 'masks', 'labels') + (('overlays',) if overlay else ()): os.makedirs(os.path.join(out, sub), exist_ok=True)
    json.dump(classes(), open(os.path.join(out, 'classes.json'), 'w'), indent=1)
    SC = json.load(open(os.path.join(cap, 'script.json')))
    pl_skin = [int(x) for x in ((SC.get('info') or {}).get('skin') or SC.get('skins') or [])]   # player index -> skin id
    pl_char = list(SC.get('chars') or [])                                                     # player index -> brawler (as requested)
    skin_name = {int(r[0]): (r[1], r[2]) for r in json.load(open(os.path.join(rc.DATA, 'sweep_skins.json')))}
    B = dumps.Battle(cap); fr = sorted(B.meta); tr = track.Tracker(track.trajectories(B), rng=random.Random(0)); WF = track.frame_weights(B)
    sp = None; pt = None; n = 0; k_seen = 0
    for f in fr:
        s = B.state(f)
        if s is None: sp = None; continue
        if pt is not None and s['tick'] - pt > 1: tr.fill(s['tick'] - pt - 1); sp = None
        v = int(B.meta[f].get('player', 0)); tr.step(f, s, sp, WF.get(f, -1), track.team_of(s, v), None, v); sp = s; pt = s['tick']
        if f not in want or f not in have: continue
        k_seen += 1
        if (k_seen - 1) % every: continue
        mat = np.array(B.meta[f]['mat'], np.float64).reshape(4, 4).T; objs = BL.frame_objects(tr); R = idbuf.load(idd, f)
        img = IMG[want[f]]
        P = idlabels.labels(objs, mat, R, state=s, ended=list(tr.ended.values()), img=img)
        ids = P['ids']; ok = (ids >= 0) & (np.round(P['share'] * 255) >= SHARE_MIN)
        M = P.get('meta') or {}; MO = P.get('motion') or {}
        own_of = {o['id']: bool(o['own']) for o in objs}
        player_of = {o['id']: int(o.get('owner', -1)) for o in objs}   # projectile -> the player who fired it
        owner_player = {int(c[0]): int(c[2]) for c in s['chars']}   # character id -> its player index
        zones = P.get('zones') or {}
        inst = {}; order = []
        for k, lid in enumerate(P['lids']):
            m = ok & (ids == k)
            if not m.any(): continue
            oid = int(lid[0])
            if oid not in inst:
                me = M.get(oid) or {}
                if oid in own_of: team = 'own' if own_of[oid] else 'enemy'; pl = player_of.get(oid, -1)
                elif lid[2] == 'spawn': pl = owner_player.get(oid, -1); team = 'own' if pl == v else 'enemy'
                elif lid[2] == 'area' and oid in zones: pl = int(zones[oid][1]); team = 'own' if pl == v else 'enemy'
                else: team = None; pl = -1
                osk = pl_skin[pl] if 0 <= pl < len(pl_skin) else None
                osn, och = skin_name.get(osk, (None, pl_char[pl] if 0 <= pl < len(pl_char) else None))
                inst[oid] = dict(object_id=oid, kind=me.get('kind') or 'attack', brawler=me.get('brawler') or och, skin=me.get('skin') or osn,
                                 name=me.get('name'), team=team, owner_player=pl if pl >= 0 else None, owner_brawler=och, owner_skin=osn,
                                 owner_skin_id=osk, parts={}, mask=np.zeros(ids.shape, bool))
                order.append(oid)
            inst[oid]['parts'][lid[2]] = inst[oid]['parts'].get(lid[2], 0) + int(m.sum()); inst[oid]['mask'] |= m
        stem = '%s_f%07d' % (name, f)
        mask = np.zeros(ids.shape, np.uint16); rows = []
        for i, oid in enumerate(order, 1):
            d = inst.pop(oid); m = d.pop('mask'); mask[m] = i
            ys, xs = np.nonzero(m)
            d.update(id=i, kind_code=KIND_CODE.get(d['kind'], 1), bbox=[int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())], area=int(m.sum()))
            if motion and oid in MO:
                d['motion'] = {k2: (list(v2) if isinstance(v2, tuple) else v2) for k2, v2 in MO[oid].items()}
            rows.append({k2: d[k2] for k2 in ('id', 'object_id', 'kind', 'kind_code', 'brawler', 'skin', 'name', 'team', 'owner_player', 'owner_brawler', 'owner_skin', 'owner_skin_id', 'parts', 'bbox', 'area') + (('motion',) if 'motion' in d else ())})
        Image.fromarray(img).save(os.path.join(out, 'images', stem + '.png'))
        Image.fromarray(mask).save(os.path.join(out, 'masks', stem + '.png'))
        json.dump(dict(image='images/%s.png' % stem, mask='masks/%s.png' % stem, width=int(img.shape[1]), height=int(img.shape[0]),
                       capture=name, frame=int(f), tick=int(s['tick']), camera=mat.tolist(), instances=rows),
                  open(os.path.join(out, 'labels', stem + '.json'), 'w'), default=float)
        if overlay:
            o = img.copy()
            for r in rows: o[_edge(mask == r['id'])] = KIND_COLOR.get(r['kind_code'], (255, 0, 0))   # one outline per instance
            Image.fromarray(o).save(os.path.join(out, 'overlays', stem + '.png'))
        n += 1
    log('%s: %d frames -> %s' % (name, n, out))
    return n
