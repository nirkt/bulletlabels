"""Exact projectile geometry and per-pixel labels for a frame, from game states (definitions: LABEL_SPEC.md).

  import random, track, bulletlabels as BL
  WF = track.frame_weights(B); TR = track.trajectories(B)            (B: a battle's consecutive states, dumps.Battle)
  tr = track.Tracker(TR, rng=random.Random(seed))
  for f in frames:
      tr.step(f, s, sp, WF[f], track.team_of(s, viewer), None, viewer)
      objs = BL.frame_objects(tr)                     every projectile the client shows: parts, pellets, mesh, targets
      L = BL.labels(objs, cam)                        cam: the frame's 4 x 4 PV matrix (render coords -> clip)
      L['share'], L['ids'], L['instances']            (540, 1200) share / owner index, one row per instance

Objects (one per projectile the client shows, hidden ones included with drawn=False):
  id, t (type), own (viewer's team), drawn, start_frame, angle (logic deg), angle_status (exact / ambiguous / estimated),
  parts {'clip' | 'shadow' | 'glow': placed polygons (projdraw.place dicts, world render coords)}, pellets (root slots),
  mesh (4 x 4 or None: the 3D object transform), mesh_tris ([dict(tris (m,3,3) world, uv (m,3,2), material, skinned)]
  of the posed mesh, skinned ones with their bones: glb.py), mesh_approx (True: a non-opaque material: share from texture alpha),
  anchor (render coords of the clip origin),
  ground (render coords of the ground point under it), logic (x, y of the server position), radius (logic Radius)"""
import numpy as np
import projdraw as PD, projtab, masks, track, glow

J = track.J
CUT = 0.5      # a label pixel belongs to a bullet when the bullet owns at least half of it (LABEL_SPEC 3)
SOFT = 0.1     # pixels a bullet owns 10..50 %: 'soft' (ignore in a loss), below: background
# types whose body label is one solid region: holes enclosed by the label (translucent interior, owned < CUT) belong to
# it. Only Bibi's super bubble (every skin, overcharged too; the user's decision 2026-10-18) and Sandy's attack
# (2026-10-19), others when asked
FILL_HOLES = {6000625, 6000626, 6000627, 6000628, 6000629, 6000630, 6000631, 6000632, 6000633, 6000634, 6000635, 6000636,
              6001641, 6001642,
              6000666, 6000335}   # Sandy's and Poco's attacks: the whole cloud as one label (the user, 2026-10-19)


def frame_objects(tr):
    T = projtab.table(); out = []
    for i, e in tr.d.items():
        if e.get('disp') is None: continue
        p = e['row']; P = tr.proj(e); d = e['disp']; t = e['t']
        S = PD.sc(T[t]['file']) if T[t]['file'] else None
        cid = S.exports.get(T[t]['blue'] if e['own'] else T[t]['red']) if S is not None else None
        ox, oy = P.own_offset
        mesh = tr.mesh(e); mesh_tris = None; mesh_approx = False
        scw = T[t]['blue_scw'] if e['own'] else T[t]['red_scw']
        if mesh is not None and scw:
            import glb, os, rc
            parts_ = glb.posed_triangles(os.path.join(rc.ASSETS, scw), d.age_time)
            if parts_:
                mesh_tris = []
                for tri, mat, skinned, uvt in parts_:
                    W = (tri.reshape(-1, 3) @ mesh[:3, :3].T + mesh[:3, 3]).reshape(-1, 3, 3)
                    mesh_tris.append(dict(tris=W, uv=uvt, material=mat, skinned=skinned))
                mesh_approx = any(x[1].get('blendMode', 4) not in (4, 0) for x in parts_)
        export = T[t]['blue'] if e['own'] else T[t]['red']; parts = tr.predict_parts(e)
        for part, polys in parts.items():   # body / glow class of every drawn layer (glow.py, LABEL_SPEC 2)
            for q in polys:
                q['cls'] = ('glow' if part == 'glow' else 'ground') if part != 'clip' else glow.layer_class(t, bool(e['own']), q)
        out.append(dict(id=int(p[0]), t=t, own=bool(e['own']), drawn=not d.hidden, start_frame=e.get('start'), first=e['first'], export=export,
                        mesh_tris=mesh_tris, mesh_approx=mesh_approx,
                        angle=P.angle, angle_status=('exact', 'ambiguous', 'estimated', 'estimated')[e.get('astat', 3)],
                        parts=parts, pellets=sorted(PD.pellet_slots(S, cid)) if cid in (S.clips if S else {}) else [],
                        mesh=mesh, file=T[t]['file'],
                        anchor=np.array([P.x + ox, -(P.y + oy), PD.HEIGHT * (P.z + (P.zoffset or 0))]),
                        ground=np.array([P.x + ox, -(P.y + oy), 0.0]), logic=(int(p[J['x']]), int(p[J['y']])),
                        radius=T[t]['radius'], stage_angle=PD.stage_angle(P.angle), src=int(p[J['src']]), owner=int(p[J['owner']]),
                        target=(tuple(int(v) for v in tr.traj[int(p[0])][2:4]) if int(p[0]) in (tr.traj or {}) else None),   # logic end point (track.trajectories)
                        finished=int((e.get('ctrl') or {}).get('finished', 0)),   # the logic projectile reached its range / hit (state 'finished')
                        path=[tuple(float(v) for v in c['pos'][:2]) for c in e['hist']]))   # its drawn ground positions, one per update, oldest first (wave rows' ages)
    import wave   # wave attacks: ONE label per attack (all wave projectiles a brawler fires on one tick), its region the
    # hull of their covered paths grown by the hitbox radius, kept while their trail is on the ground (wave.py)
    byid = {o['id']: o for o in out}
    import idlabels   # also the trail-body types (idlabels.TRAIL_BODY_EXTRA): their region along the path (ID labels' search area)
    ents = [(i, e) for i, e in tr.d.items() if e.get('disp') is not None and idlabels.trail_body(e['t'])]
    ents += [(i, e) for i, e in tr.gone.items()]
    G = wave.groups(ents); members = {}
    for i, e in ents: members.setdefault(G[i], []).append((i, e))
    for o in out: o['attack'] = None
    for gid, mem in members.items():
        pts = [p for _, e in mem for p in wave.path_points(e, PD.DT)]
        if not pts: continue
        r = max(T[e['t']]['radius'] for _, e in mem); reg = wave.attack_tris(pts, r)
        for i, _ in mem:
            if i in byid: byid[i]['label_id'] = gid
        if gid in byid: byid[gid]['attack'] = reg; byid[gid]['attack_members'] = sorted(i for i, _ in mem); continue
        e = dict(mem)[gid] if gid in dict(mem) else mem[0][1]
        x, y = pts[-1]
        o = dict(id=int(gid), t=e['t'], own=bool(e['own']), drawn=True, start_frame=e.get('start'), first=e['first'], export=None,
                 mesh_tris=None, mesh_approx=False, angle=e['ctrl']['angle'], angle_status='exact', parts={}, pellets=[], mesh=None,
                 file=None, anchor=np.array([x, -y, 0.0]), ground=np.array([x, -y, 0.0]), logic=(int(e['row'][J['x']]), int(e['row'][J['y']])),
                 radius=r, stage_angle=PD.stage_angle(e['ctrl']['angle']), ended=True, attack=reg, label_id=gid,
                 attack_members=sorted(i for i, _ in mem), src=int(e['row'][J['src']]), owner=int(e['row'][J['owner']]))
        out.append(o); byid[gid] = o
    # the client's draw order of projectile objects: the order it first showed them, then id [V 157 of 157 frames of
    # pose_p1 / lib1_90100002: first draw index of every projectile's matched quads]
    out.sort(key=lambda o: (o['first'], o['id']))
    return out


def labels(objs, cam, half=True, cut=CUT, ground=True, occluders=None):
    """per-pixel ownership of every drawn projectile part, glow excluded (LABEL_SPEC 2), plus a separate glow map.
    Label ids: (projectile id, pellet, part) with pellet -1 for a one-bullet clip and part 'body' (the clip's painting
    layers, the mesh; for 'wave' types every layer by coverage, plus the swept band of their trail: wave.py), 'ground'
    (shadow) or 'glow' (the clip's light layers, the ground glow, additive mesh materials).
    Composited in the client's order: 3D bodies, every ground part, then each object's clip (objs in the client's
    object order, frame_objects), its polygons in clip order (the client draws each clip contiguously: 583 of 583
    objects checked).
      body / ground: composited WITHOUT the glow layers: a glow pixel takes the label of what lies under the glow, as if
                     the glow had not been drawn, and a glow drawn over a body does not lower the body's share;
      glow map:      composited with everything (glow, body, ground in draw order): the share each glow layer still
                     determines where it is drawn (a body drawn later over a glow hides it).
    occluders: optional (H, W) or (H, W, 3) full-resolution transmittance of what the scene draws over the projectiles.
    -> dict(share (H, W) float32, ids (H, W) int32 into lids (-1: none or share < cut), lids [(id, pellet, part)],
            instances [rows, see instance_rows; body / ground lids, then glow lids],
            glow_share (H, W) float32, glow_ids (H, W) int32 into glow_lids (-1: share < SOFT), glow_lids)"""
    L = masks.Layers(cam); LG = masks.Layers(cam)
    for o in objs:   # 3D bodies first (the main pass draws meshes before the effect sprites [R]); opaque: own every covered pixel
        if not o['drawn'] or o.get('mesh_tris') is None: continue
        for pm in o['mesh_tris']:
            mat = pm['material'] or {}; tex = None; ch = 2
            if mat.get('blendMode', 4) != 4:   # translucent: alpha = diffuse.a x opacityTex.b (uber.fsh); missing textures: coverage
                tx = (mat.get('variables') or {}).get('textures') or {}
                import sctx
                for key, c_ in (('opacityTex2D', 2), ('diffuseTex2D', 3)):
                    try: tex = sctx.texture(tx.get(key) or '')
                    except Exception: tex = None
                    if tex is not None: ch = c_; break
            add = mat.get('blendMode') in (1, 6); cls = glow.mesh_class(mat)
            if cls == 'body': L.add_triangles((o['id'], -1, 'body'), pm['tris'], uv=pm['uv'], tex=tex, channel=ch, additive=add)
            LG.add_triangles((o['id'], -1, cls), pm['tris'], uv=pm['uv'], tex=tex, channel=ch, additive=add)
    for o in objs:   # wave attacks: their trail on the ground (GroundLow), the whole attack as one body label; drawn
        # whether or not the projectile's own (empty) clip is still shown: the trail outlives it
        if o.get('attack') is not None and len(o['attack']):
            L.add_triangles((_lid(o), -1, 'body'), o['attack']); LG.add_triangles((_lid(o), -1, 'body'), o['attack'])
    if ground:   # all ground parts before all clips [V 69 of 69 frames of pose_p1]
        for o in objs:
            if not o['drawn'] or not o['file']: continue
            S = PD.sc(o['file'])
            if 'shadow' in o['parts']:
                L.add((o['id'], -1, 'ground'), o['parts']['shadow'], S); LG.add((o['id'], -1, 'ground'), o['parts']['shadow'], S)
            if 'glow' in o['parts']: LG.add((o['id'], -1, 'glow'), o['parts']['glow'], S)
    for o in objs:
        if not o['drawn']: continue
        S = PD.sc(o['file']) if o['file'] else None
        polys = o['parts'].get('clip', []) if S is not None else []
        # runs of consecutive polygons of one (pellet, class), in clip order
        run = []
        for q in polys + [None]:
            k = None if q is None else (q['pellet'], q.get('cls', 'body'))
            if run and (q is None or k != (run[0]['pellet'], run[0].get('cls', 'body'))):
                pel, cls = run[0]['pellet'], run[0].get('cls', 'body')
                if cls == 'wave':   # the wave is its hitbox: every pixel it changes is body (glow.type_kind), one label per attack
                    L.add_coverage((_lid(o), -1, 'body'), run, S); LG.add_coverage((_lid(o), -1, 'body'), run, S)
                else:
                    if cls == 'body': L.add((o['id'], pel, 'body'), unfaded(o['t'], run), S)
                    LG.add((o['id'], pel, cls), run, S)
                run = []
            if q is not None: run.append(q)
    if occluders is not None: L.occlude(occluders); LG.occlude(occluders)
    share, ids = L.labels(half, cut, fill=_fill_lids(L, objs))
    gl = [lid for lid in LG.ids if lid[2] == 'glow']
    gshare, gids = _glow_map(LG, gl, half)
    rows = instance_rows(objs, L, cam, half, cut) + instance_rows(objs, LG, cam, half, cut, only=set(gl))
    return dict(share=share, ids=ids, lids=list(L.ids), instances=rows, glow_share=gshare, glow_ids=gids, glow_lids=gl)


def unfaded(t, run):
    """glow.UNFADE layers (Lola's default attack rings): labelled by the bitmap's own shape, the colour transform's fade
    (layer alpha) ignored, so every ring the clip draws is in the label (the user's decision 2026-10-18)"""
    sh = glow.UNFADE.get(t)
    if not sh: return run
    out = []
    for q in run:
        if q['shape'] in sh:
            m = np.array(q['mul'], copy=True); m.reshape(-1)[3] = 255; q = dict(q, mul=m)
        out.append(q)
    return out


def _fill_lids(L, objs):
    """body label ids of FILL_HOLES types"""
    t = {o['id']: o['t'] for o in objs}; t.update({_lid(o): o['t'] for o in objs})
    return [lid for lid in L.ids if lid[2] == 'body' and t.get(lid[0]) in FILL_HOLES]


def _lid(o):
    """the instance id of an object's labels: its projectile id, or for a wave projectile its attack's id (wave.groups)"""
    return o.get('label_id', o['id'])


def _glow_map(LG, gl, half=True):
    """(share (H, W) of the glow layer that determines the pixel most, its index into gl or -1 below SOFT)"""
    H, W = (masks.H_FULL // 2, masks.W_FULL // 2) if half else (masks.H_FULL, masks.W_FULL)
    best = np.zeros((H, W), np.float32); ids = np.full((H, W), -1, np.int32); sm = LG.share_maps(half)
    for i, lid in enumerate(gl):
        x0, y0, s = sm[lid]
        ax0, ay0 = max(0, x0), max(0, y0); ax1, ay1 = min(W, x0 + s.shape[1]), min(H, y0 + s.shape[0])
        if ax1 <= ax0 or ay1 <= ay0: continue
        ss = s[ay0 - y0:ay1 - y0, ax0 - x0:ax1 - x0]; bb = best[ay0:ay1, ax0:ax1]; m = ss >= bb
        bb[m] = ss[m]; ids[ay0:ay1, ax0:ax1][m] = i
    ids[best < SOFT] = -1
    return best, ids


def instance_rows(objs, L, cam, half=True, cut=CUT, only=None):
    """geometric targets per label id: px = full-resolution pixels (x right, y down) of a 2400 x 1080 frame.
      anchor_px  projected clip origin (where the client places the projectile's picture)
      ground_px  projected ground point under it (the hitbox centre the client draws at)
      n_px       label pixels with share >= cut (owner), n_soft pixels SOFT..cut, share_sum
      centroid / box: share-weighted centre and the oriented box (axes of the pixels' covariance) of the owned pixels
      stage_angle   the clip's rotation (deg, the client's +0x204: in the 48 x 36 px stage, turned by 180)
      screen_dir    direction of flight on screen (deg, x right / y down): the logic direction at the ground point, projected"""
    smaps = L.share_maps(half); sc = 2.0 if half else 1.0; rows = []
    byid = {o['id']: o for o in objs}
    if only is None: _, owner = L.labels(half, cut, fill=_fill_lids(L, objs))
    else:   # glow rows: owner among the glow labels only
        _, gi = _glow_map(L, [l for l in L.ids if l in only], half); gl = [l for l in L.ids if l in only]
        owner = np.where(gi >= 0, np.array([L.ids.index(l) for l in gl] + [-1], np.int32)[gi], -1)
        owner[(gi >= 0) & (_glow_map(L, gl, half)[0] < cut)] = -1
    for k, lid in enumerate(L.ids):
        if only is not None and lid not in only: continue
        o = byid[lid[0]]; x0, y0, s = smaps[lid]
        ys, xs = np.nonzero(s > 0)
        own_mask = np.zeros_like(s, bool)
        H, W = owner.shape
        ax0, ay0 = max(0, x0), max(0, y0); ax1, ay1 = min(W, x0 + s.shape[1]), min(H, y0 + s.shape[0])
        if ax1 > ax0 and ay1 > ay0: own_mask[ay0 - y0:ay1 - y0, ax0 - x0:ax1 - x0] = owner[ay0:ay1, ax0:ax1] == k
        oy, ox = np.nonzero(own_mask)
        r = dict(lid=lid, id=lid[0], pellet=lid[1], part=lid[2], t=o['t'], own=o['own'],
                 anchor_px=masks.project(cam, o['anchor'][None])[0][0], ground_px=masks.project(cam, o['ground'][None])[0][0],
                 n_px=int(len(ox)), n_soft=int(((s >= SOFT) & (s < cut)).sum()), share_sum=float(s.sum()),
                 stage_angle=float(o['stage_angle']), screen_dir=screen_dir(cam, o), angle_status=o['angle_status'])
        if len(ox):
            P = np.stack([ox + x0, oy + y0], 1).astype(np.float64) * sc + sc / 2; w = s[oy, ox].astype(np.float64)
            c = (P * w[:, None]).sum(0) / w.sum(); r['centroid'] = c
            if len(ox) >= 2:
                ev, V = np.linalg.eigh(np.cov((P - c).T, aweights=w) + np.eye(2) * 1e-9); a = V[:, 1]; b = V[:, 0]
                pa = (P - c) @ a; pb = (P - c) @ b
                r['box'] = dict(center=c + a * (pa.max() + pa.min()) / 2 + b * (pb.max() + pb.min()) / 2,
                                half_len=(pa.max() - pa.min()) / 2 + sc / 2, half_wid=(pb.max() - pb.min()) / 2 + sc / 2,
                                angle=float(np.degrees(np.arctan2(a[1], a[0]))))
        rows.append(r)
    return rows


def screen_dir(cam, o):
    a = np.radians(o['angle']); g = o['ground']
    q = masks.project(cam, np.stack([g, g + np.array([np.cos(a), -np.sin(a), 0.0]) * 100.0]))[0]
    return float(np.degrees(np.arctan2(q[1, 1] - q[0, 1], q[1, 0] - q[0, 0])))


def demo(gt_path, frame, out_png):
    """labels of one recorded frame from the states alone (start frames / ambiguous angles drawn at random), next to
    the labels of the quads the client drew there: a picture for checking by eye"""
    import random, dumps, gt
    from PIL import Image
    G = gt.GTFile(gt_path); B = dumps.Battle(G.battle)
    tr = track.Tracker(track.trajectories(B), rng=random.Random(0)); WF = track.frame_weights(B); sp = None
    for f in sorted(B.meta):
        s = B.state(f)
        if s is None: sp = None; continue
        v = int(B.meta[f].get('player', 0)); tr.step(f, s, sp, WF.get(f, -1), track.team_of(s, v), None, v); sp = s
        if f == frame: break
    objs = frame_objects(tr); R = labels(objs, G.cams[frame])
    pal = np.array([[230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200], [245, 130, 48], [145, 30, 180], [70, 240, 240], [240, 50, 230]])
    img = np.zeros(R['share'].shape + (3,), np.uint8); m = R['ids'] >= 0
    img[m] = pal[R['ids'][m] % len(pal)]; soft = (R['share'] >= SOFT) & ~m; img[soft] = 70
    Image.fromarray(img).save(out_png)
    return objs, R


if __name__ == '__main__':
    import sys
    o, R = demo(sys.argv[1], int(sys.argv[2]), sys.argv[3])
    for r in R['instances']: print({k: (np.round(v, 1).tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items() if k != 'box'}, r.get('box') and round(r['box']['angle'], 1))
