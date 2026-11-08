#!/usr/bin/env python3
"""(Copied from rendclone py/posecap.py for the bullet-label round: ring shots aimed sideways / inward too, so that they
cross the screen; other_round_running copied.)
Pose / attack capture (rendclone round): one real-client battle with up to 10 chosen brawlers + skins, all under our
control, placed on an ellipse around player 0 in an open area of the map (the camera follows player 0), running a
script: idle, walking in 8 directions, attacks aimed outward, supers (super charge filled), walk + attack. The ring is
rotated between rounds so every brawler is seen facing many directions. Bots are healed by every teleport.
Output = a battle folder in the collector's format (chunkNNNN.json meta with gl_swap / mat / tick + chunkNNNN.states.npz,
map.json, map.npy) + script.json (per tick: the action of every player); the renderer records the GL stream.
  python3 framegen/posecap.py --chars Sniper,Gunslinger,... --skins SniperDefault,... --out ../data/pose/p1"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'labeller'))
import rc   # repo paths: vendor/ on sys.path, game / host / GL host defaults (labeller/rc.py)
from nbpc.instance import Instance
from nbpc import diversity as D
from nbpc import proto as P


def other_round_running():
    """(original tool: refused to start while another project's training round used the machine; not needed here)"""
    return False


def open_center(tiles, w=30, h=18, W=60, H=60):
    """tile (cx, cy) maximising open tiles ('Open' = 27000000) in a w x h window (any map size W x H)"""
    T = (np.asarray(tiles).reshape(H, W) == 27000000).astype(np.int32)
    w, h = min(w, W - 2), min(h, H - 2)
    S = np.pad(T.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    best = None
    for y in range(0, H - h):
        for x in range(0, W - w):
            n = S[y + h, x + w] - S[y, x + w] - S[y + h, x] + S[y, x]
            if best is None or n > best[0]: best = (n, x + w // 2, y + h // 2)
    return best[1], best[2], best[0] / (w * h)


def theme_area(theme, MW, MH):
    """(x0, y0, x1, y1) tiles the theme's surroundings are built for: a 3v3 theme (Brawl Ball arena, Arcade, ...) has
    MapWidth x MapHeight = 21 x 33 in location_themes.csv and its stadium / stands surround the map's first 21 x 33 tiles; on a
    bigger (Showdown) map the brawler must stand inside that area, else it is "out of bounds" next to the stands. None: whole map."""
    if not theme: return None
    import csv
    L = list(csv.reader(open(os.path.join(rc.ASSETS, 'csv_logic', 'location_themes.csv'), encoding='utf8'))); h = L[0]
    rows = [r for r in L[2:] if r and r[0]]
    r = next((r for r in rows if r[0] == str(theme)), None)
    if r is None and str(theme).isdigit() and int(theme) // 1000000 == 47 and int(theme) % 1000000 < len(rows): r = rows[int(theme) % 1000000]
    if r is None: return None
    try: tw, th = int(r[h.index('MapWidth')]), int(r[h.index('MapHeight')])
    except (ValueError, IndexError): return None
    return (0, 0, min(tw, MW), min(th, MH)) if (tw < MW or th < MH) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--chars', required=True); ap.add_argument('--skins', default='')
    ap.add_argument('--location', default='Survival612'); ap.add_argument('--theme', default='')
    ap.add_argument('--seed', type=int, default=92000000); ap.add_argument('--rounds', type=int, default=3)
    ap.add_argument('--out', required=True)
    ap.add_argument('--solo', action='store_true', help='player 0 alone on screen: idle, 4 attacks (0, 90, 180, 270 deg) 40 '
                    'ticks apart, a super, idle; the others idle at far corners (isolated attack effects, wave.py round 2)')
    ap.add_argument('--solo-angles', default='0,90,180,270', help='--solo: the attack angles (deg, comma separated)')
    ap.add_argument('--turns', action='store_true', help='ring: the brawlers take turns (one attacker at a time, the others idle)')
    ap.add_argument('--turn-gap', type=int, default=25, help='--turns: idle ticks after each turn')
    ap.add_argument('--fast-save', action='store_true', help='frames.npz uncompressed (CPU; for captures deleted soon after)')
    ap.add_argument('--id-dir', default='', help="the client's per-draw ID pass (glhost_id_rec idpass.c, on-demand mode) for every "
                    'frame saved with --pix-every: OUT_DIR/<renderer>/fNNNNNNN_{id,cv,uv,wd}.z (py/idbuf.py; bullet-label round 3)')
    ap.add_argument('--pix-every', type=int, default=0, help='real frames: every N-th attack / super / walkattack tick, 2400 x 1080 '
                    'read back, 2 x 2 box -> 1200 x 540 RGB, saved to OUT/frames.npz (bullet-label round 2)')
    ap.add_argument('--batch', default='', help='JSON list of {chars, skins, out}: one game instance, a reset per entry (no boot '
                    'per skin); with --id-dir each entry\'s ID dumps move to <out>/id/<renderer>/; prints "JOBDONE <out>" per entry')
    a = ap.parse_args()
    if other_round_running(): sys.exit('pctrain round running: no rendering instances')
    jobs = json.load(open(a.batch)) if a.batch else [dict(chars=a.chars, skins=a.skins, out=a.out)]
    if a.id_dir:   # read by the renderer at start: on-demand ID dumps of the frames we request
        os.environ['NBGL_ID_DIR'] = os.path.abspath(a.id_dir); os.environ['NBGL_ID_EVERY'] = '0'
    os.makedirs(a.out if not a.batch else os.path.dirname(os.path.abspath(a.batch)), exist_ok=True)
    inst = Instance(0, tag='pc%d' % os.getpid(), gl=True, size='2400x1080',
                    log_dir=os.path.join(a.out if not a.batch else os.path.dirname(os.path.abspath(a.batch)), 'instances'))
    idreq = [None]

    def request_id(swap):
        """ask the renderer for the ID dump of frame `swap` (IdReq {int64 w, r; int64 f[1000]} in <shm>.idreq)"""
        import mmap, struct
        if idreq[0] is None:
            fn = inst.gl_shm + '.idreq'
            if not os.path.exists(fn): return
            fh = open(fn, 'r+b'); idreq[0] = mmap.mmap(fh.fileno(), 8 * 1002)
        m = idreq[0]; w = struct.unpack_from('<q', m, 0)[0]
        struct.pack_into('<q', m, 16 + 8 * (w % 1000), int(swap)); struct.pack_into('<q', m, 0, w + 1)
    try:
        inst.start(); inst.wait_ready()
        for job in jobs:
            if other_round_running(): print('pctrain round running: batch stopped', flush=True); break
            j = argparse.Namespace(**{**vars(a), **job})
            try: swaps = capture(j, inst, request_id)
            except Exception as ex:
                if 'InstanceDead' in type(ex).__name__: raise
                print('job failed', job['out'], repr(ex)[:300], flush=True); swaps = []
            if a.batch and a.id_dir:   # this entry's dumps (f<swap>_*) into its own folder
                import glob, shutil
                for sub in glob.glob(os.path.join(os.path.abspath(a.id_dir), '*', '')):
                    dst = os.path.join(job['out'], 'id', os.path.basename(sub.rstrip('/'))); os.makedirs(dst, exist_ok=True)
                    for sw in swaps:
                        for fn in glob.glob(os.path.join(sub, 'f%07d_*' % sw)): shutil.move(fn, dst)
            print('JOBDONE', job['out'], flush=True)
    finally:
        inst.close()


def capture(a, inst, request_id):
    """one battle (reset with this entry's brawlers and skins) on a started instance -> a.out; the saved frames' swaps"""
    chars = [int(c) if c.isdigit() else c for c in a.chars.split(',')]; skins = [int(s) if s.isdigit() else s for s in a.skins.split(',')] if a.skins else [0] * len(chars)
    n = len(chars)
    div = D.make_div(location=a.location, theme=a.theme or None, chr=chars + [chars[0]] * (10 - n), skin=[s for s in skins] + [skins[0]] * (10 - n))
    os.makedirs(a.out, exist_ok=True)
    meta, states, script = [], [], []
    frames = {}
    t0 = time.time()
    if True:
        inst.reset(a.seed, div=div, gas=0, flags=P.F_STATE)
        info = inst.battle_info(); tiles = inst.map_tiles()
        D.save_battle_map(a.out, info, tiles, extra=dict(seed=a.seed, kind='posecap'))
        MW, MH = (int(info.get('map_w') or 60), int(info.get('map_h') or 60)) if info else (60, 60)   # map size in tiles (3v3 maps are smaller)
        AREA = theme_area(a.theme, MW, MH)
        if AREA:   # 3v3 theme: everything happens on its pitch (the ring around the pitch's centre)
            cx, cy, frac = (AREA[0] + AREA[2]) // 2, (AREA[1] + AREA[3]) // 2, 0.0; print('theme area (tiles)', AREA)
        else: cx, cy, frac = open_center(tiles, W=MW, H=MH)
        X0, Y0 = (cx + 0.5) * 300, (cy + 0.5) * 300
        print('open centre', cx, cy, 'open fraction %.2f' % frac)
        others = list(range(1, 10))

        def ring(rot):
            pos = {0: (X0, Y0)}
            for k, i in enumerate(others):
                ang = rot + 2 * np.pi * k / len(others)
                pos[i] = (X0 + 2600 * np.cos(ang), Y0 + 1500 * np.sin(ang))
            return {i: (int(x), int(y)) for i, (x, y) in pos.items()}

        def outward(pos, i):
            if i == 0: return float((step.n * 37) % 360)   # player 0 (the camera's brawler): rotating aim
            dx, dy = pos[i][0] - X0, pos[i][1] - Y0
            return float(np.degrees(np.arctan2(dy, dx)))

        def step(actions, tag):
            step.n += 1
            want = a.pix_every and tag in ('attack', 'super', 'walkattack') and step.n % a.pix_every == 0
            if os.environ.get('POSECAP_IDLE_LOCAL'):   # the local player idle: its input reaches the simulation through the
                actions = dict(actions); actions[0] = (-1, 0, 0.0, 0.0)   # client (wall-clock frame time): not repeatable
            r = inst.step(actions=actions, ticks=1, flags=P.F_STATE | P.F_PIXELS, native_mask=0, pix=(2400, 1080) if want else (300, 135))
            rsp = inst.resp
            if want:
                px = inst.pixels_rgba()
                if px is not None and px.shape[:2] == (1080, 2400):
                    im = px[::-1, :, :3].astype(np.uint16)   # GL readback is bottom-up
                    frames[int(rsp.reserved[6])] = ((im[0::2, 0::2] + im[1::2, 0::2] + im[0::2, 1::2] + im[1::2, 1::2] + 2) // 4).astype(np.uint8)
                    if a.id_dir: request_id(int(rsp.reserved[6]))
            meta.append(dict(tick=int(rsp.tick), gl_swap=int(rsp.reserved[6]), mat=[float(v) for v in rsp.cam_mat], phase=tag,
                             cam=[float(rsp.cam_x), float(rsp.cam_y)], local=int(rsp.local_index), player=0))
            states.append(np.frombuffer(inst.state_bytes(), np.int32).copy())
            script.append(dict(tick=int(rsp.tick), phase=tag, act={str(i): list(v) for i, v in actions.items()}))

        step.n = 0
        inst.step(ticks=40, flags=P.F_STATE, native_mask=0)
        allp = list(range(10))
        if a.solo:
            far = [(1, 1), (MW - 2, 1), (1, MH - 2), (MW - 2, MH - 2), (MW // 2, 1), (MW // 2, MH - 2), (1, MH // 2), (MW - 2, MH // 2), (MW // 4, 1)]
            if AREA and MW - 2 > AREA[2] + 12:   # 3v3 theme: the idle players wait well outside its pitch (off screen)
                gx = [AREA[2] + 12 + k * ((MW - 2 - (AREA[2] + 12)) // 2) for k in range(3)]
                far = [(x, y) for y in (1, MH // 2, MH - 2) for x in gx]
            Tl = (np.asarray(tiles).reshape(MH, MW) == 27000000)   # the open tile farthest from any non-open tile
            rx0, ry0, rx1, ry1 = AREA if AREA else (0, 0, MW, MH)
            mid = (rx0 + rx1) // 2
            ys, xs = np.nonzero(~Tl); best = (-1, mid, (ry0 + ry1) // 2)
            for ty in range(ry0 + 4, ry1 - 4):
                for tx in (range(mid - 2, mid + 3) if AREA else range(rx0 + 4, rx1 - 4)):   # 3v3 theme: the pitch's middle columns (the camera then shows both sides, as in the real mode)
                    if not Tl[ty, tx]: continue
                    dd = int(np.min(np.maximum(np.abs(xs - tx), np.abs(ys - ty)))) if len(xs) else 99
                    if dd > best[0]: best = (dd, tx, ty)
            X0, Y0 = (best[1] + 0.5) * 300, (best[2] + 0.5) * 300; print('solo spot', best)
            pos = {0: (int(X0), int(Y0))}
            pos.update({i: (int((far[i - 1][0] + 0.5) * 300), int((far[i - 1][1] + 0.5) * 300)) for i in range(1, 10)})
            SH = int(os.environ.get('POSECAP_SHOOTER', '0'))   # 10-28: another (enemy, red-tinted) player fires next to the camera's brawler
            if SH:
                dx, dy = (float(v) for v in os.environ.get('POSECAP_SHOOTER_OFF', '0,450').split(','))
                pos[SH] = (int(X0 + dx), int(Y0 + dy)); print('enemy shooter', SH, pos[SH])
            inst.setup(positions=pos, ai_off=allp, unprotect=allp)
            idle = {i: (-1, 0, 0.0, 0.0) for i in allp}
            for _ in range(20): step(idle, 'idle')
            for ang in [float(x) for x in a.solo_angles.split(',') if x]:
                inst.setup(positions=pos, ai_off=allp)
                HOLD = int(os.environ.get('POSECAP_HOLD', '1'))   # 11-03: >1 holds the attack button for HOLD ticks (FireDude stream)
                for k in range(int(os.environ.get('POSECAP_ATTACK_TICKS', '40'))): acts = dict(idle); acts[SH] = (-1, 1 if k < HOLD else 0, ang, 0.0); step(acts, 'attack')
            inst.setup(positions=pos, ai_off=allp, super_fill=allp)
            for k in range(70): acts = dict(idle); acts[SH] = (-1, 2 if k == 0 else 0, 45.0, 0.7); step(acts, 'super')
            a.rounds = 0
        for rnd in range(a.rounds if a.turns else 0):
            # --turns: the ring's brawlers take turns, ONE attacker at a time (the others stand still), so no two brawlers' attacks
            # are on screen together and every label belongs to the one brawler that is firing (repo, 2026-11-08)
            rot = rnd * 2 * np.pi / (3 * len(others))
            pos = ring(rot); idle = {i: (-1, 0, 0.0, 0.0) for i in allp}
            inst.setup(positions=pos, ai_off=allp, unprotect=allp)
            for _ in range(30): step(idle, 'idle')
            for i in allp:
                inst.setup(positions=pos, ai_off=allp)
                for off in (90, -90, 0):   # attacks: sideways (cross the screen) and outward
                    for t in range(18):
                        acts = dict(idle); acts[i] = (-1, 1 if t == 0 else 0, outward(pos, i) + (off if i else 0), 0.0); step(acts, 'attack')
                inst.setup(positions=pos, ai_off=allp, super_fill=allp)
                for t in range(40):
                    acts = dict(idle); acts[i] = (-1, 2 if t == 0 else 0, outward(pos, i) + (90 if i else 0), 0.7); step(acts, 'super')
                for _ in range(a.turn_gap): step(idle, 'idle')   # the turn's shots end before the next brawler fires
        for rnd in range(0 if a.turns else a.rounds):
            rot = rnd * 2 * np.pi / (3 * len(others))
            pos = ring(rot)
            inst.setup(positions=pos, ai_off=allp, unprotect=allp)
            for _ in range(30): step({i: (-1, 0, 0.0, 0.0) for i in allp}, 'idle')
            for d8 in range(0, 16, 2):   # walk 8 directions
                inst.setup(positions=pos, ai_off=allp)
                for _ in range(14): step({i: (d8, 0, 0.0, 0.0) for i in allp}, 'walk%d' % d8)
            inst.setup(positions=pos, ai_off=allp)
            for k, off in enumerate((90, -90, 150, -150, 0, 180)):   # attacks: sideways / inward cross the screen
                for t in range(18):
                    acts = {i: (-1, 1 if t == 0 else 0, outward(pos, i) + (off if i else 0), 0.0) for i in allp}
                    step(acts, 'attack')
            for k, off in enumerate((90, -90, 160)):   # supers
                inst.setup(positions=pos, ai_off=allp, super_fill=allp)
                for t in range(30):
                    acts = {i: (-1, 2 if t == 0 else 0, outward(pos, i) + (off if i else 0), 0.7) for i in allp}
                    step(acts, 'super')
            inst.setup(positions=pos, ai_off=allp)
            for k in range(4):   # walking while attacking
                d8 = (rnd * 4 + k * 4) % 16
                for t in range(16):
                    acts = {i: (d8, 1 if t % 8 == 0 else 0, outward(pos, i), 0.0) for i in allp}
                    step(acts, 'walkattack')
        if a.id_dir: inst.step(ticks=4, flags=P.F_STATE, native_mask=0)   # the renderer writes requested dumps at a later frame end
        # chunk files (the collector's format, without frames)
        off = np.cumsum([0] + [len(s) for s in states]).astype(np.int64)
        np.savez(os.path.join(a.out, 'chunk0000.states.npz'), states=np.concatenate(states), offsets=off)
        json.dump(dict(shape=[len(meta), 0, 0, 3], codec='none', meta=meta), open(os.path.join(a.out, 'chunk0000.json'), 'w'))
        json.dump(dict(script=script, centre=[X0, Y0], chars=chars, skins=skins, info=info), open(os.path.join(a.out, 'script.json'), 'w'))
        if frames:
            ks = sorted(frames); (np.savez if a.fast_save else np.savez_compressed)(os.path.join(a.out, 'frames.npz'), swap=np.array(ks), img=np.stack([frames[k] for k in ks]))
        print('posecap done', len(meta), 'ticks, %.0f s' % (time.time() - t0), 'skins', info['skin'][:10] if info else None, flush=True)
    return sorted(frames)


if __name__ == '__main__':
    main()
