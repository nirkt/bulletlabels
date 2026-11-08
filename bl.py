#!/usr/bin/env python3
"""bulletlabels: frames of the real Null's Brawl client + exact labels of every attack, super, spawnable and zone.

  python3 bl.py doctor                                   check the setup (game files, binaries, Python packages)
  python3 bl.py skins [--brawler Knight]                 list brawlers / skins (index, skin id, skin name, brawler)
  python3 bl.py capture --skin KnightPinata OUT          capture one battle (solo: the brawler alone, attacks + supers)
  python3 bl.py capture --ring --skin A,B,C,... OUT      ring: up to 10 brawlers around player 0 (NOT REVIEWED YET, see README)
  python3 bl.py label CAPTURE DATASET [--motion]         labels -> images/ masks/ labels/ (+ overlays/ with --overlay)
  python3 bl.py classes DATASET --mode brawler_team      class scheme: all | team | kind[_team] | brawler[_team|_kind_team] | skin[...] | type[_team]
  python3 bl.py pages CAPTURE OUT_DIR                    the review contact sheets (raw | outlined), as used to check the rules
  python3 bl.py run --skin KnightPinata,RuffsBrawloween DATASET [--ring] [--keep]
                                                         capture + label each, captures deleted after (unless --keep)
Captures: ~30 s per solo battle, ~0.5-1 GB each on disk (frames + ID pass) until deleted. See README.md."""
import argparse, json, os, shutil, subprocess, sys, time

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO, 'labeller'))


def skins_table():
    """[(index, skin id, skin name, brawler)] of every skin the capture can use (data/sweep_skins.json)"""
    S = json.load(open(os.path.join(REPO, 'data', 'sweep_skins.json')))
    return [(i, int(r[0]), r[1], r[2]) for i, r in enumerate(S)]


def resolve_skin(x):
    """skin name, skin id or table index -> (skin id, skin name, brawler)"""
    T = skins_table()
    for i, sid, nm, ch in T:
        if x == nm or x == str(sid) or (x.isdigit() and int(x) == i and len(x) <= 3): return sid, nm, ch
    lo = [r for r in T if r[2].lower() == x.lower()]
    if lo: return lo[0][1:]
    raise SystemExit('unknown skin %r (python3 bl.py skins)' % x)


def cmd_doctor(a):
    import rc
    ok = True
    def chk(cond, what, hint=''):
        nonlocal ok
        print(('  ok   ' if cond else '  MISSING ') + what + ('' if cond or not hint else '   -> ' + hint)); ok &= bool(cond)
    print('repo', REPO); print('game', rc.GAME)
    chk(os.path.exists(rc.LIBG), 'game library %s' % rc.LIBG, './setup.sh (downloads and unpacks the APK)')
    chk(os.path.isdir(os.path.join(rc.ASSETS, 'csv_logic')), 'game assets %s' % rc.ASSETS, './setup.sh')
    if os.path.exists(rc.LIBG):
        import hashlib
        h = hashlib.sha256(open(rc.LIBG, 'rb').read()).hexdigest()
        want = open(os.path.join(REPO, 'native', 'LIBG_SHA256')).read().split()[0]
        chk(h == want, 'game version (libg.so sha256 %s...)' % h[:12], 'this repo needs the APK whose libg.so sha256 is %s' % want)
    chk(os.access(os.environ['NBHOST_BIN'], os.X_OK), 'game host %s' % os.environ['NBHOST_BIN'])
    chk(os.access(os.environ['NBGL_BIN'], os.X_OK), 'GL host %s' % os.environ['NBGL_BIN'], 'native/glhost/build.sh')
    q = os.path.exists(rc.QEMU_NB) or shutil.which('qemu-aarch64')
    chk(q, 'qemu (%s)' % (rc.QEMU_NB if os.path.exists(rc.QEMU_NB) else shutil.which('qemu-aarch64') or 'none'), 'native/qemu/build.sh, or: sudo apt install qemu-user')
    chk(shutil.which('unshare'), 'unshare (util-linux)')
    chk(bool(os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')), 'a display for the GL host (DISPLAY)', 'run under a desktop session or xvfb-run -a')
    for m in ('numpy', 'scipy', 'PIL', 'zstandard', 'flatbuffers', 'texture2ddecoder'):
        try: __import__(m); chk(True, 'python ' + m)
        except Exception: chk(False, 'python ' + m, 'pip install -r requirements.txt')
    print('ready' if ok else 'not ready'); return 0 if ok else 1


def cmd_skins(a):
    for i, sid, nm, ch in skins_table():
        if a.brawler and a.brawler.lower() != ch.lower(): continue
        print('%3d  %d  %-32s %s' % (i, sid, nm, ch))


def capture(skins, out, ring=False, angles='0,90,180,270', seed=94000030, rounds=1, mixed=False, log=print):
    """run framegen/posecap.py; skins = [(skin id, name, brawler)] (solo: the first one) -> True when frames were saved"""
    out = os.path.abspath(out)
    if ring: log('NOTE: ring mode has NOT been reviewed yet (solo mode is the reviewed one); its labels can be wrong. Default: '
                 'the brawlers take turns (one attacker at a time); --mixed: everyone at once.')
    if ring:
        cyc = [skins[k % len(skins)] for k in range(10)]   # 10 ring places: the chosen skins in turn
        chars = [s[2] for s in cyc]; sk = [str(s[0]) for s in cyc]
        args = ['--chars', ','.join(chars), '--skins', ','.join(sk), '--rounds', str(rounds)] + ([] if mixed else ['--turns'])
    else:
        sid, nm, ch = skins[0]
        args = ['--chars', ','.join([ch] * 10), '--skins', ','.join([str(sid)] * 10), '--solo', '--solo-angles', angles]
    cmd = [sys.executable, os.path.join(REPO, 'framegen', 'posecap.py')] + args + ['--seed', str(seed), '--pix-every', '2', '--out', out,
                                                                                  '--id-dir', os.path.join(out, 'id'), '--fast-save']
    t0 = time.time(); r = subprocess.run(cmd, cwd=os.path.join(REPO, 'framegen'), capture_output=True, text=True)
    good = os.path.exists(os.path.join(out, 'frames.npz'))
    log('capture %s: %s (%.0f s)' % (os.path.basename(out), 'ok' if good else 'FAILED', time.time() - t0))
    if not good: log((r.stdout + r.stderr)[-3000:])
    return good


def cmd_capture(a):
    sk = [resolve_skin(x) for x in a.skin.split(',')]
    return 0 if capture(sk, a.out, ring=a.ring, angles=a.angles, seed=a.seed, rounds=a.rounds, mixed=a.mixed) else 1


def _motion_env(on):
    if on:
        for k in ('IDLABELS_MOTION_DIR', 'IDLABELS_MOTION_END', 'IDLABELS_MOTION_ENDTIME'): os.environ.setdefault(k, '1')


def cmd_label(a):
    _motion_env(a.motion)
    import export
    export.export_capture(a.capture, a.out, motion=a.motion, overlay=a.overlay, every=a.every)


def cmd_classes(a):
    import classes
    classes.build(a.dataset, a.mode, kinds=set(a.kinds.split(',')) if a.kinds else None, masks=a.masks, class_list=a.class_list)


def cmd_pages(a):
    import seqsheet
    os.makedirs(a.out, exist_ok=True)
    for p in seqsheet.exact_pages(os.path.join(a.out, os.path.basename(os.path.abspath(a.capture).rstrip('/'))), a.capture): print(p)


def cmd_run(a):
    _motion_env(a.motion)
    import export
    caps = os.path.join(os.path.abspath(a.out), '_captures'); os.makedirs(caps, exist_ok=True)
    groups = [[resolve_skin(x) for x in a.skin.split(',')]] if a.ring else [[resolve_skin(x)] for x in a.skin.split(',')]
    for g in groups:
        cap = os.path.join(caps, ('ring_' if a.ring else '') + '_'.join(s[1] for s in g)[:80] + '_s%d' % a.seed)
        if capture(g, cap, ring=a.ring, angles=a.angles, seed=a.seed, rounds=a.rounds, mixed=a.mixed):
            export.export_capture(cap, a.out, motion=a.motion, overlay=a.overlay, every=a.every)
        if not a.keep: shutil.rmtree(cap, ignore_errors=True)
    if not a.keep: shutil.rmtree(caps, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('doctor').set_defaults(fn=cmd_doctor)
    p = sub.add_parser('skins'); p.add_argument('--brawler', default=''); p.set_defaults(fn=cmd_skins)
    for name, fn in (('capture', cmd_capture), ('run', cmd_run)):
        p = sub.add_parser(name); p.add_argument('--skin', required=True, help='skin names / ids / indices, comma separated')
        p.add_argument('out'); p.add_argument('--ring', action='store_true'); p.add_argument('--angles', default='0,90,180,270', help='solo: attack angles')
        p.add_argument('--seed', type=int, default=94000030); p.add_argument('--rounds', type=int, default=1, help='ring: rounds')
        p.add_argument('--mixed', action='store_true', help='ring: everyone attacks at once (default: brawlers take turns)')
        if name == 'run':
            p.add_argument('--motion', action='store_true'); p.add_argument('--overlay', action='store_true'); p.add_argument('--every', type=int, default=1)
            p.add_argument('--keep', action='store_true', help='keep the captures (OUT/_captures)')
        p.set_defaults(fn=fn)
    p = sub.add_parser('label'); p.add_argument('capture'); p.add_argument('out'); p.add_argument('--motion', action='store_true')
    p.add_argument('--overlay', action='store_true'); p.add_argument('--every', type=int, default=1, help='every N-th labelled frame'); p.set_defaults(fn=cmd_label)
    p = sub.add_parser('classes', help='class scheme for a dataset (labeller/classes.py)'); p.add_argument('dataset')
    p.add_argument('--mode', required=True, help='all | team | kind | kind_team | brawler | brawler_team | brawler_kind_team | skin | skin_team | skin_kind_team | type | type_team')
    p.add_argument('--kinds', default='', help='only these kinds, e.g. attack,super'); p.add_argument('--masks', action='store_true')
    p.add_argument('--class-list', default='', help='reuse the ids of an earlier classes.json'); p.set_defaults(fn=cmd_classes)
    p = sub.add_parser('pages'); p.add_argument('capture'); p.add_argument('out'); p.set_defaults(fn=cmd_pages)
    a = ap.parse_args(); sys.exit(a.fn(a) or 0)


if __name__ == '__main__':
    main()
