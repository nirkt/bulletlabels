"""Shared helpers of the debug tools: walk a capture frame by frame with the labeller (the same loop as labeller/export.py)."""
import glob, os, random, sys
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'labeller'))
KIND_COLOR = {1: (255, 0, 0), 2: (255, 0, 255), 3: (0, 140, 255), 4: (255, 230, 0), 5: (0, 255, 255), 6: (255, 140, 0),
              7: (255, 255, 255), 8: (255, 150, 200), 9: (150, 90, 255)}
KIND_CODE = dict(attack=1, super=2, spawnable=3, zone=4, gadget=5, starpower=5, spawnable_attack=6, target=7, throwable=8, pull=9)


def edge(m):
    import numpy as np
    e = np.zeros_like(m)
    e[1:] |= m[1:] & ~m[:-1]; e[:-1] |= m[:-1] & ~m[1:]; e[:, 1:] |= m[:, 1:] & ~m[:, :-1]; e[:, :-1] |= m[:, :-1] & ~m[:, 1:]
    return e


def frames(cap, want_frame=None):
    """yields (frame, state, objs, P, img) for every frame of the capture that has an ID pass; want_frame(frame, tick) -> bool
    picks which ones are labelled (the tracker still steps through all of them)."""
    import numpy as np
    import rc, dumps, track, bulletlabels as BL, idbuf, idlabels
    sys.setrecursionlimit(5000)
    idd = sorted(glob.glob(os.path.join(cap, 'id', '*', '')))[0]; have = set(idbuf.frames(idd))
    z = np.load(os.path.join(cap, 'frames.npz')); want = {int(s): i for i, s in enumerate(z['swap'])}; IMG = z['img']
    B = dumps.Battle(cap); fr = sorted(B.meta); tr = track.Tracker(track.trajectories(B), rng=random.Random(0)); WF = track.frame_weights(B)
    sp = None; pt = None
    for f in fr:
        s = B.state(f)
        if s is None: sp = None; continue
        if pt is not None and s['tick'] - pt > 1: tr.fill(s['tick'] - pt - 1); sp = None
        v = int(B.meta[f].get('player', 0)); tr.step(f, s, sp, WF.get(f, -1), track.team_of(s, v), None, v); sp = s; pt = s['tick']
        if f not in want or f not in have: continue
        if want_frame and not want_frame(f, int(s['tick'])): continue
        cam = np.array(B.meta[f]['mat'], np.float64).reshape(4, 4).T; objs = BL.frame_objects(tr); R = idbuf.load(idd, f)
        img = IMG[want[f]]
        P = idlabels.labels(objs, cam, R, state=s, ended=list(tr.ended.values()), img=img)
        yield f, s, objs, P, img
