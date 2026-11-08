"""Exact simulation of a projectile's MovieClip instance tree, as the client runs it:
  creation       createInstance 0xcb0ca4: every child slot gets its own instance (recursively), frame 0 applied
  frame apply    0xcb12f4 / 0xcb16dc: the frame's element list attaches children in order (matrix, colour), the
                 rest are detached; applying the current frame again does nothing
  render         0xcb13f8 -> 0xcc66a8: a MovieClip that is rendered (attached AND visible) first advances when its
                 accumulated time >= 1/fps (playback mode 0: one frame, the remainder kept), then adds dt while it is
                 playing; then renders its attached visible children (so a clip only advances while it is drawn)
  gotoAndStop    0xcb46cc (stop, time 0), gotoAndPlay 0xcb45e4 (play; time reset only when it was stopped)
  colours        child mul = parent mul * mul * 0x1010102 >> 32 (/255), add = min(255, parent add + add) (0xcc66a8)
  blend          a slot's mode (clip blend byte & 0x3f) applies to its subtree; 0 (normal) inherits the parent's
Shapes are leaves: their polygons are output with the concatenated matrix and colour.
  sim = ClipSim(S, cid); sim.goto_stop(path, f); sim.render(dt); sim.polys() -> [(path, sid, cmd, tex, uv, xy, mul4, add3, blend)]"""
import numpy as np
from sc6 import _mul

F32 = np.float32


class Inst:
    __slots__ = ('cid', 'n', 'dur', 'frame', 'time', 'state', 'kids', 'attached', 'mat', 'col', 'visible', 'blend')

    def __init__(self, S, cid):
        c = S.clips[cid]
        self.cid = cid; self.n = c['nframes']; self.dur = F32(1.0) / F32(max(1, c['fps'] & 0x7f))
        self.frame = -1; self.time = F32(0.0); self.state = 0
        self.kids = []; self.attached = []   # attached: slots in display order
        self.mat = None; self.col = None; self.visible = True; self.blend = 0
        bl = c['blend'] + [0] * (len(c['children']) - len(c['blend']))
        for slot, ch in enumerate(c['children']):
            k = Inst(S, ch) if ch in S.clips else ch    # shapes / other: the id
            if isinstance(k, Inst): k.visible = (bl[slot] & 0x40) == 0; k.blend = bl[slot] & 0x3f
            self.kids.append(k)


class ClipSim:
    def __init__(self, S, cid):
        self.S = S; self.root = Inst(S, cid); self._init(self.root)

    def _init(self, inst):
        if inst.n > 0: self.set_frame(inst, 0)
        for k in inst.kids:
            if isinstance(k, Inst): self._init(k)

    def set_frame(self, inst, f):
        if inst.frame == f: return
        inst.frame = f
        c = self.S.clips[inst.cid]; bl = c['blend'] + [0] * (len(c['children']) - len(c['blend']))
        els = self.S.frame(inst.cid, f)
        att = []; mats = {}; cols = {}
        for slot, child, m, cc, b in els:
            att.append(slot); mats[slot] = m; cols[slot] = cc
        inst.attached = att
        inst.mat = mats; inst.col = cols

    def inst_at(self, path):
        x = self.root
        for s in path: x = x.kids[s]
        return x

    def goto_stop(self, path, f):
        x = self.inst_at(path)
        if 0 <= f < x.n: self.set_frame(x, f)
        x.time = F32(0.0); x.state = 2

    def goto_play(self, path, f):
        x = self.inst_at(path)
        if 0 <= f < x.n: self.set_frame(x, f)
        if x.state != 0: x.state = 0; x.time = F32(0.0)

    def render(self, dt, inst=None):
        """one render traversal (advances the timelines of the drawn clips)"""
        x = self.root if inst is None else inst
        dt = F32(dt)
        if dt > 0 and x.time >= x.dur and x.n > 0:
            k = int(x.time / x.dur); x.time = F32(x.time - F32(k) * x.dur)
            if x.state != 1: self.set_frame(x, (x.frame + 1) % x.n)
        if x.state < 2: x.time = F32(x.time + dt)
        for slot in x.attached:
            k = x.kids[slot]
            if isinstance(k, Inst) and k.visible: self.render(dt, k)

    def polys(self):
        out = []
        self._polys(self.root, np.array([1, 0, 0, 1, 0, 0], np.float64), np.array([255, 255, 255, 255]), np.zeros(3, int), 0, (), out)
        return out

    def _polys(self, x, m, mul, add, blend, path, out):
        for slot in x.attached:
            k = x.kids[slot]
            cm = x.mat.get(slot); cc = x.col.get(slot)
            mm = m if cm is None else _mul(m, cm)
            if cc is None: mu, ad = mul, add
            else:
                mu = (mul * cc[:4].astype(np.int64) * 0x1010102) >> 32
                ad = np.minimum(255, add + cc[4:7].astype(np.int64))
            if isinstance(k, Inst):
                if not k.visible: continue
                # a blend mode set on a slot applies to everything under it; mode 0 (normal) inherits the parent's
                # [V vmask.py: shapes of an additive sub-clip are drawn with GL (ONE, ONE)]
                self._polys(k, mm, mu, ad, k.blend or blend, path + (slot,), out)
            elif k in self.S.shapes:
                c = self.S.clips[x.cid]; bl = (c['blend'][slot] if slot < len(c['blend']) else 0)
                if bl & 0x40: continue
                for ci, (tex, uv, xy) in enumerate(self.S.shapes[k]):
                    a, b, c_, d, tx, ty = mm
                    w = np.stack([a * xy[:, 0] + c_ * xy[:, 1] + tx, b * xy[:, 0] + d * xy[:, 1] + ty], 1)
                    out.append((path + (slot,), k, ci, tex, uv, w, mu.copy(), ad.copy(), (bl & 0x3f) or blend))
