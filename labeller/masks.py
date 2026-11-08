"""Per-pixel labels of projectiles, from the exact drawn geometry (projdraw.place) and the game's own textures (sctex.py),
rasterised like the client's GPU pass and reduced to the training resolution. Definitions: LABEL_SPEC.md.

What one sprite polygon does to a framebuffer pixel (client shaders, rendclone runs/dd_p0_s29000844/shaders
1143882731.vs / 2759865535.fs, and the GL blend states of the draws, out/gt/*.npz 'blends') [V]:
    v_color = (mul.rgb, 1) * mul.a          (vertex shader: premultiplied there)
    v_color_2 = add * mul.a                 (add.a = 0 on every projectile quad, checked on pose_p1)
    src = tex * v_color + v_color_2 * tex.a (texture premultiplied: ASTC payload as stored)
    normal   (ONE, ONE_MINUS_SRC_ALPHA):    F = src.rgb + (1 - src.a) B
    add      (ONE, ONE):                    F = src.rgb + B
    multiply (DST_COLOR, ONE_MINUS_SRC_ALPHA): F = (src.rgb + 1 - src.a) B
    screen   (ONE, ONE_MINUS_SRC_COLOR):    F = src.rgb + (1 - src.rgb) B
Every case is F = C + tau * B per channel (C the part's own light, tau its transmittance), so a stack of parts in draw order
composes exactly into the same form, and the share of the pixel that each labelled part owns is background-independent:
    own_k  = C_k  x prod(tau of the later parts)      (light it adds that is still visible)
    hide_k = (1 - tau_k) x prod(tau of the later)     (background it removes that is still removed)
    share_k = max over channels of max(own_k, hide_k)     in [0, 1]
The 2400 x 1080 frame is reduced 2:1 by a box filter (rendclone NOTES 5: PIL BOX on the GPU); own and hide are linear in
the sub-pixels, so the half-resolution share is computed from the 2 x 2 means.
  L = Layers(cam)               cam: 4x4 PV matrix (render coords -> clip), e.g. GTFile.cams[f] or the generator's pv
  L.add(lid, placed_polys)      lid: any hashable label id (projectile id, pellet, part); polys in draw order
  share, ids = L.labels(half=True)   (540, 1200) float32 share of the front-most owner, int32 index into L.ids (-1 none)"""
import numpy as np
import sctex, projdraw as PD

W_FULL, H_FULL = 2400, 1080
GLB = {'normal': 0, 0: 0, 'add': 8, 8: 8, 'multiply': 3, 3: 3, 'screen': 4, 4: 4}


def project(cam, P):
    """render coords (n, 3) -> full-resolution screen px (n, 2), x right, y DOWN, and clip w (n,)"""
    h = np.c_[np.asarray(P, np.float64), np.ones(len(P))] @ np.asarray(cam, np.float64).T
    w = h[:, 3]; nx = h[:, 0] / w; ny = h[:, 1] / w
    return np.stack([(nx + 1) * 0.5 * W_FULL, (1 - ny) * 0.5 * H_FULL], 1), w


def _tex_of(S, k):
    return sctex.texture(S, k)


def _sample(tex, u, v):
    """bilinear RGBA (GL_LINEAR, clamp) at normalised texture coordinates -> (n, 4) in [0, 1]"""
    h, w = tex.shape[:2]
    x = u * w - 0.5; y = v * h - 0.5
    x0 = np.floor(x).astype(np.int64); y0 = np.floor(y).astype(np.int64); fx = (x - x0)[:, None]; fy = (y - y0)[:, None]
    def g(yy, xx): return tex[np.clip(yy, 0, h - 1), np.clip(xx, 0, w - 1)].astype(np.float32)
    r = g(y0, x0) * (1 - fx) * (1 - fy) + g(y0, x0 + 1) * fx * (1 - fy) + g(y0 + 1, x0) * (1 - fx) * fy + g(y0 + 1, x0 + 1) * fx * fy
    return r / 255.0


def raster_poly(S, q, cam, clip_box=None):
    """one placed polygon (dict of projdraw.place) -> (x0, y0, C (h, w, 3), tau (h, w, 3)) over its pixel box, or None.
    Pixels whose centre lies inside one of its triangles are shaded. The client draws a shape command's points as a
    triangle strip, (0, 1, 2), (1, 2, 3), ... (GL_TRIANGLE_STRIP of quads joined by degenerate vertices, rendclone
    py/quads.py; the stored point order is the strip order: a 4-point command is TL, TR, BL, BR, zero signed area)
    [V round 2: real frames, out/glow/; a fan (0, t, t + 1) misses half of every such quad]."""
    xy, w = project(cam, q['world'])
    if (w <= 0).any(): return None
    x0 = int(np.floor(xy[:, 0].min())); x1 = int(np.ceil(xy[:, 0].max())); y0 = int(np.floor(xy[:, 1].min())); y1 = int(np.ceil(xy[:, 1].max()))
    bx0, by0, bx1, by1 = clip_box or (0, 0, W_FULL, H_FULL)
    x0 = max(x0, bx0); y0 = max(y0, by0); x1 = min(x1, bx1); y1 = min(y1, by1)
    if x1 <= x0 or y1 <= y0: return None
    gx, gy = np.meshgrid(np.arange(x0, x1) + 0.5, np.arange(y0, y1) + 0.5)
    px = gx.ravel(); py = gy.ravel()
    uvw = np.asarray(q['uv'], np.float64) / 65535.0
    iw = 1.0 / w
    U = np.full(len(px), np.nan); V = np.full(len(px), np.nan); hit = np.zeros(len(px), bool)
    n = len(xy)
    for t in range(n - 2):   # strip (t, t + 1, t + 2)
        a, b, c = xy[t], xy[t + 1], xy[t + 2]
        d = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        if abs(d) < 1e-12: continue
        l1 = ((px - a[0]) * (c[1] - a[1]) - (py - a[1]) * (c[0] - a[0])) / d
        l2 = ((b[0] - a[0]) * (py - a[1]) - (b[1] - a[1]) * (px - a[0])) / d
        l0 = 1 - l1 - l2
        m = (l0 >= 0) & (l1 >= 0) & (l2 >= 0) & ~hit
        if not m.any(): continue
        # perspective-correct attributes (the plane faces the camera, so w is nearly constant; done properly anyway)
        k0, k1, k2 = l0[m] * iw[t], l1[m] * iw[t + 1], l2[m] * iw[t + 2]; s = k0 + k1 + k2
        U[m] = (k0 * uvw[t, 0] + k1 * uvw[t + 1, 0] + k2 * uvw[t + 2, 0]) / s
        V[m] = (k0 * uvw[t, 1] + k1 * uvw[t + 1, 1] + k2 * uvw[t + 2, 1]) / s
        hit |= m
    if not hit.any(): return None
    tx = _sample(_tex_of(S, q['tex']), U[hit], V[hit])
    mul = np.asarray(q['mul'], np.float32) / 255.0; add = np.asarray(q['add'], np.float32) / 255.0
    a = mul[3]
    src_rgb = tx[:, :3] * (mul[:3] * a) + (add * a) * tx[:, 3:4]
    src_a = tx[:, 3:4] * a
    bl = GLB.get(q['blend'], 0)
    if bl == 8: C = src_rgb; tau = np.ones_like(src_rgb)
    elif bl == 3: C = np.zeros_like(src_rgb); tau = src_rgb + 1 - src_a
    elif bl == 4: C = src_rgb; tau = 1 - src_rgb
    else: C = src_rgb; tau = np.repeat(1 - src_a, 3, 1)
    h_, w_ = y1 - y0, x1 - x0
    Cf = np.zeros((h_ * w_, 3), np.float32); Tf = np.ones((h_ * w_, 3), np.float32)
    Cf[hit] = C; Tf[hit] = tau
    return x0, y0, Cf.reshape(h_, w_, 3), Tf.reshape(h_, w_, 3)


class Layers:
    """composites labelled parts in draw order; keeps per label id the still-visible own light and hidden background"""
    def __init__(self, cam):
        self.cam = np.asarray(cam, np.float64); self.ids = []; self._L = {}

    def _get(self, lid, box):
        x0, y0, x1, y1 = box
        if lid not in self._L:
            self._L[lid] = [x0, y0, np.zeros((y1 - y0, x1 - x0, 3), np.float32), np.zeros((y1 - y0, x1 - x0, 3), np.float32)]
            self.ids.append(lid)
        e = self._L[lid]
        ex0, ey0, eo, eh = e; ey1 = ey0 + eo.shape[0]; ex1 = ex0 + eo.shape[1]
        if x0 < ex0 or y0 < ey0 or x1 > ex1 or y1 > ey1:   # grow the window
            nx0, ny0, nx1, ny1 = min(x0, ex0), min(y0, ey0), max(x1, ex1), max(y1, ey1)
            no = np.zeros((ny1 - ny0, nx1 - nx0, 3), np.float32); nh = np.zeros_like(no)
            no[ey0 - ny0:ey1 - ny0, ex0 - nx0:ex1 - nx0] = eo; nh[ey0 - ny0:ey1 - ny0, ex0 - nx0:ex1 - nx0] = eh
            e[:] = [nx0, ny0, no, nh]
        return e

    def add(self, lid, placed, S):
        """placed polygons of one label id (one projectile / pellet / part), in draw order; S: their sprite file"""
        for q in placed:
            r = raster_poly(S, q, self.cam)
            if r is None: continue
            x0, y0, C, T = r; y1 = y0 + C.shape[0]; x1 = x0 + C.shape[1]
            for k, e in self._L.items():   # the earlier parts under this polygon: attenuated
                if k == lid: continue
                ex0, ey0, eo, eh = e
                ax0, ay0 = max(x0, ex0), max(y0, ey0); ax1, ay1 = min(x1, ex0 + eo.shape[1]), min(y1, ey0 + eo.shape[0])
                if ax1 <= ax0 or ay1 <= ay0: continue
                t = T[ay0 - y0:ay1 - y0, ax0 - x0:ax1 - x0]
                eo[ay0 - ey0:ay1 - ey0, ax0 - ex0:ax1 - ex0] *= t; eh[ay0 - ey0:ay1 - ey0, ax0 - ex0:ax1 - ex0] *= t
            e = self._get(lid, (x0, y0, x1, y1)); ex0, ey0, eo, eh = e
            so = eo[y0 - ey0:y1 - ey0, x0 - ex0:x1 - ex0]; sh = eh[y0 - ey0:y1 - ey0, x0 - ex0:x1 - ex0]
            so *= T; so += C; sh *= T; sh += 1 - T

    def add_coverage(self, lid, placed, S, eps=1.0 / 255):
        """placed polygons labelled by COVERAGE: every full-resolution pixel a polygon changes by at least eps (its own
        light or its hidden background, any channel; default one 8-bit level) is owned outright (hide = 1, own = 0).
        Used for 'wave' types (glow.type_kind): the drawing is exactly the hitbox, so every pixel it touches is bullet."""
        for q in placed:
            r = raster_poly(S, q, self.cam)
            if r is None: continue
            x0, y0, C, T = r
            m = (C.max(-1) >= eps) | ((1 - T).max(-1) >= eps)
            if not m.any(): continue
            Tc = np.where(m[..., None], 0.0, 1.0).astype(np.float32)
            self._add_raw(lid, x0, y0, np.zeros_like(C), Tc)

    def add_triangles(self, lid, tris, opacity=1.0, uv=None, tex=None, channel=2, additive=False):
        """a triangle set, world render coords (m, 3, 3): pixel centres inside a triangle get hide = opacity, own = 0
        (an opaque mesh owns its pixels whatever its colour; LABEL_SPEC 2). With uv (m, 3, 2) and tex (an RGBA texture):
        hide = opacity x tex[channel] at the perspective-correct uv (repeat wrap, bilinear): the uber shader's
        opacity = opacityTex.b of premultiplied mesh materials."""
        if not len(tris): return
        xy, w = project(self.cam, tris.reshape(-1, 3))
        if (w <= 0).any(): return
        xy = xy.reshape(-1, 3, 2)
        x0 = max(0, int(np.floor(xy[..., 0].min()))); x1 = min(W_FULL, int(np.ceil(xy[..., 0].max())))
        y0 = max(0, int(np.floor(xy[..., 1].min()))); y1 = min(H_FULL, int(np.ceil(xy[..., 1].max())))
        if x1 <= x0 or y1 <= y0: return
        gx, gy = np.meshgrid(np.arange(x0, x1) + 0.5, np.arange(y0, y1) + 0.5); px = gx.ravel(); py = gy.ravel()
        hit = np.zeros(len(px), bool); op = np.zeros(len(px), np.float32)
        wv = w.reshape(-1, 3)
        for ti, (a, b, c) in enumerate(xy):
            d = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
            if abs(d) < 1e-12: continue
            bx0, bx1 = min(a[0], b[0], c[0]), max(a[0], b[0], c[0]); by0, by1 = min(a[1], b[1], c[1]), max(a[1], b[1], c[1])
            m = (px >= bx0) & (px <= bx1) & (py >= by0) & (py <= by1)
            if not m.any(): continue
            l1 = ((px[m] - a[0]) * (c[1] - a[1]) - (py[m] - a[1]) * (c[0] - a[0])) / d
            l2 = ((b[0] - a[0]) * (py[m] - a[1]) - (b[1] - a[1]) * (px[m] - a[0])) / d
            inside = (l1 >= 0) & (l2 >= 0) & (l1 + l2 <= 1)
            idx = np.nonzero(m)[0][inside]
            if not len(idx): continue
            if tex is not None and uv is not None:   # perspective-correct uv, then the texture (the front-most triangle is not resolved: max)
                L1, L2 = l1[inside], l2[inside]; L0 = 1 - L1 - L2; iw = 1.0 / wv[ti]
                k0, k1, k2 = L0 * iw[0], L1 * iw[1], L2 * iw[2]; ss = k0 + k1 + k2
                U = (k0 * uv[ti, 0, 0] + k1 * uv[ti, 1, 0] + k2 * uv[ti, 2, 0]) / ss
                V = (k0 * uv[ti, 0, 1] + k1 * uv[ti, 1, 1] + k2 * uv[ti, 2, 1]) / ss
                smp = _sample(tex, np.mod(U, 1.0), np.mod(V, 1.0))
                o = (smp[:, :3].max(1) if additive else smp[:, channel]) * opacity
                op[idx] = np.maximum(op[idx], o)
            else:
                op[idx] = opacity
            hit[idx] = True
        if not hit.any(): return
        h_, w_ = y1 - y0, x1 - x0
        C = np.zeros((h_ * w_, 3), np.float32); T = np.ones((h_ * w_, 3), np.float32)
        if additive: C[hit] = op[hit][:, None]          # adds light (its brightest channel), hides nothing
        else: T[hit] = (1.0 - op[hit])[:, None]
        self._add_raw(lid, x0, y0, C.reshape(h_, w_, 3), T.reshape(h_, w_, 3))

    def _add_raw(self, lid, x0, y0, C, T):
        y1 = y0 + C.shape[0]; x1 = x0 + C.shape[1]
        for k, e in self._L.items():
            if k == lid: continue
            ex0, ey0, eo, eh = e
            ax0, ay0 = max(x0, ex0), max(y0, ey0); ax1, ay1 = min(x1, ex0 + eo.shape[1]), min(y1, ey0 + eo.shape[0])
            if ax1 <= ax0 or ay1 <= ay0: continue
            t = T[ay0 - y0:ay1 - y0, ax0 - x0:ax1 - x0]
            eo[ay0 - ey0:ay1 - ey0, ax0 - ex0:ax1 - ex0] *= t; eh[ay0 - ey0:ay1 - ey0, ax0 - ex0:ax1 - ex0] *= t
        e = self._get(lid, (x0, y0, x1, y1)); ex0, ey0, eo, eh = e
        so = eo[y0 - ey0:y1 - ey0, x0 - ex0:x1 - ex0]; sh = eh[y0 - ey0:y1 - ey0, x0 - ex0:x1 - ex0]
        so *= T; so += C; sh *= T; sh += 1 - T

    def occlude(self, tau_full):
        """attenuate every label by a later non-projectile layer's transmittance (H, W, 3) or (H, W) at full resolution"""
        t = tau_full if tau_full.ndim == 3 else tau_full[..., None]
        for e in self._L.values():
            x0, y0, eo, eh = e; s = t[y0:y0 + eo.shape[0], x0:x0 + eo.shape[1]]
            eo *= s; eh *= s

    def share_maps(self, half=True):
        """{lid: (x0, y0, share (h, w))} at half resolution (2 x 2 box means of own / hide), or full resolution"""
        out = {}
        for lid, (x0, y0, eo, eh) in self._L.items():
            if half:
                hx0, hy0 = x0 // 2, y0 // 2; hx1 = (x0 + eo.shape[1] + 1) // 2; hy1 = (y0 + eo.shape[0] + 1) // 2
                po = np.zeros(((hy1 - hy0) * 2, (hx1 - hx0) * 2, 3), np.float32); ph = np.zeros_like(po)
                po[y0 - 2 * hy0:y0 - 2 * hy0 + eo.shape[0], x0 - 2 * hx0:x0 - 2 * hx0 + eo.shape[1]] = eo
                ph[y0 - 2 * hy0:y0 - 2 * hy0 + eo.shape[0], x0 - 2 * hx0:x0 - 2 * hx0 + eo.shape[1]] = eh
                mo = po.reshape(hy1 - hy0, 2, hx1 - hx0, 2, 3).mean((1, 3)); mh = ph.reshape(hy1 - hy0, 2, hx1 - hx0, 2, 3).mean((1, 3))
                out[lid] = (hx0, hy0, np.clip(np.maximum(mo, mh).max(-1), 0, 1))
            else:
                out[lid] = (x0, y0, np.clip(np.maximum(eo, eh).max(-1), 0, 1))
        return out

    def labels(self, half=True, cut=0.5, fill=()):
        """(share (H, W) float32 of the owner, ids (H, W) int32 index into self.ids or -1): owner = the label with the
        largest share (the front-most one when shares tie); pixels with share < cut get -1 but keep their share.
        fill: label ids whose enclosed holes (unowned pixels surrounded by the label) are given to the label (Bibi's
        bubble: its translucent centre is one label with the rim, the user's decision 2026-10-18)"""
        H, W = (H_FULL // 2, W_FULL // 2) if half else (H_FULL, W_FULL)
        best = np.zeros((H, W), np.float32); ids = np.full((H, W), -1, np.int32)
        for i, (lid, (x0, y0, s)) in enumerate(self.share_maps(half).items()):
            ax0, ay0 = max(0, x0), max(0, y0); ax1, ay1 = min(W, x0 + s.shape[1]), min(H, y0 + s.shape[0])
            if ax1 <= ax0 or ay1 <= ay0: continue
            ss = s[ay0 - y0:ay1 - y0, ax0 - x0:ax1 - x0]; bb = best[ay0:ay1, ax0:ax1]; m = ss >= bb
            bb[m] = ss[m]; ids[ay0:ay1, ax0:ax1][m] = self.ids.index(lid)
        ids[best < cut] = -1
        if fill:
            from scipy.ndimage import binary_fill_holes
            for lid in fill:
                k = self.ids.index(lid); m = ids == k
                if m.any(): ids[binary_fill_holes(m) & (ids == -1)] = k
        return best, ids


def point_labels(L, cut=0.6, skip_additive=True):
    """compatibility view of rendclone / expertbot's convention (single sample at full-res pixel (2x+1, 2y+1), coverage =
    alpha of the front-most non-additive draw, COV_CUT 0.6): not the definition of LABEL_SPEC, provided for comparison"""
    raise NotImplementedError('see LABEL_SPEC.md section 6; use Layers.share_maps(half=False)[..][1::2, 1::2]')
