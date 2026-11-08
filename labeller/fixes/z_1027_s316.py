# sweep 10-27 fix for skin 316 (Sonnet revision, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s316/)
# Sweep 2026-10-27 task s316 (second attempt): skin 316 FireDudeFrost, type 6000833 = FireDude003Projectile (frost wave,
# attack 0 deg and 90 deg rows, pairs 5901-5911).
# Cause: the frost wave is drawn ADDITIVELY (blend 1); its bright pixels have no usable body sprite in the ID pass, so
# sprite matching leaves only a few specks of the trail sprite. The visible attack is the bright cyan-white streak.
# Why attempt 1 failed: it clipped the streak to ctx['region'] (two diamonds around the predicted points) which cuts the
# streak in pieces at ticks 309-315 (pinch between the diamonds, upper half outside them).
# Fix: body = the frame's bright frost-coloured pixels (max(R,G,B) >= 140, R >= 120, 20 <= G-R <= 90), NOT clipped by the
# region; white cores (bright, near-white pixels touching that mask) are added, the mask is closed 2 px, holes filled,
# opened by 1 px (spurs / sparkle dots off) and only the connected piece that touches the projectile's region and is the
# largest is kept: the streak's own solid outline, one piece. This drops the lavender walls (G-R ~ 5), the blue water
# (R ~ 30), the soft glow (below the brightness cut) and the sparkle arc (small separate dots). Only type 6000833 is
# changed; the super 6000834 and other types are untouched.
import numpy as _np
from scipy.ndimage import label as _cc, binary_fill_holes as _bfh, binary_opening as _bop
FROST_WAVE = dict(T=140, RMIN=120, DMIN=20, DMAX=90)


def _frost_wave_body(body, ctx):
    img = ctx['img']; region = ctx['region']
    if img is None: return body
    im = img.astype(_np.int16); mx = im.max(-1); mn = im.min(-1); dgr = im[..., 1] - im[..., 0]
    m = (mx >= FROST_WAVE['T']) & (im[..., 0] >= FROST_WAVE['RMIN']) & (dgr >= FROST_WAVE['DMIN']) & (dgr <= FROST_WAVE['DMAX'])
    white = (mn >= 180) & (dgr >= 0) & (dgr < FROST_WAVE['DMIN'])      # white-hot core pixels of the streak
    m |= white & _dilate(m, 2)
    m = _bfh(_close(m, 3))
    m0 = _bop(m, structure=_np.ones((3, 3), bool))
    yy, xx = _np.mgrid[-2:3, -2:3]; disk = (yy * yy + xx * xx) <= 5      # radius-2 opening rounds off spurs and bumps
    m1 = _bop(m0, structure=disk)
    m = m1 if m1.sum() >= 0.8 * m0.sum() else m0                         # never eat the thin early streak
    L, n = _cc(m)
    if not n: return body
    sz = _np.bincount(L.ravel())[1:]
    touch = _np.bincount(L[region].ravel(), minlength=n + 1)[1:] > 0
    if not touch.any(): return body
    k = 1 + int(_np.argmax(_np.where(touch, sz, 0)))
    m = _bfh(L == k)
    return m if m.any() else body


POST_BODY[6000833] = _frost_wave_body
