# human review 2026-10-28 fix for skin 045 (Haiku fix, reviewed by the coordinator, installed 10-29; source out/sweep2/haikutasks/human1028/s045/fix.py; tested via IDLABELS_OVERRIDE after all fixes incl. z_1099_cleanup, hence z_1100_)
# s045 fix (human review 2026-10-28), skin 045 SpawnerDudeDefault, super row 45 deg. Keyed to explicit type / character ids.
#
# (a) User finding, tick 345 "too small / cut off": the super ball SpawnerDudeUltiProjectile (type 6000697, object 2000004)
#     is half covered by a translucent cyan burst and a white flash in front of it. The ID pass sees the burst on top, so the
#     drawn body is only the ball's left part (296 px) and its right / lower part lies outside the label. The hidden-parts rule
#     applies: when at least 20 % of the ball's predicted disc has no sprite of its own in front (hidden), the label is the
#     whole predicted disc (the logic circle). At ticks 347-359 the ball is fully drawn and the drawn body equals the prediction
#     (671 vs 672 px at 347), so their labels do not change (no hidden share there).
#
# (b) User findings, ticks 371, 389, 411 "too big" (and 411: "only the fully visible circle should be labeled, not that extra
#     part on the bottom"): the super's turret, character SpawnerDudeTurret (did 16000173, CollisionRadius 120 in
#     characters.csv), is a spawnable whose 3D-model label (spawn) takes in its dark shadow on the bottom-right (ShadowSkew 25
#     in the data: a ground shadow offset to the lower right), and at 411 the black bump above the blue dome. The turret's
#     footprint is its collision circle on the ground: the 120-unit circle projected at its logic x, y (the gold ring's centre
#     and size match this projection at every tick 363-411: centre (694.2, 336.2) on screen, gold ring centroid (693.6-693.9,
#     335.5-335.7) at ticks 365-389). The label keeps only the turret's own model pixels inside that ellipse, so the blue dome
#     and the ring stay, the shadow and the bump go. Only this turret (did 16000173) is changed; its characters are untouched
#     by the other fixes, the projectiles are not touched.

S045_BALL = 6000697                 # SpawnerDudeUltiProjectile (the default super ball of SpawnerDude skins)
S045_HIDDEN_MIN = 0.2               # share of the predicted disc with no sprite in front from which the ball is the full disc
S045_TURRET = 16000173              # SpawnerDudeTurret (characters.csv)
S045_TURRET_R = 120                 # its CollisionRadius (logic units): the footprint circle on the ground
S045_ELL_PTS = 180                  # polygon points for the footprint


def _s045_ball_disc(body, ctx):
    """SpawnerDudeUltiProjectile: hidden under a flash / burst (predicted pixels with no sprite in front >= S045_HIDDEN_MIN):
    the label is the whole predicted disc (the ball's own circle); otherwise unchanged."""
    pb = ctx['pred_body']; C = ctx['C']
    if not pb.any(): return body
    if (C[pb] == 0).mean() < S045_HIDDEN_MIN: return body
    return body | pb


POST_BODY[S045_BALL] = _s045_ball_disc


def _s045_footprint(cam, x, y, shape):
    """(H, W) bool: the inside of the turret's collision circle (S045_TURRET_R logic units, on the ground z = 0) projected
    to the screen (the frame's 1200 x 540 px)."""
    from PIL import Image as _Img, ImageDraw as _Draw
    ang = np.linspace(0, 2 * np.pi, S045_ELL_PTS, endpoint=False)
    pts = [_proj(cam, x + S045_TURRET_R * np.cos(a), y + S045_TURRET_R * np.sin(a), 0.0) for a in ang]
    im = _Img.new('L', (shape[1], shape[0]), 0); _Draw.Draw(im).polygon(pts, fill=1)
    return np.array(im, bool)


_s045_prev_labels = labels


def labels(objs, cam, R, state=None, ended=(), img=None):
    P = _s045_prev_labels(objs, cam, R, state=state, ended=ended, img=img)
    if state is None: return P
    ids, lids, share = P['ids'], P['lids'], P['share']
    for c in state['chars']:
        cid, did = int(c[0]), int(c[1])
        if did != S045_TURRET: continue
        ks = [k for k, l in enumerate(lids) if l[0] == cid and l[2] == 'spawn']
        if not ks: continue
        k = ks[0]; m = ids == k
        if not m.any(): continue
        E = _s045_footprint(cam, int(c[3]), int(c[4]), ids.shape)
        out = m & ~E
        if out.any():
            ids[out] = -1; share[out] = 0
    P['ids'], P['share'] = ids, share
    return P
