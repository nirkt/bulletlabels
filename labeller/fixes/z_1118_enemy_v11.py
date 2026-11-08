# 2026-11-08 enemy-label fix, group v11 (skins 420 446 459 478 505). Global prefix _f11_. Install name z_1118_enemy_v11.py.
# Own (blue) labels are never touched: every rule below is keyed to the red clips / ctx['own'] False.
import functools as _f11_ft

# ---- 420 KickerDudeDemon, Fang008Projectile2 (6001118) ----
# CAUSE: the trail sprites of the type (idlabels.sprites: effect fang_008_atk_trail -> its emitter clips 'fang_008_atk_reached_meteor_NN') are
# taken by their BLUE names only. The red shot draws the red variants (fang_008_atk_reached_meteor_NN_red: other shape ids 506 / 508 instead of
# 507 / 509, same art in red). The dark lumps of the ball (blue: shape 509, class 'trail over', body again where the aligned prediction says
# body) were unknown sprites in the red run (code 0), so they were not body and the enemy label kept the rest of the blob ragged / lopsided
# (area 589 / 723 against 382 / 395 own).
# FIX: for the RED art of this type the red variants of those trail clips are added as trail 'over' sprites too (sprites() wrapper, enemy only).
_f11_FANG2 = 6001118
_f11_sprites_prev = sprites


@_f11_ft.lru_cache(maxsize=None)
def _f11_sprites_cached(t, own):
    out = {k: list(v) for k, v in _f11_sprites_prev(t, own).items()}
    if t != _f11_FANG2 or own: return out
    for swf, name in _trail_clips(t):
        S = PD.sc(swf); cid = S.exports.get(name + '_red')
        if cid is None or name + '_red' == name: continue
        for sid, k in _clip_shapes(S, cid):
            tex, uv, xy = S.shapes[sid][k]
            out.setdefault((swf, tex), []).append((np.asarray(uv, np.float64) / 65535.0, 'trail', 'over'))
    return out


def sprites(t, own):
    return _f11_sprites_cached(t, own) if t == _f11_FANG2 and not own else _f11_sprites_prev(t, own)


# ---- 446 ControllerXmas, Nani007Projectile (6000782; the column of three jellies, one object each) ----
# CAUSE: at the first drawn ticks of a shot (attack 0 tick 267, attack 90 tick 307) the RED shot has a huge launch burst (a white-pink spiky star, an
# enemy-only effect) drawn with shape 2 of the SAME atlas as the jellies' own halves (it is one single draw of 1100-2100 px in the region, the jelly
# draws are 40-550 px), so every pixel of it is 'clip body': the enemy label swallowed the burst (area 2198 against 1054 own). The blue shot shows
# no such burst.
# FIX (enemy objects only): body pixels of draws bigger than _f11_NANI_BIG px (whole frame; the burst is one draw of 1100-2100 px, jelly draws 40-600 px) are dropped, the part of the jelly the burst hides is the hull of the predicted body (hidden-part rule as z_1109_enemy_pl), specks
# under _f11_NANI_SPECK px (burst sparkles) dropped. Ticks without such a draw, and all own shots: unchanged.
_f11_NANI = 6000782
_f11_NANI_BIG = 900
_f11_NANI_SPECK = 12
_f11_nani_prev = POST_BODY.get(_f11_NANI)


def _f11_nani_post(body, ctx):
    if _f11_nani_prev is not None: body = _f11_nani_prev(body, ctx)
    if ctx.get('own') or not body.any(): return body
    from scipy.ndimage import label as _cc
    ih = ctx['ids_h']; region = ctx['region']
    ds = np.unique(ih[body & (ih > 0)]); big = [int(d) for d in ds if (ih == d).sum() >= _f11_NANI_BIG]
    if not big: return body
    res = body & ~np.isin(ih, big)
    pb = ctx['pred_body']
    if pb.any(): res = res | (hull_mask(pb) & region)
    L, n = _cc(res, structure=np.ones((3, 3), int))
    if n:
        sz = np.bincount(L.ravel()); small = np.nonzero((sz < _f11_NANI_SPECK) & (np.arange(len(sz)) > 0))[0]
        res = res & ~np.isin(L, small)
    return res


POST_BODY[_f11_NANI] = _f11_nani_post


# ---- 459 HookDudeStrikers, Gene007Projectile (6000517; the hand / bubble ball; only this skin) ----
# CAUSE: as 446: at the first drawn tick of a shot (attack 0 tick 265, attack 90 tick 305) the RED shot has an enemy-only pink-white launch burst
# (spiky splash round the shooter, the blue shot has none) drawn with sprites of the type's own clip (code 1), so 400-450 px of it became body next to
# the ball (label 723 / 746 px against 390 / 428 px own; the ball and the label of the own run both follow the predicted body to within 1 %).
# FIX (enemy objects only): when 4+ px outside the hull of the predicted body hold at least max(_f11_GENE_MIN, 0.6 x the predicted body) body pixels (a
# burst, not the ball's glow edge), the body is cut to the hull of the predicted body grown by _f11_GENE_K px (plus the hull itself, hidden part).
# Other ticks (incl. the hit tick 311 whose small predicted body leaves 50 px outside in both runs) and all own shots are untouched.
_f11_GENE = 6000517
_f11_GENE_K = 4
_f11_GENE_MIN = 120
_f11_gene_prev = POST_BODY.get(_f11_GENE)


def _f11_gene_post(body, ctx):
    if _f11_gene_prev is not None: body = _f11_gene_prev(body, ctx)
    if ctx.get('own') or not body.any(): return body
    pb = ctx['pred_body']
    if not pb.any(): return body
    hp = hull_mask(pb); keep = _dilate(hp, _f11_GENE_K)
    out = (body & ~keep).sum()
    if out < max(_f11_GENE_MIN, 0.6 * pb.sum()): return body
    return (body & keep) | (hp & ctx['region'])


POST_BODY[_f11_GENE] = _f11_gene_post


# ---- 478 WhirlwindPirate, Carl005Projectile (6000611; the anchor) ----
# CAUSE: z_1103_samurai_whirl drops the chain (links + rod, the user's rule 'anchor only, no chain') by the BLUE clip's shape ids 58408 / 58409. The red
# clip (carl_005_atk_projectile_red) has the same layout with the twin shapes 58405 (links) / 58406 (rod) (58404 = anchor, as 58407): nothing was
# dropped in the red run, the brown chain stayed in the label (area 1473-1534 against 1151-1186 own).
# FIX: the red chain shapes are added to SAM_CHAIN_SHAPES (the list z_1103's anchor rule reads). Red ids are never drawn in own shots: own labels identical.
SAM_CHAIN_SHAPES = tuple(SAM_CHAIN_SHAPES) + tuple(s for s in (58405, 58406) if s not in SAM_CHAIN_SHAPES)
