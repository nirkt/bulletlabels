# AI-review batch 2026-11-01 fix (Haiku fix, reviewed by the coordinator, installed 11-01 LOCAL ONLY; source out/sweep2/haikutasks/siteai/bulldude/fix.py)
# AI-reviewer findings on skins 492, 506, 516, 517, 531, 603 (2026-11-01), verified by hand on the new captures
# sw2_506, sw2_516, sw2_517, sw2_603 (labels per object, shape stats). Each block says what it fixes and why.
# Chained onto the installed hooks (POST_BODY of z_1027_s603 for 6000051, a2_agent2 for 6000046), not replacing them.
# exec'd in idlabels' namespace (IDLABELS_OVERRIDE). Verdicts: see report.md in this folder.

def _chain_post(t, fn):
    prev = POST_BODY.get(t)
    if prev is None: POST_BODY[t] = fn
    else: POST_BODY[t] = lambda body, ctx, _p=prev, _f=fn: _f(_p(body, ctx), ctx)


def _drop_specks(min_px):
    """body without its tiny detached bits: connected parts under min_px pixels are dropped (the largest part stays)"""
    def fn(body, ctx):
        from scipy.ndimage import label as cc
        if not body.any(): return body
        L, n = cc(body)
        if n <= 1: return body
        sz = np.bincount(L.ravel()); keep = sz >= min_px; keep[0] = False; keep[np.argmax(sz[1:]) + 1] = True
        return keep[L]
    return fn


# ---- 506 BullDudeTerminator, Bull004Projectile (6000045): attack 90 orb cluster, t305 (AI: 'sawtooth, torn top edge').
# Each orb is one object (2000015-19). Its top carries cyan flame wisps drawn with the orb's own clip sprite, and at t305
# the outline of 2-3 orbs had 1-2 px detached bits (comps [191, 2, 1] ...) on the flame tops: specks, plus a notched top.
# Fix: the tiny detached bits go (parts under 12 px), then the orb's gaps / notches between the flame tongues are closed
# (ONE_LABEL closing, 3 px, holes filled, inside the object's region). The orb body and its flame tongues stay.
_chain_post(6000045, _drop_specks(12))
ONE_LABEL[6000045] = 3


# ---- 516 BullDudeLion, Bull006Projectile (6000047): attack 90, t307 (AI: 'small separate outline on a puff beside the
# wave'). Object 2000015 carried a 1 px detached speck at (618,324) at t307 and 1-2 px specks at t305 (2000015, 2000017,
# 2000018): the 'small outline' the reviewer saw. Each puff stays its own object / label (5 projectiles, 2000015-19, end one by
# one: separate labels, not one hull). Fix: specks under 12 px dropped.
_chain_post(6000047, _drop_specks(12))


# ---- 517 BullDudeMang, Bull005Projectile (6000046): attack 90, t305 (AI: 'label over the cyan cloud has an inner loop').
# The union of its labels has a 15 px unlabelled pocket inside the bright cloud at (598,292), enclosed by three neighbouring
# puffs (per object there is no hole; the pocket lies between them). Fix: closing of each object's body (3 px, holes
# filled, inside its region) closes the pocket from the touching sides. Tiny specks (under 12 px) dropped as well.
_chain_post(6000046, _drop_specks(12))
ONE_LABEL[6000046] = 3


# ---- 603 BullDudeDemon, Bull011Projectile (6000051): attack 0 t271 / t273, attack 90 t305 and t307-t311 (AI: 'shot pieces
# outlined as many small fragments and loops' -> hull; 'holes / ragged' at t305). Measured on sw2_603: 1 px specks and loops
# at t271-273 (a 1 px ring beside the bubble at t271), 1 px holes in the labels at t305-307 (2000017, 2000018), detached
# parts 39-43 px (2000011 at t271/t273). Fix here: specks under 12 px dropped, holes of 4 px or less filled (the
# labels' own holes). NO hull: a hull per object took in the empty space between the bubbles of the column (tried, too
# big: the outline covered the whole column at t267-t311); the hull question is written up as needs_user.
_chain_post(6000051, _drop_specks(12))
_chain_post(6000051, lambda body, ctx: body | small_holes(body, 4) if body.any() else body)
