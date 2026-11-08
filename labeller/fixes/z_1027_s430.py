# sweep 10-27 fix for skin 430 (Haiku fix, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s430/)
# sweep 10-27 fix for skin 430 ArtilleryGalBrawlAcademy, attack 90 deg (task out/sweep2/haikutasks/sweep1027/s430/; pairs 8088-8091,
# ticks 309-315). Type 6000465 (Penny008Projectile, clip penny_008_atk_projectile).
# Cause: the clip draws a translucent gradient column above the shell (9 px wide, coverage 120-180, atlas v 0.96-0.98,
# u 0.18-0.24), which widens into a translucent flare over the top of the crown. The ID pass gives it clip-body codes, so the
# label took in a spike rising 20-70 px above the shell into the trail glow (rules 2 and 3: the glow and the trail are not
# the attack). The shell itself is opaque (coverage 255).
# Fix, only while the column is drawn (its translucent atlas band is in the body): the translucent body pixels (coverage
# below 180) that lie ABOVE the opaque shell in their column are dropped, which takes the column, its edges and the flare.
# Translucent pixels within _P008_RESTORE rows of each column's opaque top come back (the crown's top edge under the flare's
# base is visible shell, not glow): the label keeps the crown's edge at the level of its opaque top.
# Nothing else changes: frames without the column (ticks 305, 307), the translucent lower parts of the shell, other types.
# Only this type (Penny 008 projectile; see report.md for the other skins that use it).
_P008_U = (0.15, 0.27)     # atlas u band of the translucent column sprite
_P008_V = (0.95, 1.0)      # atlas v band of the translucent column sprite
_P008_CV = 180             # coverage below which a drawn pixel is translucent (the opaque shell is 255)
_P008_RESTORE = 4          # rows: translucent pixels this close above a column's opaque top are kept (crown top under the flare)


def _fix_6000465(body, ctx):
    if not body.any(): return body
    uv = ctx['R']['uv']; cv = ctx['cv']
    band = (uv[..., 0] >= _P008_U[0]) & (uv[..., 0] <= _P008_U[1]) & (uv[..., 1] >= _P008_V[0]) & (uv[..., 1] <= _P008_V[1])
    tr = body & (cv < _P008_CV)
    if not (tr & band).any(): return body          # no translucent column drawn in this frame
    opaque = body & (cv >= _P008_CV)
    has = opaque.any(0)
    if not has.any(): return body
    top = np.where(has, np.argmax(opaque, 0), body.shape[0])
    above = (np.arange(body.shape[0])[:, None] < top[None, :]) & has[None, :]
    remove = tr & above
    keep = body & ~remove
    # restore only the few rows just under the crown's opaque top in each column (the flare over the crown's top edge)
    depth = np.arange(body.shape[0])[:, None] - top[None, :]
    keep = keep | (remove & has[None, :] & (depth >= -_P008_RESTORE) & (depth < _P008_RESTORE))
    return keep


POST_BODY[6000465] = _fix_6000465
