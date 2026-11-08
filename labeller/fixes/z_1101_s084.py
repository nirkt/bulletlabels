# hosted review 2026-10-29 fix (Haiku fix, reviewed by the coordinator, installed 10-29 LOCAL ONLY; source out/sweep2/haikutasks/site1029/s084/fix.py; tested via IDLABELS_OVERRIDE after all fixes incl. z_1100_*, hence z_1101_)
# Skin 084 ConductorDefault, attack 0 deg and attack 90 deg (hosted review 2026-10-29, task s084; user: "part of the smoke
# isnt labeled"; coordinator: bring the WHOLE smoke cloud in, as one filled outline per projectile).
# Type 6001369 ConductorProjectile (attack, sc/effects_brawler.sc chuck_def_atk_projectile). The cloud is drawn from one
# puff sprite (shape 63 of that clip). The label so far took only the bright puffs: the darker puffs round them and behind
# them are the same shape in the same draws, and the per-draw / per-coverage rules left them out.
# Fix: the label is the cloud's silhouette, solid:
#  1. body = the label as it was, plus every puff pixel (shape 63 of this clip, coverage >= 128) inside the projectile's region;
#  2. gaps between the puffs closed (3 px), enclosed holes filled;
#  3. opened with a 4 px disk: narrow spurs (a small puff touching the cloud by a neck) are cut off, the cloud's rounded
#     outline is kept; enclosed holes filled again.
# Detached puffs (separate bits, the 10-27 speck rule in z_1027_s084.py) are not added here: they stay out.
# Keyed to this one type id only. Runs as POST_BODY, before the speck rule of z_1027_s084.py (which still applies).

_T84_TYPE = 6001369
_T84_FILE = 'sc/effects_brawler.sc'
_T84_CLIPS = ('chuck_def_atk_projectile', 'chuck_def_atk_projectile_red')
_T84_PUFF = 63 + 1         # shape id + 1, as _a4_shape_pixels gives it
_T84_CUT = 128             # coverage from which a puff pixel is cloud
_T84_CLOSE = 3             # px: gaps between the puffs closed
_T84_OPEN = 4              # px: radius of the opening that cuts spurs


def _t84_disk(r):
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r + r // 2


def _t84_body(body, ctx):
    """6001369: the whole puff cloud of this projectile, one solid piece, without thin spurs"""
    from scipy.ndimage import binary_fill_holes, binary_opening
    SP = _a4_shape_pixels(ctx['R'], _T84_FILE, _T84_CLIPS)
    puff = (SP == _T84_PUFF) & (ctx['cv'] >= _T84_CUT) & ctx['region']
    m = body | puff
    if not m.any(): return body
    m = binary_fill_holes(_close(m, _T84_CLOSE))
    o = binary_opening(np.pad(m, _T84_OPEN + 1), structure=_t84_disk(_T84_OPEN))[_T84_OPEN + 1:-_T84_OPEN - 1,
                                                                                _T84_OPEN + 1:-_T84_OPEN - 1]
    if o.any(): m = o
    return binary_fill_holes(m) & (ctx['region'] | body)


POST_BODY[_T84_TYPE] = _t84_body
