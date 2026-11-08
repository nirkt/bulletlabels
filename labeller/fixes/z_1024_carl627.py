# Carl Strikers 627 attack: disc holes filled (Haiku task 21, reviewed); also its flip-clip type 6000624
# Task 21 (2026-10-23): skin 627 WhirlwindStrikers (Carl 011), type 6000623 Carl011Projectile (clip carl_011_atk_projectile).
# The frisbee disc's label had holes inside it (two dark swirl slots at t265-271). The pixels there are either opaque
# draws of the floor showing through the disc's centre (no shape, coverage 255) or translucent body pixels (coverage
# 38-122, below CUT). The user's decision (binding): each projectile's label is its outline, fully labelled inside,
# one piece (no holes). Fix: every hole enclosed by the label is filled, within the projectile's region. A hole is
# enclosed by the label's own outline, so nothing outside the disc is added. Only this type (6000623) is touched.
def _t21_carl011_disc_fill(body, ctx):
    from scipy.ndimage import binary_fill_holes
    if not body.any(): return body
    return binary_fill_holes(body) & ctx['region']


POST_BODY[6000623] = _t21_carl011_disc_fill

POST_BODY[6000624] = POST_BODY[6000623]   # Carl011Projectile2 (flip clip of the same skin): same fill (coordinator)
