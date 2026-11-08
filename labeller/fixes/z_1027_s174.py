# sweep 10-27 fix for skin 174 (Sonnet revision, Sonnet-reviewed, installed by the coordinator; task out/sweep2/haikutasks/sweep1027/s174/)
# Skin 174 ClusterBombDudeCerberus (sweep 2026-10-27, finding pair 3012, extra_label on the bush debris), second version.
# The pet (ClusterBombPet, character 16000167, spawnable) label took bush-leaf debris and a pale dust cloud above the bush
# (ticks 387-397). Cause: spawn_labels gives every 3D model draw (runtime texture) to the nearest spawnable character within
# CHAR_NEAR = 80 px; the pet is small on screen (base-top segment 20-30 px), so the debris effect draws were taken as its body.
# What differs (all ticks 353-399 checked): the pet's own body draws all use fragment shader 1680474519 (program-independent
# hash, stable between captures) and have their centroid 1-20 px from the base-top segment; the debris draws use the
# translucent effect shader 4132660303 and sit 43-95 px from the segment.
# Fix: from the pet's spawn label remove the pixels of every model draw assigned to the pet that either uses an effect shader
# (EFFECT_FS) or whose centroid is more than PET_NEAR px from the pet's segment. Pixels of the pet's body draws are not touched
# (their label pixel counts are identical before/after). Only spawn_labels of 16000167 is changed: the thrown pet at ticks
# 345-349 is a projectile label (6000535), not a spawn label, so those frames are not affected at all.
# Keyed to the type id (16000167). The pet's body is not labelled where it is hidden in the bush (ticks 395-399: no body draw).
import numpy as _np

PET_DID = 16000167        # ClusterBombPet (owners.character: kind spawnable)
PET_NEAR = 30             # px (1200 x 540 frame): body draws are within ~20 px of the pet's segment, debris 43+ px
EFFECT_FS = {4132660303}  # fragment shader of the translucent bush-debris / dust model draws

_spawn_labels_base = spawn_labels


def spawn_labels(state, cam, R, taken):
    out = _spawn_labels_base(state, cam, R, taken)
    if not out: return out
    chars = []
    for c in state['chars']:
        x, y, z = int(c[3]), int(c[4]), int(c[5])
        a = _np.array(_proj(cam, x, y, z)); b = _np.array(_proj(cam, x, y, z + CHAR_TOP)); chars.append((int(c[0]), int(c[1]), int(c[2]), a, b))
    ids_h = R['id'][::2, ::2]
    for d, dr in R['draws'].items():
        if dr.get('th') != MODEL_TEX: continue
        m = ids_h == d
        if not m.any(): continue
        ys, xs = _np.nonzero(m); p = _np.array([xs.mean(), ys.mean()]); best = (CHAR_NEAR, None)
        for cid, did, own, a, b in chars:   # the same nearest-character assignment as spawn_labels
            ab = b - a; t = _np.clip(_np.dot(p - a, ab) / max(1e-6, _np.dot(ab, ab)), 0, 1); dist = _np.linalg.norm(a + t * ab - p)
            if dist < best[0]: best = (dist, (cid, did, own))
        if best[1] is None: continue
        cid, did, own = best[1]
        if did != PET_DID or own < 0 or cid not in out: continue
        if dr.get('fs') in EFFECT_FS or best[0] > PET_NEAR:
            out[cid] = out[cid] & ~m
    for cid in list(out):
        if not out[cid].any(): del out[cid]
    return out
