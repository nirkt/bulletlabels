# Owner 2026-11-08: "drop glow layer" - the glow-class labels (P['glow_ids'] / P['glow_share']: predicted clip glow, not the bullet) are
# removed from the labeller output. They were never shown in the owner's review (the review app hides the green glow outlines) and the
# offset scan found ghost glow polygons where nothing is drawn (Dynamike 224/225, Penny turret, Crab 410, Splitter 077, Tara 303 ...).
# Runs LAST (after zz_1108_fill_smooth and zz_1109_motion), so rules that use the glow internally are unaffected.
# Toggle: IDLABELS_GLOW=1 keeps the glow layer (old behaviour). Body labels (P['ids'], P['share'], meta) are never touched.
import os as _ng_os
import numpy as _ng_np

_ng_prev_labels = labels


def labels(objs, cam, R, state=None, ended=(), img=None):
    P = _ng_prev_labels(objs, cam, R, state=state, ended=ended, img=img)
    if _ng_os.environ.get('IDLABELS_GLOW', '0') == '1' or not isinstance(P, dict): return P
    if P.get('glow_ids') is not None:
        P['glow_ids'] = _ng_np.full_like(P['glow_ids'], -1)
    if P.get('glow_share') is not None:
        P['glow_share'] = _ng_np.zeros_like(P['glow_share'])
    return P
