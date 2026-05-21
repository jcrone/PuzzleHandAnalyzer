"""Shared vision helpers used by puzzle_pieces.py and puzzle_clusters.py.

Currently exposes hand_mask: a binary mask of pixels occupied by detected
hands (convex hull + dilation), used to ignore hand pixels when looking
for puzzle-piece blobs.
"""

import numpy as np
import cv2


_DEFAULT_HAND_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))


def hand_mask(W, H, hands_pts, kernel=None):
    """Return an HxW uint8 mask with 255 where hands are, 0 elsewhere.

    hands_pts is a list of hands, each a list of (x, y) tuples in
    normalized 0-1 image coordinates. Hands with fewer than 3 points are
    skipped (convexHull needs >= 3). The resulting mask is dilated by an
    ellipse kernel (default 35x35) to swallow shadows around the hand.
    """
    m = np.zeros((H, W), dtype=np.uint8)
    for pts in hands_pts:
        if not pts or len(pts) < 3:
            continue
        arr = np.array(
            [[int(x * W), int(y * H)] for x, y in pts], dtype=np.int32)
        hull = cv2.convexHull(arr)
        cv2.fillConvexPoly(m, hull, 255)
    if m.any():
        m = cv2.dilate(m, kernel if kernel is not None else _DEFAULT_HAND_KERNEL)
    return m
