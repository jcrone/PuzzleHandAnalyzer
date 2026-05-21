"""Tests for the hand_mask helper extracted into puzzle_vision_utils."""
import unittest
import numpy as np
import cv2

from puzzle_vision_utils import hand_mask


class TestHandMask(unittest.TestCase):

    def test_empty_hands_returns_zero_mask(self):
        m = hand_mask(W=100, H=80, hands_pts=[])
        self.assertEqual(m.shape, (80, 100))
        self.assertEqual(m.dtype, np.uint8)
        self.assertEqual(int(m.sum()), 0)

    def test_single_hand_fills_convex_hull(self):
        # 5 points roughly bounding a 40x40 region centred at (50, 40)
        pts = [(0.3, 0.4), (0.7, 0.4), (0.7, 0.6), (0.3, 0.6), (0.5, 0.5)]
        m = hand_mask(W=100, H=80, hands_pts=[pts])
        # centre should be filled
        self.assertEqual(m[40, 50], 255)
        # corner far from the hand should be empty
        self.assertEqual(m[0, 0], 0)

    def test_short_point_list_is_ignored(self):
        # convexHull needs >= 3 points; 2-point hand should be dropped
        pts = [(0.4, 0.5), (0.6, 0.5)]
        m = hand_mask(W=100, H=80, hands_pts=[pts])
        self.assertEqual(int(m.sum()), 0)

    def test_dilation_extends_mask_outward(self):
        # A tiny hand in the centre should grow outward after dilation
        pts = [(0.49, 0.49), (0.51, 0.49), (0.50, 0.51)]
        # default kernel is 35x35 ellipse — the mask should extend at least
        # 10 pixels from the original hull
        m = hand_mask(W=200, H=200, hands_pts=[pts])
        self.assertEqual(m[100, 100], 255)
        self.assertEqual(m[110, 100], 255)  # 10px below original hull centre


if __name__ == "__main__":
    unittest.main()
