"""Tests for the cluster mapper: DBSCAN, IoU matching, blob survey."""
import unittest
import numpy as np

from puzzle_clusters import (
    dbscan_2d, bbox_iou, blobs_to_clusters,
)


class TestDbscan2D(unittest.TestCase):

    def test_two_clusters_of_three(self):
        # Two well-separated clusters, eps small enough to keep them apart
        pts = np.array([
            (0.10, 0.10), (0.11, 0.11), (0.12, 0.10),   # cluster A
            (0.80, 0.80), (0.81, 0.80), (0.80, 0.81),   # cluster B
        ])
        labels = dbscan_2d(pts, eps=0.05, min_samples=2)
        # Two distinct labels, each appearing 3 times; no -1 (noise)
        unique = sorted(set(labels.tolist()))
        self.assertEqual(len(unique), 2)
        self.assertNotIn(-1, unique)
        counts = sorted([(labels == u).sum() for u in unique])
        self.assertEqual(counts, [3, 3])

    def test_isolated_point_is_noise(self):
        pts = np.array([
            (0.10, 0.10), (0.11, 0.11),    # cluster
            (0.90, 0.90),                  # noise
        ])
        labels = dbscan_2d(pts, eps=0.05, min_samples=2)
        # The isolated point gets -1
        self.assertEqual((labels == -1).sum(), 1)
        # The other two share a label
        non_noise = labels[labels != -1]
        self.assertEqual(len(set(non_noise.tolist())), 1)

    def test_empty_input(self):
        labels = dbscan_2d(np.empty((0, 2)), eps=0.05, min_samples=2)
        self.assertEqual(len(labels), 0)


class TestBboxIou(unittest.TestCase):

    def test_identical_boxes(self):
        b = (0.1, 0.1, 0.5, 0.5)
        self.assertAlmostEqual(bbox_iou(b, b), 1.0)

    def test_disjoint_boxes(self):
        a = (0.0, 0.0, 0.2, 0.2)
        b = (0.5, 0.5, 0.8, 0.8)
        self.assertEqual(bbox_iou(a, b), 0.0)

    def test_half_overlap(self):
        # b is the right half of a: area_a=1, area_b=0.5, inter=0.5
        # union = 1 + 0.5 - 0.5 = 1.0; IoU = 0.5 / 1.0 = 0.5
        a = (0.0, 0.0, 1.0, 1.0)
        b = (0.5, 0.0, 1.0, 1.0)
        self.assertAlmostEqual(bbox_iou(a, b), 0.5, places=5)


class TestBlobsToClusters(unittest.TestCase):

    def test_groups_nearby_blobs(self):
        blobs = [
            {"xy": (0.10, 0.10), "area": 0.001},
            {"xy": (0.11, 0.11), "area": 0.001},
            {"xy": (0.12, 0.10), "area": 0.001},
            {"xy": (0.80, 0.80), "area": 0.001},
            {"xy": (0.81, 0.80), "area": 0.001},
        ]
        clusters = blobs_to_clusters(blobs, eps=0.05, min_samples=2)
        # Two clusters of size 3 and 2
        sizes = sorted(c["count"] for c in clusters)
        self.assertEqual(sizes, [2, 3])
        # Each cluster has a bbox (use assertLessEqual: collinear blobs
        # can produce a degenerate bbox where x1==x2 or y1==y2)
        for c in clusters:
            x1, y1, x2, y2 = c["bbox"]
            self.assertLessEqual(x1, x2)
            self.assertLessEqual(y1, y2)

    def test_single_isolated_blob_dropped(self):
        # min_samples=2 means a lone blob is noise
        blobs = [{"xy": (0.5, 0.5), "area": 0.001}]
        clusters = blobs_to_clusters(blobs, eps=0.05, min_samples=2)
        self.assertEqual(clusters, [])


class TestClusterMapperUpdate(unittest.TestCase):

    def _synth_frame(self, W, H, blob_positions):
        """Make a black frame with white squares at the given positions."""
        import cv2
        f = np.zeros((H, W, 3), dtype=np.uint8)
        for (cx, cy) in blob_positions:
            x = int(cx * W); y = int(cy * H)
            cv2.rectangle(f, (x - 6, y - 6), (x + 6, y + 6), (255, 255, 255), -1)
        return f

    def test_update_short_circuits_off_survey_frames(self):
        from puzzle_clusters import ClusterMapper
        m = ClusterMapper(W=400, H=400, eff_fps=10.0,
                          survey_interval_s=5.0,
                          area_min_frac=0.0001, area_max_frac=0.5,
                          cluster_eps=0.10)
        frame = self._synth_frame(400, 400, [(0.40, 0.35), (0.50, 0.35)])
        m.update(frame, hands_pts=[], frame_idx=0, t=0.0)
        # survey stride = 5s * 10fps = 50 frames; frame 1 should be a no-op
        n_after_first = len(m.surveys)
        m.update(frame, hands_pts=[], frame_idx=1, t=0.1)
        self.assertEqual(len(m.surveys), n_after_first)

    def test_two_consecutive_surveys_share_cluster_id(self):
        from puzzle_clusters import ClusterMapper
        # Three blobs: pairwise distances 0.10, 0.086, 0.086 — all within
        # eps=0.10 of at least one other, so DBSCAN groups them.
        # W=H=400 ensures the 12px squares are far enough apart after
        # dilation to produce three distinct contours rather than one merged one.
        m = ClusterMapper(W=400, H=400, eff_fps=10.0,
                          survey_interval_s=5.0,
                          area_min_frac=0.0001, area_max_frac=0.5,
                          cluster_eps=0.10)
        blobs = [(0.40, 0.35), (0.50, 0.35), (0.45, 0.42)]
        f0 = self._synth_frame(400, 400, blobs)
        f1 = self._synth_frame(400, 400, blobs)
        m.update(f0, [], frame_idx=0, t=0.0)
        m.update(f1, [], frame_idx=50, t=5.0)   # next survey window
        # Same blobs in same place across two surveys -> one persistent ID
        self.assertEqual(len(m.tracks), 1)
        tr = list(m.tracks.values())[0]
        self.assertGreaterEqual(len(tr["history_seconds"]), 2)


if __name__ == "__main__":
    unittest.main()
