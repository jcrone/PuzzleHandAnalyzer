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


    def test_dormant_track_clears_on_rematch(self):
        from puzzle_clusters import ClusterMapper
        m = ClusterMapper(W=400, H=400, eff_fps=10.0,
                          survey_interval_s=5.0,
                          area_min_frac=0.0001, area_max_frac=0.5,
                          cluster_eps=0.10, dormant_after_surveys=2)
        # Seed one cluster
        blobs = [(0.40, 0.35), (0.50, 0.35), (0.45, 0.42)]
        f_seed = self._synth_frame(400, 400, blobs)
        m.update(f_seed, [], frame_idx=0, t=0.0)
        # Two consecutive blank surveys -> the track should go dormant
        f_blank = self._synth_frame(400, 400, [])
        m.update(f_blank, [], frame_idx=50, t=5.0)
        m.update(f_blank, [], frame_idx=100, t=10.0)
        tid = next(iter(m.tracks))
        self.assertTrue(m.tracks[tid]["dormant"])
        # Now the cluster returns -> track should re-activate
        m.update(f_seed, [], frame_idx=150, t=15.0)
        self.assertFalse(m.tracks[tid]["dormant"])
        self.assertEqual(m.tracks[tid]["consecutive_misses"], 0)


class TestClassification(unittest.TestCase):

    def _make_mapper_with_synthetic_tracks(self, tracks):
        """Build a ClusterMapper, then poke synthetic tracks into it
        directly (bypassing update). tracks is a list of dicts."""
        from puzzle_clusters import ClusterMapper
        m = ClusterMapper(W=200, H=200, num_pieces=100, eff_fps=10.0)
        for tr in tracks:
            m.tracks[tr["id"]] = tr
        m.next_id = max(t["id"] for t in tracks) + 1
        return m

    def _track(self, tid, initial, final, peak=None, bbox=(0.2, 0.2, 0.7, 0.7),
               born_t=10.0, last_t=100.0):
        if peak is None:
            peak = max(initial, final)
        return {
            "id": tid, "born_t": born_t, "last_seen_t": last_t,
            "initial_count": initial, "peak_count": peak,
            "final_count": final, "bbox_at_peak": bbox,
            "last_bbox": bbox,
            "history_seconds": [born_t, last_t],
            "history_counts": [initial, final],
            "history_bboxes": [bbox, bbox],
            "consecutive_misses": 0, "dormant": False,
            "merged_into": None, "merged_from": [],
        }

    def test_assembly_cluster_wins_over_sort_pile(self):
        # Pile: starts at 200, ends at 10 (net growth -190)
        # Assembly: starts at 2, ends at 80 (net growth +78)
        pile = self._track(1, initial=200, final=10,
                           bbox=(0.0, 0.0, 0.3, 0.4))
        asm = self._track(2, initial=2, final=80,
                          bbox=(0.4, 0.2, 0.9, 0.8))
        m = self._make_mapper_with_synthetic_tracks([pile, asm])
        clusters, _milestones = m.finalize(n_frames=1000)
        main = [c for c in clusters if c.get("is_main")]
        self.assertEqual(len(main), 1)
        self.assertEqual(main[0]["id"], 2)

    def test_no_main_when_nothing_grows(self):
        # Only sort pile present; no assembly cluster
        pile = self._track(1, initial=200, final=10)
        m = self._make_mapper_with_synthetic_tracks([pile])
        clusters, _ = m.finalize(n_frames=1000)
        main = [c for c in clusters if c.get("is_main")]
        self.assertEqual(len(main), 0)
        self.assertIsNone(m.inferred_board())

    def test_aspect_ratio_tiebreak(self):
        # Both clusters have similar positive growth. The one whose
        # aspect ratio is closer to a typical puzzle (1:2 .. 2:1) wins.
        # ar 4:1 -> elongated, looks like a pile shoved against an edge
        a = self._track(1, initial=2, final=50,
                        bbox=(0.0, 0.45, 0.8, 0.55))   # aspect 8:1
        # ar 1:1 -> classic puzzle shape
        b = self._track(2, initial=2, final=48,
                        bbox=(0.2, 0.2, 0.7, 0.7))     # aspect 1:1
        m = self._make_mapper_with_synthetic_tracks([a, b])
        clusters, _ = m.finalize(n_frames=1000)
        main = [c for c in clusters if c.get("is_main")]
        self.assertEqual(len(main), 1)
        self.assertEqual(main[0]["id"], 2)
        # cluster_summary should record the tiebreak path
        self.assertTrue(m.last_summary["tiebreak_used"])

    def test_prunes_tiny_noise_tracks(self):
        # A track that never grew past peak_count=2 should be removed
        noise = self._track(1, initial=2, final=2, peak=2)
        asm = self._track(2, initial=2, final=50, peak=50)
        m = self._make_mapper_with_synthetic_tracks([noise, asm])
        clusters, _ = m.finalize(n_frames=1000)
        ids = {c["id"] for c in clusters}
        self.assertNotIn(1, ids)
        self.assertIn(2, ids)

    def test_low_count_tiebreak_uses_absolute_floor(self):
        # Two clusters with small net growth. Without the 10-piece floor,
        # a 1-piece survey noise would trigger an AR tiebreak. With the
        # floor, the higher-growth cluster wins outright.
        a = self._track(1, initial=2, final=11,
                        bbox=(0.0, 0.45, 0.8, 0.55))   # net 9, ar 8:1
        b = self._track(2, initial=2, final=20,
                        bbox=(0.2, 0.2, 0.7, 0.7))     # net 18, ar 1:1
        m = self._make_mapper_with_synthetic_tracks([a, b])
        clusters, _ = m.finalize(n_frames=1000)
        main = [c for c in clusters if c.get("is_main")]
        self.assertEqual(len(main), 1)
        # With absolute floor: top_growth=18, threshold=max(0.9*18, 18-10)=max(16.2, 8)=16.2.
        # Cluster a has net=9 which is below 16.2, so b wins outright,
        # no tiebreak. Without the floor: 0.9*18=16.2 still excludes a,
        # so this test mainly confirms b wins without the AR rescue.
        self.assertEqual(main[0]["id"], 2)
        self.assertFalse(m.last_summary["tiebreak_used"])


class TestInferredBoard(unittest.TestCase):

    def test_returns_main_cluster_bbox_at_peak(self):
        from puzzle_clusters import ClusterMapper
        m = ClusterMapper(W=200, H=200, num_pieces=100, eff_fps=10.0)
        m.tracks[2] = {
            "id": 2, "born_t": 10.0, "last_seen_t": 100.0,
            "initial_count": 2, "peak_count": 80, "final_count": 80,
            "bbox_at_peak": (0.1, 0.2, 0.7, 0.8),
            "last_bbox": (0.1, 0.2, 0.7, 0.8),
            "history_seconds": [10.0, 100.0],
            "history_counts": [2, 80],
            "history_bboxes": [(0.1, 0.2, 0.7, 0.8), (0.1, 0.2, 0.7, 0.8)],
            "consecutive_misses": 0, "dormant": False,
            "merged_into": None, "merged_from": [],
        }
        m.next_id = 3
        m.finalize(n_frames=1000)   # populates is_main
        board = m.inferred_board()
        self.assertEqual(board, [0.1, 0.2, 0.7, 0.8])


class TestMilestones(unittest.TestCase):

    def _mapper_with_growth_history(self, num_pieces, history):
        """history is [(t, count, bbox), ...] for a single growing cluster."""
        from puzzle_clusters import ClusterMapper
        m = ClusterMapper(W=200, H=200, num_pieces=num_pieces, eff_fps=10.0)
        history_seconds = [h[0] for h in history]
        history_counts = [h[1] for h in history]
        history_bboxes = [h[2] for h in history]
        m.tracks[1] = {
            "id": 1, "born_t": history[0][0],
            "last_seen_t": history[-1][0],
            "initial_count": history[0][1],
            "peak_count": max(history_counts),
            "final_count": history[-1][1],
            "bbox_at_peak": history_bboxes[history_counts.index(max(history_counts))],
            "last_bbox": history_bboxes[-1],
            "history_seconds": history_seconds,
            "history_counts": history_counts,
            "history_bboxes": history_bboxes,
            "consecutive_misses": 0, "dormant": False,
            "merged_into": None, "merged_from": [],
        }
        m.next_id = 2
        return m

    def test_first_piece_joined_fires(self):
        bbox = (0.2, 0.2, 0.7, 0.7)
        m = self._mapper_with_growth_history(
            100, [(10.0, 2, bbox), (20.0, 50, bbox)])
        _, milestones = m.finalize(n_frames=1000)
        types = [ms["type"] for ms in milestones]
        self.assertIn("first_piece_joined", types)
        fp = next(ms for ms in milestones if ms["type"] == "first_piece_joined")
        self.assertAlmostEqual(fp["t"], 10.0)
        self.assertEqual(fp["cluster_id"], 1)

    def test_25_50_75_pct_thresholds(self):
        bbox = (0.2, 0.2, 0.7, 0.7)
        # num_pieces=100 -> 25/50/75 pieces = thresholds
        history = [(10.0 * i, 5 * i, bbox) for i in range(1, 21)]   # 5,10,...,100
        m = self._mapper_with_growth_history(100, history)
        _, milestones = m.finalize(n_frames=1000)
        types = [ms["type"] for ms in milestones]
        for t in ("cluster_25pct", "cluster_50pct", "cluster_75pct"):
            self.assertIn(t, types)
        # 25% threshold crossed at count >= 25 -> first such survey is i=5 (count=25)
        ms25 = next(m for m in milestones if m["type"] == "cluster_25pct")
        self.assertEqual(ms25["details"]["count"], 25)

    def test_pct_milestones_omitted_when_num_pieces_none(self):
        bbox = (0.2, 0.2, 0.7, 0.7)
        m = self._mapper_with_growth_history(
            None, [(10.0, 2, bbox), (20.0, 50, bbox), (30.0, 80, bbox)])
        m.num_pieces = None  # explicit
        _, milestones = m.finalize(n_frames=1000)
        types = [ms["type"] for ms in milestones]
        for t in ("cluster_25pct", "cluster_50pct", "cluster_75pct"):
            self.assertNotIn(t, types)
        # but first_piece_joined still present
        self.assertIn("first_piece_joined", types)

    def test_frame_complete_fires_on_ring_shape(self):
        from puzzle_clusters import ClusterMapper, _perimeter_fraction
        # Build a roughly 25x20 ring of piece positions inside a bbox
        bbox = (0.1, 0.1, 0.9, 0.7)
        x1, y1, x2, y2 = bbox
        cols, rows = 25, 20
        ring = []
        for c in range(cols):
            for r in range(rows):
                if c == 0 or c == cols - 1 or r == 0 or r == rows - 1:
                    fx = x1 + (c + 0.5) * (x2 - x1) / cols
                    fy = y1 + (r + 0.5) * (y2 - y1) / rows
                    ring.append((fx, fy))
        count = len(ring)   # 2*(25+20)-4 = 86
        m = self._mapper_with_growth_history(
            500, [(10.0, 2, bbox), (1800.0, count, bbox)])
        # Patch history_members on the track so frame-complete can inspect it
        m.tracks[1]["history_members"] = [ring, ring]
        _, milestones = m.finalize(n_frames=1000)
        types = [ms["type"] for ms in milestones]
        self.assertIn("frame_complete", types)

    def test_frame_complete_does_not_fire_on_filled_rectangle(self):
        # A 25x20 fully-filled rectangle (500 pieces) — high interior fraction
        bbox = (0.1, 0.1, 0.9, 0.7)
        x1, y1, x2, y2 = bbox
        cols, rows = 25, 20
        filled = []
        for c in range(cols):
            for r in range(rows):
                fx = x1 + (c + 0.5) * (x2 - x1) / cols
                fy = y1 + (r + 0.5) * (y2 - y1) / rows
                filled.append((fx, fy))
        m = self._mapper_with_growth_history(
            500, [(10.0, 2, bbox), (1800.0, len(filled), bbox)])
        m.tracks[1]["history_members"] = [filled, filled]
        _, milestones = m.finalize(n_frames=1000)
        types = [ms["type"] for ms in milestones]
        self.assertNotIn("frame_complete", types)

    def test_islands_merged_event_becomes_milestone(self):
        from puzzle_clusters import ClusterMapper
        m = ClusterMapper(W=200, H=200, num_pieces=100, eff_fps=10.0)
        bbox = (0.2, 0.2, 0.7, 0.7)
        m.tracks[1] = {
            "id": 1, "born_t": 10.0, "last_seen_t": 100.0,
            "initial_count": 2, "peak_count": 30, "final_count": 30,
            "bbox_at_peak": bbox, "last_bbox": bbox,
            "history_seconds": [10.0, 100.0],
            "history_counts": [2, 30],
            "history_bboxes": [bbox, bbox],
            "consecutive_misses": 0, "dormant": False,
            "merged_into": None, "merged_from": [],
        }
        m.tracks[2] = {
            "id": 2, "born_t": 50.0, "last_seen_t": 200.0,
            "initial_count": 2, "peak_count": 60, "final_count": 60,
            "bbox_at_peak": bbox, "last_bbox": bbox,
            "history_seconds": [50.0, 200.0],
            "history_counts": [2, 60],
            "history_bboxes": [bbox, bbox],
            "consecutive_misses": 0, "dormant": False,
            "merged_into": None, "merged_from": [1],
        }
        m.next_id = 3
        m.events.append(("merged", 180.0,
                         {"from_id": 1, "into_id": 2,
                          "from_peak": 30, "into_peak": 60}))
        _, milestones = m.finalize(n_frames=1000)
        types = [ms["type"] for ms in milestones]
        self.assertIn("islands_merged", types)

    def test_sort_pile_cleared_fires(self):
        from puzzle_clusters import ClusterMapper
        m = ClusterMapper(W=200, H=200, num_pieces=100, eff_fps=10.0)
        bbox_pile = (0.0, 0.0, 0.3, 0.4)
        bbox_asm = (0.4, 0.2, 0.9, 0.8)
        # Pile shrinks 200 -> 15 (below 20% of peak=200, i.e. 40)
        m.tracks[1] = {
            "id": 1, "born_t": 0.0, "last_seen_t": 900.0,
            "initial_count": 200, "peak_count": 200, "final_count": 15,
            "bbox_at_peak": bbox_pile, "last_bbox": bbox_pile,
            "history_seconds": [0.0, 300.0, 600.0, 900.0],
            "history_counts": [200, 100, 50, 15],
            "history_bboxes": [bbox_pile] * 4,
            "consecutive_misses": 0, "dormant": False,
            "merged_into": None, "merged_from": [],
        }
        m.tracks[2] = {
            "id": 2, "born_t": 100.0, "last_seen_t": 1800.0,
            "initial_count": 2, "peak_count": 80, "final_count": 80,
            "bbox_at_peak": bbox_asm, "last_bbox": bbox_asm,
            "history_seconds": [100.0, 1800.0],
            "history_counts": [2, 80],
            "history_bboxes": [bbox_asm] * 2,
            "consecutive_misses": 0, "dormant": False,
            "merged_into": None, "merged_from": [],
        }
        m.next_id = 3
        _, milestones = m.finalize(n_frames=2000)
        types = [ms["type"] for ms in milestones]
        self.assertIn("sort_pile_cleared", types)
        sp = next(ms for ms in milestones if ms["type"] == "sort_pile_cleared")
        # Pile drops below 20% of peak (= 40) somewhere between count=50 and count=15
        # Expected at t=900 (where count dropped to 15).
        self.assertEqual(sp["details"]["count"], 15)


    def test_islands_merged_suppressed_for_tiny_clusters(self):
        from puzzle_clusters import ClusterMapper
        m = ClusterMapper(W=200, H=200, num_pieces=100, eff_fps=10.0)
        bbox = (0.2, 0.2, 0.7, 0.7)
        # Both clusters have peak_count well below ISLAND_MIN_PEAK (10)
        m.tracks[1] = {
            "id": 1, "born_t": 10.0, "last_seen_t": 100.0,
            "initial_count": 2, "peak_count": 4, "final_count": 4,
            "bbox_at_peak": bbox, "last_bbox": bbox,
            "history_seconds": [10.0, 100.0],
            "history_counts": [2, 4],
            "history_bboxes": [bbox, bbox],
            "consecutive_misses": 0, "dormant": False,
            "merged_into": None, "merged_from": [],
        }
        m.tracks[2] = {
            "id": 2, "born_t": 50.0, "last_seen_t": 200.0,
            "initial_count": 2, "peak_count": 6, "final_count": 6,
            "bbox_at_peak": bbox, "last_bbox": bbox,
            "history_seconds": [50.0, 200.0],
            "history_counts": [2, 6],
            "history_bboxes": [bbox, bbox],
            "consecutive_misses": 0, "dormant": False,
            "merged_into": None, "merged_from": [1],
        }
        m.next_id = 3
        m.events.append(("merged", 180.0,
                         {"from_id": 1, "into_id": 2,
                          "from_peak": 4, "into_peak": 6}))
        _, milestones = m.finalize(n_frames=1000)
        types = [ms["type"] for ms in milestones]
        self.assertNotIn("islands_merged", types)


class TestAssemblyOnset(unittest.TestCase):

    def test_clean_phase_boundary_high_confidence(self):
        from puzzle_clusters import assembly_onset
        secs = [0.0, 20.0, 40.0, 60.0, 80.0, 100.0]
        cnts = [0,    1,    1,    5,    20,   40]
        t, conf, note = assembly_onset(secs, cnts)
        self.assertEqual(t, 60.0)
        self.assertEqual(conf, "high")

    def test_gradual_ramp_is_medium(self):
        from puzzle_clusters import assembly_onset
        secs = [0.0, 20.0, 40.0, 60.0]
        cnts = [4,   8,    14,   22]
        t, conf, note = assembly_onset(secs, cnts)
        self.assertEqual(t, 0.0)
        self.assertEqual(conf, "medium")

    def test_no_sustained_growth_unavailable(self):
        from puzzle_clusters import assembly_onset
        secs = [0.0, 20.0, 40.0, 60.0]
        cnts = [0,   1,    0,    1]
        t, conf, note = assembly_onset(secs, cnts)
        self.assertIsNone(t)
        self.assertEqual(conf, "unavailable")

    def test_empty_history_unavailable(self):
        from puzzle_clusters import assembly_onset
        t, conf, note = assembly_onset([], [])
        self.assertIsNone(t)
        self.assertEqual(conf, "unavailable")


class TestPlacementSplits(unittest.TestCase):
    pass


class TestDetectStalls(unittest.TestCase):
    pass


if __name__ == "__main__":
    unittest.main()
